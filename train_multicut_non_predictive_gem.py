# ============================================================
# MULTICUT COSTS + GAEC + CODEC-AWARE REINFORCE
#
# Stream codec (lossless):
#   - Labels: zlib-compressed label map
#   - Residuals: JPEG-LS / LOCO-I style MED predictor + context-adaptive Rice
#
# Key speed changes vs naive Python:
#   (1) Threshold selection:
#       - Evaluate many thresholds on a downsample proxy (cheap)
#       - Evaluate only TOP-K thresholds on full-res (expensive)
#   (2) JPEG-LS style coder:
#       - Unary coding uses bulk zero emission (byte-level), not per-bit loops
#       - No per-pixel np.array allocations; scalar neighbor reads only
#       - Faster Rice k adaptation (LOCO-I-like): choose smallest k s.t. (N<<k) >= A
#   (3) Validation speed:
#       - Full images are center-cropped to FULL_VALIDATE_CROP size during validation
#       - Final tests still use full images
# ============================================================

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import glob
import time
import io
import struct
import zlib
import heapq
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


# =========================
# CONFIG
# =========================
class Config:
    DATA_DIR = "./DIV2K_train_HR"

    PATCH = 64
    BATCH = 2
    EPOCHS = 80
    LR = 5e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VAL_EVERY = 5

    # DataLoader
    NUM_WORKERS = 8
    PIN_MEMORY = True
    DROP_LAST = True

    # REINFORCE
    SAMPLES_PER_IMAGE = 1
    BASELINE_DECAY = 0.99

    # Edge-cost sampling
    COST_TEMPERATURE = 1.0
    LOGIT_CLAMP = 8.0
    COST_STD = 0.12
    SATURATION_BETA = 0.02

    # Patch threshold search to stabilize region count (training only)
    TARGET_REGIONS_PATCH = 120
    THR_SEARCH_ITERS = 5
    THR_SEARCH_RANGE = (-1.0, 1.0)
    MERGE_THRESHOLD_PATCH_DEFAULT = -0.2

    # Region penalty (training only)
    REGION_MIN_PATCH = 20
    REGION_MAX_PATCH = 800
    REGION_PENALTY_LAMBDA = 80.0

    # Full-image threshold candidates (proxy ranks them; full evaluates only TOP-K)
    FULL_THRESHOLDS_TO_TRY = [-0.95, -0.8, -0.65, -0.5, -0.35, -0.2, -0.05, 0.0]
    MAX_FULL_THRESH_TRIES = 8  # applies to list above
    PROXY_MAX_SIDE = 512       # downsample full image so max(H,W)<=512 for proxy
    PROXY_TOPK = 2             # evaluate only these many thresholds on full-res

    # Validation speed: crop full images to this size (set None to disable)
    FULL_VALIDATE_CROP = 512   # center-crop validation images to 512×512 for speed

    # Guardrails
    MAX_REGIONS_SKIP_POST = 200_000

    # Tiny region merge (full-res)
    INFER_TINY_REGION_SIZE = 256
    TINY_MERGE_ITERS = 2
    MAX_TINY_REGIONS_TO_MERGE = 120_000

    # Tiny region merge (training patches)
    TRAIN_TINY_REGION_SIZE = 128

    # Labels coding
    ZLIB_LEVEL = 9

    # JPEG-LS style residual coder
    JL_G_THR = 4            # gradient quantization threshold
    JL_NUM_CTX = 27 * 4     # 27 gradient bins * 4 segmentation flags
    JL_MAX_K = 10
    JL_RESCALE_AT = 1 << 15

    # Debug: histograms are expensive to compute (not printing; updating counts)
    JL_COLLECT_KHIST = True  # set False for speed


# =========================
# LOGGING
# =========================
_log_file = None

def init_log_file():
    global _log_file
    os.makedirs("./logs", exist_ok=True)
    fn = f"./logs/log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    _log_file = open(fn, "w", buffering=1)
    log(f"Log file created: {fn}")
    return fn

def log(msg: str):
    global _log_file
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file is not None:
        _log_file.write(line + "\n")
        _log_file.flush()
        try:
            os.fsync(_log_file.fileno())
        except Exception:
            pass

def close_log_file():
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None


# =========================
# DATASET
# =========================
class DIV2KDataset(Dataset):
    def __init__(self, root_dir, patch_size):
        os.makedirs(root_dir, exist_ok=True)
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.png")))
        if len(self.files) == 0:
            self.files = ["__dummy__"] * 200

        self.patch_size = patch_size
        self.t = transforms.Compose([
            transforms.RandomCrop(patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
        self.test_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.files[idx] == "__dummy__":
            return torch.rand(3, self.patch_size, self.patch_size)
        try:
            img = Image.open(self.files[idx]).convert("RGB")
            return self.t(img)
        except Exception:
            return torch.rand(3, self.patch_size, self.patch_size)

    def load_full_image(self, idx):
        if self.files[idx] == "__dummy__":
            return torch.rand(3, 256, 256)
        try:
            img = Image.open(self.files[idx]).convert("RGB")
            return self.test_transform(img)
        except Exception:
            return torch.rand(3, 256, 256)


# =========================
# IMAGE UTILS
# =========================
def center_crop_tensor(img_tensor, crop_size):
    """Center crop [3,H,W] tensor to [3,crop_size,crop_size]. If already smaller, return as-is."""
    _, H, W = img_tensor.shape
    if H <= crop_size and W <= crop_size:
        return img_tensor, H, W
    
    crop_h = min(crop_size, H)
    crop_w = min(crop_size, W)
    
    y0 = (H - crop_h) // 2
    x0 = (W - crop_w) // 2
    
    cropped = img_tensor[:, y0:y0+crop_h, x0:x0+crop_w]
    return cropped, crop_h, crop_w


# =========================
# MODELS
# =========================
class CompressionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):
        f1 = F.relu(self.bn1(self.conv1(x)))
        f2 = F.relu(self.bn2(self.conv2(f1)))
        f3 = F.relu(self.bn3(self.conv3(f2)))
        return f3

class EdgeCostPredictor(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * feat_dim + 3, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, feat_u, feat_v, rgb_diff):
        x = torch.cat([feat_u, feat_v, rgb_diff], dim=1)
        return self.net(x)


# =========================
# GRID EDGES
# =========================
def grid_edges(H, W, device):
    rs = torch.arange(H, device=device)
    cs = torch.arange(W, device=device)
    r, c = torch.meshgrid(rs, cs, indexing="ij")

    mh = c < (W - 1)
    uh = (r[mh] * W + c[mh]).reshape(-1)
    vh = (r[mh] * W + (c[mh] + 1)).reshape(-1)

    mv = r < (H - 1)
    uv = (r[mv] * W + c[mv]).reshape(-1)
    vv = ((r[mv] + 1) * W + c[mv]).reshape(-1)

    u = torch.cat([uh, uv], dim=0)
    v = torch.cat([vh, vv], dim=0)
    return u, v


# =========================
# MERGE SMALL REGIONS (FAST, GUARDED)
# =========================
def merge_small_regions_fast(labels2d, min_size=256, max_iters=2, verbose=True, max_tiny_regions=120_000):
    H, W = labels2d.shape
    labels = labels2d.astype(np.int32, copy=False)

    for it in range(max_iters):
        flat = labels.reshape(-1)
        K = int(flat.max()) + 1
        sizes = np.bincount(flat, minlength=K)
        tiny = np.where((sizes > 0) & (sizes < min_size))[0]

        if tiny.size == 0:
            if verbose:
                log(f"[MERGE] No tiny regions found (all >= {min_size} pixels)")
            return labels

        if verbose:
            tiny_pixels = int(sizes[tiny].sum())
            tiny_pct = 100.0 * tiny_pixels / float(H * W)
            log(f"[MERGE] Found {tiny.size} tiny regions (< {min_size} pixels)")
            log(f"[MERGE] Tiny regions contain: {tiny_pixels:,} pixels ({tiny_pct:.1f}% of image)")

        if tiny.size > max_tiny_regions:
            if verbose:
                log(f"[MERGE] SKIP tiny-merge (tiny regions {tiny.size:,} > guard {max_tiny_regions:,})")
            return labels

        votes = defaultdict(lambda: defaultdict(int))

        # Horizontal boundaries
        a = labels[:, :-1].ravel()
        b = labels[:, 1:].ravel()
        m = (a != b)
        a = a[m]; b = b[m]
        for x, y in zip(a.tolist(), b.tolist()):
            votes[x][y] += 1
            votes[y][x] += 1

        # Vertical boundaries
        a = labels[:-1, :].ravel()
        b = labels[1:, :].ravel()
        m = (a != b)
        a = a[m]; b = b[m]
        for x, y in zip(a.tolist(), b.tolist()):
            votes[x][y] += 1
            votes[y][x] += 1

        remap = np.arange(K, dtype=np.int32)
        merged_count = 0

        tiny_set = set(int(t) for t in tiny.tolist())
        for r in tiny_set:
            if sizes[r] == 0:
                continue
            if r not in votes or len(votes[r]) == 0:
                continue

            best_nb = None
            best_cnt = -1
            best_size = -1
            for nb, cnt in votes[r].items():
                nb_size = int(sizes[nb])
                if (cnt > best_cnt) or (cnt == best_cnt and nb_size > best_size):
                    best_nb = nb
                    best_cnt = cnt
                    best_size = nb_size

            if best_nb is not None and best_nb != r:
                remap[r] = int(best_nb)
                merged_count += 1

        labels = remap[labels]

        uniq = np.unique(labels)
        lut = np.zeros(int(uniq.max()) + 1, dtype=np.int32)
        lut[uniq] = np.arange(uniq.size, dtype=np.int32)
        labels = lut[labels]

        if verbose:
            log(f"[MERGE] Iter {it}: merged {merged_count} tiny regions -> now {int(labels.max())+1:,} regions")

    if verbose:
        log(f"[MERGE] Reached max_iters={max_iters}. Final regions: {int(labels.max())+1:,}")
    return labels


# =========================
# GAEC (ADDITIVE)
# =========================
def gaec_additive(num_nodes, u_np, v_np, cost_np, merge_threshold=0.0):
    if len(u_np) == 0:
        return np.arange(num_nodes, dtype=np.int32), 0, {}

    maxidx = max(int(u_np.max()), int(v_np.max()))
    if maxidx >= num_nodes:
        num_nodes = maxidx + 1

    fulladj = {i: {} for i in range(num_nodes)}
    for i in range(len(u_np)):
        U = int(u_np[i]); V = int(v_np[i]); C = float(cost_np[i])
        if U == V:
            continue
        fulladj[U][V] = fulladj[U].get(V, 0.0) + C
        fulladj[V][U] = fulladj[V].get(U, 0.0) + C

    parent = np.arange(num_nodes, dtype=np.int32)

    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for p in path:
            parent[p] = i
        return i

    heap = []
    for u in range(num_nodes):
        for v, w in fulladj[u].items():
            if u < v and w > merge_threshold:
                heapq.heappush(heap, (-w, u, v))

    mergescount = 0
    edgesinheap = len(heap)

    while heap:
        negw, u, v = heapq.heappop(heap)
        w = -negw
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        if rv not in fulladj[ru]:
            continue
        currw = fulladj[ru][rv]
        if abs(currw - w) > 1e-6:
            continue
        if w <= merge_threshold:
            continue

        parent[rv] = ru
        mergescount += 1

        for nb, wv in list(fulladj[rv].items()):
            if nb == ru:
                if rv in fulladj[ru]:
                    del fulladj[ru][rv]
                continue

            if rv in fulladj[nb]:
                del fulladj[nb][rv]

            old = fulladj[ru].get(nb, 0.0)
            neww = old + wv
            fulladj[ru][nb] = neww
            fulladj[nb][ru] = neww

            if neww > merge_threshold:
                a, b = (ru, nb) if ru < nb else (nb, ru)
                heapq.heappush(heap, (-neww, a, b))

        fulladj[rv].clear()

    labels = np.array([find(i) for i in range(num_nodes)], dtype=np.int32)
    uniq = np.unique(labels)
    mapping = {int(x): i for i, x in enumerate(uniq)}
    finallabels = np.array([mapping[int(x)] for x in labels], dtype=np.int32)

    edgestats = {
        "total_edges": int(len(u_np)),
        "edges_in_heap_init": int(edgesinheap),
        "merges": int(mergescount),
        "final_regions": int(len(uniq)),
        "positive_edges": int((cost_np > merge_threshold).sum()),
        "nonpositive_edges": int((cost_np <= merge_threshold).sum()),
    }
    return finallabels, mergescount, edgestats


# =========================
# PATCH THRESHOLD SEARCH (REGION COUNT)
# =========================
def find_threshold_for_target_regions(cost_np, u_np, v_np, num_nodes, target_regions,
                                     lo=-1.0, hi=1.0, iters=7):
    if np.allclose(cost_np.min(), cost_np.max()):
        return float(Config.MERGE_THRESHOLD_PATCH_DEFAULT)

    best_thr = float(Config.MERGE_THRESHOLD_PATCH_DEFAULT)
    best_err = 1e18

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        labels, _, st = gaec_additive(num_nodes, u_np, v_np, cost_np, merge_threshold=mid)
        K = int(st.get("final_regions", int(labels.max()) + 1))
        err = abs(K - target_regions)
        if err < best_err:
            best_err = err
            best_thr = mid
        if K > target_regions:
            hi = mid
        else:
            lo = mid

    return float(best_thr)


# =========================
# JPEG-LS STYLE CODEC (FAST BIT I/O + MED + CTX RICE)
# =========================
class BitWriter:
    __slots__ = ("buf", "acc", "nbits")
    def __init__(self):
        self.buf = bytearray()
        self.acc = 0
        self.nbits = 0

    def write_bit(self, b: int):
        self.acc = (self.acc << 1) | (b & 1)
        self.nbits += 1
        if self.nbits == 8:
            self.buf.append(self.acc & 0xFF)
            self.acc = 0
            self.nbits = 0

    def write_bits(self, v: int, n: int):
        for i in range(n - 1, -1, -1):
            self.write_bit((v >> i) & 1)

    def write_zero_bits(self, n: int):
        if n <= 0:
            return

        if self.nbits != 0:
            take = min(n, 8 - self.nbits)
            self.acc <<= take
            self.nbits += take
            n -= take
            if self.nbits == 8:
                self.buf.append(self.acc & 0xFF)
                self.acc = 0
                self.nbits = 0

        if n >= 8:
            nb = n // 8
            self.buf.extend(b"\x00" * nb)
            n -= nb * 8

        if n > 0:
            self.acc <<= n
            self.nbits += n
            if self.nbits == 8:
                self.buf.append(self.acc & 0xFF)
                self.acc = 0
                self.nbits = 0

    def write_unary_zeros_then_one(self, q: int):
        self.write_zero_bits(q)
        self.write_bit(1)

    def finish(self) -> bytes:
        if self.nbits != 0:
            self.acc <<= (8 - self.nbits)
            self.buf.append(self.acc & 0xFF)
            self.acc = 0
            self.nbits = 0
        return bytes(self.buf)


class BitReader:
    __slots__ = ("data", "i", "acc", "nbits")
    def __init__(self, data: bytes):
        self.data = data
        self.i = 0
        self.acc = 0
        self.nbits = 0

    def _refill(self):
        if self.i >= len(self.data):
            self.acc = 0
            self.nbits = 8
            return
        self.acc = self.data[self.i]
        self.i += 1
        self.nbits = 8

    def read_bit(self) -> int:
        if self.nbits == 0:
            self._refill()
        self.nbits -= 1
        return (self.acc >> self.nbits) & 1

    def read_bits(self, n: int) -> int:
        v = 0
        for _ in range(n):
            v = (v << 1) | self.read_bit()
        return v

    def read_unary_zeros_then_one(self) -> int:
        q = 0
        while True:
            if self.nbits == 0:
                self._refill()

            seg = self.acc & ((1 << self.nbits) - 1)
            if seg == 0:
                q += self.nbits
                self.nbits = 0
                continue

            lz = self.nbits - seg.bit_length()
            q += lz
            self.nbits -= (lz + 1)
            return q


def med_predictor(A, B, C):
    if C >= A and C >= B:
        return A if A < B else B
    if C <= A and C <= B:
        return A if A > B else B
    return A + B - C


def _q3(g, T):
    if g < -T: return -1
    if g >  T: return  1
    return 0


def jls_context_id(labels, y, x, A, B, C, D, T):
    # 27 gradient contexts
    g1 = int(D) - int(B)
    g2 = int(B) - int(C)
    g3 = int(C) - int(A)
    q1 = _q3(g1, T) + 1
    q2 = _q3(g2, T) + 1
    q3 = _q3(g3, T) + 1
    base = (q1 * 9) + (q2 * 3) + q3  # 0..26

    # segmentation flags (affect entropy context)
    left_same = 1 if (x > 0 and labels[y, x] == labels[y, x - 1]) else 0
    up_same = 1 if (y > 0 and labels[y, x] == labels[y - 1, x]) else 0
    flags = left_same | (up_same << 1)  # 0..3

    return base * 4 + flags  # 0..107


def rice_map_signed(err: int) -> int:
    return (err << 1) if err >= 0 else ((-err << 1) - 1)


def rice_unmap_signed(u: int) -> int:
    return (u >> 1) if (u & 1) == 0 else -(u >> 1) - 1


def rice_k_from_stats(sum_abs: np.ndarray, cnt: np.ndarray, ctx: int, ch: int) -> int:
    # LOCO-I-like: pick smallest k with (N << k) >= A, where A=sum_abs, N=count.
    # This is fast (no floats) and adapts more smoothly than bit_length(mean).
    A = int(sum_abs[ctx, ch])
    N = int(cnt[ctx, ch])
    if N <= 0:
        return 0

    k = 0
    while k < Config.JL_MAX_K and ((N << k) < A):
        k += 1
    return k


def rice_write(bw: BitWriter, u: int, k: int):
    q = u >> k
    r = u & ((1 << k) - 1) if k > 0 else 0
    bw.write_unary_zeros_then_one(q)
    if k > 0:
        bw.write_bits(r, k)


def rice_read(br: BitReader, k: int) -> int:
    q = br.read_unary_zeros_then_one()
    r = br.read_bits(k) if k > 0 else 0
    return (q << k) | r


def labels_to_bytes(labels_2d, use_u32=False):
    if use_u32:
        return labels_2d.astype(np.uint32, copy=False).tobytes(order="C")
    return labels_2d.astype(np.uint16, copy=False).tobytes(order="C")


def labels_from_bytes(buf, H, W, use_u32=False):
    if use_u32:
        return np.frombuffer(buf, dtype=np.uint32).reshape(H, W).astype(np.int32)
    return np.frombuffer(buf, dtype=np.uint16).reshape(H, W).astype(np.int32)


def encode_segpred_stream_jpegls(img_u8, labels_2d, zlevel=9, collect_stats=False):
    H, W, C = img_u8.shape
    assert C == 3

    labels = labels_2d.astype(np.int32, copy=False)
    num_regions = int(labels.max()) + 1

    use_u32 = (num_regions > 65535)
    labels_raw = labels_to_bytes(labels, use_u32=use_u32)
    labels_z = zlib.compress(labels_raw, zlevel)

    sum_abs = np.ones((Config.JL_NUM_CTX, 3), dtype=np.int64) * 4
    cnt = np.ones((Config.JL_NUM_CTX, 3), dtype=np.int64)

    bw = BitWriter()
    recon = np.zeros_like(img_u8, dtype=np.uint8)

    do_hist = bool(collect_stats) and bool(Config.JL_COLLECT_KHIST)
    if do_hist:
        k_hist = np.zeros((3, Config.JL_MAX_K + 1), dtype=np.int64)

    T = int(Config.JL_G_THR)

    # Scalar neighbor reads only (no np.array allocations)
    for y in range(H):
        for x in range(W):
            for ch in range(3):
                A = int(recon[y, x - 1, ch]) if x > 0 else 0
                B = int(recon[y - 1, x, ch]) if y > 0 else 0
                Cn = int(recon[y - 1, x - 1, ch]) if (y > 0 and x > 0) else 0
                D = int(recon[y - 1, x + 1, ch]) if (y > 0 and x < W - 1) else 0

                ctx = jls_context_id(labels, y, x, A, B, Cn, D, T)
                k = rice_k_from_stats(sum_abs, cnt, ctx, ch)

                predv = med_predictor(A, B, Cn)
                pix = int(img_u8[y, x, ch])
                err = pix - predv

                uval = rice_map_signed(int(err))
                rice_write(bw, uval, k)

                recon[y, x, ch] = (predv + err) & 255

                ae = abs(int(err))
                sum_abs[ctx, ch] += ae
                cnt[ctx, ch] += 1
                if cnt[ctx, ch] >= Config.JL_RESCALE_AT:
                    sum_abs[ctx, ch] >>= 1
                    cnt[ctx, ch] >>= 1
                    if cnt[ctx, ch] == 0:
                        cnt[ctx, ch] = 1

                if do_hist:
                    k_hist[ch, k] += 1

    residual_bytes = bw.finish()

    magic = b"MC02"
    flags = 1 if use_u32 else 0
    header = struct.pack("<4sHHBBII", magic, H, W, C, flags, len(labels_z), len(residual_bytes))
    bitstream = header + labels_z + residual_bytes

    meta = {
        "H": H, "W": W, "C": C,
        "labels_z_len": len(labels_z),
        "residual_len": len(residual_bytes),
        "total": len(bitstream),
        "num_regions": num_regions,
        "labels_u32": use_u32,
        "codec": "MC02_JPEG_LS_STYLE",
    }
    if do_hist:
        meta["k_hist"] = k_hist
    return bitstream, meta


def decode_segpred_stream_jpegls(bitstream):
    magic, H, W, C, flags, labels_len, residual_len = struct.unpack("<4sHHBBII", bitstream[:18])
    assert magic == b"MC02"
    assert C == 3

    use_u32 = bool(flags & 1)
    off = 18
    labels_z = bitstream[off:off + labels_len]; off += labels_len
    residual_bytes = bitstream[off:off + residual_len]; off += residual_len

    labels_raw = zlib.decompress(labels_z)
    labels = labels_from_bytes(labels_raw, H, W, use_u32=use_u32)

    sum_abs = np.ones((Config.JL_NUM_CTX, 3), dtype=np.int64) * 4
    cnt = np.ones((Config.JL_NUM_CTX, 3), dtype=np.int64)

    br = BitReader(residual_bytes)
    recon = np.zeros((H, W, 3), dtype=np.uint8)

    T = int(Config.JL_G_THR)

    for y in range(H):
        for x in range(W):
            for ch in range(3):
                A = int(recon[y, x - 1, ch]) if x > 0 else 0
                B = int(recon[y - 1, x, ch]) if y > 0 else 0
                Cn = int(recon[y - 1, x - 1, ch]) if (y > 0 and x > 0) else 0
                D = int(recon[y - 1, x + 1, ch]) if (y > 0 and x < W - 1) else 0

                ctx = jls_context_id(labels, y, x, A, B, Cn, D, T)
                k = rice_k_from_stats(sum_abs, cnt, ctx, ch)

                predv = med_predictor(A, B, Cn)
                uval = rice_read(br, k)
                err = rice_unmap_signed(int(uval))

                recon[y, x, ch] = (predv + err) & 255

                ae = abs(int(err))
                sum_abs[ctx, ch] += ae
                cnt[ctx, ch] += 1
                if cnt[ctx, ch] >= Config.JL_RESCALE_AT:
                    sum_abs[ctx, ch] >>= 1
                    cnt[ctx, ch] >>= 1
                    if cnt[ctx, ch] == 0:
                        cnt[ctx, ch] = 1

    return recon, labels


def compute_compression_size_segpred(img_u8, labels_2d, verbose=True):
    collect_stats = bool(verbose) and bool(Config.JL_COLLECT_KHIST)
    stream, meta = encode_segpred_stream_jpegls(img_u8, labels_2d, zlevel=Config.ZLIB_LEVEL, collect_stats=collect_stats)
    size = meta["total"]

    if verbose:
        segsize = meta["labels_z_len"]
        ressize = meta["residual_len"]
        overhead = size - segsize - ressize
        bpp = (8.0 * ressize) / float(img_u8.shape[0] * img_u8.shape[1])
        log(f"[CODEC] {meta['codec']} | ctx={Config.JL_NUM_CTX} | g_thr={Config.JL_G_THR} | max_k={Config.JL_MAX_K}")
        log(f"[SIZE BREAKDOWN] Total={size:,} bytes | Seg={segsize:,} | Residual={ressize:,} | Overhead={overhead:,} | Residual_bpp={bpp:.3f}")
        if collect_stats and ("k_hist" in meta):
            kh = meta["k_hist"]
            for ch, name in enumerate(["R", "G", "B"]):
                top = np.argsort(-kh[ch])[:4]
                top_str = ", ".join([f"k={int(k)}:{int(kh[ch, k]):,}" for k in top if kh[ch, k] > 0])
                log(f"[RICE K HIST] {name}: {top_str}")

    return size, stream, meta


# =========================
# PNG BASELINE
# =========================
def save_png_bytes_from_u8(img_u8):
    buf = io.BytesIO()
    try:
        Image.fromarray(img_u8).save(buf, format="PNG", optimize=True, compress_level=9)
    except TypeError:
        Image.fromarray(img_u8).save(buf, format="PNG", optimize=True, compresslevel=9)
    return buf.getvalue()


# =========================
# VISUALS
# =========================
def boundary_mask(labels2d):
    b = np.zeros(labels2d.shape, dtype=bool)
    b[:, :-1] |= (labels2d[:, :-1] != labels2d[:, 1:])
    b[:-1, :] |= (labels2d[:-1, :] != labels2d[1:, :])
    return b

def visualize_segmentation(img_u8, labels2d, title="Segmentation"):
    K = int(labels2d.max()) + 1
    b = boundary_mask(labels2d)
    overlay = img_u8.copy()
    overlay[b] = np.array([255, 0, 0], dtype=np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_u8); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(overlay); axes[1].set_title(f"Boundaries overlay (K={K})"); axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(f"segmentation_gem_{title}.png", dpi=120, bbox_inches="tight")
    plt.close()
    log(f"Saved: segmentation_gem_{title}.png")

def visualize_reconstruction_from_stream(original_u8, labels2d, bitstream, title="Reconstruction"):
    recon_u8, labels_dec = decode_segpred_stream_jpegls(bitstream)
    ok = np.array_equal(recon_u8, original_u8)

    diff = original_u8.astype(np.float32) - recon_u8.astype(np.float32)
    mse = float(np.mean(diff ** 2))
    psnr = float("inf") if mse == 0.0 else float(20.0 * np.log10(255.0 / np.sqrt(mse)))
    mae = float(np.mean(np.abs(diff)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(original_u8); axes[0].set_title("Original", fontsize=12, fontweight="bold"); axes[0].axis("off")
    axes[1].imshow((labels2d * 53) % 255, cmap="tab20"); axes[1].set_title(f"Labels (K={int(labels2d.max())+1})", fontsize=12, fontweight="bold"); axes[1].axis("off")
    axes[2].imshow(recon_u8)
    psnr_text = "Lossless" if np.isinf(psnr) else f"{psnr:.2f} dB"
    axes[2].set_title(f"Reconstructed\nPSNR={psnr_text}, MAE={mae:.6f}", fontsize=12, fontweight="bold")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(f"reconstruction_gem_{title}.png", dpi=120, bbox_inches="tight")
    plt.close()

    same_labels = (labels_dec.shape == labels2d.shape) and np.array_equal(labels_dec.astype(np.int32), labels2d.astype(np.int32))
    log(f"Saved: reconstruction_gem_{title}.png")
    log(f" Reconstruction OK: {ok}")
    log(f" Labels match: {same_labels}")
    log(f" PSNR: {'Lossless' if np.isinf(psnr) else f'{psnr:.2f} dB'}")
    log(f" MAE: {mae:.6f}")
    return ok


# =========================
# THRESHOLD PROXY + FULL EVAL
# =========================
def make_proxy_tensor(img_tensor, max_side):
    H, W = int(img_tensor.shape[1]), int(img_tensor.shape[2])
    m = max(H, W)
    if m <= max_side:
        return img_tensor, H, W, 1.0
    scale = float(max_side) / float(m)
    Hp = max(16, int(round(H * scale)))
    Wp = max(16, int(round(W * scale)))
    imgp = F.interpolate(img_tensor.unsqueeze(0), size=(Hp, Wp), mode="area").squeeze(0)
    return imgp, Hp, Wp, scale


@torch.no_grad()
def compute_edge_costs_for_tensor(img_tensor, enc, pred, device):
    H, W = int(img_tensor.shape[1]), int(img_tensor.shape[2])
    img_device = img_tensor.unsqueeze(0).to(device)

    feat = enc(img_device)
    feat_flat = feat.permute(0, 2, 3, 1).reshape(1, -1, 128)

    u, v = grid_edges(H, W, device)
    rgb_flat = img_device.permute(0, 2, 3, 1).reshape(1, -1, 3)

    fu = feat_flat[0, u]
    fv = feat_flat[0, v]
    rgb_diff = rgb_flat[0, u] - rgb_flat[0, v]

    raw = pred(fu, fv, rgb_diff).squeeze(1)
    raw = (raw / float(Config.COST_TEMPERATURE)).clamp(-Config.LOGIT_CLAMP, Config.LOGIT_CLAMP)
    costs = torch.tanh(raw).cpu().numpy().astype(np.float32)

    img_u8 = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return costs, u.detach().cpu().numpy(), v.detach().cpu().numpy(), H, W, img_u8


def choose_threshold_full(costs_full, u_full_np, v_full_np, H, W, img_u8, proxy_pack=None):
    thrs = Config.FULL_THRESHOLDS_TO_TRY[:Config.MAX_FULL_THRESH_TRIES]

    # ---- Stage A: proxy ranking (cheap) ----
    shortlist = thrs
    if proxy_pack is not None:
        costs_p, u_p, v_p, Hp, Wp, imgp_u8, scale = proxy_pack
        proxy_scores = []

        log(f"[THR-PROXY] Proxy {Hp}×{Wp} (scale≈{scale:.3f}), evaluating {len(thrs)} thresholds")
        for thr in thrs:
            labels_1d, merges, _ = gaec_additive(Hp * Wp, u_p, v_p, costs_p, merge_threshold=float(thr))
            labels2d = labels_1d.reshape(Hp, Wp).astype(np.int32)

            min_size_p = max(16, int(round(Config.INFER_TINY_REGION_SIZE * (scale ** 2))))
            labels2d = merge_small_regions_fast(
                labels2d,
                min_size=min_size_p,
                max_iters=1,
                verbose=False,
                max_tiny_regions=min(30_000, Config.MAX_TINY_REGIONS_TO_MERGE),
            )

            seg_bytes, _, _ = compute_compression_size_segpred(imgp_u8, labels2d, verbose=False)
            K = int(labels2d.max()) + 1
            proxy_scores.append((int(seg_bytes), float(thr), int(K)))

        proxy_scores.sort(key=lambda x: x[0])
        shortlist = [t for (_, t, _) in proxy_scores[:Config.PROXY_TOPK]]

        log(f"[THR-PROXY] Best proxy bytes={proxy_scores[0][0]:,} at thr={proxy_scores[0][1]:.3f} (K={proxy_scores[0][2]:,})")
        log(f"[THR-PROXY] Shortlist TOP-{Config.PROXY_TOPK}: {', '.join([f'{t:.3f}' for t in shortlist])}")

    # ---- Stage B: full evaluation (expensive) ----
    best = None
    log(f"[THR-FULL] Evaluating {len(shortlist)} thresholds on FULL {H}×{W}")

    for thr in shortlist:
        t0 = time.time()
        labels_1d, merges, _ = gaec_additive(H * W, u_full_np, v_full_np, costs_full, merge_threshold=float(thr))
        gaec_t = time.time() - t0

        labels2d = labels_1d.reshape(H, W).astype(np.int32)
        K0 = int(labels2d.max()) + 1
        log(f"[INFERENCE] GAEC thr={thr:.3f} merges={merges:,} regions={K0:,} (t={gaec_t:.1f}s)")
        log(f"[POST-PROCESS] Starting with {K0:,} regions")

        if K0 > Config.MAX_REGIONS_SKIP_POST:
            log(f"[MERGE] SKIP tiny-merge (regions {K0:,} > cap {Config.MAX_REGIONS_SKIP_POST:,})")
            continue

        labels2d = merge_small_regions_fast(
            labels2d,
            min_size=int(Config.INFER_TINY_REGION_SIZE),
            max_iters=int(Config.TINY_MERGE_ITERS),
            verbose=True,
            max_tiny_regions=int(Config.MAX_TINY_REGIONS_TO_MERGE),
        )

        # For non-winning thresholds, keep codec logs off to reduce overhead.
        seg_bytes, stream, meta = compute_compression_size_segpred(img_u8, labels2d, verbose=False)
        log(f"[SIZE] thr={thr:.3f} Total={seg_bytes:,} (Seg={meta['labels_z_len']:,}, Resid={meta['residual_len']:,})")

        if best is None or seg_bytes < best["seg_bytes"]:
            best = {"thr": float(thr), "labels": labels2d, "seg_bytes": int(seg_bytes), "stream": stream}

    if best is None:
        thr = float(thrs[0])
        labels_1d, merges, _ = gaec_additive(H * W, u_full_np, v_full_np, costs_full, merge_threshold=float(thr))
        labels2d = labels_1d.reshape(H, W).astype(np.int32)
        seg_bytes, stream, _ = compute_compression_size_segpred(img_u8, labels2d, verbose=False)
        best = {"thr": thr, "labels": labels2d, "seg_bytes": int(seg_bytes), "stream": stream}

    # One detailed codec log for the winner
    _ = compute_compression_size_segpred(img_u8, best["labels"], verbose=True)
    log(f"[INFERENCE] Best thr={best['thr']:.3f} -> regions={int(best['labels'].max())+1:,}, bytes={best['seg_bytes']:,}")
    return best


# =========================
# REINFORCE LOSS (CODEC BYTES)
# =========================
def reinforce_codec_loss(img_batch, edge_cost_raw_batched, u, v, baseline_ema):
    device = img_batch.device
    B, C, H, W = img_batch.shape
    E = int(u.numel())

    u_np = u.detach().cpu().numpy()
    v_np = v.detach().cpu().numpy()

    raw = (edge_cost_raw_batched / float(Config.COST_TEMPERATURE)).clamp(-Config.LOGIT_CLAMP, Config.LOGIT_CLAMP)
    mu = torch.tanh(raw)

    saturation = (mu.abs() > 0.95).float().mean()
    saturation_penalty = Config.SATURATION_BETA * saturation

    sigma = float(Config.COST_STD)
    loss = torch.tensor(0.0, device=device)
    bytes_list = []

    log_norm = np.log(sigma * np.sqrt(2.0 * np.pi))

    for b in range(B):
        mu_b = mu[b]
        for _ in range(Config.SAMPLES_PER_IMAGE):
            eps = torch.randn_like(mu_b)
            sample = (mu_b + sigma * eps).clamp(-1.0, 1.0)

            z = (sample - mu_b) / sigma
            logp = (-0.5 * (z * z) - log_norm).sum() / float(E)

            cost_np = sample.detach().cpu().numpy().astype(np.float32)

            thr = find_threshold_for_target_regions(
                cost_np, u_np, v_np, H * W,
                target_regions=int(Config.TARGET_REGIONS_PATCH),
                lo=float(Config.THR_SEARCH_RANGE[0]),
                hi=float(Config.THR_SEARCH_RANGE[1]),
                iters=int(Config.THR_SEARCH_ITERS),
            )

            labels_1d, _, _ = gaec_additive(H * W, u_np, v_np, cost_np, merge_threshold=thr)
            labels2d = labels_1d.reshape(H, W).astype(np.int32)

            labels2d = merge_small_regions_fast(
                labels2d,
                min_size=int(Config.TRAIN_TINY_REGION_SIZE),
                max_iters=1,
                verbose=False,
                max_tiny_regions=10_000,
            )

            img_u8 = (img_batch[b].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
            seg_bytes, _, _ = compute_compression_size_segpred(img_u8, labels2d, verbose=False)

            K = int(labels2d.max()) + 1
            penalty = 0.0
            if K < Config.REGION_MIN_PATCH:
                penalty += Config.REGION_PENALTY_LAMBDA * float(Config.REGION_MIN_PATCH - K)
            if K > Config.REGION_MAX_PATCH:
                penalty += Config.REGION_PENALTY_LAMBDA * float(K - Config.REGION_MAX_PATCH)

            total_bytes = float(seg_bytes + penalty)
            bytes_list.append(total_bytes)

            advantage = float(total_bytes - baseline_ema)
            loss = loss + (torch.tensor(advantage, device=device).detach() * logp)

    denom = float(B * Config.SAMPLES_PER_IMAGE)
    policy_loss = loss / denom
    total_loss = policy_loss + saturation_penalty

    mean_bytes = float(np.mean(bytes_list)) if bytes_list else 0.0
    return total_loss, mean_bytes


# =========================
# INFERENCE (CALLED IN MAIN)
# =========================
@torch.no_grad()
def inference_on_image(img_tensor, enc, pred, H, W, device):
    enc.eval()
    pred.eval()

    costs_full, u_full_np, v_full_np, Hf, Wf, img_u8 = compute_edge_costs_for_tensor(img_tensor, enc, pred, device)
    log(f"[INFERENCE] Image: {Hf}×{Wf}, Edges: {len(u_full_np):,}")
    log(f"[INFERENCE] Costs Mean={costs_full.mean():.3f}, Std={costs_full.std():.3f}, Min={costs_full.min():.3f}, Max={costs_full.max():.3f}")

    # Proxy evaluation pack (downsample)
    proxy_pack = None
    img_proxy, Hp, Wp, scale = make_proxy_tensor(img_tensor, Config.PROXY_MAX_SIDE)
    if (Hp != Hf) or (Wp != Wf):
        costs_p, u_p, v_p, _, _, imgp_u8 = compute_edge_costs_for_tensor(img_proxy, enc, pred, device)
        proxy_pack = (costs_p, u_p, v_p, Hp, Wp, imgp_u8, scale)

    best = choose_threshold_full(costs_full, u_full_np, v_full_np, Hf, Wf, img_u8, proxy_pack=proxy_pack)

    png_bytes = len(save_png_bytes_from_u8(img_u8))
    ratio = float(best["seg_bytes"]) / float(png_bytes) if png_bytes > 0 else float("inf")
    gain = (1.0 - ratio) * 100.0

    log(f"[INFERENCE] Final result: {int(best['labels'].max())+1:,} regions, {int(best['seg_bytes']):,} bytes")
    log(f"[INFERENCE] Baseline PNG={png_bytes:,} bytes | Ratio={ratio:.3f} | Gain={gain:+.1f}%")

    return {
        "labels": best["labels"],
        "num_regions": int(best["labels"].max()) + 1,
        "seg_bytes": int(best["seg_bytes"]),
        "png_bytes": int(png_bytes),
        "ratio": float(ratio),
        "gain_pct": float(gain),
        "original_img": img_u8,
        "stream": best["stream"],
        "best_thr": float(best["thr"]),
    }


# =========================
# MAIN
# =========================
def main():
    init_log_file()
    log(f"Device: {Config.DEVICE}")
    log("=" * 70)
    log("MULTICUT COSTS + GAEC + CODEC-AWARE REINFORCE (JPEG-LS style residuals)")
    log("=" * 70)

    ds = DIV2KDataset(Config.DATA_DIR, Config.PATCH)
    log(f"Total images: {len(ds)}")

    n_val = max(5, int(len(ds) * 0.1))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])

    train_dl = DataLoader(
        train_ds,
        batch_size=Config.BATCH,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=Config.DROP_LAST,
    )

    enc = CompressionEncoder().to(Config.DEVICE)
    pred = EdgeCostPredictor().to(Config.DEVICE)

    opt = optim.AdamW(
        list(enc.parameters()) + list(pred.parameters()),
        lr=Config.LR,
        weight_decay=1e-5,
        betas=(0.9, 0.999),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=Config.EPOCHS)

    u_patch, v_patch = grid_edges(Config.PATCH, Config.PATCH, Config.DEVICE)
    log(f"Training: {Config.PATCH}×{Config.PATCH} patches, edges={int(u_patch.numel()):,}")
    log(f"Target regions (patch): {Config.TARGET_REGIONS_PATCH}")
    log(f"Train tiny merge size={Config.TRAIN_TINY_REGION_SIZE}, infer tiny merge size={Config.INFER_TINY_REGION_SIZE}")
    log(f"Cost std exploration={Config.COST_STD}")
    if Config.FULL_VALIDATE_CROP is not None:
        log(f"Validation crop size={Config.FULL_VALIDATE_CROP}×{Config.FULL_VALIDATE_CROP} (final tests use full images)")

    baseline_ema = 10_000.0
    best_ratio = float("inf")

    val_indices = list(range(min(3, len(ds))))

    for ep in range(Config.EPOCHS):
        enc.train()
        pred.train()

        total_loss = 0.0
        n_batches = 0

        for bi, img_batch in enumerate(train_dl):
            img_batch = img_batch.to(Config.DEVICE, non_blocking=True)
            B, C, H, W = img_batch.shape

            feat = enc(img_batch)
            feat_flat = feat.permute(0, 2, 3, 1).reshape(B, -1, 128)
            rgb_flat = img_batch.permute(0, 2, 3, 1).reshape(B, -1, 3)

            costs_raw_list = []
            for i in range(B):
                fu = feat_flat[i, u_patch]
                fv = feat_flat[i, v_patch]
                rgb_diff = rgb_flat[i, u_patch] - rgb_flat[i, v_patch]
                edge_raw = pred(fu, fv, rgb_diff).squeeze(1)
                costs_raw_list.append(edge_raw)
            edge_cost_raw_batched = torch.stack(costs_raw_list, dim=0)  # [B,E]

            loss, mean_bytes = reinforce_codec_loss(img_batch, edge_cost_raw_batched, u_patch, v_patch, baseline_ema)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(pred.parameters()), 1.0)
            opt.step()

            total_loss += float(loss.item())
            n_batches += 1

            baseline_ema = Config.BASELINE_DECAY * baseline_ema + (1.0 - Config.BASELINE_DECAY) * float(mean_bytes)

            if bi % 20 == 0:
                log(f"Ep{ep} B{bi}/{len(train_dl)} Loss={loss.item():.4f} MeanBytes={mean_bytes:.1f} Baseline={baseline_ema:.1f}")

        scheduler.step()
        log(f"Ep{ep} DONE - Loss: {total_loss / max(1, n_batches):.4f}")

        # ==================== VALIDATION (WITH CROP) ====================
        if (ep + 1) % Config.VAL_EVERY == 0 or ep == (Config.EPOCHS - 1):
            val_idx = val_indices[ep % len(val_indices)]
            log(f"\n[VALIDATION Ep{ep} on image {val_idx}]")

            val_img_tensor_full = ds.load_full_image(val_idx)
            
            # Crop if configured
            if Config.FULL_VALIDATE_CROP is not None:
                val_img_tensor, Hf, Wf = center_crop_tensor(val_img_tensor_full, Config.FULL_VALIDATE_CROP)
                log(f"[VALIDATION] Cropped from {val_img_tensor_full.shape[1]}×{val_img_tensor_full.shape[2]} to {Hf}×{Wf}")
            else:
                val_img_tensor = val_img_tensor_full
                Hf, Wf = int(val_img_tensor.shape[1]), int(val_img_tensor.shape[2])

            res = inference_on_image(val_img_tensor, enc, pred, Hf, Wf, Config.DEVICE)

            log(f"[VAL] Compressed={res['seg_bytes']:,} bytes, PNG={res['png_bytes']:,} bytes, Ratio={res['ratio']:.3f}, Gain={res['gain_pct']:+.1f}%\n")

            visualize_segmentation(res["original_img"], res["labels"], title=f"ep{ep}")
            visualize_reconstruction_from_stream(res["original_img"], res["labels"], res["stream"], title=f"ep{ep}")

            if res["ratio"] < best_ratio:
                best_ratio = res["ratio"]
                torch.save(enc.state_dict(), "enc_best.pth")
                torch.save(pred.state_dict(), "pred_best.pth")
                log(f"[BEST] New best ratio: {best_ratio:.3f}\n")

    torch.save(enc.state_dict(), "enc_final.pth")
    torch.save(pred.state_dict(), "pred_final.pth")
    log("Models saved!")

    # ==================== FINAL TESTS (FULL IMAGES) ====================
    log("\n" + "=" * 70)
    log("TESTING ON FULL IMAGES (no crop)")
    log("=" * 70)

    for idx in range(min(2, len(ds))):
        log(f"\n[TEST {idx}]")
        test_img = ds.load_full_image(idx)
        Ht, Wt = int(test_img.shape[1]), int(test_img.shape[2])
        res = inference_on_image(test_img, enc, pred, Ht, Wt, Config.DEVICE)

        log(f"[RESULT] Regions={res['num_regions']:,}, Compressed={res['seg_bytes']:,} bytes, PNG={res['png_bytes']:,} bytes, Ratio={res['ratio']:.3f}, Gain={res['gain_pct']:+.1f}%")
        visualize_segmentation(res["original_img"], res["labels"], title=f"test{idx}")
        visualize_reconstruction_from_stream(res["original_img"], res["labels"], res["stream"], title=f"test{idx}")


if __name__ == "__main__":
    try:
        main()
    finally:
        close_log_file()
