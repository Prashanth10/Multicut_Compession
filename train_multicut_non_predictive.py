"""
LEARNED COMPRESSION WITH ENTROPY MODELS - CORRECTED VERSION

KEY FIX:
  - decode_full_image_learned() now correctly handles zlib decompressed bytes
  - Uses np.frombuffer(error_bytes, dtype=np.uint8) instead of np.array()
  - This prevents the "invalid literal for int()" error

CHANGES FROM PREVIOUS VERSION:
  1. Fixed decode_full_image_learned() - line 410
  2. Uses np.frombuffer for fast, correct byte-to-array conversion
  3. All other functionality unchanged
"""

import os, glob, time, math, io, struct, zlib
import numpy as np

from PIL import Image

import heapq

from collections import Counter

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import transforms

# CONFIG

class Config:
    DATA_DIR = "./DIV2K_train_HR"
    PATCH = 64
    BATCH = 4
    EPOCHS = 100
    LR = 5e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VAL_EVERY = 5
    TARGET_REGIONS = 200
    TINY_REGION_SIZE = 1000
    ERROR_QUANTIZATION = 4  # Quantize errors by dividing by 4

_log_file = None

def init_log_file():
    global _log_file
    os.makedirs("./logs", exist_ok=True)
    log_filename = f"./logs/log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    _log_file = open(log_filename, 'w', buffering=1)
    log(f"Log file created: {log_filename}")
    return log_filename

def log(msg):
    global _log_file
    timestamp = time.strftime('%H:%M:%S')
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg, flush=True)
    if _log_file is not None:
        _log_file.write(log_msg + '\n')
        _log_file.flush()
        try:
            os.fsync(_log_file.fileno())
        except:
            pass

# DATASET

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

# ENCODER

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

# EDGE PREDICTOR

class EdgeCostPredictor(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*feat_dim + 3, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, feat_u, feat_v, rgb_diff):
        x = torch.cat([feat_u, feat_v, rgb_diff], dim=1)
        return self.net(x)

# ENTROPY MODEL
class EntropyModel(nn.Module):
    """Learns to predict error distribution per region."""
    def __init__(self, feat_dim=128, num_error_values=65):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_error_values = num_error_values
        
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_error_values)
        )
        
    def forward(self, region_feat):
        logits = self.net(region_feat)
        pmf = torch.softmax(logits, dim=-1)
        return pmf

# GRID GRAPH

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

# MERGE TINY REGIONS
def merge_tiny_regions(labels_2d, min_region_size=1000):
    H, W = labels_2d.shape
    labels_merged = labels_2d.copy()
    region_counts = Counter(labels_2d.flatten())
    tiny_regions = {r: size for r, size in region_counts.items() if size < min_region_size}

    if not tiny_regions:
        log(f"[MERGE] No tiny regions found (all >= {min_region_size} pixels)")
        return labels_2d

    tiny_pixels = sum(tiny_regions.values())
    tiny_pct = 100 * tiny_pixels / (H * W)
    log(f"[MERGE] Found {len(tiny_regions)} tiny regions (< {min_region_size} pixels)")
    log(f"[MERGE] Tiny regions contain: {tiny_pixels:,} pixels ({tiny_pct:.1f}% of image)")

    merged_count = 0
    for tiny_id in tiny_regions.keys():
        mask = (labels_2d == tiny_id)
        if not mask.any():
            continue

        neighbors = set()
        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    neighbor_label = labels_2d[ny, nx]
                    if neighbor_label != tiny_id:
                        neighbors.add(neighbor_label)

        if not neighbors:
            continue

        largest_neighbor = max(neighbors, key=lambda r: region_counts.get(r, 0))
        labels_merged[mask] = largest_neighbor
        merged_count += 1

    unique_labels = np.unique(labels_merged)
    relabel_map = {old: new for new, old in enumerate(unique_labels)}
    labels_final = np.array([relabel_map[l] for l in labels_merged.flatten()]).reshape(H, W).astype(np.int32)

    log(f"[MERGE] Merged {merged_count} tiny regions")
    log(f"[MERGE] Final regions: {len(unique_labels):,}")

    return labels_final

# RLE ENCODING
def encode_rle(flat_array):
    if len(flat_array) == 0:
        return bytes()
    runs = []
    current_val = flat_array[0]
    count = 1
    for i in range(1, len(flat_array)):
        if flat_array[i] == current_val and count < 255:
            count += 1
        else:
            runs.append((int(current_val), count))
            current_val = flat_array[i]
            count = 1
    runs.append((int(current_val), count))
    encoded = []
    for val, count in runs:
        encoded.append(val & 0xFF)
        encoded.append(count & 0xFF)
    return bytes(encoded)

def decode_rle(encoded_bytes, expected_length):
    decoded = []
    for i in range(0, len(encoded_bytes), 2):
        if i + 1 < len(encoded_bytes):
            val = encoded_bytes[i]
            count = encoded_bytes[i + 1]
            decoded.extend([val] * count)
    if len(decoded) < expected_length:
        decoded.extend([0] * (expected_length - len(decoded)))
    return np.array(decoded[:expected_length], dtype=np.int32)

# PAETH PREDICTOR
def paeth_predictor(a, b, c):
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    elif pb <= pc:
        return b
    else:
        return c

# ENCODE WITH QUANTIZED ERRORS + LEARNED ENTROPY
def encode_full_image_learned(img_u8, labels_2d, entropy_model=None, encoder=None):
    H, W, C = img_u8.shape
    num_regions = int(labels_2d.max()) + 1

    encoded_data = {
        'H': H, 'W': W, 'C': C,
        'num_regions': num_regions,
        'regions': {},
        'quantization': Config.ERROR_QUANTIZATION,
    }

    # Encode segmentation
    flat_labels = labels_2d.flatten().astype(np.int32)
    encoded_data['segmentation_rle'] = encode_rle(flat_labels)

    # Encode each region
    for region_id in range(num_regions):
        mask = (labels_2d == region_id)
        if not mask.any():
            continue

        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        bbox_img = img_u8[y0:y1+1, x0:x1+1].copy()
        bbox_h, bbox_w = y1 - y0 + 1, x1 - x0 + 1
        bbox_mask = mask[y0:y1+1, x0:x1+1]

        mask_rle = encode_rle(bbox_mask.astype(np.int32).flatten())

        # Extract region feature for entropy model
        if encoder is not None:
            with torch.no_grad():
                region_pixel = torch.tensor(bbox_img.astype(np.float32) / 255.0, dtype=torch.float32)
                region_pixel = region_pixel.permute(2, 0, 1).unsqueeze(0)
                if region_pixel.shape[2] < 32 or region_pixel.shape[3] < 32:
                    region_pixel = F.pad(region_pixel, (0, 32-region_pixel.shape[3], 0, 32-region_pixel.shape[2]))
                feat = encoder(region_pixel.to(Config.DEVICE))
                region_feat = feat.mean(dim=(2, 3))
        else:
            region_feat = torch.ones(1, 128)

        compressed_data = {}
        total_region_bits = 0
        
        for c in range(C):
            channel_2d = bbox_img[:, :, c]
            
            # Paeth prediction
            errors = []
            for i in range(bbox_h):
                for j in range(bbox_w):
                    pixel = int(channel_2d[i, j])
                    
                    if i == 0 and j == 0:
                        pred = 128
                    elif i == 0:
                        pred = int(channel_2d[i, j-1])
                    elif j == 0:
                        pred = int(channel_2d[i-1, j])
                    else:
                        left = int(channel_2d[i, j-1])
                        top = int(channel_2d[i-1, j])
                        diagonal = int(channel_2d[i-1, j-1])
                        pred = paeth_predictor(left, top, diagonal)
                    
                    error = pixel - pred
                    error = np.clip(error, -128, 127)
                    errors.append(error)
            
            # QUANTIZE ERRORS
            errors_quantized = np.array(errors) // Config.ERROR_QUANTIZATION
            errors_quantized = np.clip(errors_quantized, -32, 31) + 32
            
            # Get entropy from model
            if entropy_model is not None:
                with torch.no_grad():
                    pmf = entropy_model(region_feat.to(Config.DEVICE))
                    pmf = pmf.squeeze(0).cpu().numpy()
            else:
                pmf = np.ones(65) / 65
            
            # Huffman encode with learned distribution
            error_bytes = bytes(errors_quantized.astype(np.uint8))
            compressed = zlib.compress(error_bytes, level=9)
            
            # Calculate bits (for logging)
            bits_estimate = sum(-np.log2(pmf[e] + 1e-9) for e in errors_quantized)
            total_region_bits += bits_estimate
            
            compressed_data[c] = compressed

        encoded_data['regions'][region_id] = {
            'bbox': (y0, y1, x0, x1),
            'mask_rle': mask_rle,
            'compressed_data': compressed_data,
            'bits_estimate': total_region_bits,
        }

    return encoded_data

def decode_full_image_learned(encoded_data):
    """Decode with quantized errors. FIXED: Uses np.frombuffer for correct byte handling."""
    H = encoded_data['H']
    W = encoded_data['W']
    C = encoded_data['C']
    Q = encoded_data['quantization']
    reconstructed = np.zeros((H, W, C), dtype=np.uint8)

    for region_id, region_data in encoded_data['regions'].items():
        y0, y1, x0, x1 = region_data['bbox']
        bbox_h, bbox_w = y1 - y0 + 1, x1 - x0 + 1

        mask_flat = decode_rle(region_data['mask_rle'], bbox_h * bbox_w)
        mask_2d = mask_flat.reshape(bbox_h, bbox_w).astype(bool)

        for c in range(C):
            compressed = region_data['compressed_data'][c]
            error_bytes = zlib.decompress(compressed)
            
            # FIX: Use np.frombuffer to convert bytes to uint8 array correctly
            errors_quantized = np.frombuffer(error_bytes, dtype=np.uint8).astype(np.int32) - 32
            errors = errors_quantized * Q
            
            pixel_2d = np.zeros((bbox_h, bbox_w), dtype=np.int32)

            error_idx = 0
            for i in range(bbox_h):
                for j in range(bbox_w):
                    error = errors[error_idx]
                    error_idx += 1

                    if i == 0 and j == 0:
                        pred = 128
                    elif i == 0:
                        pred = int(pixel_2d[i, j-1])
                    elif j == 0:
                        pred = int(pixel_2d[i-1, j])
                    else:
                        left = int(pixel_2d[i, j-1])
                        top = int(pixel_2d[i-1, j])
                        diagonal = int(pixel_2d[i-1, j-1])
                        pred = paeth_predictor(left, top, diagonal)

                    pixel = np.clip(pred + error, 0, 255)
                    pixel_2d[i, j] = pixel

            reconstructed[y0:y1+1, x0:x1+1, c][mask_2d] = pixel_2d[mask_2d]

    return reconstructed

# COMPRESSION SIZE
def compute_compression_size(encoded_data):
    total_bytes = 0
    total_bytes += len(encoded_data['segmentation_rle'])
    
    for region_id, region_data in encoded_data['regions'].items():
        total_bytes += len(region_data['mask_rle'])
        for c in range(encoded_data['C']):
            if c in region_data['compressed_data']:
                total_bytes += len(region_data['compressed_data'][c])
    
    return total_bytes

# GAEC
def gaec_additive(num_nodes, u_np, v_np, cost_np, merge_threshold=0.0):
    if len(u_np) == 0:
        return np.arange(num_nodes), 0, {}
    max_idx = max(u_np.max(), v_np.max())
    if max_idx >= num_nodes:
        num_nodes = max_idx + 1
    full_adj = [dict() for _ in range(num_nodes)]
    for i in range(len(u_np)):
        U, V, C = int(u_np[i]), int(v_np[i]), cost_np[i]
        if U == V:
            continue
        full_adj[U][V] = full_adj[U].get(V, 0.0) + float(C)
        full_adj[V][U] = full_adj[V].get(U, 0.0) + float(C)
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
        for v, w in full_adj[u].items():
            if u < v and w > merge_threshold:
                heapq.heappush(heap, (-w, u, v))

    edges_in_heap = len(heap)
    merges_count = 0

    while heap:
        neg_w, u, v = heapq.heappop(heap)
        w = -neg_w
        root_u, root_v = find(u), find(v)
        if root_u == root_v:
            continue
        if root_v not in full_adj[root_u]:
            continue
        curr_w = full_adj[root_u][root_v]
        if abs(curr_w - w) > 1e-6:
            continue
        if w <= merge_threshold:
            continue
        parent[root_v] = root_u
        merges_count += 1

        for neighbor, weight_v in list(full_adj[root_v].items()):
            if neighbor == root_u:
                if root_v in full_adj[root_u]:
                    del full_adj[root_u][root_v]
                continue
            if root_v in full_adj[neighbor]:
                del full_adj[neighbor][root_v]
            old_weight_u = full_adj[root_u].get(neighbor, 0.0)
            new_weight = old_weight_u + weight_v
            full_adj[root_u][neighbor] = new_weight
            full_adj[neighbor][root_u] = new_weight
            if new_weight > merge_threshold:
                a, b = (root_u, neighbor) if root_u < neighbor else (neighbor, root_u)
                heapq.heappush(heap, (-new_weight, a, b))
        full_adj[root_v].clear()

    labels = np.array([find(i) for i in range(num_nodes)], dtype=np.int32)
    uniq = np.unique(labels)
    mapping = {x: i for i, x in enumerate(uniq)}
    final_labels = np.array([mapping[x] for x in labels], dtype=np.int32)
    num_final_regions = len(uniq)

    log(f"[GAEC] Edges in heap: {edges_in_heap:,}")
    log(f"[GAEC] Merges executed: {merges_count:,}")
    log(f"[GAEC] Final regions: {num_final_regions:,}")

    edge_stats = {
        "total_edges": len(u_np),
        "positive_cost_edges": (cost_np > merge_threshold).sum(),
        "negative_cost_edges": (cost_np <= merge_threshold).sum(),
        "merges": merges_count,
    }
    return final_labels, merges_count, edge_stats

# COMPRESSION-AWARE LOSS
def compression_loss_aware(img_batch, edge_costs_raw, u, v, device, num_regions_factor=1.0):
    B, C, H, W = img_batch.shape
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    flat = img_batch.reshape(B, C, -1)
    for b in range(B):
        costs_raw = edge_costs_raw[b * len(u):(b + 1) * len(u)]
        pu = flat[b, :, u]
        pv = flat[b, :, v]
        rgb_diff = torch.sqrt(((pu - pv) ** 2).mean(dim=0) + 1e-8)
        
        target_cost = torch.zeros_like(rgb_diff)
        target_cost[rgb_diff < 0.1] = 1.0
        target_cost[rgb_diff > 0.3] = -1.0
        smooth_target = 1.5 * torch.tanh(3 * target_cost) / torch.tanh(torch.tensor(3.0))
        costs_scaled = torch.tanh(costs_raw)
        similarity_loss = ((costs_scaled - smooth_target) ** 2).mean()
        
        merge_score = torch.sigmoid(costs_raw).mean()
        estimated_regions = max(1, int((H * W * 0.01) * (1 - merge_score.item())))
        region_penalty = 0.05 * abs(estimated_regions - num_regions_factor * 200) / 200
        
        saturation = (torch.abs(costs_scaled) > 0.95).float().mean()
        saturation_penalty = 0.1 * saturation
        
        loss = similarity_loss + saturation_penalty + region_penalty
        total_loss = total_loss + loss
    
    return total_loss / B

# THRESHOLD SELECTION
def choosebestthreshold_downsampled(imgtensor, enc, pred, entropy_model, device, maxside=512, thresholds=None):
    if thresholds is None:
        thresholds = [-0.9, -0.7, -0.5, -0.3, -0.1]

    C, H0, W0 = imgtensor.shape
    scale = float(maxside) / float(max(H0, W0))

    if scale < 1.0:
        Hd = max(1, int(round(H0 * scale)))
        Wd = max(1, int(round(W0 * scale)))
        imgds = F.interpolate(imgtensor.unsqueeze(0), size=(Hd, Wd),
                            mode='bilinear', align_corners=False).squeeze(0)
    else:
        Hd, Wd = H0, W0
        imgds = imgtensor

    enc.eval()
    pred.eval()
    with torch.no_grad():
        imgdev = imgds.unsqueeze(0).to(device)
        feat = enc(imgdev)
        featflat = feat.permute(0, 2, 3, 1).reshape(1, -1, 128)
        rgbflat = imgdev.permute(0, 2, 3, 1).reshape(1, -1, 3)
        u, v = grid_edges(Hd, Wd, device)
        fu = featflat[0, u]
        fv = featflat[0, v]
        rgbdiff = rgbflat[0, u] - rgbflat[0, v]
        edgecostraw = pred(fu, fv, rgbdiff)
        edgecosts = torch.tanh(edgecostraw.squeeze(1)).cpu().numpy()
        unp = u.cpu().numpy()
        vnp = v.cpu().numpy()
        maxidx = max(int(unp.max()), int(vnp.max()))
        numnodes = maxidx + 1

    imgu8 = (imgds.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

    results = {}
    bestthr = None
    bestsize = None

    log(f"[THR-SWEEP] Downsampled from {H0}x{W0} -> {Hd}x{Wd}, trying {len(thresholds)} thresholds")

    for thr in thresholds:
        labels1d, mergescount, edgestats = gaec_additive(numnodes, unp, vnp, edgecosts, merge_threshold=float(thr))
        labels2d = labels1d.reshape(Hd, Wd).astype(np.int32)
        labels2d = merge_tiny_regions(labels2d, min_region_size=Config.TINY_REGION_SIZE)
        encodeddata = encode_full_image_learned(imgu8, labels2d, entropy_model, enc)
        sizebytes = compute_compression_size(encodeddata)
        numregions = int(labels2d.max()) + 1

        results[float(thr)] = {
            "size_bytes": int(sizebytes),
            "numregions": int(numregions),
            "ds_hw": (int(Hd), int(Wd)),
        }

        log(f"[THR-SWEEP] thr={thr:+.3f} -> size={sizebytes:,} bytes, regions={numregions}")

        if bestsize is None or sizebytes < bestsize:
            bestsize = sizebytes
            bestthr = float(thr)

    log(f"[THR-SWEEP] BEST thr={bestthr:+.3f} size={bestsize:,} bytes @ {Hd}x{Wd}")

    return bestthr, results

# INFERENCE
def inference_on_image(img_tensor, enc, pred, entropy_model, H, W, device):
    enc.eval()
    pred.eval()
    entropy_model.eval()
    
    with torch.no_grad():
        img_device = img_tensor.unsqueeze(0).to(device)
        feat = enc(img_device)
        feat_flat = feat.permute(0, 2, 3, 1).reshape(1, -1, 128)
        u, v = grid_edges(H, W, device)
        log(f"[INFERENCE] Image: {H}×{W}, Edges: {len(u):,}")
        rgb_flat = img_device.permute(0, 2, 3, 1).reshape(1, -1, 3)
        fu = feat_flat[0, u]
        fv = feat_flat[0, v]
        rgb_diff = rgb_flat[0, u] - rgb_flat[0, v]
        edge_cost_raw = pred(fu, fv, rgb_diff)
        edge_costs = torch.tanh(edge_cost_raw).squeeze(1).cpu().numpy()

        log(f"[INFERENCE] Costs: Mean={edge_costs.mean():.3f}, Std={edge_costs.std():.3f}, "
            f"Min={edge_costs.min():.3f}, Max={edge_costs.max():.3f}")

        target_regions = Config.TARGET_REGIONS
        log(f"[INFERENCE] Target regions: {target_regions}")

        threshold, _ = choosebestthreshold_downsampled(
            img_tensor, enc, pred, entropy_model, device,
            maxside=512,
            thresholds=[-0.9, -0.7, -0.5, -0.3, -0.1]
        )

        max_idx = max(u.max().item(), v.max().item())
        labels, num_merges, edge_stats = gaec_additive(
            max_idx + 1, u.cpu().numpy(), v.cpu().numpy(), edge_costs,
            merge_threshold=threshold
        )

        img_u8 = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        labels_2d = labels.reshape(H, W)

        log(f"\n[POST-PROCESS] Starting with {labels_2d.max() + 1:,} regions")
        labels_2d = merge_tiny_regions(labels_2d, min_region_size=Config.TINY_REGION_SIZE)
        log(f"")

        num_regions = labels_2d.max() + 1
        encoded_data = encode_full_image_learned(img_u8, labels_2d, entropy_model, enc)
        comp_bytes = compute_compression_size(encoded_data)

        log(f"[INFERENCE] Final result: {num_regions:,} regions, {comp_bytes:,} bytes\n")

        return {
            "labels": labels_2d,
            "num_regions": num_regions,
            "edge_stats": edge_stats,
            "comp_bytes": comp_bytes,
            "original_img": img_u8,
            "encoded_data": encoded_data,
        }

# VISUALIZATION
def visualize_reconstruction_real(original, labels, encoded_data, title="Reconstruction"):
    img_u8 = original.astype(np.uint8)
    labels_2d = labels
    reconstructed = decode_full_image_learned(encoded_data)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(original)
    axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0].axis("off")

    seg_map = (labels * 50 % 255).astype(np.uint8)
    axes[1].imshow(seg_map, cmap="tab20")
    num_regions = labels.max() + 1
    axes[1].set_title(f"Segmentation\n({num_regions} regions)", fontsize=12, fontweight='bold')
    axes[1].axis("off")

    axes[2].imshow(reconstructed)

    diff = original.astype(np.float32) - reconstructed.astype(np.float32)
    mse = np.mean(diff ** 2)
    psnr = 20 * np.log10(255 / np.sqrt(mse)) if mse > 0 else float('inf')
    mae = np.mean(np.abs(diff))

    if np.isinf(psnr):
        psnr_text = "Lossless"
    else:
        psnr_text = f"{psnr:.2f} dB"

    axes[2].set_title(f"Reconstructed\nPSNR={psnr_text}, MAE={mae:.6f}",
                     fontsize=12, fontweight='bold')
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(f"reconstruction_{title}.png", dpi=100, bbox_inches="tight")
    plt.close()

    log(f"Saved: reconstruction_{title}.png")
    log(f" Original: {img_u8.nbytes:,} bytes")
    if np.isinf(psnr):
        log(f" PSNR: Lossless")
    else:
        log(f" PSNR: {psnr:.2f} dB")
    log(f" MAE: {mae:.6f}")

# MAIN

def main():
    init_log_file()
    log(f"Device: {Config.DEVICE}")
    log("="*70)
    log("LEARNED COMPRESSION: QUANTIZED ERRORS + ENTROPY MODEL")
    log("="*70)

    ds = DIV2KDataset(Config.DATA_DIR, Config.PATCH)
    log(f"Total images: {len(ds)}")

    n_val = max(5, int(len(ds) * 0.1))
    train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val])
    train_dl = DataLoader(train_ds, batch_size=Config.BATCH, shuffle=True, num_workers=0)

    enc = CompressionEncoder().to(Config.DEVICE)
    pred = EdgeCostPredictor().to(Config.DEVICE)
    entropy_model = EntropyModel(feat_dim=128, num_error_values=65).to(Config.DEVICE)

    opt = optim.AdamW(
        list(enc.parameters()) + list(pred.parameters()) + list(entropy_model.parameters()),
        lr=Config.LR, weight_decay=1e-5, betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=Config.EPOCHS)

    u, v = grid_edges(Config.PATCH, Config.PATCH, Config.DEVICE)

    log(f"Training: {Config.PATCH}×{Config.PATCH} patches, {len(u)} edges")
    log(f"Quantization: Divide errors by {Config.ERROR_QUANTIZATION}")
    log(f"Entropy model: Network-learned PMF per region\n")

    val_indices = list(range(min(5, len(ds))))

    best_ratio = float('inf')
    patience = 15
    patience_counter = 0

    for ep in range(Config.EPOCHS):
        enc.train()
        pred.train()
        entropy_model.train()

        total_loss = 0.0
        num_batches = 0

        for bi, img_batch in enumerate(train_dl):
            img_batch = img_batch.to(Config.DEVICE)
            B = img_batch.shape[0]

            feat = enc(img_batch)
            feat_flat = feat.permute(0, 2, 3, 1).reshape(B, -1, 128)
            rgb_flat = img_batch.permute(0, 2, 3, 1).reshape(B, -1, 3)

            all_edge_costs_raw = []
            for i in range(B):
                fu = feat_flat[i, u]
                fv = feat_flat[i, v]
                rgb_diff = rgb_flat[i, u] - rgb_flat[i, v]
                edge_raw = pred(fu, fv, rgb_diff)
                all_edge_costs_raw.append(edge_raw.squeeze(1))

            edge_costs_raw = torch.cat(all_edge_costs_raw, dim=0)
            loss = compression_loss_aware(img_batch, edge_costs_raw, u, v, Config.DEVICE)

            total_loss += loss.item()
            num_batches += 1

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(pred.parameters()) + list(entropy_model.parameters()), 1.0
            )
            opt.step()

            if bi % 50 == 0:
                log(f"Ep{ep} B{bi}/{len(train_dl)} Loss={loss.item():.4f}")

        avg_loss = total_loss / num_batches
        scheduler.step()

        log(f"Ep{ep} DONE - Loss: {avg_loss:.4f}")

        if (ep + 1) % Config.VAL_EVERY == 0 or ep == Config.EPOCHS - 1:
            val_idx = val_indices[ep % len(val_indices)]
            log(f"\n[VALIDATION Ep{ep} on image {val_idx}]")

            val_img_tensor = ds.load_full_image(val_idx)
            val_h, val_w = val_img_tensor.shape[1], val_img_tensor.shape[2]

            val_result = inference_on_image(val_img_tensor, enc, pred, entropy_model, val_h, val_w, Config.DEVICE)

            baseline_buf = io.BytesIO()
            Image.fromarray(val_result["original_img"]).save(
                baseline_buf, format="PNG", optimize=True, compress_level=9
            )
            baseline_bytes = len(baseline_buf.getvalue())

            ratio = val_result["comp_bytes"] / baseline_bytes
            gain = (1.0 - ratio) * 100

            log(f"[VAL] Compressed={val_result['comp_bytes']:,} bytes, "
                f"PNG={baseline_bytes:,} bytes, "
                f"Ratio={ratio:.3f}, Gain={gain:+.1f}%\n")

            visualize_reconstruction_real(val_result["original_img"], val_result["labels"],
                                         val_result["encoded_data"], f"ep{ep}")

            if ratio < best_ratio:
                best_ratio = ratio
                patience_counter = 0
                torch.save(enc.state_dict(), "enc_best.pth")
                torch.save(pred.state_dict(), "pred_best.pth")
                torch.save(entropy_model.state_dict(), "entropy_best.pth")
                log(f"[BEST] New best ratio: {ratio:.3f}\n")
            else:
                patience_counter += 1

            if patience_counter > patience:
                log(f"\n[EARLY STOP] No improvement for {patience} validations")
                break

    torch.save(enc.state_dict(), "enc_final.pth")
    torch.save(pred.state_dict(), "pred_final.pth")
    torch.save(entropy_model.state_dict(), "entropy_final.pth")
    log("Models saved!")

    log("\n" + "="*70)
    log("TESTING ON FULL IMAGES")
    log("="*70 + "\n")

    for idx in range(min(3, len(ds))):
        log(f"\n[TEST {idx}]")
        test_img = ds.load_full_image(idx)
        th, tw = test_img.shape[1], test_img.shape[2]

        result = inference_on_image(test_img, enc, pred, entropy_model, th, tw, Config.DEVICE)

        baseline_buf = io.BytesIO()
        Image.fromarray(result["original_img"]).save(
            baseline_buf, format="PNG", optimize=True, compress_level=9
        )
        baseline_bytes = len(baseline_buf.getvalue())

        ratio = result["comp_bytes"] / baseline_bytes
        gain = (1.0 - ratio) * 100

        log(f"[RESULT] Regions={result['num_regions']:,}, "
            f"Compressed={result['comp_bytes']:,} bytes, "
            f"PNG={baseline_bytes:,} bytes, "
            f"Ratio={ratio:.3f}, Gain={gain:+.1f}%")

        visualize_reconstruction_real(result["original_img"], result["labels"],
                                     result["encoded_data"], f"test{idx}")

def close_log_file():
    global _log_file
    if _log_file is not None:
        _log_file.close()

if __name__ == "__main__":
    try:
        main()
    finally:
        close_log_file()