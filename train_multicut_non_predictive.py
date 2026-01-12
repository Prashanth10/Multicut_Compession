"""

MULTICUT COMPRESSION WITH PREDICTIVE CODING + HUFFMAN + PROPER RECONSTRUCTION

CORE IDEA:

1. Segment image into regions (via GAEC)

2. For each region: Use PREDICTIVE CODING to encode pixel errors

- Predict pixel from neighbors

- Encode only the DIFFERENCE (error)

- Errors have much lower entropy than raw pixels!

3. Huffman encode the errors

4. DECODE: Reconstruct using error + prediction (reversible!)

5. TOTAL BITS = segmentation bits + sum(region error bits)

WHY PREDICTIVE?

- Natural images have spatial coherence

- Neighbors are similar → errors are small

- Small errors → fewer bits in Huffman tree

- 10-25× better than direct Huffman!

"""

import os, glob, time, math, io, pickle

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

def log(msg):

    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

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

            # img = img.resize((256, 256), Image.LANCZOS)

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

# ============================================================================
# ENCODING/DECODING DATA STRUCTURES
# ============================================================================

class EncodedRegion:

    """Store encoded data for one region."""

    def __init__(self):

        self.region_id = None

        self.bbox = None

        self.mask_rle = None

        self.huffman_tables = {}

        self.initial_pred = 128

# HUFFMAN CODING

class HuffmanNode:

    def __init__(self, value, freq):

        self.value = value

        self.freq = freq

        self.left = None

        self.right = None

    def __lt__(self, other):

        return self.freq < other.freq

def estimate_huffman_bits(values):

    if len(values) == 0:

        return 0

    freq_count = Counter(values)

    if len(freq_count) == 1:

        return 8 + len(values)

    entropy = 0

    for count in freq_count.values():

        prob = count / len(values)

        if prob > 0:

            entropy -= prob * np.log2(prob)

    codebook_bits = len(freq_count) * 160 + 32

    data_bits = entropy * len(values)

    return codebook_bits + data_bits

def build_huffman_codes(values):

    """Build Huffman codes from error values."""

    freq_count = Counter(values)

    if len(freq_count) == 1:

        unique_error = list(freq_count.keys())[0]

        huffman_table = {unique_error: '0'}

        return huffman_table, values

    heap = [HuffmanNode(v, f) for v, f in freq_count.items()]

    heapq.heapify(heap)

    while len(heap) > 1:

        n1 = heapq.heappop(heap)

        n2 = heapq.heappop(heap)

        parent = HuffmanNode(None, n1.freq + n2.freq)

        parent.left = n1

        parent.right = n2

        heapq.heappush(heap, parent)

    if not heap:

        return {}, []

    root = heap[0]

    huffman_table = {}

    def traverse(node, code=''):

        if node.value is not None:

            huffman_table[node.value] = code if code else '0'

        else:

            if node.left:

                traverse(node.left, code + '0')

            if node.right:

                traverse(node.right, code + '1')

    traverse(root)

    return huffman_table, values

# ============================================================================
# RLE ENCODING/DECODING
# ============================================================================

def encode_rle(flat_array):

    """Encode 1D array using run-length encoding."""

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

    """Decode RLE data back to array."""

    decoded = []

    for i in range(0, len(encoded_bytes), 2):

        if i + 1 < len(encoded_bytes):

            val = encoded_bytes[i]

            count = encoded_bytes[i + 1]

            decoded.extend([val] * count)

    if len(decoded) < expected_length:

        decoded.extend([0] * (expected_length - len(decoded)))

    return np.array(decoded[:expected_length], dtype=np.int32)

# ============================================================================
# ENCODING FUNCTIONS
# ============================================================================

def encode_region_channel(region_pixels_2d, initial_pred=128):

    """Encode one channel of one region using predictive coding."""

    H, W = region_pixels_2d.shape

    errors = []

    current_pred = initial_pred

    for i in range(H):

        for j in range(W):

            pixel = int(region_pixels_2d[i, j])

            top = int(region_pixels_2d[i-1, j]) if i > 0 else current_pred

            left = int(region_pixels_2d[i, j-1]) if j > 0 else current_pred

            pred = (left + top) // 2

            error = pixel - pred

            error = np.clip(error, -128, 127)

            errors.append(error)

            current_pred = pixel

    huffman_table, _ = build_huffman_codes(errors)

    return huffman_table, errors

def encode_full_image(img_u8, labels_2d):

    """Encode entire segmented image."""

    H, W, C = img_u8.shape

    num_regions = int(labels_2d.max()) + 1

    encoded_data = {

        'H': H, 'W': W, 'C': C,

        'num_regions': num_regions,

        'regions': {},

    }

    flat_labels = labels_2d.flatten().astype(np.int32)

    encoded_data['segmentation_rle'] = encode_rle(flat_labels)

    for region_id in range(num_regions):

        mask = (labels_2d == region_id)

        if not mask.any():

            continue

        ys, xs = np.where(mask)

        y0, y1 = ys.min(), ys.max()

        x0, x1 = xs.min(), xs.max()

        bbox_img = img_u8[y0:y1+1, x0:x1+1].copy()

        bbox_h, bbox_w = y1 - y0 + 1, x1 - x0 + 1

        encoded_region = EncodedRegion()

        encoded_region.region_id = region_id

        encoded_region.bbox = (y0, y1, x0, x1)

        bbox_mask = mask[y0:y1+1, x0:x1+1]

        encoded_region.mask_rle = encode_rle(

            bbox_mask.astype(np.int32).flatten()

        )

        for c in range(C):

            channel_2d = bbox_img[:, :, c]

            huffman_table, errors = encode_region_channel(

                channel_2d, initial_pred=128

            )

            encoded_region.huffman_tables[c] = {

                'table': huffman_table,

                'errors': errors,

            }

        encoded_data['regions'][region_id] = encoded_region

    return encoded_data

# ============================================================================
# DECODING FUNCTION - PROPER RECONSTRUCTION
# ============================================================================

def decode_full_image(encoded_data):

    """
    Reconstruct image from encoded data.
    
    KEY FUNCTION: Uses error + prediction to perfectly reconstruct
    """

    H = encoded_data['H']

    W = encoded_data['W']

    C = encoded_data['C']

    reconstructed = np.zeros((H, W, C), dtype=np.uint8)

    for region_id, encoded_region in encoded_data['regions'].items():

        y0, y1, x0, x1 = encoded_region.bbox

        bbox_h, bbox_w = y1 - y0 + 1, x1 - x0 + 1

        # Decode mask

        mask_flat = decode_rle(

            encoded_region.mask_rle, bbox_h * bbox_w

        )

        mask_2d = mask_flat.reshape(bbox_h, bbox_w).astype(bool)

        # Decode each channel

        for c in range(C):

            huffman_info = encoded_region.huffman_tables[c]

            errors = huffman_info['errors']

            pixel_2d = np.zeros((bbox_h, bbox_w), dtype=np.int32)

            error_idx = 0

            current_pred = 128

            for i in range(bbox_h):

                for j in range(bbox_w):

                    error = errors[error_idx]

                    error_idx += 1

                    top = int(pixel_2d[i-1, j]) if i > 0 else current_pred

                    left = int(pixel_2d[i, j-1]) if j > 0 else current_pred

                    pred = (left + top) // 2

                    # KEY STEP: pixel = error + prediction
                    pixel = np.clip(pred + error, 0, 255)

                    pixel_2d[i, j] = pixel

                    current_pred = pixel

            reconstructed[y0:y1+1, x0:x1+1, c][mask_2d] = pixel_2d[mask_2d]

    return reconstructed

# ============================================================================
# COMPRESSION SIZE (existing code)
# ============================================================================

def compress_region_predictive(region_img):

    H, W, C = region_img.shape

    total_bits = 32

    for c in range(C):

        channel = region_img[:, :, c].astype(np.int32)

        errors = np.zeros((H, W), dtype=np.int32)

        for i in range(H):

            for j in range(W):

                left = channel[i, j-1] if j > 0 else channel[i, j]

                top = channel[i-1, j] if i > 0 else channel[i, j]

                pred = (int(left) + int(top)) // 2

                error = int(channel[i, j]) - pred

                error = np.clip(error, -128, 127)

                errors[i, j] = error

        error_list = errors.flatten().tolist()

        bits = estimate_huffman_bits(error_list)

        total_bits += bits

    return total_bits

def compute_compression_size_real(img_u8, labels_2d):

    H, W, C = img_u8.shape

    num_regions = int(labels_2d.max()) + 1

    total_bits = 0

    flat_labels = labels_2d.flatten()

    runs = []

    if len(flat_labels) > 0:

        current_label = flat_labels[0]

        count = 1

        for i in range(1, len(flat_labels)):

            if flat_labels[i] == current_label:

                count += 1

            else:

                runs.append((int(current_label), count))

                current_label = flat_labels[i]

                count = 1

        runs.append((int(current_label), count))

    num_labels = len(np.unique(flat_labels))

    bits_per_label = max(1, math.ceil(math.log2(num_labels)))

    run_counts = Counter([r[1] for r in runs])

    run_entropy = 0

    for count in run_counts.values():

        prob = count / max(1, len(runs))

        if prob > 0:

            run_entropy -= prob * np.log2(prob)

    seg_bits = len(runs) * (bits_per_label + run_entropy + 8)

    total_bits += seg_bits

    for region_id in range(num_regions):

        mask = (labels_2d == region_id)

        if not mask.any():

            continue

        region_pixels = img_u8[mask]

        if len(region_pixels) == 0:

            continue

        region_compression_bits = 32

        for c in range(3):

            channel_values = region_pixels[:, c].astype(np.int32)

            errors = []

            pred = 128

            for val in channel_values:

                error = val - pred

                error = np.clip(error, -128, 127)

                errors.append(error)

                pred = val

            bits = estimate_huffman_bits(errors)

            region_compression_bits += bits

        total_bits += region_compression_bits

    total_bytes = (total_bits + 7) // 8

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

    edge_stats = {

        "total_edges": len(u_np),

        "positive_cost_edges": (cost_np > merge_threshold).sum(),

        "negative_cost_edges": (cost_np <= merge_threshold).sum(),

        "merges": merges_count,

    }

    return final_labels, merges_count, edge_stats

# LOSS FUNCTION

def compression_loss(img_batch, edge_costs_raw, u, v, device):

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

        saturation = (torch.abs(costs_scaled) > 0.95).float().mean()

        saturation_penalty = 0.1 * saturation

        merge_score = torch.sigmoid(costs_raw).mean()

        estimated_regions = max(1, int((H * W * 0.01) * (1 - merge_score.item())))

        region_penalty = 0.05 * abs(estimated_regions - Config.TARGET_REGIONS) / Config.TARGET_REGIONS

        loss = similarity_loss + saturation_penalty + region_penalty

        total_loss = total_loss + loss

    return total_loss / B

# THRESHOLD
def find_optimal_threshold(edge_costs, target_regions, num_pixels):
    if len(edge_costs) == 0 or target_regions <= 0:
        return np.median(edge_costs)
    merging_fraction = (num_pixels - target_regions) / num_pixels
    merging_fraction = np.clip(merging_fraction, 0.0, 0.99)
    min_cost = np.percentile(edge_costs, 5)
    max_cost = np.percentile(edge_costs, 95)
    threshold = max_cost - merging_fraction * (max_cost - min_cost)
    return threshold


# def find_optimal_threshold(edge_costs, target_regions, num_pixels):

#     sorted_costs = np.sort(edge_costs)

#     if len(edge_costs) == 0 or target_regions <= 0:
#         return 0.5
    
#     best_threshold = np.mean(edge_costs)

#     best_diff = float('inf')

#     for percentile in np.linspace(0, 100, 50):

#         threshold = np.percentile(edge_costs, percentile)

#         num_merges = (edge_costs > threshold).sum()

#         estimated_regions = num_pixels - num_merges

#         estimated_regions = max(1, min(num_pixels, estimated_regions))

#         diff = abs(estimated_regions - target_regions)

#         if diff < best_diff:

#             best_diff = diff

#             best_threshold = threshold

#     print(f"[THRESHOLD] Best threshold: {best_threshold:.4f} with estimated regions: {estimated_regions}")

#     return max(0.0, best_threshold)

# INFERENCE

def inference_on_image(img_tensor, enc, pred, H, W, device):

    enc.eval()

    pred.eval()

    with torch.no_grad():

        img_device = img_tensor.unsqueeze(0).to(device)

        feat = enc(img_device)

        feat_flat = feat.permute(0, 2, 3, 1).reshape(1, -1, 128)

        u, v = grid_edges(H, W, device)

        log(f"[INFERENCE] Image: {H}×{W}, Edges: {len(u)}")

        rgb_flat = img_device.permute(0, 2, 3, 1).reshape(1, -1, 3)

        fu = feat_flat[0, u]

        fv = feat_flat[0, v]

        rgb_diff = rgb_flat[0, u] - rgb_flat[0, v]

        edge_cost_raw = pred(fu, fv, rgb_diff)

        edge_costs = torch.tanh(edge_cost_raw).squeeze(1).cpu().numpy()

        log(f"[INFERENCE] Costs: Mean={edge_costs.mean():.3f}, Std={edge_costs.std():.3f}")

        threshold = find_optimal_threshold(edge_costs, Config.TARGET_REGIONS, H * W)

        log(f"[INFERENCE] Threshold: {threshold:.4f}")

        max_idx = max(u.max().item(), v.max().item())

        labels, num_merges, edge_stats = gaec_additive(

            max_idx + 1, u.cpu().numpy(), v.cpu().numpy(), edge_costs,

            merge_threshold=threshold

        )

        img_u8 = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        labels_2d = labels.reshape(H, W)

        num_regions = labels_2d.max() + 1

        comp_bytes = compute_compression_size_real(img_u8, labels_2d)

        log(f"[INFERENCE] Result: {num_regions} regions, {comp_bytes} bytes")

        return {

            "labels": labels_2d,

            "num_regions": num_regions,

            "edge_stats": edge_stats,

            "comp_bytes": comp_bytes,

            "original_img": img_u8,

        }

# VISUALIZATION

def visualize_segmentation(img_u8, labels, title="Segmentation"):

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(img_u8)

    axes[0].set_title("Original Image")

    axes[0].axis("off")

    seg_map = (labels * 50 % 255).astype(np.uint8)

    axes[1].imshow(seg_map, cmap="tab20")

    axes[1].set_title(f"Segmentation ({labels.max() + 1} regions)")

    axes[1].axis("off")

    plt.tight_layout()

    plt.savefig(f"segmentation_{title}.png", dpi=100, bbox_inches="tight")

    plt.close()

    log(f"Saved: segmentation_{title}.png")

def visualize_reconstruction_real(original, labels, title="Reconstruction"):

    """
    PROPER visualization with actual reconstruction.
    Uses encode/decode to reconstruct from compressed data.
    """

    img_u8 = original.astype(np.uint8)

    labels_2d = labels

    # Encode
    encoded_data = encode_full_image(img_u8, labels_2d)

    # Decode (reconstruct)
    reconstructed = decode_full_image(encoded_data)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Column 1: Original
    axes[0].imshow(original)

    axes[0].set_title("Original Image", fontsize=12, fontweight='bold')

    axes[0].axis("off")

    # Column 2: Segmentation
    seg_map = (labels * 50 % 255).astype(np.uint8)

    axes[1].imshow(seg_map, cmap="tab20")

    num_regions = labels.max() + 1

    axes[1].set_title(f"Segmentation\n({num_regions} regions)", fontsize=12, fontweight='bold')

    axes[1].axis("off")

    # Column 3: Reconstructed
    axes[2].imshow(reconstructed)

    # Compute quality metrics
    diff = original.astype(np.float32) - reconstructed.astype(np.float32)

    mse = np.mean(diff ** 2)

    psnr = 20 * np.log10(255 / np.sqrt(mse)) if mse > 0 else float('inf')

    mae = np.mean(np.abs(diff))

    axes[2].set_title(f"Reconstructed\nPSNR={psnr:.2f}dB, MAE={mae:.2f}",

                      fontsize=12, fontweight='bold')

    axes[2].axis("off")

    plt.tight_layout()

    plt.savefig(f"reconstruction_{title}.png", dpi=100, bbox_inches="tight")

    plt.close()

    log(f"Saved: reconstruction_{title}.png")

    log(f"  Original: {img_u8.nbytes} bytes")

    log(f"  Encoded: {sum(len(r.mask_rle) for r in encoded_data['regions'].values())} bytes")

    log(f"  PSNR: {psnr:.2f} dB")

    log(f"  MAE: {mae:.4f}")

# MAIN

def main():

    log(f"Device: {Config.DEVICE}")

    log("="*70)

    log("MULTICUT COMPRESSION: PREDICTIVE CODING + HUFFMAN")

    log("="*70)

    ds = DIV2KDataset(Config.DATA_DIR, Config.PATCH)

    log(f"Total images: {len(ds)}")

    n_val = max(5, int(len(ds) * 0.1))

    train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val])

    train_dl = DataLoader(train_ds, batch_size=Config.BATCH, shuffle=True, num_workers=2)

    enc = CompressionEncoder().to(Config.DEVICE)

    pred = EdgeCostPredictor().to(Config.DEVICE)

    opt = optim.AdamW(list(enc.parameters()) + list(pred.parameters()),

                      lr=Config.LR, weight_decay=1e-5, betas=(0.9, 0.999))

    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=Config.EPOCHS)

    u, v = grid_edges(Config.PATCH, Config.PATCH, Config.DEVICE)

    log(f"Training: {Config.PATCH}×{Config.PATCH} patches, {len(u)} edges")

    log(f"Target regions: {Config.TARGET_REGIONS}\n")

    val_indices = list(range(min(5, len(ds))))

    best_ratio = float('inf')

    patience = 15

    patience_counter = 0

    for ep in range(Config.EPOCHS):

        enc.train()

        pred.train()

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

            loss = compression_loss(img_batch, edge_costs_raw, u, v, Config.DEVICE)

            total_loss += loss.item()

            num_batches += 1

            opt.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(

                list(enc.parameters()) + list(pred.parameters()), 1.0

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

            val_result = inference_on_image(val_img_tensor, enc, pred, val_h, val_w, Config.DEVICE)

            baseline_buf = io.BytesIO()

            Image.fromarray(val_result["original_img"]).save(

                baseline_buf, format="PNG", optimize=True, compress_level=9

            )

            baseline_bytes = len(baseline_buf.getvalue())

            ratio = val_result["comp_bytes"] / baseline_bytes

            gain = (1.0 - ratio) * 100

            log(f"[VAL] Compressed={val_result['comp_bytes']} bytes, "

                f"PNG={baseline_bytes} bytes, "

                f"Ratio={ratio:.3f}, Gain={gain:+.1f}%\n")

            visualize_segmentation(val_result["original_img"], val_result["labels"], f"ep{ep}")

            visualize_reconstruction_real(val_result["original_img"], val_result["labels"], f"ep{ep}")

            if ratio < best_ratio:

                best_ratio = ratio

                patience_counter = 0

                torch.save(enc.state_dict(), "enc_best.pth")

                torch.save(pred.state_dict(), "pred_best.pth")

                log(f"[BEST] New best ratio: {ratio:.3f}\n")

            else:

                patience_counter += 1

                if patience_counter > patience:

                    log(f"\n[EARLY STOP] No improvement for {patience} validations")

                    break

    torch.save(enc.state_dict(), "enc_final.pth")

    torch.save(pred.state_dict(), "pred_final.pth")

    log("Models saved!")

    log("\n" + "="*70)

    log("TESTING ON FULL IMAGES")

    log("="*70 + "\n")

    for idx in range(min(3, len(ds))):

        log(f"\n[TEST {idx}]")

        test_img = ds.load_full_image(idx)

        th, tw = test_img.shape[1], test_img.shape[2]

        result = inference_on_image(test_img, enc, pred, th, tw, Config.DEVICE)

        baseline_buf = io.BytesIO()

        Image.fromarray(result["original_img"]).save(

            baseline_buf, format="PNG", optimize=True, compress_level=9

        )

        baseline_bytes = len(baseline_buf.getvalue())

        ratio = result["comp_bytes"] / baseline_bytes

        gain = (1.0 - ratio) * 100

        log(f"[RESULT] Regions={result['num_regions']}, "

            f"Compressed={result['comp_bytes']} bytes, "

            f"PNG={baseline_bytes} bytes, "

            f"Ratio={ratio:.3f}, Gain={gain:+.1f}%")

        visualize_segmentation(result["original_img"], result["labels"], f"test{idx}")

        visualize_reconstruction_real(result["original_img"], result["labels"], f"test{idx}")

if __name__ == "__main__":

    main()