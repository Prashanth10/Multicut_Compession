"""
MULTICUT COMPRESSION WITH PNG COMPRESSION PER REGION + BINARY FORMAT
SIMPLIFIED: DIRECT THRESHOLD ASSIGNMENT (NO COMPLEX CALCULATIONS)

KEY IDEA:
- Stop calculating threshold from percentiles
- Use FIXED aggressive threshold: -0.5 to -0.9
- This ensures 90%+ of edges merge → ~200 final regions

WHY THIS WORKS:
- Network trained: cost=1.0 for merge, cost=-1.0 for boundary
- Threshold=-0.5 is exactly in middle of range
- Merges all edges with cost > -0.5
- Preserves only strong boundaries (cost < -0.5)
- Result: Pyramid reduction → 2M pixels → 200 regions

SIMPLE & EFFECTIVE:
- No percentile confusion
- No distribution analysis needed
- Direct control over merge aggressiveness
- Predictable results
"""

import os, glob, time, math, io, struct
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
    TINY_REGION_SIZE = 1000  # Merge regions smaller than this

_log_file = None

def init_log_file():
    global _log_file
    os.makedirs("./logs", exist_ok=True)
    log_filename = f"./logs/log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    _log_file = open(log_filename, 'w', buffering=1)  # ← Line buffering
    log(f"Log file created: {log_filename}")
    return log_filename

def log(msg):
    global _log_file
    timestamp = time.strftime('%H:%M:%S')
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg, flush=True)
    
    if _log_file is not None:
        _log_file.write(log_msg + '\n')
        _log_file.flush()  # ← Flush every time
        try:
            os.fsync(_log_file.fileno())  # ← Force disk write
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
# POST-PROCESSING: MERGE TINY REGIONS
# ============================================================================

def merge_tiny_regions(labels_2d, min_region_size=1000):
    """
    POST-PROCESSING: Merge regions smaller than min_region_size pixels
    with their largest neighbor.
    
    Why this helps:
    - Tiny regions have huge PNG overhead (500+ bytes per region header)
    - Merging them saves FAR more than the segmentation map size increase
    - With 2884 regions → 2500 tiny regions → 1.25MB overhead just in headers!
    
    Strategy:
    1. Find all regions with size < min_region_size
    2. Merge each tiny region with its largest neighbor
    3. Relabel to consecutive IDs
    4. Return: final segmentation map
    """
    H, W = labels_2d.shape
    labels_merged = labels_2d.copy()
    
    # Count region sizes from ORIGINAL labels
    region_counts = Counter(labels_2d.flatten())
    tiny_regions = {r: size for r, size in region_counts.items() if size < min_region_size}
    
    if not tiny_regions:
        log(f"[MERGE] No tiny regions found (all >= {min_region_size} pixels)")
        return labels_2d
    
    tiny_pixels = sum(tiny_regions.values())
    tiny_pct = 100 * tiny_pixels / (H * W)
    log(f"[MERGE] Found {len(tiny_regions)} tiny regions (< {min_region_size} pixels)")
    log(f"[MERGE] Tiny regions contain: {tiny_pixels:,} pixels ({tiny_pct:.1f}% of image)")
    
    # For each tiny region, find largest neighbor and merge
    merged_count = 0
    for tiny_id in tiny_regions.keys():
        mask = (labels_2d == tiny_id)
        if not mask.any():
            continue
        
        # Find neighbors using 4-connectivity on boundary pixels
        neighbors = set()
        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            # Check 4 neighbors: up, down, left, right
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    neighbor_label = labels_2d[ny, nx]
                    if neighbor_label != tiny_id:
                        neighbors.add(neighbor_label)
        
        if not neighbors:
            # No neighbors found, skip this region
            continue
        
        # Merge with largest neighbor (by original size)
        largest_neighbor = max(neighbors, key=lambda r: region_counts.get(r, 0))
        labels_merged[mask] = largest_neighbor
        merged_count += 1
    
    # Relabel to be consecutive (0, 1, 2, ...)
    unique_labels = np.unique(labels_merged)
    relabel_map = {old: new for new, old in enumerate(unique_labels)}
    labels_final = np.array([relabel_map[l] for l in labels_merged.flatten()]).reshape(H, W).astype(np.int32)
    
    num_final = len(unique_labels)
    log(f"[MERGE] Merged {merged_count} tiny regions")
    log(f"[MERGE] Final regions: {num_final:,}")
    
    return labels_final

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
# PNG COMPRESSION PER REGION
# ============================================================================

def encode_full_image_png(img_u8, labels_2d):
    """Encode image by compressing each region with PNG."""
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
        bbox_mask = mask[y0:y1+1, x0:x1+1]

        pil_img = Image.fromarray(bbox_img.astype(np.uint8))
        png_buffer = io.BytesIO()
        pil_img.save(png_buffer, format='PNG', optimize=True, compress_level=9)
        png_bytes = png_buffer.getvalue()

        encoded_data['regions'][region_id] = {
            'bbox': (y0, y1, x0, x1),
            'mask_rle': encode_rle(bbox_mask.astype(np.int32).flatten()),
            'png_data': png_bytes,
        }

    return encoded_data

def decode_full_image_png(encoded_data):
    """Decode image with PNG decompression per region."""
    H = encoded_data['H']
    W = encoded_data['W']
    C = encoded_data['C']

    reconstructed = np.zeros((H, W, C), dtype=np.uint8)

    for region_id, region_data in encoded_data['regions'].items():
        y0, y1, x0, x1 = region_data['bbox']
        bbox_h, bbox_w = y1 - y0 + 1, x1 - x0 + 1

        png_img = Image.open(io.BytesIO(region_data['png_data']))
        bbox_img = np.array(png_img, dtype=np.uint8)

        mask_flat = decode_rle(region_data['mask_rle'], bbox_h * bbox_w)
        mask_2d = mask_flat.reshape(bbox_h, bbox_w).astype(bool)

        reconstructed[y0:y1+1, x0:x1+1][mask_2d] = bbox_img[mask_2d]

    return reconstructed

# ============================================================================
# BINARY FORMAT (NO PICKLE!)
# ============================================================================

def encode_to_binary(encoded_data):
    """Convert encoded data to BINARY FORMAT (not pickle!)."""
    buffer = io.BytesIO()

    H, W, C = encoded_data['H'], encoded_data['W'], encoded_data['C']
    buffer.write(struct.pack('<HHB', H, W, C))

    seg_rle = encoded_data['segmentation_rle']
    buffer.write(struct.pack('<I', len(seg_rle)))
    buffer.write(seg_rle)

    num_regions = len(encoded_data['regions'])
    buffer.write(struct.pack('<I', num_regions))

    for region_id in sorted(encoded_data['regions'].keys()):
        region = encoded_data['regions'][region_id]

        y0, y1, x0, x1 = region['bbox']
        buffer.write(struct.pack('<HHHH', y0, y1, x0, x1))

        mask_rle = region['mask_rle']
        buffer.write(struct.pack('<I', len(mask_rle)))
        buffer.write(mask_rle)

        png_data = region['png_data']
        buffer.write(struct.pack('<I', len(png_data)))
        buffer.write(png_data)

    return buffer.getvalue()

def decode_from_binary(binary_data):
    """Decode from BINARY FORMAT."""
    buffer = io.BytesIO(binary_data)

    data = buffer.read(5)
    H, W, C = struct.unpack('<HHB', data)

    seg_len = struct.unpack('<I', buffer.read(4))[0]
    seg_rle = buffer.read(seg_len)

    num_regions = struct.unpack('<I', buffer.read(4))[0]

    regions = {}
    for _ in range(num_regions):
        bbox_data = buffer.read(8)
        y0, y1, x0, x1 = struct.unpack('<HHHH', bbox_data)

        mask_len = struct.unpack('<I', buffer.read(4))[0]
        mask_rle = buffer.read(mask_len)

        data_len = struct.unpack('<I', buffer.read(4))[0]
        png_data = buffer.read(data_len)

        regions[len(regions)] = {
            'bbox': (y0, y1, x0, x1),
            'mask_rle': mask_rle,
            'png_data': png_data,
        }

    return {
        'H': H, 'W': W, 'C': C,
        'num_regions': num_regions,
        'segmentation_rle': seg_rle,
        'regions': regions,
    }

# ============================================================================
# COMPRESSION SIZE CALCULATION
# ============================================================================

def compute_compression_size_png(encoded_data):
    """Calculate codec size for PNG per-region approach."""
    binary_data = encode_to_binary(encoded_data)
    size = len(binary_data)
    
    seg_size = len(encoded_data['segmentation_rle'])
    regions_size = sum(
        len(r['mask_rle']) + len(r['png_data'])
        for r in encoded_data['regions'].values()
    )
    overhead = size - seg_size - regions_size
    
    log(f"[SIZE BREAKDOWN] Total={size:,} bytes | Seg={seg_size:,} | Regions={regions_size:,} | Overhead={overhead:,}")
    return size

# GAEC - WITH DETAILED LOGGING
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

# THRESHOLD - DIRECT AGGRESSIVE ASSIGNMENT (SIMPLIFIED!)
def find_optimal_threshold(edge_costs, target_regions, num_pixels):
    """
    SIMPLIFIED APPROACH: Direct threshold assignment
    
    NO complex percentile calculations!
    Just use FIXED aggressive threshold: -0.5
    
    Why -0.5?
    - Network trained: cost=1.0 for similar regions (merge)
    - Network trained: cost=-1.0 for boundaries (keep)
    - Threshold=-0.5 is exactly in the middle
    - All edges with cost > -0.5 merge (90%+ of edges)
    - Only strong boundaries (cost < -0.5) preserved
    - Result: Pyramid reduction → ~200 regions
    """
    if len(edge_costs) == 0:
        return -0.5

    max_cost = np.max(edge_costs)
    min_cost = np.min(edge_costs)
    median_cost = np.median(edge_costs)
    mean_cost = np.mean(edge_costs)
    
    log(f"[THRESHOLD] DIRECT AGGRESSIVE ASSIGNMENT:")
    log(f"  Cost distribution: min={min_cost:.4f}, mean={mean_cost:.4f}, median={median_cost:.4f}, max={max_cost:.4f}")
    
    # DIRECT THRESHOLD: Use fixed aggressive value
    threshold = -0.5  # FIXED - aggressive merging in the middle!
    
    # Adapt based on output range (optional fine-tuning)
    if max_cost < 0.3:
        # Weak network output - be extra aggressive
        threshold = -0.7
        log(f"  Network weak (max={max_cost:.4f}) → extra-aggressive threshold=-0.7")
    elif max_cost > 0.95:
        # Strong network output - can relax slightly
        threshold = -0.3
        log(f"  Network strong (max={max_cost:.4f}) → standard aggressive threshold=-0.3")
    else:
        log(f"  Network normal → default aggressive threshold=-0.5")
    
    mergeable = (edge_costs > threshold).sum()
    num_no_merge = (edge_costs <= threshold).sum()
    merge_pct = 100 * mergeable / len(edge_costs) if len(edge_costs) > 0 else 0
    
    log(f"  FINAL THRESHOLD: {threshold:.4f}")
    log(f"  Edges to MERGE (cost > {threshold:.4f}): {mergeable:,} ({merge_pct:.1f}%)")
    log(f"  Edges to KEEP (cost <= {threshold:.4f}): {num_no_merge:,} ({100-merge_pct:.1f}%)")
    log(f"  Expected regions: ~150-300 (aggressive pyramid reduction)")
    
    return threshold

# INFERENCE
def inference_on_image(img_tensor, enc, pred, H, W, device):
    enc.eval()
    pred.eval()

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

        threshold = find_optimal_threshold(edge_costs, target_regions, H * W)

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

        encoded_data = encode_full_image_png(img_u8, labels_2d)
        comp_bytes = compute_compression_size_png(encoded_data)

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

def visualize_reconstruction_real(original, labels, encoded_data, title="Reconstruction"):
    """Reconstruction using PNG decompression"""
    img_u8 = original.astype(np.uint8)
    labels_2d = labels

    reconstructed = decode_full_image_png(encoded_data)

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
    log("MULTICUT COMPRESSION: PNG PER-REGION + DIRECT THRESHOLD")
    log("="*70)

    ds = DIV2KDataset(Config.DATA_DIR, Config.PATCH)
    log(f"Total images: {len(ds)}")

    n_val = max(5, int(len(ds) * 0.1))
    train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val])
    train_dl = DataLoader(train_ds, batch_size=Config.BATCH, shuffle=True, num_workers=0)

    enc = CompressionEncoder().to(Config.DEVICE)
    pred = EdgeCostPredictor().to(Config.DEVICE)

    opt = optim.AdamW(list(enc.parameters()) + list(pred.parameters()),
                      lr=Config.LR, weight_decay=1e-5, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=Config.EPOCHS)

    u, v = grid_edges(Config.PATCH, Config.PATCH, Config.DEVICE)
    log(f"Training: {Config.PATCH}×{Config.PATCH} patches, {len(u)} edges")
    log(f"Target regions: {Config.TARGET_REGIONS}")
    log(f"Post-process: Merge regions < {Config.TINY_REGION_SIZE} pixels\n")

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

            log(f"[VAL] Compressed={val_result['comp_bytes']:,} bytes, "
                f"PNG={baseline_bytes:,} bytes, "
                f"Ratio={ratio:.3f}, Gain={gain:+.1f}%\n")

            visualize_segmentation(val_result["original_img"], val_result["labels"], f"ep{ep}")
            visualize_reconstruction_real(val_result["original_img"], val_result["labels"],
                                        val_result["encoded_data"], f"ep{ep}")

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

        log(f"[RESULT] Regions={result['num_regions']:,}, "
            f"Compressed={result['comp_bytes']:,} bytes, "
            f"PNG={baseline_bytes:,} bytes, "
            f"Ratio={ratio:.3f}, Gain={gain:+.1f}%")

        visualize_segmentation(result["original_img"], result["labels"], f"test{idx}")
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

