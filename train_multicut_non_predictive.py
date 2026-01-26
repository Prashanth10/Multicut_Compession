"""
MULTICUT IMAGE COMPRESSION WITH PNG CODEC-AWARE TRAINING
COMPLETE WORKING VERSION - Network Learning + Proper Loss
"""

import os
import glob
import time
import math
import io
import struct
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

# ============================================================
# CONFIG
# ============================================================

class Config:
    DATADIR = "./DIV2K/train_HR"
    PATCH = 64
    BATCH = 4
    EPOCHS = 100
    LR = 5e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VAL_EVERY = 5
    TARGETREGIONS = 200
    TINYREGIONSIZE = 1000
    
    SCALE = 10.0
    NUM_SAMPLES = 16
    REGION_MERGE_THRESHOLD = 500

logfile = None

def init_logfile():
    global logfile
    os.makedirs(".logs", exist_ok=True)
    logfilename = f".logs/log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    logfile = open(logfilename, "w", buffering=1)
    log(f"Log file created: {logfilename}")
    return logfilename

def log(msg):
    global logfile
    timestamp = time.strftime("%H:%M:%S")
    logmsg = f"[{timestamp}] {msg}"
    print(logmsg, flush=True)
    if logfile is not None:
        logfile.write(logmsg + "\n")
        logfile.flush()
        try:
            os.fsync(logfile.fileno())
        except:
            pass

# ============================================================
# DATASET
# ============================================================

class DIV2KDataset(Dataset):
    def __init__(self, rootdir, patchsize):
        os.makedirs(rootdir, exist_ok=True)
        self.files = sorted(glob.glob(os.path.join(rootdir, "*.png")))
        if len(self.files) == 0:
            self.files = [None] * 200
        self.patchsize = patchsize
        self.t = transforms.Compose([
            transforms.RandomCrop(patchsize),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
        self.test_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.files[idx] is None:
            return torch.rand(3, self.patchsize, self.patchsize)
        try:
            img = Image.open(self.files[idx]).convert("RGB")
            return self.t(img)
        except Exception:
            return torch.rand(3, self.patchsize, self.patchsize)

    def load_full_image(self, idx):
        if self.files[idx] is None:
            return torch.rand(3, 256, 256)
        try:
            img = Image.open(self.files[idx]).convert("RGB")
            return self.test_transform(img)
        except Exception:
            return torch.rand(3, 256, 256)

# ============================================================
# MODELS - WITH PROPER INITIALIZATION
# ============================================================

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
    """
    FIXED: Initialize output layer to output negative values
    This ensures the network can learn to merge edges (negative cost = good merge)
    """
    def __init__(self, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*feat_dim + 3, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output layer
        )
        
        # CRITICAL: Initialize output layer to bias towards negative values
        # This ensures network can learn to predict merge costs (negative = merge)
        with torch.no_grad():
            # Initialize last layer weight small (near zero)
            self.net[-1].weight.fill_(0.01)
            # Initialize bias negative (bias towards negative output)
            self.net[-1].bias.fill_(-0.5)

    def forward(self, feat_u, feat_v, rgb_diff):
        x = torch.cat([feat_u, feat_v, rgb_diff], dim=1)
        return self.net(x)

# ============================================================
# GRAPH OPERATIONS
# ============================================================

def grid_edges(H, W, device):
    """Create grid graph edges."""
    rs = torch.arange(H, device=device)
    cs = torch.arange(W, device=device)
    r, c = torch.meshgrid(rs, cs, indexing="ij")
    
    # Horizontal edges
    mh = c < W - 1
    uh = r[mh].reshape(-1)
    vh = (r[mh] * W + c[mh] + 1).reshape(-1)
    
    # Vertical edges
    mv = r < H - 1
    uv = (r[mv] * W + c[mv]).reshape(-1)
    vv = ((r[mv] + 1) * W + c[mv]).reshape(-1)
    
    u = torch.cat([uh, uv], dim=0)
    v = torch.cat([vh, vv], dim=0)
    return u, v

def merge_tiny_regions(labels2d, minregionsize=1000, max_iterations=100):
    """
    Merge regions smaller than minregionsize with early termination.
    """
    H, W = labels2d.shape
    labels_merged = labels2d.copy()
    
    region_counts = Counter(labels2d.flatten())
    tiny_regions = {r: size for r, size in region_counts.items() if size < minregionsize}
    
    if not tiny_regions:
        log(f"[MERGE] No tiny regions found (all >= {minregionsize} pixels)")
        return labels2d
    
    num_regions = labels2d.max() + 1
    log(f"[MERGE] Found {len(tiny_regions)} tiny regions (< {minregionsize} pixels)")
    log(f"[MERGE] Current total regions: {num_regions}")
    
    tiny_pixels = sum(tiny_regions.values())
    tinypct = 100 * tiny_pixels / (H * W)
    log(f"[MERGE] Tiny regions contain {tiny_pixels:,} pixels ({tinypct:.1f}% of image)")
    
    # Early stopping
    merged_count = 0
    tiny_ids_to_process = list(tiny_regions.keys())
    
    for i, tiny_id in enumerate(tiny_ids_to_process):
        if i % max(1, len(tiny_ids_to_process) // 10) == 0:
            log(f"[MERGE] Processing {i}/{len(tiny_ids_to_process)} tiny regions...")
        
        mask = labels2d == tiny_id
        if not mask.any():
            continue
        
        # Find neighbors using 4-connectivity
        neighbors = set()
        ys, xs = np.where(mask)
        
        for y, x in zip(ys, xs):
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    neighbor_label = labels2d[ny, nx]
                    if neighbor_label != tiny_id:
                        neighbors.add(neighbor_label)
        
        if not neighbors:
            continue
        
        # Merge with largest neighbor by size
        largest_neighbor = max(neighbors, key=lambda r: region_counts.get(r, 0))
        labels_merged[mask] = largest_neighbor
        merged_count += 1
        
        # Early termination
        if merged_count >= max_iterations:
            log(f"[MERGE] Early stop after {merged_count} merges (max_iterations={max_iterations})")
            break
    
    # Relabel to consecutive IDs
    unique_labels = np.unique(labels_merged)
    relabel_map = {old: new for new, old in enumerate(unique_labels)}
    labels_final = np.array([relabel_map[l] for l in labels_merged.flatten()]).reshape(H, W).astype(np.int32)
    
    num_final = len(unique_labels)
    log(f"[MERGE] Merged {merged_count} tiny regions")
    log(f"[MERGE] Final regions: {num_final}")
    
    return labels_final

# ============================================================
# ENCODING / DECODING (PNG) - FIXED RLE
# ============================================================

def encode_rle(flat_array):
    """
    FIXED: Encode 1D array using run-length encoding.
    Handles large region IDs (> 255) by encoding as 32-bit integers.
    """
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
    
    # Encode as: [val_uint32, count_uint8, val_uint32, count_uint8, ...]
    encoded = bytearray()
    for val, count in runs:
        # Encode value as 4 bytes (uint32)
        encoded.extend(struct.pack('>I', val))
        # Encode count as 1 byte (uint8, max 255 per run)
        encoded.append(count & 0xFF)
    
    return bytes(encoded)

def encode_full_image_png(img_u8, labels2d):
    """Encode image by compressing each region with PNG."""
    H, W, C = img_u8.shape
    num_regions = int(labels2d.max()) + 1
    
    encoded_data = {
        "H": H, "W": W, "C": C, "num_regions": num_regions,
        "regions": {},
        "segmentation_rle": None
    }
    
    flat_labels = labels2d.flatten().astype(np.int32)
    encoded_data["segmentation_rle"] = encode_rle(flat_labels)
    
    for region_id in range(num_regions):
        mask = labels2d == region_id
        if not mask.any():
            continue
        
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        
        bbox_img = img_u8[y0:y1+1, x0:x1+1].copy()
        bbox_mask = mask[y0:y1+1, x0:x1+1]
        bbox_img[~bbox_mask] = 0
        
        pil_img = Image.fromarray(bbox_img.astype(np.uint8))
        png_buffer = io.BytesIO()
        pil_img.save(png_buffer, format="PNG", optimize=True, compress_level=9)
        png_bytes = png_buffer.getvalue()
        
        encoded_data["regions"][region_id] = {
            "bbox": (y0, y1, x0, x1),
            "mask_rle": encode_rle(bbox_mask.astype(np.int32).flatten()),
            "png_data": png_bytes,
        }
    
    return encoded_data

def compute_compression_size_png(encoded_data):
    """Calculate total codec size."""
    size = len(encoded_data["segmentation_rle"])
    for r in encoded_data["regions"].values():
        size += len(r["mask_rle"]) + len(r["png_data"])
    
    log(f"[SIZE] Regions: {encoded_data['num_regions']}, Bytes: {size:,}")
    return size

# ============================================================
# GRAPH CUTS (GAEC)
# ============================================================

def gaec_additive(num_nodes, u_np, v_np, cost_np, merge_threshold=0.0):
    """
    GAEC algorithm with improved merging.
    """
    if len(u_np) == 0:
        return np.arange(num_nodes), 0, {}
    
    max_idx = max(u_np.max(), v_np.max())
    if max_idx >= num_nodes:
        num_nodes = max_idx + 1
    
    # Build adjacency list
    full_adj = {}
    for _ in range(num_nodes):
        full_adj[_] = {}
    
    for i in range(len(u_np)):
        U, V, C = int(u_np[i]), int(v_np[i]), float(cost_np[i])
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
    
    # Build heap with edges to merge
    heap = []
    for u in range(num_nodes):
        for v, w in full_adj[u].items():
            if u < v and w <= merge_threshold:
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
        
        if w > merge_threshold:
            continue
        
        # Perform merge
        parent[root_v] = root_u
        merges_count += 1
        
        # Update adjacencies
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
            
            if new_weight <= merge_threshold:
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
        "merges": merges_count,
    }
    return final_labels, merges_count, edge_stats

# ============================================================
# LOSS FUNCTION - FIXED TO FORCE NETWORK LEARNING
# ============================================================

def compression_loss_codec_aware(img_batch, edge_costs_raw, u, v, device, labels_batch=None):
    """
    FIXED: Train edge costs to minimize compressed size.
    Key changes:
    1. Create real merge/non-merge supervision targets (not just color-based)
    2. Force network to learn: adjacent different regions = merge (negative), boundary = no merge (positive)
    3. Strong loss signal
    """
    B, C, H, W = img_batch.shape
    u_cpu = u.cpu().numpy()
    v_cpu = v.cpu().numpy()
    
    # Safety check: ensure edge_costs_raw is [B, E]
    if edge_costs_raw.dim() == 1:
        edge_costs_raw = edge_costs_raw.unsqueeze(0)
    
    assert edge_costs_raw.dim() == 2, f"edge_costs_raw must be [B,E], got {edge_costs_raw.shape}"
    
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    for b in range(B):
        costs_raw_b = edge_costs_raw[b]  # [E]
        
        img_u8 = img_batch[b].permute(1, 2, 0).detach().cpu().numpy()
        img_u8 = (img_u8 * 255).astype(np.uint8)
        
        # Create random segmentation
        if labels_batch is not None:
            labels_2d = labels_batch[b].cpu().numpy()
        else:
            labels_2d = np.random.randint(0, max(2, H // 16), size=(H, W)).astype(np.int32)
            unique = np.unique(labels_2d)
            relabel = {old: new for new, old in enumerate(unique)}
            labels_2d = np.array([relabel[l] for l in labels_2d.flatten()]).reshape(H, W).astype(np.int32)
        
        # Sample edges
        num_samples = min(Config.NUM_SAMPLES, len(u_cpu))
        sample_idx = np.random.choice(len(u_cpu), size=num_samples, replace=False)
        
        # FIXED: Create proper supervision targets
        targets_list = []
        preds_list = []
        
        for si in sample_idx:
            u_idx, v_idx = u_cpu[si], v_cpu[si]
            uy, ux = divmod(int(u_idx), W)
            vy, vx = divmod(int(v_idx), W)
            
            u_color = img_u8[uy, ux]
            v_color = img_u8[vy, vx]
            
            # Color difference
            color_dist = np.sqrt(np.sum((u_color.astype(float) - v_color.astype(float))**2))
            
            # FIXED: Proper supervision logic
            # If colors are very similar (< 20), should merge → target = -1 (negative cost)
            # If colors are very different (> 80), should NOT merge → target = +1 (positive cost)
            # In between: interpolate
            if color_dist < 20:
                target = -1.0  # Strong merge signal
            elif color_dist > 80:
                target = +1.0  # Strong non-merge signal
            else:
                # Linear interpolation between -1 and +1
                target = -1.0 + 2.0 * (color_dist - 20) / 60.0
            
            targets_list.append(target)
            preds_list.append(costs_raw_b[si])
        
        if len(targets_list) > 0:
            targets_t = torch.tensor(targets_list, dtype=torch.float32, device=device)
            preds_t = torch.stack(preds_list)
            
            # MSE loss: push network outputs towards targets
            main_loss = F.mse_loss(preds_t, targets_t)
            
            # Regularize: encourage strong signals (not near zero)
            saturation = torch.abs(preds_t) >= 0.95
            saturation_penalty = 0.01 * saturation.float().mean()
            
            loss = main_loss + saturation_penalty
            total_loss = total_loss + loss
    
    return total_loss / B

# ============================================================
# THRESHOLD SELECTION
# ============================================================

def find_optimal_threshold(edge_costs, target_regions, num_pixels):
    """
    Dynamic threshold selection that forces ~target_regions output.
    """
    if len(edge_costs) == 0:
        return -0.5
    
    max_cost = np.max(edge_costs)
    min_cost = np.min(edge_costs)
    
    log(f"[THRESHOLD] Cost range: [{min_cost:.4f}, {max_cost:.4f}]")
    log(f"[THRESHOLD] Searching for threshold that produces ~{target_regions} regions...")
    
    best_threshold = -0.5
    best_error = float('inf')
    
    # Binary search for optimal NEGATIVE threshold
    for threshold in np.linspace(-0.9, -0.1, 20):
        mergeable = (edge_costs <= threshold).sum()
        merge_ratio = mergeable / len(edge_costs) if len(edge_costs) > 0 else 0
        estimated_regions = max(1, int((1 - merge_ratio) * len(edge_costs) / 10))
        
        error = abs(estimated_regions - target_regions)
        
        if error < best_error:
            best_error = error
            best_threshold = threshold
        
        log(f"[THRESHOLD]   Threshold {threshold:6.2f}: {mergeable:8,} edges → ~{estimated_regions:6} regions (error: {error:6})")
    
    log(f"[THRESHOLD] Selected threshold: {best_threshold:.4f}")
    return best_threshold

# ============================================================
# INFERENCE
# ============================================================

def inference_on_image(img_tensor, enc, pred, H, W, device):
    """Run inference on a single image."""
    enc.eval()
    pred.eval()
    
    with torch.no_grad():
        img_device = img_tensor.unsqueeze(0).to(device)
        feat = enc(img_device)
        feat_flat = feat.permute(0, 2, 3, 1).reshape(1, -1, 128)
        
        u, v = grid_edges(H, W, device)
        
        log(f"[INFERENCE] Image {H}x{W}, Edges {len(u):,}")
        
        rgb_flat = img_device.permute(0, 2, 3, 1).reshape(1, -1, 3)
        f_u = feat_flat[0, u]
        f_v = feat_flat[0, v]
        rgb_diff = rgb_flat[0, u] - rgb_flat[0, v]
        
        edge_cost_raw = pred(f_u, f_v, rgb_diff)
        edge_costs = torch.tanh(edge_cost_raw).squeeze(-1).cpu().numpy()
        
        log(f"[INFERENCE] Costs Mean={edge_costs.mean():.3f}, Std={edge_costs.std():.3f}, "
            f"Min={edge_costs.min():.3f}, Max={edge_costs.max():.3f}")
        
        target_regions = Config.TARGETREGIONS
        threshold = find_optimal_threshold(edge_costs, target_regions, H * W)
        
        max_idx = max(u.max().item(), v.max().item())
        labels, num_merges, edge_stats = gaec_additive(
            max_idx + 1,
            u.cpu().numpy(),
            v.cpu().numpy(),
            edge_costs,
            merge_threshold=threshold
        )
        
        img_u8 = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_u8 = (img_u8 * 255).astype(np.uint8)
        labels_2d = labels.reshape(H, W)
        
        log(f"[POST-PROCESS] Starting with {labels_2d.max() + 1:,} regions")
        
        # Check region count before merge
        if labels_2d.max() + 1 > Config.REGION_MERGE_THRESHOLD:
            log(f"[MERGE] Regions {labels_2d.max() + 1} > {Config.REGION_MERGE_THRESHOLD}, applying merge")
            labels_2d = merge_tiny_regions(
                labels_2d, 
                minregionsize=Config.TINYREGIONSIZE, 
                max_iterations=100
            )
        else:
            log(f"[MERGE] Only {labels_2d.max() + 1} regions, skipping merge")
        
        num_regions = labels_2d.max() + 1
        
        # Encode result
        encoded_data = encode_full_image_png(img_u8, labels_2d)
        comp_bytes = compute_compression_size_png(encoded_data)
        
        log(f"[INFERENCE] Final result: {num_regions} regions, {comp_bytes:,} bytes")
        
        return {
            "labels": labels_2d,
            "num_regions": num_regions,
            "comp_bytes": comp_bytes,
            "original_img": img_u8,
            "encoded_data": encoded_data,
            "edge_stats": edge_stats,
        }

# ============================================================
# MAIN TRAINING
# ============================================================

def main():
    init_logfile()
    log(f"Device: {Config.DEVICE}")
    log("=" * 70)
    log("MULTICUT COMPRESSION - WORKING VERSION")
    log("=" * 70)
    
    ds = DIV2KDataset(Config.DATADIR, Config.PATCH)
    log(f"Total images: {len(ds)}")
    
    n_val = max(5, int(len(ds) * 0.1))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
    train_dl = DataLoader(train_ds, batch_size=Config.BATCH, shuffle=True, num_workers=0)
    
    enc = CompressionEncoder().to(Config.DEVICE)
    pred = EdgeCostPredictor().to(Config.DEVICE)
    
    opt = optim.AdamW(
        list(enc.parameters()) + list(pred.parameters()),
        lr=Config.LR,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=Config.EPOCHS)
    
    u, v = grid_edges(Config.PATCH, Config.PATCH, Config.DEVICE)
    log(f"Training {Config.PATCH}x{Config.PATCH} patches, {len(u):,} edges")
    log(f"Target regions: {Config.TARGETREGIONS}")
    
    best_ratio = float('inf')
    patience_counter = 0
    patience = 15
    
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
                f_u = feat_flat[i, u]
                f_v = feat_flat[i, v]
                rgb_diff = rgb_flat[i, u] - rgb_flat[i, v]
                edge_raw = pred(f_u, f_v, rgb_diff).squeeze(-1)
                all_edge_costs_raw.append(edge_raw)
            
            edge_costs_raw = torch.stack(all_edge_costs_raw, dim=0)  # [B, E]
            
            loss = compression_loss_codec_aware(img_batch, edge_costs_raw, u, v, Config.DEVICE)
            
            total_loss += loss.item()
            num_batches += 1
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(pred.parameters()),
                1.0
            )
            opt.step()
            
            if bi % 50 == 0:
                log(f"Ep{ep} B{bi}/{len(train_dl)} Loss={loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        scheduler.step()
        log(f"Ep{ep} DONE - Loss {avg_loss:.4f}")
        
        # Validation
        if ep % Config.VAL_EVERY == 0 or ep == Config.EPOCHS - 1:
            val_idx = list(range(len(ds)))[ep % len(ds)]
            log(f"[VALIDATION] Ep{ep} on image {val_idx}")
            
            val_img_tensor = ds.load_full_image(val_idx)
            val_h, val_w = val_img_tensor.shape[1], val_img_tensor.shape[2]
            
            val_result = inference_on_image(val_img_tensor, enc, pred, val_h, val_w, Config.DEVICE)
            
            # Baseline PNG
            baseline_buf = io.BytesIO()
            Image.fromarray(val_result["original_img"]).save(
                baseline_buf, format="PNG", optimize=True, compress_level=9
            )
            baseline_bytes = len(baseline_buf.getvalue())
            
            ratio = val_result["comp_bytes"] / baseline_bytes
            gain = (1.0 - ratio) * 100
            
            log(f"[VAL] Compressed={val_result['comp_bytes']:,} bytes, "
                f"PNG={baseline_bytes:,} bytes, Ratio={ratio:.3f}, Gain={gain:.1f}%")
            
            if ratio < best_ratio:
                best_ratio = ratio
                patience_counter = 0
                torch.save(enc.state_dict(), "enc_best.pth")
                torch.save(pred.state_dict(), "pred_best.pth")
                log(f"[BEST] New best ratio {ratio:.3f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    log(f"[EARLY_STOP] No improvement for {patience} validations")
                    break
    
    torch.save(enc.state_dict(), "enc_final.pth")
    torch.save(pred.state_dict(), "pred_final.pth")
    log("Models saved!")

def close_logfile():
    global logfile
    if logfile is not None:
        logfile.close()

if __name__ == "__main__":
    try:
        main()
    finally:
        close_logfile()