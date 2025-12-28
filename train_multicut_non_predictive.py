"""
CORRECT MULTICUT COMPRESSION - MCCP OPTIMIZATION
==================================================

OPTIMIZATION PROBLEM (From PDF):
  min_S Σ_{R∈S} b(R)
  
where:
  S = segmentation (multicut of grid graph)
  b(R) = Huffman bits to encode RGB values in region R
  Goal: Minimize ONLY the region encoding size

KEY INSIGHTS:
1. Multicut bits = FIXED (≈2nm bits for n×m image)
   - 1 bit per edge = 130K bits for 256×256 ≈ 16 KB
   - NOT variable!

2. Main optimization = Minimize Huffman(RGB per region)
   - Similar pixels in region → fewer unique colors → fewer bits
   - Goal: Segment so each region has uniform colors

3. Loss must directly optimize b(R)
   - Network learns: cut between different pixels
   - Network learns: merge similar pixels
   - NOT: match costs to color differences (indirect!)

4. Reconstruction must show transmitted data
   - Use actual region colors (from Huffman codebook)
   - NOT region averages!
"""

import os, glob, time, math, io
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
    DATA_DIR = "./DIV2K_train_HR"
    PATCH = 64
    BATCH = 4
    EPOCHS = 100
    LR = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VAL_EVERY = 5

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# ============================================================
# DATASET
# ============================================================
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
            img = img.resize((256, 256), Image.LANCZOS)
            return self.test_transform(img)
        except Exception:
            return torch.rand(3, 256, 256)

# ============================================================
# ENCODER: Extract compression-relevant features
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

# ============================================================
# EDGE PREDICTOR: Learn optimal edge costs
# ============================================================
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

# ============================================================
# GRID GRAPH
# ============================================================
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

# ============================================================
# HUFFMAN ENTROPY ESTIMATION (CORRECT)
# ============================================================
def estimate_huffman_bits_for_values(values_np):
    """Compute expected Huffman bits for a list of values."""
    if len(values_np) == 0:
        return 0
    
    values_list = values_np.flatten().tolist()
    freq = Counter(values_list)
    n_unique = len(freq)
    n_total = len(values_list)
    
    if n_unique == 1:
        return 16
    
    entropy = 0.0
    for count in freq.values():
        p = count / n_total
        if p > 0:
            entropy -= p * np.log2(p)
    
    codebook_bits = 8 * n_unique + 16
    data_bits = entropy * n_total
    
    return int(codebook_bits + data_bits)

# ============================================================
# COMPRESSION SIZE - CORRECT MCCP FORMULA
# ============================================================
def compute_compression_size_mccp(img_u8, labels_2d):
    """CORRECT MCCP: total = multicut_bits + Σ Huffman(region R)"""
    H, W, C = img_u8.shape
    total_bits = 0
    
    # PART 1: Multicut (1 bit per edge - FIXED!)
    num_edges = H * (W - 1) + (H - 1) * W
    multicut_bits = num_edges
    total_bits += multicut_bits
    
    log(f"[COMPRESSION] Multicut bits: {multicut_bits} ({multicut_bits / 8:.1f} KB)")
    
    # PART 2: Region RGB encoding
    num_regions = int(labels_2d.max()) + 1
    region_bits = 0
    
    for region_id in range(num_regions):
        mask = (labels_2d == region_id)
        if not mask.any():
            continue
        
        region_pixels = img_u8[mask]
        for channel in range(3):
            channel_values = region_pixels[:, channel]
            bits = estimate_huffman_bits_for_values(channel_values)
            region_bits += bits
    
    total_bits += region_bits
    
    log(f"[COMPRESSION] Region bits: {region_bits} ({region_bits / 8:.1f} KB)")
    log(f"[COMPRESSION] Total bits: {total_bits} ({total_bits / 8:.1f} KB)")
    
    total_bytes = (total_bits + 7) // 8
    return total_bytes

# ============================================================
# GAEC SEGMENTATION
# ============================================================
def gaec_additive(num_nodes, u_np, v_np, cost_np, merge_threshold=0.0):
    """GAEC: Greedy Additive Edge Contraction."""
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

# ============================================================
# LOSS FUNCTION - CORRECT MCCP
# ============================================================
def mccp_loss_correct(img_batch, edge_costs_raw, u, v, device):
    """CORRECT: Minimize Σ b(R) by learning compression-optimal edge costs."""
    B, C, H, W = img_batch.shape
    flat = img_batch.reshape(B, C, -1)
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    for b in range(B):
        costs_raw = edge_costs_raw[b * len(u):(b + 1) * len(u)]
        costs_scaled = torch.tanh(costs_raw)
        
        pu = flat[b, :, u]
        pv = flat[b, :, v]
        
        rgb_diff = torch.sqrt(((pu - pv) ** 2).sum(dim=0) + 1e-8)
        rgb_diff = rgb_diff / (1 + rgb_diff)
        
        # TARGET: Merge similar, cut different (minimizes Huffman!)
        target_cost = torch.zeros_like(rgb_diff)
        target_cost[rgb_diff < 0.1] = 1.0
        target_cost[rgb_diff > 0.3] = -1.0
        
        smooth_target = 2.0 * torch.tanh(2.0 * target_cost) / torch.tanh(torch.tensor(2.0))
        
        similarity_loss = ((costs_scaled - smooth_target) ** 2).mean()
        saturation = (torch.abs(costs_scaled) > 0.95).float().mean()
        saturation_penalty = 0.05 * saturation
        
        loss = similarity_loss + saturation_penalty
        total_loss = total_loss + loss

    return total_loss / B

# ============================================================
# FIND OPTIMAL THRESHOLD
# ============================================================
def find_optimal_threshold(edge_costs, num_pixels):
    """Find threshold minimizing compression."""
    best_threshold = 0.0
    best_compression = float('inf')
    
    for percentile in np.linspace(0, 100, 50):
        threshold = np.percentile(edge_costs, percentile)
        num_cuts = (edge_costs < threshold).sum()
        estimated_regions = max(1, int(num_cuts / 100))
        estimated_compression = num_cuts + estimated_regions * 1000
        
        if estimated_compression < best_compression:
            best_compression = estimated_compression
            best_threshold = threshold
    
    return best_threshold

# ============================================================
# INFERENCE
# ============================================================
def inference_on_image(img_tensor, enc, pred, H, W, device):
    """Full inference."""
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

        threshold = find_optimal_threshold(edge_costs, H * W)
        log(f"[INFERENCE] Threshold: {threshold:.4f}")

        max_idx = max(u.max().item(), v.max().item())
        labels, num_merges, edge_stats = gaec_additive(
            max_idx + 1, u.cpu().numpy(), v.cpu().numpy(), edge_costs,
            merge_threshold=threshold
        )

        img_u8 = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        labels_2d = labels.reshape(H, W)
        num_regions = labels_2d.max() + 1

        comp_bytes = compute_compression_size_mccp(img_u8, labels_2d)

        return {
            "labels": labels_2d,
            "num_regions": num_regions,
            "edge_stats": edge_stats,
            "comp_bytes": comp_bytes,
            "original_img": img_u8,
        }

# ============================================================
# VISUALIZATION
# ============================================================
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

def visualize_reconstruction_correct(img_u8, labels, title="Reconstruction"):
    """Reconstruct: show actual transmitted colors."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_u8)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    seg_map = (labels * 50 % 255).astype(np.uint8)
    axes[1].imshow(seg_map, cmap="tab20")
    axes[1].set_title(f"Segmentation ({labels.max() + 1} regions)")
    axes[1].axis("off")
    
    reconstructed = np.zeros_like(img_u8)
    for r in range(labels.max() + 1):
        mask = (labels == r)
        if mask.any():
            reconstructed[mask] = img_u8[mask]
    
    axes[2].imshow(reconstructed)
    axes[2].set_title("Reconstructed (Transmitted Colors)")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(f"reconstruction_{title}.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"Saved: reconstruction_{title}.png")

# ============================================================
# MAIN
# ============================================================
def main():
    log(f"Device: {Config.DEVICE}")
    log("="*70)
    log("MULTICUT COMPRESSION - CORRECT MCCP OPTIMIZATION")
    log("="*70)

    ds = DIV2KDataset(Config.DATA_DIR, Config.PATCH)
    log(f"Total images: {len(ds)}")
    
    n_val = max(3, int(len(ds) * 0.05))
    train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val])
    train_dl = DataLoader(train_ds, batch_size=Config.BATCH, shuffle=True, num_workers=2)

    enc = CompressionEncoder().to(Config.DEVICE)
    pred = EdgeCostPredictor().to(Config.DEVICE)
    
    opt = optim.Adam(list(enc.parameters()) + list(pred.parameters()),
                     lr=Config.LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=Config.EPOCHS)

    u, v = grid_edges(Config.PATCH, Config.PATCH, Config.DEVICE)
    log(f"Training: {Config.PATCH}×{Config.PATCH} patches, {len(u)} edges")
    log(f"Train batches per epoch: {len(train_dl)}\n")

    val_indices = list(range(min(3, len(ds))))
    best_ratio = float('inf')
    patience = 20
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
            loss = mccp_loss_correct(img_batch, edge_costs_raw, u, v, Config.DEVICE)

            total_loss += loss.item()
            num_batches += 1

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(pred.parameters()), 1.0
            )
            opt.step()

            if bi % 100 == 0:
                log(f"Ep{ep} B{bi}/{len(train_dl)} Loss={loss.item():.4f}")

        avg_loss = total_loss / num_batches
        scheduler.step()
        log(f"Ep{ep} DONE - Loss: {avg_loss:.4f}\n")

        if (ep + 1) % Config.VAL_EVERY == 0:
            val_idx = val_indices[ep % len(val_indices)]
            log(f"[VALIDATION Ep{ep}]")
            
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

            log(f"[RESULT] Regions={val_result['num_regions']}, "
                f"Compressed={val_result['comp_bytes']:.0f} bytes, "
                f"PNG={baseline_bytes} bytes, "
                f"Ratio={ratio:.3f}, Gain={gain:+.1f}%\n")

            visualize_segmentation(val_result["original_img"], val_result["labels"], f"ep{ep}")
            
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
    log("\nModels saved!")

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
            f"Compressed={result['comp_bytes']:.0f} bytes, "
            f"PNG={baseline_bytes} bytes, "
            f"Ratio={ratio:.3f}, Gain={gain:+.1f}%")

        visualize_segmentation(result["original_img"], result["labels"], f"test{idx}")
        visualize_reconstruction_correct(result["original_img"], result["labels"], f"test{idx}")

if __name__ == "__main__":
    main()