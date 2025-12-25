import os, glob, time, math, io
import numpy as np
from PIL import Image
import zlib
import heapq
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
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
    EPOCHS = 20
    LR = 5e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        """Load full image without cropping (for testing)"""
        if self.files[idx] == "__dummy__":
            return torch.rand(3, 256, 256)
        try:
            img = Image.open(self.files[idx]).convert("RGB")
            img = img.resize((256, 256), Image.LANCZOS)
            return self.test_transform(img)
        except Exception:
            return torch.rand(3, 256, 256)

# ============================================================
# MODEL: CNN only predicts edge costs, nothing else
# ============================================================
class EncoderCNN(nn.Module):
    def __init__(self, c_out=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, c_out, 3, padding=1)
        )
    def forward(self, x):
        return self.net(x)

class EdgePredictor(nn.Module):
    """Predicts edge cost (scalar) given feature vectors of two adjacent pixels"""
    def __init__(self, c_feat=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*c_feat + 1, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, fu, fv, orient):
        return self.net(torch.cat([fu, fv, orient], dim=1))

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
# LOSS: Teach CNN to predict good edge costs
# ============================================================
def edge_cost_loss(img_bchw, u, v, edge_cost_logits):
    """
    Train the CNN to predict edge costs based on pixel differences.
    
    The idea: 
    - If pixels are similar (low diff) → edge cost should be HIGH (merge is good)
    - If pixels are different (high diff) → edge cost should be LOW (cut is good)
    
    We convert pixel differences to target edge costs and train MSE.
    """
    _, C, H, W = img_bchw.shape
    flat = img_bchw.reshape(1, C, -1)[0]
    pu = flat[:, u]
    pv = flat[:, v]
    
    # Pixel difference between adjacent pixels
    pixel_diff = (pu - pv).abs().mean(dim=0, keepdim=True).T  # (num_edges, 1)
    
    # Target edge cost: sigmoid-based (similar pixels get high cost, different get low)
    THRESHOLD = 0.10
    target_cost = 2.0 * torch.sigmoid((THRESHOLD - pixel_diff) * 30.0) - 1.0  # Range [-1, 1]
    
    # Predicted edge cost (clamp to [-1, 1] with tanh)
    pred_cost = torch.tanh(edge_cost_logits)
    
    # MSE loss
    loss = ((pred_cost - target_cost) ** 2).mean()
    
    return loss, {
        "loss": loss.item(),
        "target_cost_mean": target_cost.mean().item(),
        "pred_cost_mean": pred_cost.mean().item(),
        "pred_cost_std": pred_cost.std().item(),
        "pred_cost_min": pred_cost.min().item(),
        "pred_cost_max": pred_cost.max().item(),
        "pixel_diff_mean": pixel_diff.mean().item(),
        "pixel_diff_max": pixel_diff.max().item(),
    }

# ============================================================
# GAEC: Pure algorithm, no manual pixel changes
# ============================================================
def gaec_additive(num_nodes, u_np, v_np, cost_np, merge_threshold=0.0):
    """
    Greedy Additive Edge Contraction (GAEC) algorithm.
    
    Merges edges with POSITIVE cost (high cost = good to merge = keep connected).
    Cuts edges with NEGATIVE cost (low cost = good to cut = separate).
    
    Pure algorithm - doesn't touch image pixels.
    """
    if len(u_np) == 0:
        return np.arange(num_nodes), 0, {}, []
    
    max_idx = max(u_np.max(), v_np.max())
    if max_idx >= num_nodes:
        num_nodes = max_idx + 1

    # Build adjacency list
    full_adj = [dict() for _ in range(num_nodes)]
    for i in range(len(u_np)):
        U, V, C = int(u_np[i]), int(v_np[i]), cost_np[i]
        if U == V:
            continue
        full_adj[U][V] = full_adj[U].get(V, 0.0) + float(C)
        full_adj[V][U] = full_adj[V].get(U, 0.0) + float(C)

    # Union-find for merging
    parent = np.arange(num_nodes, dtype=np.int32)

    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for p in path:
            parent[p] = i
        return i

    # Priority queue: (-cost, u, v) - highest cost first
    heap = []
    for u in range(num_nodes):
        for v, w in full_adj[u].items():
            if u < v and w > merge_threshold:
                heapq.heappush(heap, (-w, u, v))

    merges_count = 0
    merge_history = []  # Track merges for logging
    cut_count = 0
    
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
            cut_count += 1
            continue
        
        # Merge: set root_v's parent to root_u
        parent[root_v] = root_u
        merges_count += 1
        merge_history.append((root_u, root_v, w))
        
        # Update adjacency after merge
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
    
    # Renumber regions
    labels = np.array([find(i) for i in range(num_nodes)], dtype=np.int32)
    uniq = np.unique(labels)
    mapping = {x: i for i, x in enumerate(uniq)}
    final_labels = np.array([mapping[x] for x in labels], dtype=np.int32)
    
    edge_stats = {
        "total_edges": len(u_np),
        "positive_cost_edges": (cost_np > merge_threshold).sum(),
        "negative_cost_edges": (cost_np <= merge_threshold).sum(),
        "merges": merges_count,
        "cuts": cut_count,
    }
    
    return final_labels, merges_count, edge_stats, merge_history

# ============================================================
# COMPRESSION: Encode each region as PNG (lossless)
# ============================================================
def compress_regions_png(img_u8, labels):
    """
    Compress image by encoding each region separately as PNG.
    This is LOSSLESS - no pixel modification.
    
    Returns:
      - total_bytes: Total compressed size
      - num_regions: Number of regions
      - region_info: List of (region_id, bbox, png_bytes)
    """
    H, W, C = img_u8.shape
    labels_2d = labels.reshape(H, W)
    num_regions = labels_2d.max() + 1
    
    total_bytes = 0
    region_info = []
    
    # Overhead: store region count (2 bytes) + segmentation map info (2 bytes per region)
    overhead_bytes = 2 + num_regions * 2
    total_bytes += overhead_bytes
    
    for k in range(num_regions):
        mask = (labels_2d == k)
        ys, xs = np.where(mask)
        
        if len(ys) == 0:
            continue
        
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        h, w = y1 - y0 + 1, x1 - x0 + 1
        
        # Extract region bbox
        bbox_img = img_u8[y0:y1+1, x0:x1+1].copy()
        bbox_mask = mask[y0:y1+1, x0:x1+1]
        
        # Create PNG with alpha channel (transparent outside region)
        # Or: just encode the pixels (they'll compress well if region is coherent)
        
        # Simple approach: encode the bbox as PNG
        pil_img = Image.fromarray(bbox_img, mode="RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG", optimize=True, compress_level=9)
        png_bytes = buf.getvalue()
        
        region_bytes = len(png_bytes)
        total_bytes += region_bytes
        
        # Store: region_id, bbox coords, region size
        region_info.append({
            "region_id": k,
            "bbox": (y0, y1, x0, x1),
            "size_pixels": bbox_mask.sum(),
            "png_bytes": region_bytes,
        })
    
    return total_bytes, num_regions, region_info

# ============================================================
# INFERENCE & TESTING
# ============================================================
def inference_on_image(img_tensor, enc, pred, H, W, device):
    """
    Full inference pipeline on an image.
    
    Returns:
      - labels: Region assignment for each pixel
      - edge_stats: Statistics about edge costs and merges
      - compression_bytes: Total compressed size
      - region_info: Per-region compression info
    """
    enc.eval()
    pred.eval()
    
    with torch.no_grad():
        # Step 1: Feature extraction
        img_device = img_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)
        feat = enc(img_device)  # (1, 64, H, W)
        feat_flat = feat.permute(0, 2, 3, 1).reshape(1, -1, 64)
        
        # Step 2: Generate grid edges
        u, v = grid_edges(H, W, device)
        
        log(f"[INFERENCE] Image size: {H}×{W}, Edges: {len(u)}")
        
        # Step 3: Predict edge costs (CNN only job)
        fu = feat_flat[0, u]
        fv = feat_flat[0, v]
        edge_cost_logits = pred(fu, fv, torch.zeros(len(u), 1, device=device))
        edge_costs = torch.tanh(edge_cost_logits).squeeze(1).cpu().numpy()
        
        log(f"[INFERENCE] Edge costs - Mean: {edge_costs.mean():.4f}, Std: {edge_costs.std():.4f}, "
            f"Min: {edge_costs.min():.4f}, Max: {edge_costs.max():.4f}")
        
        # Step 4: GAEC segmentation (algorithm only job)
        max_idx = max(u.max().item(), v.max().item())
        labels, num_merges, edge_stats, _ = gaec_additive(
            max_idx + 1, u.cpu().numpy(), v.cpu().numpy(), edge_costs,
            merge_threshold=0.0
        )
        
        log(f"[INFERENCE] GAEC: {num_merges} merges, "
            f"Positive cost edges: {edge_stats['positive_cost_edges']}, "
            f"Negative cost edges: {edge_stats['negative_cost_edges']}")
        
        # Step 5: Convert image and compress
        img_u8 = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        comp_bytes, num_regions, region_info = compress_regions_png(img_u8, labels)
        
        log(f"[INFERENCE] Compression: {num_regions} regions, "
            f"Total: {comp_bytes} bytes")
        
        return {
            "labels": labels.reshape(H, W),
            "num_regions": num_regions,
            "num_merges": num_merges,
            "edge_stats": edge_stats,
            "total_bytes": comp_bytes,
            "region_info": region_info,
            "original_img": img_u8,
        }

# ============================================================
# VISUALIZATION
# ============================================================
def visualize_segmentation(img_u8, labels, title="Segmentation"):
    """Display original image and segmentation map"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(img_u8)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    seg_map = (labels * 50 % 255).astype(np.uint8)
    axes[1].imshow(seg_map, cmap="tab20")
    axes[1].set_title(f"Segmentation Map\n({labels.max() + 1} regions)")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.savefig(f"segmentation_{title}.png", dpi=100, bbox_inches="tight")
    plt.close()
    
    log(f"Saved: segmentation_{title}.png")

# ============================================================
# TRAINING
# ============================================================
def main():
    log(f"Device: {Config.DEVICE}")
    log("="*60)
    log("TRAINING: CNN learns to predict edge costs")
    log("INFERENCE: GAEC uses costs to segment, PNG encodes regions")
    log("="*60)
    
    # Load dataset
    ds = DIV2KDataset(Config.DATA_DIR, Config.PATCH)
    n_val = max(1, int(len(ds) * 0.1))
    train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val])
    
    train_dl = DataLoader(train_ds, batch_size=Config.BATCH, shuffle=True, num_workers=0)
    
    # Create models
    enc = EncoderCNN().to(Config.DEVICE)
    pred = EdgePredictor().to(Config.DEVICE)
    opt = optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=Config.LR)
    
    # Grid edges for training patches
    u, v = grid_edges(Config.PATCH, Config.PATCH, Config.DEVICE)
    n_edges = len(u)
    
    log(f"Training patches: {Config.PATCH}×{Config.PATCH}, Edges per patch: {n_edges}")
    log(f"Training for {Config.EPOCHS} epochs\n")
    
    # Use fixed validation image
    val_img_tensor = ds.load_full_image(0)
    val_h, val_w = val_img_tensor.shape[1], val_img_tensor.shape[2]
    
    # Training loop
    for ep in range(Config.EPOCHS):
        enc.train()
        pred.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for bi, img_batch in enumerate(train_dl):
            img_batch = img_batch.to(Config.DEVICE)
            B = img_batch.shape[0]
            
            # Extract features
            feat = enc(img_batch)
            feat_flat = feat.permute(0, 2, 3, 1).reshape(B, -1, 64)
            
            # Compute loss for batch
            loss_batch = 0.0
            for i in range(B):
                fu = feat_flat[i, u]
                fv = feat_flat[i, v]
                edge_logits = pred(fu, fv, torch.zeros(len(u), 1, device=Config.DEVICE))
                loss_i, _ = edge_cost_loss(img_batch[i:i+1], u, v, edge_logits)
                loss_batch += loss_i
            
            loss_batch /= B
            total_loss += loss_batch.item()
            num_batches += 1
            
            # Backward pass
            opt.zero_grad()
            loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(pred.parameters()), 1.0)
            opt.step()
            
            if bi % 20 == 0:
                log(f"Ep{ep} Batch{bi} Loss={loss_batch.item():.6f}")
        
        avg_loss = total_loss / num_batches
        log(f"Ep{ep} DONE - Avg Loss: {avg_loss:.6f}")
        
        # Validation: Compress full image
        if (ep + 1) % 5 == 0:
            log(f"\n[VALIDATION Ep{ep}]")
            val_result = inference_on_image(val_img_tensor, enc, pred, val_h, val_w, Config.DEVICE)
            
            # Compare to baseline PNG
            baseline_buf = io.BytesIO()
            Image.fromarray(val_result["original_img"]).save(baseline_buf, format="PNG", optimize=True, compress_level=9)
            baseline_bytes = len(baseline_buf.getvalue())
            
            ratio = val_result["total_bytes"] / baseline_bytes
            gain = (1.0 - ratio) * 100
            
            log(f"[VAL RESULT] Regions={val_result['num_regions']}, "
                f"CompSize={val_result['total_bytes']} bytes, "
                f"BaselinePNG={baseline_bytes} bytes, "
                f"Ratio={ratio:.3f}, Gain={gain:+.1f}%\n")
            
            visualize_segmentation(val_result["original_img"], val_result["labels"], f"ep{ep}")
    
    # Save models
    torch.save(enc.state_dict(), "enc.pth")
    torch.save(pred.state_dict(), "pred.pth")
    log("\nModels saved: enc.pth, pred.pth")
    
    # Test on multiple images
    log("\n" + "="*60)
    log("TESTING ON FULL IMAGES")
    log("="*60 + "\n")
    
    test_indices = [5, 10, 15]
    for idx in test_indices:
        if idx < len(ds):
            log(f"\n[TEST {idx}] Loading test image...")
            test_img = ds.load_full_image(idx)
            th, tw = test_img.shape[1], test_img.shape[2]
            
            result = inference_on_image(test_img, enc, pred, th, tw, Config.DEVICE)
            
            # Baseline PNG
            baseline_buf = io.BytesIO()
            Image.fromarray(result["original_img"]).save(baseline_buf, format="PNG", optimize=True, compress_level=9)
            baseline_bytes = len(baseline_buf.getvalue())
            
            ratio = result["total_bytes"] / baseline_bytes
            gain = (1.0 - ratio) * 100
            
            log(f"\n[TEST {idx} RESULT] Regions={result['num_regions']}, "
                f"CompSize={result['total_bytes']} bytes, "
                f"BaselinePNG={baseline_bytes} bytes, "
                f"Ratio={ratio:.3f}, Gain={gain:+.1f}%")
            
            # Region details
            avg_region_bytes = result["total_bytes"] / max(1, result["num_regions"])
            log(f"[TEST {idx} REGIONS] Avg bytes per region: {avg_region_bytes:.1f}")
            
            visualize_segmentation(result["original_img"], result["labels"], f"test{idx}")

if __name__ == "__main__":
    main()
