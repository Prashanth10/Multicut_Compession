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
        # Store full images for testing
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
            # Resize to reasonable size for testing
            img = img.resize((256, 256), Image.LANCZOS)
            return self.test_transform(img)
        except Exception:
            return torch.rand(3, 256, 256)

# ============================================================
# MODEL
# ============================================================
class EncoderCNN(nn.Module):
    def __init__(self, c_out=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, c_out, 3, padding=1)
        )
    def forward(self, x): return self.net(x)

class EdgePredictor(nn.Module):
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
# LOSS
# ============================================================
def merge_gain_loss(img_bchw, u, v, merge_gain_logits):
    _, C, H, W = img_bchw.shape
    flat = img_bchw.reshape(1, C, -1)[0]
    pu = flat[:, u]
    pv = flat[:, v]
    diff = (pu - pv).abs().mean(dim=0, keepdim=True).T
    
    THRESHOLD = 0.10
    target_gain = 2.0 * torch.sigmoid((THRESHOLD - diff) * 30.0) - 1.0
    
    pred_gain = torch.tanh(merge_gain_logits)
    loss = ((pred_gain - target_gain) ** 2).mean()
    
    prob_cut = torch.sigmoid(-merge_gain_logits)
    n_high_diff = (diff > THRESHOLD).sum().item()
    
    return loss, {
        "loss": loss.item(),
        "target_gain_mean": target_gain.mean().item(),
        "pred_gain_mean": pred_gain.mean().item(),
        "pred_gain_std": pred_gain.std().item(),
        "prob_cut_mean": prob_cut.mean().item(),
        "n_high_diff_edges": n_high_diff,
        "diff_mean": diff.mean().item(),
    }

# ============================================================
# GAEC
# ============================================================
def gaec_additive(num_nodes, u_np, v_np, gain_np, merge_threshold=0.0):
    if len(u_np) == 0: return np.arange(num_nodes), 0
    max_idx = max(u_np.max(), v_np.max())
    if max_idx >= num_nodes: num_nodes = max_idx + 1

    full_adj = [dict() for _ in range(num_nodes)]
    for i in range(len(u_np)):
        U, V, G = int(u_np[i]), int(v_np[i]), gain_np[i]
        if U == V: continue
        full_adj[U][V] = full_adj[U].get(V, 0.0) + float(G)
        full_adj[V][U] = full_adj[V].get(U, 0.0) + float(G)

    parent = np.arange(num_nodes, dtype=np.int32)

    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for p in path: parent[p] = i
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
        if root_u == root_v: continue
        
        if root_v not in full_adj[root_u]: continue 
        curr_w = full_adj[root_u][root_v]
        if abs(curr_w - w) > 1e-6: continue
            
        if w <= merge_threshold: break
        
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
    return np.array([mapping[x] for x in labels], dtype=np.int32), merges_count

# ============================================================
# ENCODING & RECONSTRUCTION
# ============================================================
def encode_and_reconstruct(img_u8, labels, reconstruct_mode="mean_color"):
    """
    Encode image with segmentation + optionally reconstruct.
    
    reconstruct_mode:
      "mean_color": Lossy - fill each region with mean color
      "original": Lossless - store original pixels (via PNG per region)
    """
    H, W, C = img_u8.shape
    labels_2d = labels.reshape(H, W)
    K = labels_2d.max() + 1
    
    # Segmentation overhead
    seg_overhead = K * 10
    total_payload = 0
    
    # Reconstruction image
    reconstructed = np.zeros_like(img_u8)
    
    region_stats = []
    
    for k in range(K):
        mask = (labels_2d == k)
        ys, xs = np.where(mask)
        if len(ys) == 0: continue
        
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        crop = img_u8[y0:y1+1, x0:x1+1].copy()
        m = mask[y0:y1+1, x0:x1+1]
        crop[~m] = 0
        
        if reconstruct_mode == "mean_color":
            # Compute mean color in this region
            region_pixels = img_u8[mask]
            mean_color = region_pixels.mean(axis=0).astype(np.uint8)
            reconstructed[mask] = mean_color
            
            # For "payload", estimate as if we stored mean color (3 bytes) + region description
            region_bytes = 3  # RGB mean
            total_payload += region_bytes
            
        elif reconstruct_mode == "original":
            # Store original pixels via PNG encoding
            region_img = Image.fromarray(crop, mode="RGB")
            buf = io.BytesIO()
            region_img.save(buf, format="PNG", optimize=True)
            total_payload += len(buf.getvalue())
            
            # Reconstruct with original pixels
            reconstructed[mask] = img_u8[mask]
        
        region_stats.append({
            "region_id": k,
            "size_pixels": mask.sum(),
            "mean_color": mean_color if reconstruct_mode == "mean_color" else None
        })
    
    total_bytes = (seg_overhead + total_payload) // 8
    
    return total_bytes, K, reconstructed, region_stats

# ============================================================
# INFERENCE FUNCTION (For New Images)
# ============================================================
def inference_compress_reconstruct(img_tensor, enc, pred, H, W, device, reconstruct_mode="mean_color"):
    """
    Full pipeline for a new image:
    1. Load image
    2. Extract features (Encoder)
    3. Predict edge weights (EdgePredictor)
    4. Run GAEC segmentation
    5. Encode regions
    6. Reconstruct image
    """
    enc.eval()
    pred.eval()
    
    with torch.no_grad():
        # Feature extraction
        img_device = img_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)
        feat = enc(img_device)  # (1, 64, H, W)
        feat_flat = feat.permute(0, 2, 3, 1).reshape(1, -1, 64)
        
        # Generate grid edges
        u, v = grid_edges(H, W, device)
        
        # Predict merge gains
        fu = feat_flat[0, u]
        fv = feat_flat[0, v]
        merge_logits = pred(fu, fv, torch.zeros(len(u), 1, device=device))
        gains = merge_logits.squeeze(1).cpu().numpy()
        
        # GAEC segmentation
        max_idx = max(u.max().item(), v.max().item())
        labels, merges = gaec_additive(max_idx+1, u.cpu().numpy(), v.cpu().numpy(), gains, 
                                      merge_threshold=0.0)
        
        # Convert image to uint8
        img_u8 = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        # Encode and reconstruct
        total_bytes, K, reconstructed, region_stats = encode_and_reconstruct(
            img_u8, labels, reconstruct_mode=reconstruct_mode
        )
        
        return {
            "original_img": img_u8,
            "reconstructed_img": reconstructed,
            "segmentation_map": labels.reshape(H, W),
            "total_bytes": total_bytes,
            "num_regions": K,
            "num_merges": merges,
            "region_stats": region_stats
        }

# ============================================================
# VISUALIZATION
# ============================================================
def visualize_compression_result(result, title="Compression Result"):
    """Display original, reconstructed, and segmentation map side-by-side"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(result["original_img"])
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(result["reconstructed_img"])
    axes[1].set_title(f"Reconstructed\n({result['num_regions']} regions)")
    axes[1].axis("off")
    
    seg_map = (result["segmentation_map"] * 50 % 255).astype(np.uint8)
    axes[2].imshow(seg_map, cmap="tab20")
    axes[2].set_title(f"Segmentation Map\n({result['total_bytes']} bytes)")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(f"test_result_{title}.png", dpi=100, bbox_inches="tight")
    plt.close()
    
    log(f"Saved visualization: test_result_{title}.png")

# ============================================================
# MAIN: Train + Test
# ============================================================
def main():
    log(f"Device: {Config.DEVICE}")
    
    ds = DIV2KDataset(Config.DATA_DIR, Config.PATCH)
    n_val = max(1, int(len(ds) * 0.1))
    train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val])
    
    train_dl = DataLoader(train_ds, batch_size=Config.BATCH, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    enc = EncoderCNN().to(Config.DEVICE)
    pred = EdgePredictor().to(Config.DEVICE)
    opt = optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=Config.LR)
    
    u, v = grid_edges(Config.PATCH, Config.PATCH, Config.DEVICE)
    n_edges = len(u)
    
    # Use fixed validation image
    val_img_fixed = ds.load_full_image(0).to(Config.DEVICE)  # Get first image
    val_h, val_w = val_img_fixed.shape[1], val_img_fixed.shape[2]
    
    log(f"Fixed validation image: {val_h}×{val_w}")
    
    for ep in range(Config.EPOCHS):
        enc.train()
        pred.train()
        
        for bi, img in enumerate(train_dl):
            img = img.to(Config.DEVICE)
            B = img.shape[0]
            
            feat = enc(img)
            feat_flat = feat.permute(0,2,3,1).reshape(B, -1, 64)
            
            loss_batch = 0
            for i in range(B):
                fu = feat_flat[i, u]
                fv = feat_flat[i, v]
                merge_logits = pred(fu, fv, torch.zeros(len(u), 1, device=Config.DEVICE))
                loss_i, _ = merge_gain_loss(img[i:i+1], u, v, merge_logits)
                loss_batch += loss_i
            
            loss_batch /= B
            opt.zero_grad()
            loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(pred.parameters()), 1.0)
            opt.step()
            
            if bi % 50 == 0:
                log(f"Ep{ep} B{bi} Loss={loss_batch.item():.6f}")

        # Validation on FIXED image
        enc.eval()
        pred.eval()
        
        result = inference_compress_reconstruct(val_img_fixed, enc, pred, val_h, val_w, Config.DEVICE, 
                                               reconstruct_mode="mean_color")
        
        base_buf = io.BytesIO()
        Image.fromarray(result["original_img"]).save(base_buf, format="PNG", optimize=True)
        base_size = len(base_buf.getvalue())
        
        ratio = result["total_bytes"] / base_size
        gain = (1 - ratio) * 100
        
        log(f"[VAL Ep{ep}] Regs={result['num_regions']} Size={result['total_bytes']} Base={base_size} Ratio={ratio:.3f} Gain={gain:+.1f}%")
        
        if ep % 5 == 0:
            visualize_compression_result(result, f"ep{ep}")

    # Final: Test on multiple test images
    log("\n=== TESTING ON TEST IMAGES ===")
    torch.save(enc.state_dict(), "enc.pth")
    torch.save(pred.state_dict(), "pred.pth")
    
    test_indices = [5, 10, 15]  # Test on 3 different images
    for idx in test_indices:
        if idx < len(ds):
            test_img = ds.load_full_image(idx).to(Config.DEVICE)
            th, tw = test_img.shape[1], test_img.shape[2]
            
            result = inference_compress_reconstruct(test_img, enc, pred, th, tw, Config.DEVICE,
                                                   reconstruct_mode="mean_color")
            
            base_buf = io.BytesIO()
            Image.fromarray(result["original_img"]).save(base_buf, format="PNG", optimize=True)
            base_size = len(base_buf.getvalue())
            
            ratio = result["total_bytes"] / base_size
            gain = (1 - ratio) * 100
            
            log(f"[TEST {idx}] Size={result['total_bytes']} Base={base_size} Ratio={ratio:.3f} Gain={gain:+.1f}%")
            visualize_compression_result(result, f"test{idx}")

if __name__ == "__main__":
    main()
