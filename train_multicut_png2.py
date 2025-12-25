import os, glob, time, math, io
import numpy as np
from PIL import Image
import zlib
import heapq

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
    LR = 5e-4  # Slightly higher learning rate

    # NEW: Direct merge-gain regression (not sigmoid probabilities!)
    # Positive gain = "Merge is beneficial"
    # Negative gain = "Cut is beneficial"
    # Large magnitude = "Confident decision"
    
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
    """
    Outputs MERGE GAIN logits (unbounded, not sigmoid-clamped!)
    Large positive = "Merge is very good"
    Large negative = "Cut is very good"
    """
    def __init__(self, c_feat=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*c_feat + 1, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)  # Unbounded output!
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
# LOSS: Direct Merge-Gain Regression
# ============================================================
def merge_gain_loss(img_bchw, u, v, merge_gain_logits):
    """
    KEY CHANGE: Regress to target merge gains directly!
    
    Target Merge Gain:
    - If pixels are similar (low diff): target = +1.0 (merge is good)
    - If pixels are different (high diff): target = -1.0 (cut is good)
    
    The network learns to predict these gains, which GAEC then uses.
    """
    _, C, H, W = img_bchw.shape
    flat = img_bchw.reshape(1, C, -1)[0]
    pu = flat[:, u]
    pv = flat[:, v]
    diff = (pu - pv).abs().mean(dim=0, keepdim=True).T  # (E, 1)
    
    # Target: High diff -> Cut (gain = -1), Low diff -> Merge (gain = +1)
    # Use smooth transition with sigmoid
    THRESHOLD = 0.10
    target_gain = 2.0 * torch.sigmoid((THRESHOLD - diff) * 30.0) - 1.0  # Maps to [-1, 1]
    
    # Regression loss: L2 between predicted and target gains
    pred_gain = torch.tanh(merge_gain_logits)  # Clamp to [-1, 1]
    loss = ((pred_gain - target_gain) ** 2).mean()
    
    # Debug stats
    prob_cut = torch.sigmoid(-merge_gain_logits)  # Inverse of merge prob
    n_high_diff = (diff > THRESHOLD).sum().item()
    
    return loss, {
        "loss": loss.item(),
        "target_gain_mean": target_gain.mean().item(),
        "pred_gain_mean": pred_gain.mean().item(),
        "pred_gain_std": pred_gain.std().item(),
        "pred_gain_min": pred_gain.min().item(),
        "pred_gain_max": pred_gain.max().item(),
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
# REAL PNG ENCODING
# ============================================================
def encode_regions_rgb_png(img_u8, labels):
    H, W, C = img_u8.shape
    labels_2d = labels.reshape(H, W)
    K = labels_2d.max() + 1
    
    seg_overhead = K * 10
    total_payload = 0
    
    for k in range(K):
        mask = (labels_2d == k)
        ys, xs = np.where(mask)
        if len(ys) == 0: continue
        
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        crop = img_u8[y0:y1+1, x0:x1+1].copy()
        m = mask[y0:y1+1, x0:x1+1]
        crop[~m] = 0
        
        region_img = Image.fromarray(crop, mode="RGB")
        buf = io.BytesIO()
        region_img.save(buf, format="PNG", optimize=True)
        total_payload += len(buf.getvalue())
    
    return (seg_overhead + total_payload) // 8, K

# ============================================================
# MAIN
# ============================================================
def main():
    log(f"Device: {Config.DEVICE}")
    log(f"Loss: Direct Merge-Gain Regression (target: high-diff->-1, low-diff->+1)")
    
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
    log(f"Grid: {Config.PATCH}x{Config.PATCH}, Total Edges: {n_edges}")
    
    for ep in range(Config.EPOCHS):
        enc.train(); pred.train()
        
        for bi, img in enumerate(train_dl):
            img = img.to(Config.DEVICE)
            B = img.shape[0]
            
            feat = enc(img)
            feat_flat = feat.permute(0,2,3,1).reshape(B, -1, 64)
            
            loss_batch = 0
            debug_batch = None
            
            for i in range(B):
                fu = feat_flat[i, u]
                fv = feat_flat[i, v]
                merge_logits = pred(fu, fv, torch.zeros(len(u), 1, device=Config.DEVICE))
                loss_i, dbg = merge_gain_loss(img[i:i+1], u, v, merge_logits)
                loss_batch += loss_i
                if debug_batch is None:
                    debug_batch = dbg
            
            loss_batch /= B
            opt.zero_grad()
            loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(pred.parameters()), 1.0)
            opt.step()
            
            if bi % 20 == 0:
                log(f"Ep{ep} B{bi} Loss={loss_batch.item():.6f} | "
                    f"Target={debug_batch['target_gain_mean']:+.3f} Pred={debug_batch['pred_gain_mean']:+.3f} "
                    f"[min={debug_batch['pred_gain_min']:+.3f}, max={debug_batch['pred_gain_max']:+.3f}, std={debug_batch['pred_gain_std']:.3f}] | "
                    f"HighDiffEdges={debug_batch['n_high_diff_edges']}/{n_edges}")

        enc.eval(); pred.eval()
        if len(val_dl) > 0:
            with torch.no_grad():
                img_val = next(iter(val_dl)).to(Config.DEVICE)
                feat = enc(img_val)
                feat_flat = feat.permute(0,2,3,1).reshape(1, -1, 64)
                
                fu = feat_flat[0, u]
                fv = feat_flat[0, v]
                merge_logits = pred(fu, fv, torch.zeros(len(u), 1, device=Config.DEVICE))
                gains = merge_logits.squeeze(1).cpu().numpy()
                
                max_idx = max(u.max().item(), v.max().item())
                labels, merges = gaec_additive(max_idx+1, u.cpu().numpy(), v.cpu().numpy(), gains, 
                                             merge_threshold=0.0)
                
                img_u8 = (img_val[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                sz, K = encode_regions_rgb_png(img_u8, labels)
                
                buf = io.BytesIO()
                Image.fromarray(img_u8).save(buf, format="PNG", optimize=True)
                base = len(buf.getvalue())
                
                ratio = sz / base if base > 0 else 1.0
                gain_pct = (1.0 - ratio) * 100
                log(f"[VAL] Regs={K} Merges={merges}/{n_edges-1} Size={sz} Base={base} Ratio={ratio:.3f} Gain={gain_pct:+.1f}%")
                
                os.makedirs("viz", exist_ok=True)
                viz = (labels.reshape(Config.PATCH, Config.PATCH) * 50 % 255).astype(np.uint8)
                Image.fromarray(viz, mode='L').save(f"viz/ep{ep}_val.png")

    torch.save(enc.state_dict(), "enc.pth")
    torch.save(pred.state_dict(), "pred.pth")
    log("Done")

if __name__ == "__main__":
    main()
