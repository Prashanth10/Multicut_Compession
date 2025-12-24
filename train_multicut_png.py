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
# 1) CONFIG
# ============================================================
class Config:
    DATA_DIR = "./DIV2K_train_HR"
    PATCH = 64
    BATCH = 4
    EPOCHS = 10
    LR = 1e-4

    # Loss weights
    LAMBDA_BOUNDARY = 0.5    # Penalize cuts
    LAMBDA_CUTRATE = 5.0     # Regularize cut percentage
    CUTRATE_TARGET = 0.15    # Target ~15% cuts
    PIX_SCALE = 15.0         # Sharpness of pixel-diff proxy

    # Inference Tuning
    GAEC_MERGE_THRESHOLD = 0.5 
    ZLIB_LEVEL = 6
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# ============================================================
# 2) DATASET
# ============================================================
class DIV2KDataset(Dataset):
    def __init__(self, root_dir, patch_size):
        os.makedirs(root_dir, exist_ok=True)
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.png")))
        if len(self.files) == 0:
            log(f"WARNING: No images found in {root_dir}. Using DUMMY noise.")
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
# 3) MODEL
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
        x = torch.cat([fu, fv, orient], dim=1)
        return self.net(x)

# ============================================================
# 4) GRID GRAPH
# ============================================================
def grid_edges(H, W, device):
    rs = torch.arange(H, device=device)
    cs = torch.arange(W, device=device)
    r, c = torch.meshgrid(rs, cs, indexing="ij")

    # Horizontal
    mh = c < (W - 1)
    uh = (r[mh] * W + c[mh]).reshape(-1)
    vh = (r[mh] * W + (c[mh] + 1)).reshape(-1)
    oh = torch.full((uh.numel(), 1), -1.0, device=device)

    # Vertical
    mv = r < (H - 1)
    uv = (r[mv] * W + c[mv]).reshape(-1)
    vv = ((r[mv] + 1) * W + c[mv]).reshape(-1)
    ov = torch.full((uv.numel(), 1), +1.0, device=device)

    u = torch.cat([uh, uv], dim=0)
    v = torch.cat([vh, vv], dim=0)
    o = torch.cat([oh, ov], dim=0)
    return u, v, o

# ============================================================
# 5) LOSS
# ============================================================
def proxy_loss(img_bchw, u, v, merge_logits):
    prob_merge = torch.sigmoid(merge_logits)
    prob_cut = 1.0 - prob_merge

    loss_boundary = prob_cut.mean()

    _, C, H, W = img_bchw.shape
    flat = img_bchw.reshape(1, C, -1)[0]
    pu = flat[:, u]; pv = flat[:, v]
    diff = (pu - pv).abs().mean(dim=0, keepdim=True).T
    
    target_cut = torch.tanh(diff * Config.PIX_SCALE)
    loss_pixel = nn.MSELoss()(prob_cut, target_cut)

    cutrate = prob_cut.mean()
    loss_cutrate = (cutrate - Config.CUTRATE_TARGET) ** 2

    total = loss_pixel + Config.LAMBDA_BOUNDARY * loss_boundary + Config.LAMBDA_CUTRATE * loss_cutrate
    return total, {"pix": loss_pixel.item(), "bnd": loss_boundary.item(), "rate": cutrate.item()}

# ============================================================
# 6) GAEC (ROBUST & FIXED)
# ============================================================
def gaec_additive(num_nodes, u_np, v_np, gain_np, merge_threshold=0.0):
    if len(u_np) == 0: return np.arange(num_nodes), 0
    max_idx = max(u_np.max(), v_np.max())
    if max_idx >= num_nodes: num_nodes = max_idx + 1

    full_adj = [dict() for _ in range(num_nodes)]
    for i in range(len(u_np)):
        U, V, G = u_np[i], v_np[i], gain_np[i]
        if U == V: continue
        full_adj[U][V] = full_adj[U].get(V, 0.0) + float(G)
        full_adj[V][U] = full_adj[V].get(U, 0.0) + float(G)

    parent = np.arange(num_nodes, dtype=np.int32)
    alive = np.ones(num_nodes, dtype=bool)

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
        alive[root_v] = False
        merges_count += 1
        
        neighbors_v = list(full_adj[root_v].items())
        for neighbor, weight_v in neighbors_v:
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
# 7) ENCODING (FIXED SHAPE)
# ============================================================
def paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p-a), abs(p-b), abs(p-c)
    if pa <= pb and pa <= pc: return a
    if pb <= pc: return b
    return c

def encode_regions(img_u8, labels):
    H, W, C = img_u8.shape
    
    # CRITICAL FIX: Reshape 1D labels to 2D image grid
    labels_2d = labels.reshape(H, W)
    K = labels_2d.max() + 1
    
    seg_bits = (H*(W-1) + (H-1)*W)
    seg_bytes = (seg_bits + 7) // 8
    
    payload = 0
    for k in range(K):
        mask = (labels_2d == k) # Now 2D match
        ys, xs = np.where(mask) # Now returns 2 arrays
        if len(ys) == 0: continue
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        
        crop = img_u8[y0:y1+1, x0:x1+1]
        m = mask[y0:y1+1, x0:x1+1]
        
        data = bytearray()
        for y in range(crop.shape[0]):
            data.append(4)
            for x in range(crop.shape[1]):
                if not m[y, x]:
                    data.extend([0,0,0])
                    continue
                for c in range(3):
                    val = int(crop[y, x, c])
                    l = int(crop[y, x-1, c]) if x>0 else 0
                    u = int(crop[y-1, x, c]) if y>0 else 0
                    ul= int(crop[y-1, x-1, c]) if x>0 and y>0 else 0
                    pred = paeth(l, u, ul)
                    data.append((val - pred) & 0xFF)
        
        payload += len(zlib.compress(data, level=Config.ZLIB_LEVEL))
        
    return seg_bytes + payload, K

# ============================================================
# 8) MAIN
# ============================================================
def main():
    log(f"Device: {Config.DEVICE}")
    ds = DIV2KDataset(Config.DATA_DIR, Config.PATCH)
    
    n_val = int(len(ds) * 0.1)
    train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val])
    
    train_dl = DataLoader(train_ds, batch_size=Config.BATCH, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    enc = EncoderCNN().to(Config.DEVICE)
    pred = EdgePredictor().to(Config.DEVICE)
    opt = optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=Config.LR)
    
    u, v, orient = grid_edges(Config.PATCH, Config.PATCH, Config.DEVICE)
    
    for ep in range(Config.EPOCHS):
        enc.train(); pred.train()
        for bi, img in enumerate(train_dl):
            img = img.to(Config.DEVICE)
            B = img.shape[0]
            
            feat = enc(img)
            feat_flat = feat.permute(0,2,3,1).reshape(B, -1, 64)
            
            total_loss = 0
            for i in range(B):
                fu = feat_flat[i, u]
                fv = feat_flat[i, v]
                logits = pred(fu, fv, orient)
                loss, _ = proxy_loss(img[i:i+1], u, v, logits)
                total_loss += loss
            
            total_loss /= B
            opt.zero_grad(); total_loss.backward(); opt.step()
            
            if bi % 20 == 0:
                log(f"Ep{ep} B{bi} Loss={total_loss.item():.4f}")

        enc.eval(); pred.eval()
        if len(val_dl) > 0:
            with torch.no_grad():
                img = next(iter(val_dl)).to(Config.DEVICE)
                feat = enc(img)
                feat_flat = feat.permute(0,2,3,1).reshape(1, -1, 64)
                
                fu = feat_flat[0, u]; fv = feat_flat[0, v]
                logits = pred(fu, fv, orient).squeeze(1)
                gains = logits.cpu().numpy()
                
                max_idx = max(u.max().item(), v.max().item())
                labels, merges = gaec_additive(max_idx+1, u.cpu().numpy(), v.cpu().numpy(), gains, 
                                             merge_threshold=Config.GAEC_MERGE_THRESHOLD)
                
                img_u8 = (img[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                sz, K = encode_regions(img_u8, labels)
                
                buf = io.BytesIO()
                Image.fromarray(img_u8).save(buf, format="PNG")
                base = len(buf.getvalue())
                
                log(f"[VAL] Regs={K} Merges={merges} Size={sz} Base={base}")
                
                os.makedirs("viz", exist_ok=True)
                viz = (labels.reshape(Config.PATCH, Config.PATCH) * 50 % 255).astype(np.uint8)
                Image.fromarray(viz, mode='L').save(f"viz/ep{ep}_val.png")

    torch.save(enc.state_dict(), "enc.pth")
    torch.save(pred.state_dict(), "pred.pth")
    log("Done")

if __name__ == "__main__":
    main()
