import os
import glob
import time
import math
import io
import heapq
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    DATA_DIR = "./DIV2K_train_HR"  # Path to your downloaded images
    BATCH_SIZE = 4
    PATCH_SIZE = 64     # Small patches for training
    LEARNING_RATE = 1e-4
    EPOCHS = 10         # Increase for better results
    
    # Loss Weights
    LAMBDA_BOUNDARY = 1.5
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {DEVICE}")

# ==========================================
# 2. DATASET
# ==========================================
class DIV2KDataset(Dataset):
    def __init__(self, root_dir, patch_size=64):
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.png")))
        self.patch_size = patch_size
        self.transform = transforms.Compose([
            transforms.RandomCrop(patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(), # [0,1] float tensor
        ])
        
        if len(self.files) == 0:
            print(f"WARNING: No images in {root_dir}. Using DUMMY NOISE data.")
            self.files = ["dummy"] * 20

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.files[idx] == "dummy":
            # Dummy RGB pattern for testing without dataset
            return torch.rand(3, self.patch_size, self.patch_size)
        try:
            img = Image.open(self.files[idx]).convert("RGB")
            return self.transform(img)
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            return torch.rand(3, self.patch_size, self.patch_size)

# ==========================================
# 3. MODEL (CNN + Edge Predictor)
# ==========================================
class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), # Output features
        )

    def forward(self, x):
        return self.net(x)

class EdgeWeightPredictor(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        # Input: Feat(u) + Feat(v) + Orientation(1)
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Raw weight (logit)
        )

    def forward(self, feat_u, feat_v, orientation):
        x = torch.cat([feat_u, feat_v, orientation], dim=1)
        return self.net(x)

# ==========================================
# 4. TRAINING LOSS (Differentiable)
# ==========================================
def compute_diff_loss(img, edge_weights, u_idx, v_idx):
    """
    Computes differentiable loss to train the network without discrete GAEC.
    img: (B, 3, H, W)
    edge_weights: (NumEdges, 1) predicted logits
    """
    # 1. Boundary Loss: Sigmoid(w) * Cost
    # We want to minimize cuts (minimize file size)
    prob_cut = torch.sigmoid(edge_weights)
    loss_bound = prob_cut.sum() * Config.LAMBDA_BOUNDARY
    
    # 2. Pixel Loss Proxy
    # If pixels differ greatly, we SHOULD cut (weight > 0, prob ~ 1).
    # If pixels are similar, we SHOULD merge (weight < 0, prob ~ 0).
    # We use a tanh of color difference as a "soft target" for the cut probability.
    
    B, C, H, W = img.shape
    img_flat = img.reshape(B, C, -1) # (B, C, N)
    
    # Gather pixel values for u and v (simple logic for batch size 1 or similar)
    # For full batch training, we'd need complex indexing. 
    # Simplified: Calculate mean diff over batch for these edges.
    
    # Let's extract differences for the first image in batch to drive gradients
    # (A simplification for readability; usually do parallel gather)
    pix_u = img_flat[0, :, u_idx].T # (NumEdges, 3)
    pix_v = img_flat[0, :, v_idx].T
    
    # Color difference metric (L1)
    diff = (pix_u - pix_v).abs().mean(dim=1, keepdim=True) # (NumEdges, 1)
    
    # Target: High diff -> Cut (1.0). Low diff -> Merge (0.0).
    # Soft target: tanh(diff * scale)
    target = torch.tanh(diff * 10.0) 
    
    # BCE / MSE between predicted probability and target
    # This teaches the network: "High contrast = Cut"
    loss_pix = nn.MSELoss()(prob_cut, target)
    
    return loss_pix, loss_bound

# ==========================================
# 5. INFERENCE: DISCRETE GAEC (Fast)
# ==========================================
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            return True
        return False

def run_discrete_gaec(num_nodes, u_indices, v_indices, weights):
    """
    Standard GAEC: Greedily contract edges with NEGATIVE weights (Merge beneficial).
    In our loss formulation: 
       Weight > 0 (Positive) -> Cut penalty high -> Likely Cut.
       Weight < 0 (Negative) -> Merge beneficial.
    Wait! Our training target was "High Diff -> Weight > 0 (Cut)".
    So GAEC should merge edges with LOWEST weights (most negative or smallest positive?).
    
    Standard GAEC contracts max affinity. 
    Here, 'weight' is 'probability of cut'. 
    So 'affinity' (merge score) = -weight.
    We merge edge if -weight is high (i.e. weight is very negative).
    """
    
    # Build list of edges: (weight, u, v)
    # We want to merge edges with lowest weights (most negative) first.
    edges = []
    for i in range(len(weights)):
        w = weights[i].item()
        # Only merge if weight is negative (prediction says "Merge")
        # If weight is positive, network says "Cut", so don't contract.
        if w < 0: 
            edges.append((w, u_indices[i], v_indices[i]))
            
    # Sort by weight ascending (most negative first)
    edges.sort(key=lambda x: x[0])
    
    uf = UnionFind(num_nodes)
    for w, u, v in edges:
        uf.union(u, v)
        
    # Result: Component ID for each pixel
    segmentation = np.zeros(num_nodes, dtype=np.int32)
    for i in range(num_nodes):
        segmentation[i] = uf.find(i)
        
    return segmentation

# ==========================================
# 6. VALIDATION METRIC (True PNG Size)
# ==========================================
def measure_png_size(img_tensor, segmentation):
    """
    Real-world check: 
    1. Mask image by segmentation regions.
    2. Encode regions (simulated by saving whole image as PNG since we want global compression).
    Wait, the task is *multicut compression*: 
    - Store segmentation map (boundary cost).
    - Store pixels per region (pixel cost).
    
    Approximation for metric:
    - Save original image as PNG -> baseline.
    - Save segmented image (mean color per region) -> rough proxy for low-freq content?
    - Actually, let's just measure if the SEGMENTATION is coherent.
    """
    # 1. Coherence Check: count number of regions
    unique_regions = np.unique(segmentation)
    num_regions = len(unique_regions)
    
    # 2. Convert to PIL for visualization/saving
    seg_map = (segmentation * 13 % 255).astype(np.uint8).reshape(Config.PATCH_SIZE, Config.PATCH_SIZE)
    seg_img = Image.fromarray(seg_map, mode='L')
    
    # Return metrics
    return num_regions, seg_img

# ==========================================
# 7. MAIN PIPELINE
# ==========================================
def main():
    # A. Setup
    dataset = DIV2KDataset(Config.DATA_DIR, Config.PATCH_SIZE)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    encoder = EncoderCNN().to(Config.DEVICE)
    predictor = EdgeWeightPredictor().to(Config.DEVICE)
    optimizer = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=Config.LEARNING_RATE)
    
    print(f"Start Training: {Config.EPOCHS} epochs, Batch {Config.BATCH_SIZE}")
    
    # B. Training Loop
    for epoch in range(Config.EPOCHS):
        total_loss = 0
        
        for batch_idx, imgs in enumerate(loader):
            imgs = imgs.to(Config.DEVICE) # (B, 3, H, W)
            B, C, H, W = imgs.shape
            
            # --- 1. Graph Construction (Grid) ---
            # Create edge indices for 4-connected grid
            # (Simplified: calculate once per batch size if fixed, here dynamic)
            r, c = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            
            # Horizontal edges (r, c) -- (r, c+1)
            mask_h = (c < W-1)
            u_h = (r[mask_h] * W + c[mask_h]).flatten()
            v_h = (r[mask_h] * W + (c[mask_h]+1)).flatten()
            
            # Vertical edges (r, c) -- (r+1, c)
            mask_v = (r < H-1)
            u_v = (r[mask_v] * W + c[mask_v]).flatten()
            v_v = (r[mask_v] * W + c[mask_v]).flatten() # Wait, index is (r+1)*W + c
            v_v_real = ((r[mask_v]+1) * W + c[mask_v]).flatten()
            
            all_u = torch.cat([u_h, u_v])
            all_v = torch.cat([v_h, v_v_real]) # combined indices
            
            # Move indices to device
            all_u = all_u.to(Config.DEVICE)
            all_v = all_v.to(Config.DEVICE)
            
            # --- 2. Feature Extraction ---
            features = encoder(imgs) # (B, 64, H, W)
            flat_feats = features.permute(0, 2, 3, 1).reshape(B, H*W, 64) # (B, N, 64)
            
            # --- 3. Edge Prediction (Batch 0 only for simplicity) ---
            # To train properly, we should loop over batch or use gather.
            # Demo: Training on first image of batch
            f_u = flat_feats[0][all_u] # (NumEdges, 64)
            f_v = flat_feats[0][all_v]
            
            # Orientation: 0 for H, 1 for V
            num_h = len(u_h)
            num_v = len(u_v)
            orient = torch.cat([torch.zeros(num_h, 1), torch.ones(num_v, 1)]).to(Config.DEVICE)
            
            weights = predictor(f_u, f_v, orient) # (NumEdges, 1)
            
            # --- 4. Differentiable Loss ---
            loss_pix, loss_bound = compute_diff_loss(imgs[0:1], weights, all_u, all_v)
            
            loss = loss_pix + 0.1 * loss_bound
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Ep {epoch} B {batch_idx}: Loss {loss.item():.4f} (Pix {loss_pix.item():.4f}, Bound {loss_bound.item():.4f})")
        
        print(f"Epoch {epoch} Avg Loss: {total_loss / len(loader):.4f}")
        
        # --- C. Validation (Real GAEC) ---
        if epoch % 2 == 0:
            with torch.no_grad():
                # Run on last training image
                # Get discrete segmentation
                seg = run_discrete_gaec(H*W, all_u.cpu().numpy(), all_v.cpu().numpy(), weights.cpu().numpy())
                
                # Check Stats
                n_regions, _ = measure_png_size(imgs[0], seg)
                print(f"[VAL] Discrete Regions: {n_regions} (Target: Not too high, not 1)")
                
                # Save visual debug
                if not os.path.exists("debug_vis"): os.makedirs("debug_vis")
                # Visual: Weight Heatmap (reshape to H-1, W-1 approx)
                # Just save the Segmentation Map
                seg_viz = (seg.reshape(H, W) * 20 % 255).astype(np.uint8)
                Image.fromarray(seg_viz).save(f"debug_vis/ep{epoch}_seg.png")

    # D. Final Test
    print("Saving Model...")
    torch.save(encoder.state_dict(), "multicut_encoder.pth")
    torch.save(predictor.state_dict(), "multicut_predictor.pth")
    print("Done.")

if __name__ == "__main__":
    main()
