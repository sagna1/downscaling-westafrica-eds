# =============================================================================
# DOWNSCALING STATISTIQUE HAUTE RÉSOLUTION — AFRIQUE DE L'OUEST
# Entrée  : ERA5    0.25° ≈ 28 km  (76×136)  → upsample bilinéaire → 380×680
# Sortie  : IMERG   0.05° ≈  5.5 km (380×680) — vrai downscaling ×5
# Modèles : UNet-Simple | UNet-SE | UNet-CBAM
# Données : X_nc_data.npy (channel-first) + Y_nc_data.npy (channel-first)
# =============================================================================

import os
import sys
import random
import warnings
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

SEED         = 42
BATCH_SIZE   = 4       # plus petit — 380×680 pixels par image
NUM_EPOCHS   = 150
LR           = 3e-4
WEIGHT_DECAY = 1e-4
ES_PATIENCE  = 30
WARMUP       = 10
GRAD_CLIP    = 1.0

START_YEAR  = 2000
END_YEAR    = 2020
N_YEARS     = END_YEAR - START_YEAR + 1

# Split temporel — même logique que downscaling_unet.py
TRAIN_YEARS = list(range(2000, 2016))
VAL_YEARS   = list(range(2016, 2018))
TEST_YEARS  = list(range(2018, 2021))

# Résolution
H_ERA5,  W_ERA5  = 76,  136    # ERA5  0.25°
H_IMERG, W_IMERG = 380, 680    # IMERG 0.05°

LAT_MIN, LAT_MAX = 4.0, 23.0
LON_MIN, LON_MAX = -18.0, 16.0

# Fichiers haute résolution
X_PATH   = os.environ.get('X_PATH', '/net/nfs/ssd3/dsagna/geo_shum/X_nc_data.npy')  # (N, 18, 76, 136)
Y_PATH   = os.environ.get('Y_PATH', '/net/nfs/ssd3/dsagna/geo_shum/Y_nc_data.npy')  # (N,  1, 380, 680)
CKPT_DIR = 'checkpoints_highres'
FIG_DIR  = 'figures_highres'

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)


# =============================================================================
# 2. CHARGEMENT ET PRÉTRAITEMENT
# =============================================================================

def load_and_preprocess():
    print("\n[1/4] Chargement des données haute résolution...")
    X_raw = np.load(X_PATH)   # (N, 18, 76, 136) channel-first
    Y_raw = np.load(Y_PATH)   # (N,  1, 380, 680) channel-first, mm/j

    N = X_raw.shape[0]
    print(f"  X : {X_raw.shape}   Y : {Y_raw.shape}")
    print(f"  Facteur downscaling : {Y_raw.shape[2]//X_raw.shape[2]}× "
          f"× {Y_raw.shape[3]//X_raw.shape[3]}×")

    days_per_year = N // N_YEARS
    print(f"  Jours/saison : {days_per_year}")

    # ----- NaN -----
    print("[2/4] Nettoyage des NaN...")
    X_raw = np.nan_to_num(X_raw, nan=0.0)
    Y_raw = np.clip(np.nan_to_num(Y_raw, nan=0.0), 0.0, 300.0)

    # ----- Indices par année -----
    def years_to_idx(years):
        idx = []
        for y in years:
            s = (y - START_YEAR) * days_per_year
            e = s + days_per_year
            if e <= N:
                idx.extend(range(s, e))
        return np.array(idx)

    tr_idx  = years_to_idx(TRAIN_YEARS)
    val_idx = years_to_idx(VAL_YEARS)
    te_idx  = years_to_idx(TEST_YEARS)
    print(f"  Train : {len(tr_idx)} | Val : {len(val_idx)} | Test : {len(te_idx)}")

    # ----- Transformation Y : log1p → min-max [0,1] -----
    Y_log  = np.log1p(Y_raw)
    y_min  = Y_log[tr_idx].min()
    y_max  = Y_log[tr_idx].max()
    Y_norm = np.clip((Y_log - y_min) / (y_max - y_min + 1e-8), 0.0, 1.0)

    # ----- Normalisation X : z-score par canal (stats train) -----
    print("[3/4] Normalisation X (z-score, stats train)...")
    X_tr  = X_raw[tr_idx]          # (n_tr, 18, 76, 136)
    mu    = X_tr.mean(axis=(0, 2, 3))               # (18,)
    sigma = X_tr.std(axis=(0, 2, 3))  + 1e-8       # (18,)

    # Broadcast : (N, 18, 76, 136) ← (18,) reshape → (1, 18, 1, 1)
    X_norm = (X_raw - mu[np.newaxis, :, np.newaxis, np.newaxis]) / \
              sigma[np.newaxis, :, np.newaxis, np.newaxis]
    X_norm = np.nan_to_num(X_norm)

    np.savez(os.path.join(CKPT_DIR, 'norm_stats.npz'),
             mu=mu, sigma=sigma, y_min=float(y_min), y_max=float(y_max))

    # ----- Tenseurs float32 -----
    def to_tensor(arr, idx):
        return torch.from_numpy(arr[idx].astype(np.float32))

    X_tr_t  = to_tensor(X_norm, tr_idx)
    Y_tr_t  = to_tensor(Y_norm, tr_idx)
    X_val_t = to_tensor(X_norm, val_idx)
    Y_val_t = to_tensor(Y_norm, val_idx)
    X_te_t  = to_tensor(X_norm, te_idx)
    Y_te_t  = to_tensor(Y_norm, te_idx)

    print(f"[4/4] ERA5 {tuple(X_tr_t.shape)} | IMERG {tuple(Y_tr_t.shape)}")

    # ----- Mois JJASO -----
    month_seq = []
    for y in range(START_YEAR, START_YEAR + N_YEARS):
        for m, nd in zip([6,7,8,9,10], [30,31,31,30,31]):
            month_seq.extend([m] * nd)
    month_seq = np.array(month_seq[:N])

    Y_raw_tr = Y_raw[tr_idx, 0]    # (n_tr, 380, 680) mm/j pour QM

    return (X_tr_t, Y_tr_t, X_val_t, Y_val_t, X_te_t, Y_te_t,
            y_min, y_max, days_per_year,
            month_seq[tr_idx], month_seq[val_idx], month_seq[te_idx],
            Y_raw_tr, Y_norm[tr_idx])


# =============================================================================
# 3. DATASET AVEC UPSAMPLE BILINÉAIRE À LA VOLÉE
# =============================================================================

class DownscalingDataset(Dataset):
    """
    X (ERA5 0.25°, 76×136) est upsampleé bilinéairement vers 380×680
    pour correspondre à la résolution IMERG à l'entrée du U-Net.
    L'upsample se fait sur GPU via F.interpolate — pas de surcoût mémoire.
    """
    def __init__(self, X, Y, augment=False):
        self.X = X          # (N, 18, 76, 136)
        self.Y = Y          # (N,  1, 380, 680)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]    # (18, 76, 136)
        y = self.Y[idx]    # (1, 380, 680)

        if self.augment and random.random() < 0.5:
            # Bruit gaussien sur ERA5 (2% de l'std normalisé)
            x = x + 0.02 * torch.randn_like(x)
        if self.augment and random.random() < 0.3:
            # Translation ±2 pixels
            sy = random.randint(-2, 2)
            sx = random.randint(-2, 2)
            x = torch.roll(x, (sy, sx), dims=(1, 2))
            y = torch.roll(y, (sy*5, sx*5), dims=(1, 2))

        return x, y

    @staticmethod
    def collate_upsample(batch):
        """
        Collate function : upsample ERA5 (76×136) → IMERG (380×680)
        via interpolation bilinéaire. Fait en CPU ici, transféré sur GPU
        dans la boucle d'entraînement.
        """
        xs, ys = zip(*batch)
        X_batch = torch.stack(xs)                          # (B, 18, 76, 136)
        Y_batch = torch.stack(ys)                          # (B,  1, 380, 680)
        X_up    = F.interpolate(X_batch,
                                size=(H_IMERG, W_IMERG),
                                mode='bilinear',
                                align_corners=False)       # (B, 18, 380, 680)
        return X_up, Y_batch


# =============================================================================
# 4. ARCHITECTURE U-NET (inchangée — traite 380×680 au lieu de 76×136)
# =============================================================================

class SEBlock(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        mid = max(channels // r, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False), nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False), nn.Sigmoid())
    def forward(self, x):
        w = self.pool(x).view(x.shape[0], x.shape[1])
        return x * self.fc(w).view(x.shape[0], x.shape[1], 1, 1)


class CBAM(nn.Module):
    def __init__(self, channels, r=16, k=7):
        super().__init__()
        mid = max(channels // r, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc  = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False))
        self.ch_sig  = nn.Sigmoid()
        self.sp_conv = nn.Conv2d(2, 1, k, padding=k//2, bias=False)
        self.sp_sig  = nn.Sigmoid()
    def forward(self, x):
        ca = self.ch_sig(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))
        x  = x * ca
        sa = self.sp_sig(self.sp_conv(
            torch.cat([torch.mean(x,dim=1,keepdim=True),
                       torch.max(x,dim=1,keepdim=True)[0]], dim=1)))
        return x * sa


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.Wg  = nn.Conv2d(F_g,   F_int, 1, bias=False)
        self.Wx  = nn.Conv2d(F_l,   F_int, 1, bias=False)
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=False), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.Wg(g)
        x1 = self.Wx(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        return x * self.psi(self.relu(g1 + x1))


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c, use_se=False, use_cbam=False, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c,  out_c, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_c, affine=True), nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_c, affine=True), nn.LeakyReLU(0.01, inplace=True))
        self.attn = SEBlock(out_c) if use_se else (CBAM(out_c) if use_cbam else None)
    def forward(self, x):
        x = self.conv(x)
        return self.attn(x) if self.attn else x


class Down(nn.Module):
    def __init__(self, in_c, out_c, **kw):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_c, out_c, **kw))
    def forward(self, x): return self.block(x)


class Up(nn.Module):
    def __init__(self, in_c, out_c, use_se=False, use_cbam=False, use_ag=True):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_c, in_c//2, 1, bias=False))
        self.ag   = AttentionGate(in_c//2, in_c//2, in_c//4) if use_ag else None
        self.conv = DoubleConv(in_c, out_c, use_se=use_se, use_cbam=use_cbam)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy = x2.size(2) - x1.size(2); dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        if self.ag: x2 = self.ag(x1, x2)
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet(nn.Module):
    def __init__(self, n_in, n_out, use_se=False, use_cbam=False, use_ag=True):
        super().__init__()
        kw = dict(use_se=use_se, use_cbam=use_cbam)
        self.enc1 = DoubleConv(n_in, 64,   **kw)
        self.enc2 = Down(64,  128, **kw)
        self.enc3 = Down(128, 256, **kw)
        self.enc4 = Down(256, 512, **kw)
        self.bot  = Down(512, 1024, **kw)
        self.dec1 = Up(1024, 512, use_ag=use_ag, **kw)
        self.dec2 = Up(512,  256, use_ag=use_ag, **kw)
        self.dec3 = Up(256,  128, use_ag=use_ag, **kw)
        self.dec4 = Up(128,   64, use_ag=use_ag, **kw)
        self.head = nn.Conv2d(64, n_out, 1)
    def forward(self, x):
        e1 = self.enc1(x);  e2 = self.enc2(e1)
        e3 = self.enc3(e2); e4 = self.enc4(e3)
        b  = self.bot(e4)
        x  = self.dec1(b, e4); x = self.dec2(x, e3)
        x  = self.dec3(x, e2); x = self.dec4(x, e1)
        return self.head(x)


CONFIGS = {
    'UNet-Simple': dict(use_se=False, use_cbam=False, use_ag=False),
    'UNet-SE':     dict(use_se=True,  use_cbam=False, use_ag=True),
    'UNet-CBAM':   dict(use_se=False, use_cbam=True,  use_ag=True),
}


# =============================================================================
# 5. PERTES
# =============================================================================

def gradient_loss(pred, target):
    lx = F.l1_loss(pred[:,:,:,1:] - pred[:,:,:,:-1],
                   target[:,:,:,1:] - target[:,:,:,:-1])
    ly = F.l1_loss(pred[:,:,1:,:] - pred[:,:,:-1,:],
                   target[:,:,1:,:] - target[:,:,:-1,:])
    return lx + ly

def pinball_loss(pred, target, quantiles=(0.9, 0.95, 0.99)):
    loss = 0.0
    for q in quantiles:
        e = target - pred
        loss += torch.mean(torch.where(e >= 0, q*e, (q-1)*e))
    return loss / len(quantiles)

def publication_loss(pred, target):
    return (F.l1_loss(pred, target)
            + 0.5 * pinball_loss(pred, target, (0.90, 0.95, 0.99))
            + 0.3 * gradient_loss(pred, target))


# =============================================================================
# 6. ENTRAÎNEMENT
# =============================================================================

def train(model, tr_loader, val_loader, name, warm_start=False):
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)

    best_val, no_improve = float('inf'), 0
    history = {'train': [], 'val': []}
    best_weights = None

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*65}")
    print(f"  {name}  ({n_params:,} params)  grille IMERG {H_IMERG}×{W_IMERG}")
    print(f"{'='*65}")

    # Warm-start : charger les poids du meilleur checkpoint existant
    if warm_start:
        ckpt_path = os.path.join(CKPT_DIR, f"{name.replace('-','_')}_best.pth")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict({k: v.to(device) for k, v in ckpt['model_state_dict'].items()})
            best_val = ckpt['val_loss']
            best_weights = ckpt['model_state_dict']
            print(f"  Warm-start depuis checkpoint (val_loss={best_val:.5f})", flush=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        run, n = 0.0, 0
        for Xb, Yb in tr_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            optimizer.zero_grad()
            loss = publication_loss(torch.clamp(model(Xb), 0., 1.), Yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            run += loss.item() * Xb.size(0); n += Xb.size(0)
        tr_loss = run / n

        model.eval()
        run, n = 0.0, 0
        with torch.no_grad():
            for Xb, Yb in val_loader:
                Xb, Yb = Xb.to(device), Yb.to(device)
                run += publication_loss(
                    torch.clamp(model(Xb), 0., 1.), Yb
                ).item() * Xb.size(0)
                n += Xb.size(0)
        val_loss = run / n
        scheduler.step(epoch - 1)
        history['train'].append(tr_loss); history['val'].append(val_loss)

        marker = ''
        if epoch >= WARMUP and val_loss < best_val - 1e-6:
            best_val, no_improve = val_loss, 0
            best_weights = {k: v.cpu().clone() for k,v in model.state_dict().items()}
            torch.save({'model_state_dict': best_weights, 'val_loss': best_val},
                       os.path.join(CKPT_DIR, f"{name.replace('-','_')}_best.pth"))
            marker = '  ← best'
        else:
            no_improve += 1

        if epoch % 10 == 0 or marker:
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS}  "
                  f"Train {tr_loss:.5f}  Val {val_loss:.5f}{marker}", flush=True)

        if no_improve >= ES_PATIENCE:
            print(f"  Early stopping epoch {epoch}", flush=True); break

    if best_weights:
        model.load_state_dict({k: v.to(device) for k,v in best_weights.items()})
    print(f"  Meilleure val : {best_val:.5f}")
    return model, history


# =============================================================================
# 7. ÉVALUATION
# =============================================================================

def denorm_y(y_norm, y_min, y_max):
    return np.expm1(np.clip(y_norm, 0, 1) * (y_max - y_min) + y_min)

@torch.no_grad()
def predict_all(model, loader, y_min, y_max):
    model.eval()
    preds, targets = [], []
    for Xb, Yb in loader:
        p = torch.clamp(model(Xb.to(device)), 0., 1.).cpu().numpy()
        preds.append(p); targets.append(Yb.numpy())
    preds   = np.concatenate(preds,   axis=0)
    targets = np.concatenate(targets, axis=0)
    return denorm_y(preds, y_min, y_max), denorm_y(targets, y_min, y_max)

def compute_metrics(preds_mm, targets_mm):
    p, t  = preds_mm.ravel(), targets_mm.ravel()
    rmse  = np.sqrt(np.mean((p-t)**2))
    ss    = np.sum((p-t)**2); tot = np.sum((t-t.mean())**2)
    nse   = 1 - ss/(tot+1e-12)
    r     = np.corrcoef(p, t)[0,1]
    slope = np.polyfit(t, p, 1)[0]
    p99r  = np.percentile(p, 99) / max(np.percentile(t, 99), 0.1)
    return dict(RMSE=rmse, NSE=nse, r=r, Slope=slope, P99_ratio=p99r)

def compute_categorical(preds_mm, targets_mm, thresholds=(5, 20, 40)):
    out = {}
    for thr in thresholds:
        p_b = (preds_mm.ravel() >= thr).astype(float)
        o_b = (targets_mm.ravel() >= thr).astype(float)
        hits = np.sum((p_b==1)&(o_b==1)); miss = np.sum((p_b==0)&(o_b==1))
        fa   = np.sum((p_b==1)&(o_b==0))
        out[thr] = dict(POD=hits/(hits+miss+1e-12),
                        FAR=fa  /(hits+fa  +1e-12),
                        CSI=hits/(hits+miss+fa+1e-12))
    return out

def fss(pred_mm, obs_mm, threshold, scale):
    from scipy.ndimage import uniform_filter
    fss_vals = []
    for i in range(pred_mm.shape[0]):
        p_bin = (pred_mm[i] >= threshold).astype(float)
        o_bin = (obs_mm[i]  >= threshold).astype(float)
        size  = 2*scale+1
        fp, fo = uniform_filter(p_bin, size), uniform_filter(o_bin, size)
        num = np.mean((fp-fo)**2); den = np.mean(fp**2)+np.mean(fo**2)
        fss_vals.append(1.0 if den<1e-12 else 1.0-num/den)
    return np.mean(fss_vals)


# =============================================================================
# 8. PIPELINE PRINCIPAL
# =============================================================================

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-only', action='store_true',
                    help='Évaluation uniquement (pas d\'entraînement)')
    ap.add_argument('--model', default='all',
                    choices=['Simple', 'SE', 'CBAM', 'all'],
                    help='Modèle à entraîner (défaut: all)')
    args = ap.parse_args()

    print("="*65)
    print("  Downscaling haute résolution ERA5(0.25°) → IMERG(0.05°)")
    print(f"  Facteur 5× | {H_ERA5}×{W_ERA5} → {H_IMERG}×{W_IMERG}")
    if args.eval_only:
        print("  MODE : ÉVALUATION SEULE")
    elif args.model != 'all':
        print(f"  MODE : Entraînement {args.model} uniquement")
    print("="*65)

    # --- Données ---
    (X_tr, Y_tr, X_val, Y_val, X_te, Y_te,
     y_min, y_max, dpyr,
     months_tr, months_val, months_te,
     Y_raw_tr, Y_norm_tr) = load_and_preprocess()

    n_in  = X_tr.shape[1]   # 18
    n_out = Y_tr.shape[1]   # 1

    # DataLoaders avec collate qui fait l'upsample bilinéaire à la volée
    tr_set  = DownscalingDataset(X_tr,  Y_tr,  augment=True)
    val_set = DownscalingDataset(X_val, Y_val, augment=False)
    te_set  = DownscalingDataset(X_te,  Y_te,  augment=False)

    tr_loader  = DataLoader(tr_set,  batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=DownscalingDataset.collate_upsample,
                            num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=DownscalingDataset.collate_upsample,
                            num_workers=2, pin_memory=True)
    te_loader  = DataLoader(te_set,  batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=DownscalingDataset.collate_upsample,
                            num_workers=2, pin_memory=True)

    print(f"\n  Batch shape ERA5 upsampled : (B=4, 18, {H_IMERG}, {W_IMERG})")
    print(f"  Batch shape IMERG          : (B=4,  1, {H_IMERG}, {W_IMERG})")

    # --- Entraînement ---
    trained, histories = {}, {}

    # Filtrer les modèles à entraîner
    configs_to_run = CONFIGS if args.model == 'all' else {
        f'UNet-{args.model}': CONFIGS[f'UNet-{args.model}']}

    for mname, cfg in configs_to_run.items():
        model = UNet(n_in, n_out, **cfg).to(device)
        ckpt_path = os.path.join(CKPT_DIR, f"{mname.replace('-','_')}_best.pth")

        if args.eval_only:
            # Charger le checkpoint pour l'évaluation
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device)
                model.load_state_dict({k: v.to(device) for k, v in ckpt['model_state_dict'].items()})
                print(f"\n  {mname} : checkpoint chargé (val_loss={ckpt['val_loss']:.5f})", flush=True)
            else:
                print(f"\n  {mname} : AUCUN checkpoint trouvé, skip", flush=True)
                continue
            hist = {'train': [], 'val': []}
        else:
            # Warm-start si checkpoint existant
            warm = os.path.exists(ckpt_path)
            model, hist = train(model, tr_loader, val_loader, mname, warm_start=warm)

        trained[mname] = model; histories[mname] = hist

    # --- Évaluation ---
    print("\n" + "="*65 + "\n  ÉVALUATION TEST SET (IMERG 0.05°)\n" + "="*65)
    THRESHOLDS = (5, 20, 40)

    # Baseline bilinéaire (ce que BCSD ferait)
    bilinear_preds = []
    true_targets   = []
    with torch.no_grad():
        for Xb_up, Yb in te_loader:
            # Bilinéaire seul (ERA5 interpolé sans correction)
            bilinear_preds.append(Xb_up.numpy())   # déjà upsampled
            true_targets.append(Yb.numpy())
    # On n'a pas de Y_bilinear en mm directement (ERA5 n'est pas des précipitations)
    # → baseline = climatologie mensuelle

    results = {}
    for name, model in trained.items():
        preds_mm, tgt_mm = predict_all(model, te_loader, y_min, y_max)
        m   = compute_metrics(preds_mm, tgt_mm)
        cat = compute_categorical(preds_mm, tgt_mm, THRESHOLDS)
        f10 = fss(preds_mm[:,0], tgt_mm[:,0], threshold=10, scale=3)
        f20 = fss(preds_mm[:,0], tgt_mm[:,0], threshold=20, scale=5)
        results[name] = {**m, 'FSS_10mm':f10, 'FSS_20mm':f20, **{
            f'CSI_{t}mm': cat[t]['CSI'] for t in THRESHOLDS},  **{
            f'POD_{t}mm': cat[t]['POD'] for t in THRESHOLDS}}
        print(f"\n  {name}")
        print(f"    RMSE={m['RMSE']:.3f}  NSE={m['NSE']:.4f}  r={m['r']:.4f}  "
              f"Slope={m['Slope']:.3f}  P99={m['P99_ratio']:.2f}", flush=True)
        print(f"    FSS_10={f10:.3f}  FSS_20={f20:.3f}  "
              f"CSI_40={cat[40]['CSI']:.3f}  POD_40={cat[40]['POD']:.3f}", flush=True)

    # --- Tableau final ---
    df = pd.DataFrame(results).T.round(4)
    path = os.path.join(FIG_DIR, 'results_highres.csv')
    df.to_csv(path)
    print(f"\n  Tableau sauvé : {path}")
    print("\n  Pipeline haute résolution terminé.")
