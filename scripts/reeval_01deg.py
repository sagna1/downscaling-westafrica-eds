# =============================================================================
# RÉ-ÉVALUATION HONNÊTE À 0.1° (corrigerOS1)
# Le run d'origine a été fait sur une cible IMERG interpolée à 0.05° (380x680),
# alors qu'IMERG est nativement à 0.1° (~11 km). On ré-évalue les MÊMES modèles
# entraînés en agrégeant prédictions ET cible de 0.05° -> 0.1° (blocs 2x2),
# puis on recalcule toutes les métriques à la vraie résolution 0.1° (190x340).
# Aucun ré-entraînement : mêmes checkpoints, même normalisation.
# =============================================================================
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'src'))
import downscaling_highres as ds   # modèle, dataset, collate, predict_all, metriques

CKPT_DIR = os.environ.get('CKPT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'checkpoints_highres'))
OUT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'results')
os.makedirs(OUT_DIR, exist_ok=True)

# ---- stats de normalisation EXACTES du run (sauvegardées à l'entraînement) ----
st = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'results','norm_stats.npz'))
mu, sigma = st['mu'], st['sigma']
y_min, y_max = float(st['y_min']), float(st['y_max'])
print(f"Stats: y_min={y_min} y_max={y_max} mu/sigma shape={mu.shape}")

# ---- indices test (2018-2020), JJASO=153 j/an, 2000..2020 -> N=3213 ----
DPY = 153
te_idx = slice((2018-2000)*DPY, (2021-2000)*DPY)   # 2754:3213 -> 459 j

# ---- chargement test seulement (mmap pour la RAM) ----
print("Chargement test...")
X = np.load(ds.X_PATH, mmap_mode='r')[te_idx]      # (459,18,76,136)
Y = np.load(ds.Y_PATH, mmap_mode='r')[te_idx]      # (459, 1,380,680) mm/j
X = np.nan_to_num(np.asarray(X, np.float32))
Y = np.clip(np.nan_to_num(np.asarray(Y, np.float32)), 0., 300.)
print(f"  X_te {X.shape}  Y_te {Y.shape}")

# ---- normalisation identique à l'entraînement ----
Xn = (X - mu[None,:,None,None]) / sigma[None,:,None,None]
Xn = np.nan_to_num(Xn)
Yln = np.log1p(Y)
Yn  = np.clip((Yln - y_min) / (y_max - y_min + 1e-8), 0., 1.)

Xt = torch.from_numpy(Xn.astype(np.float32))
Yt = torch.from_numpy(Yn.astype(np.float32))

te_set = ds.DownscalingDataset(Xt, Yt, augment=False)
te_loader = DataLoader(te_set, batch_size=4, shuffle=False,
                       collate_fn=ds.DownscalingDataset.collate_upsample,
                       num_workers=2, pin_memory=True)

def to_01deg(a):
    """(N,1,380,680) 0.05° -> (N,1,190,340) 0.1° par moyenne de blocs 2x2."""
    N, C, H, W = a.shape
    return a.reshape(N, C, H//2, 2, W//2, 2).mean(axis=(3, 5))

rows05, rows01 = {}, {}
for name, cfg in ds.CONFIGS.items():
    ckpt_path = os.path.join(CKPT_DIR, f"{name.replace('-','_')}_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"  {name}: pas de checkpoint, skip"); continue
    model = ds.UNet(Xt.shape[1], Yt.shape[1], **cfg).to(ds.device)
    ckpt = torch.load(ckpt_path, map_location=ds.device)
    model.load_state_dict({k: v.to(ds.device) for k, v in ckpt['model_state_dict'].items()})
    model.eval()

    preds_mm, tgt_mm = ds.predict_all(model, te_loader, y_min, y_max)   # (459,1,380,680)

    # --- métriques 0.05° (contrôle de reproduction du log) ---
    m05  = ds.compute_metrics(preds_mm, tgt_mm)
    c05  = ds.compute_categorical(preds_mm, tgt_mm, (5,20,40))
    f205 = ds.fss(preds_mm[:,0], tgt_mm[:,0], 20, 5)
    rows05[name] = {**m05, 'FSS_20mm':f205,
                    'CSI_40mm':c05[40]['CSI'], 'CSI_20mm':c05[20]['CSI']}

    # --- agrégation -> 0.1° et métriques honnêtes ---
    p01, t01 = to_01deg(preds_mm), to_01deg(tgt_mm)
    m01  = ds.compute_metrics(p01, t01)
    c01  = ds.compute_categorical(p01, t01, (5,20,40))
    f20_s2 = ds.fss(p01[:,0], t01[:,0], 20, 2)   # voisinage ~ physique du 5px@0.05°
    f20_s5 = ds.fss(p01[:,0], t01[:,0], 20, 5)   # même nb de pixels que l'article
    rows01[name] = {**m01, 'FSS_20mm(s2)':f20_s2, 'FSS_20mm(s5)':f20_s5,
                    'CSI_40mm':c01[40]['CSI'], 'CSI_20mm':c01[20]['CSI'],
                    'POD_40mm':c01[40]['POD']}
    print(f"\n  {name}")
    print(f"    0.05° : RMSE={m05['RMSE']:.3f} NSE={m05['NSE']:.4f} r={m05['r']:.4f} "
          f"P99={m05['P99_ratio']:.2f} FSS20={f205:.3f} CSI40={c05[40]['CSI']:.3f}")
    print(f"    0.10° : RMSE={m01['RMSE']:.3f} NSE={m01['NSE']:.4f} r={m01['r']:.4f} "
          f"P99={m01['P99_ratio']:.2f} FSS20(s2)={f20_s2:.3f} CSI40={c01[40]['CSI']:.3f}")

import pandas as pd
pd.DataFrame(rows05).T.round(4).to_csv(os.path.join(OUT_DIR,'metrics_005deg_check.csv'))
pd.DataFrame(rows01).T.round(4).to_csv(os.path.join(OUT_DIR,'metrics_01deg_honest.csv'))
print(f"\nSauvé dans {OUT_DIR}/ : metrics_005deg_check.csv, metrics_01deg_honest.csv")
