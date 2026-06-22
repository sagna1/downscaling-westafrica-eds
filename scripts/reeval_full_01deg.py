# =============================================================================
# RÉ-ÉVALUATION COMPLÈTE À 0.1° (corrigerOS1)
# Tout évalué à la vraie résolution IMERG 0.1° (190x340), facteur 2.5x :
#   U-Net Simple/SE/CBAM | CBAM+QM | CBAM cGAN | CBAM cGAN+QM
#   BCSD (brut + full) + climatologie | régional (zones) | mensuel
# Agrégation 0.05°->0.1° par blocs 2x2. Mêmes checkpoints, même normalisation.
# =============================================================================
import os, sys, warnings
import numpy as np, torch, pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'src'))
import downscaling_highres as ds
from qm_highres import MonthlyQM

CKPT = os.environ.get('CKPT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'checkpoints_highres'))
OUT  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'results')
os.makedirs(OUT, exist_ok=True)
THR = (5, 20, 40)
dev = ds.device

st = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'results','norm_stats.npz'))
mu, sigma = st['mu'], st['sigma']
y_min, y_max = float(st['y_min']), float(st['y_max'])

DPY = 153
tr_sl = slice(0, 16*DPY)            # 2000-2015
te_sl = slice(18*DPY, 21*DPY)       # 2018-2020
mseq = []
for _ in range(21):
    for m, nd in zip([6,7,8,9,10], [30,31,31,30,31]):
        mseq += [m]*nd
mseq = np.array(mseq)
months_tr, months_te = mseq[tr_sl], mseq[te_sl]

Xall = np.load(ds.X_PATH, mmap_mode='r')
Yall = np.load(ds.Y_PATH, mmap_mode='r')

def Xnorm(sl):
    X = np.nan_to_num(np.asarray(Xall[sl], np.float32))
    return ((X - mu[None,:,None,None]) / sigma[None,:,None,None]).astype(np.float32)
def Yraw(sl):
    return np.clip(np.nan_to_num(np.asarray(Yall[sl], np.float32)), 0., 300.)
def Ynorm(Yr):
    return np.clip((np.log1p(Yr) - y_min)/(y_max - y_min + 1e-8), 0., 1.).astype(np.float32)
def agg01(a):                      # blocs 2x2 -> 0.1°
    if a.ndim == 4:
        n,c,H,W = a.shape; return a.reshape(n,c,H//2,2,W//2,2).mean((3,5))
    n,H,W = a.shape; return a.reshape(n,H//2,2,W//2,2).mean((2,4))

def metrics01(p01, o01, fss_scale=2):
    """p01,o01 : (N,190,340) mm -> dict de métriques 0.1°."""
    p4, o4 = p01[:,None], o01[:,None]
    m  = ds.compute_metrics(p4, o4)
    c  = ds.compute_categorical(p4, o4, THR)
    f20 = ds.fss(p01, o01, 20, fss_scale)
    mae = float(np.mean(np.abs(p01 - o01)))
    return {'RMSE':m['RMSE'],'MAE':mae,'NSE':m['NSE'],'r':m['r'],'Slope':m['Slope'],
            'P99_ratio':m['P99_ratio'],'FSS_20mm':f20,
            'CSI_20mm':c[20]['CSI'],'CSI_40mm':c[40]['CSI'],
            'POD_20mm':c[20]['POD'],'POD_40mm':c[40]['POD']}

def load_unet(cfg, ckpt_file):
    model = ds.UNet(18, 1, **cfg).to(dev)
    ck = torch.load(f'{CKPT}/{ckpt_file}', map_location=dev)
    sd = ck['model_state_dict'] if 'model_state_dict' in ck else ck
    model.load_state_dict({k: v.to(dev) for k, v in sd.items()})
    model.eval(); return model

def make_loader(Xn, Yn):
    s = ds.DownscalingDataset(torch.from_numpy(Xn), torch.from_numpy(Yn), augment=False)
    return DataLoader(s, batch_size=4, shuffle=False,
                      collate_fn=ds.DownscalingDataset.collate_upsample, num_workers=2)

# ---------- données ----------
print("Chargement test...", flush=True)
Xte = Xnorm(te_sl); Yte_r = Yraw(te_sl); Yte_n = Ynorm(Yte_r)
te_loader = make_loader(Xte, Yte_n)
# cible de référence à 0.1° (cohérente avec predict_all : denorm(Ynorm))
_, tgt_te_mm = ds.predict_all(load_unet(ds.CONFIGS['UNet-Simple'],'UNet_Simple_best.pth'),
                              te_loader, y_min, y_max)
obs01 = agg01(tgt_te_mm)[:,0]      # (Nte,190,340)
print(f"  obs01 {obs01.shape} mean={obs01.mean():.2f} mm/j", flush=True)

rows = {}
cbam_te_01 = None
# ---------- U-Nets déterministes ----------
for name, cfg, f in [('U-Net-Simple', ds.CONFIGS['UNet-Simple'], 'UNet_Simple_best.pth'),
                     ('U-Net-SE',     ds.CONFIGS['UNet-SE'],     'UNet_SE_best.pth'),
                     ('U-Net-CBAM',   ds.CONFIGS['UNet-CBAM'],   'UNet_CBAM_best.pth')]:
    p_mm,_ = ds.predict_all(load_unet(cfg, f), te_loader, y_min, y_max)
    p01 = agg01(p_mm)[:,0]
    rows[name] = metrics01(p01, obs01)
    if name == 'U-Net-CBAM': cbam_te_01 = p01
    print(f"  {name}: RMSE={rows[name]['RMSE']:.3f} CSI40={rows[name]['CSI_40mm']:.3f}", flush=True)

# ---------- QM (sur CBAM) : fit à 0.1° sur le train ----------
print("Prédictions CBAM train (pour QM)...", flush=True)
Xtr = Xnorm(tr_sl); Ytr_n = Ynorm(Yraw(tr_sl))
tr_loader = make_loader(Xtr, Ytr_n)
cbam = load_unet(ds.CONFIGS['UNet-CBAM'], 'UNet_CBAM_best.pth')
ptr_mm, otr_mm = ds.predict_all(cbam, tr_loader, y_min, y_max)
cbam_tr_01 = agg01(ptr_mm)[:,0]; obs_tr_01 = agg01(otr_mm)[:,0]
del ptr_mm, otr_mm
qm = MonthlyQM(500); qm.fit(cbam_tr_01, obs_tr_01, months_tr)
cbam_qm_te_01 = qm.transform(cbam_te_01.copy(), months_te)
rows['U-Net-CBAM + QM'] = metrics01(cbam_qm_te_01, obs01)
print(f"  CBAM+QM: RMSE={rows['U-Net-CBAM + QM']['RMSE']:.3f} FSS20={rows['U-Net-CBAM + QM']['FSS_20mm']:.3f}", flush=True)

# ---------- cGAN (sur CBAM) ----------
print("Prédictions cGAN test+train...", flush=True)
cgan = load_unet(ds.CONFIGS['UNet-CBAM'], 'UNet_CBAM_highres_cGAN_best.pth')
pg_mm,_ = ds.predict_all(cgan, te_loader, y_min, y_max)
cgan_te_01 = agg01(pg_mm)[:,0]; del pg_mm
rows['U-Net-CBAM cGAN'] = metrics01(cgan_te_01, obs01)
pgtr_mm,_ = ds.predict_all(cgan, tr_loader, y_min, y_max)
cgan_tr_01 = agg01(pgtr_mm)[:,0]; del pgtr_mm
qmg = MonthlyQM(500); qmg.fit(cgan_tr_01, obs_tr_01, months_tr)
cgan_qm_te_01 = qmg.transform(cgan_te_01.copy(), months_te)
rows['U-Net-CBAM cGAN+QM'] = metrics01(cgan_qm_te_01, obs01)
print(f"  cGAN: RMSE={rows['U-Net-CBAM cGAN']['RMSE']:.3f} CSI40={rows['U-Net-CBAM cGAN']['CSI_40mm']:.3f}", flush=True)

# ---------- BCSD à 0.1° ----------
print("BCSD (Ridge coarse -> 0.1°)...", flush=True)
Xtr_r = Xnorm(tr_sl); Xte_r = Xte                      # ERA5 normalisé (76x136)
Ytr_fine = Yraw(tr_sl)[:,0]                             # (Ntr,380,680) mm
def pool5(Yf):
    t = torch.from_numpy(Yf[:,None]); return F.avg_pool2d(t,5,5)[:,0].numpy()
Yc_tr = pool5(Ytr_fine)                                 # (Ntr,76,136)
Hc, Wc = 76, 136
pc_tr = np.zeros((Xtr_r.shape[0], Hc, Wc), np.float32)
pc_te = np.zeros((Xte_r.shape[0], Hc, Wc), np.float32)
rg = Ridge(alpha=10.0, fit_intercept=True)
for i in range(Hc):
    for j in range(Wc):
        rg.fit(Xtr_r[:,:,i,j], Yc_tr[:,i,j])
        pc_tr[:,i,j] = rg.predict(Xtr_r[:,:,i,j]).clip(0)
        pc_te[:,i,j] = rg.predict(Xte_r[:,:,i,j]).clip(0)
qmc = MonthlyQM(500); qmc.fit(pc_tr, Yc_tr, months_tr)
pc_te_qm = qmc.transform(pc_te, months_te)
def up01(arr):                                          # 76x136 -> 190x340 (0.1°)
    t = torch.from_numpy(arr[:,None].astype(np.float32))
    return F.interpolate(t, size=(190,340), mode='bilinear', align_corners=False)[:,0].numpy()
bcsd_raw_01 = up01(pc_te)
bcsd_qm_01  = up01(pc_te_qm)
fine_clim01 = obs_tr_01.mean(0)
coarse_clim01 = np.maximum(up01(pc_tr).mean(0), 0.1)
ratio = np.clip(fine_clim01/coarse_clim01, 0.1, 10.0)
bcsd_full_01 = np.clip(bcsd_qm_01 * ratio[None], 0, None)
rows['BCSD raw']  = metrics01(bcsd_raw_01, obs01)
rows['BCSD full'] = metrics01(bcsd_full_01, obs01)
# climatologie mensuelle (baseline)
clim01 = {m: obs_tr_01[months_tr==m].mean(0) for m in [6,7,8,9,10]}
clim_te_01 = np.stack([clim01[m] for m in months_te])
rows['Climatology'] = metrics01(clim_te_01, obs01)
print(f"  BCSD full: RMSE={rows['BCSD full']['RMSE']:.3f} CSI40={rows['BCSD full']['CSI_40mm']:.3f}", flush=True)

pd.DataFrame(rows).T.round(4).to_csv(f'{OUT}/table_perf_01deg.csv')
print(f"\nTable perf 0.1° -> {OUT}/table_perf_01deg.csv", flush=True)

# ---------- RÉGIONAL (CBAM, 0.1°) ----------
# row r -> lat = 4 + (r+0.5)*0.1 ; row0 = sud (4°N)
lat = 4.0 + (np.arange(190)+0.5)*0.1
zones = {'Guinean (4-10N)': lat<10, 'Sudanian (10-15N)': (lat>=10)&(lat<15), 'Sahelian (15-23N)': lat>=15}
reg = {}
for zname, msk in zones.items():
    reg[zname] = metrics01(cbam_te_01[:,msk,:], obs01[:,msk,:])
pd.DataFrame(reg).T.round(4).to_csv(f'{OUT}/table_regional_01deg.csv')
print(f"Table régional 0.1° -> {OUT}/table_regional_01deg.csv", flush=True)

# ---------- MENSUEL (CBAM base / +QM / cGAN, 0.1°) ----------
mon = {}
for m in [6,7,8,9,10]:
    idx = months_te==m
    mon[f'{m}_CBAM']    = metrics01(cbam_te_01[idx],    obs01[idx])
    mon[f'{m}_CBAM+QM'] = metrics01(cbam_qm_te_01[idx], obs01[idx])
    mon[f'{m}_cGAN']    = metrics01(cgan_te_01[idx],    obs01[idx])
pd.DataFrame(mon).T.round(4).to_csv(f'{OUT}/table_monthly_01deg.csv')
print(f"Table mensuel 0.1° -> {OUT}/table_monthly_01deg.csv", flush=True)
print("\nTERMINÉ.", flush=True)
