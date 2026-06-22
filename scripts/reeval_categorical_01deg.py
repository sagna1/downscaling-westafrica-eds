# Table catégorielle (POD/FAR/CSI à 5/20/40 mm) à 0.1° pour CBAM / CBAM+QM / cGAN
import os, sys, warnings, numpy as np, torch, pandas as pd
from torch.utils.data import DataLoader
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'src'))
import downscaling_highres as ds
from qm_highres import MonthlyQM
CKPT=os.environ.get('CKPT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'checkpoints_highres')); OUT=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'results')
st=np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'results','norm_stats.npz')); mu,sigma=st['mu'],st['sigma']
y_min,y_max=float(st['y_min']),float(st['y_max']); DPY=153
mseq=[];
for _ in range(21):
    for m,nd in zip([6,7,8,9,10],[30,31,31,30,31]): mseq+=[m]*nd
mseq=np.array(mseq); months_tr=mseq[0:16*DPY]; months_te=mseq[18*DPY:21*DPY]
Xall=np.load(ds.X_PATH,mmap_mode='r'); Yall=np.load(ds.Y_PATH,mmap_mode='r')
def Xn(sl): X=np.nan_to_num(np.asarray(Xall[sl],np.float32)); return ((X-mu[None,:,None,None])/sigma[None,:,None,None]).astype(np.float32)
def Yn(sl): Y=np.clip(np.nan_to_num(np.asarray(Yall[sl],np.float32)),0,300); return np.clip((np.log1p(Y)-y_min)/(y_max-y_min+1e-8),0,1).astype(np.float32)
def agg(a): n,c,H,W=a.shape; return a.reshape(n,c,H//2,2,W//2,2).mean((3,5))
def load(cfg,f):
    M=ds.UNet(18,1,**cfg).to(ds.device); ck=torch.load(f'{CKPT}/{f}',map_location=ds.device)
    M.load_state_dict({k:v.to(ds.device) for k,v in ck['model_state_dict'].items()}); M.eval(); return M
def loader(a,b): return DataLoader(ds.DownscalingDataset(torch.from_numpy(a),torch.from_numpy(b),augment=False),batch_size=4,shuffle=False,collate_fn=ds.DownscalingDataset.collate_upsample,num_workers=2)
te=slice(18*DPY,21*DPY); tr=slice(0,16*DPY)
te_l=loader(Xn(te),Yn(te))
cbam=load(ds.CONFIGS['UNet-CBAM'],'UNet_CBAM_best.pth')
p_mm,t_mm=ds.predict_all(cbam,te_l,y_min,y_max)
cbam01=agg(p_mm)[:,0]; obs01=agg(t_mm)[:,0]
tr_l=loader(Xn(tr),Yn(tr)); ptr,otr=ds.predict_all(cbam,tr_l,y_min,y_max)
qm=MonthlyQM(500); qm.fit(agg(ptr)[:,0],agg(otr)[:,0],months_tr)
cbamqm01=qm.transform(cbam01.copy(),months_te)
cg=load(ds.CONFIGS['UNet-CBAM'],'UNet_CBAM_highres_cGAN_best.pth')
pg,_=ds.predict_all(cg,te_l,y_min,y_max); cgan01=agg(pg)[:,0]
rows={}
for nm,arr in [('U-Net-CBAM',cbam01),('U-Net-CBAM + QM',cbamqm01),('U-Net-CBAM cGAN',cgan01)]:
    cat=ds.compute_categorical(arr[:,None],obs01[:,None],(5,20,40))
    for t in (5,20,40):
        rows[f'{nm}|{t}mm']={'POD':round(cat[t]['POD'],3),'FAR':round(cat[t]['FAR'],3),'CSI':round(cat[t]['CSI'],3)}
pd.DataFrame(rows).T.to_csv(f'{OUT}/table_categorical_01deg.csv')
print(pd.DataFrame(rows).T.to_string()); print('OK')
