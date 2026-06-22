# =============================================================================
# FIGURES QUANTITATIVES À 0.1° (corrigerOS1)
# Recalcule les champs 0.1° nécessaires (scatter/distribution/Taylor) et trace
# les 7 figures depuis les CSV honnêtes. Sortie -> corrigerOS1/figures/
# =============================================================================
import os, sys, warnings, numpy as np, torch, pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'src'))
import downscaling_highres as ds
from qm_highres import MonthlyQM

CKPT=os.environ.get('CKPT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'checkpoints_highres')); RES=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'results')
FIG=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'figures'); os.makedirs(FIG,exist_ok=True)
st=np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'results','norm_stats.npz')); mu,sigma=st['mu'],st['sigma']
y_min,y_max=float(st['y_min']),float(st['y_max']); DPY=153
mseq=[]
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

ARR=f'{RES}/arrays_01deg.npz'
if os.path.exists(ARR):
    print('Chargement arrays...'); z=np.load(ARR); D={k:z[k] for k in z.files}
else:
    print('Calcul des champs 0.1°...', flush=True)
    te_l=loader(Xn(te),Yn(te))
    _,t_mm=ds.predict_all(load(ds.CONFIGS['UNet-Simple'],'UNet_Simple_best.pth'),te_l,y_min,y_max)
    obs=agg(t_mm)[:,0]
    def pred01(cfg,f):
        p,_=ds.predict_all(load(cfg,f),te_l,y_min,y_max); return agg(p)[:,0]
    D={'obs':obs,
       'Simple':pred01(ds.CONFIGS['UNet-Simple'],'UNet_Simple_best.pth'),
       'SE':pred01(ds.CONFIGS['UNet-SE'],'UNet_SE_best.pth'),
       'CBAM':pred01(ds.CONFIGS['UNet-CBAM'],'UNet_CBAM_best.pth'),
       'cGAN':pred01(ds.CONFIGS['UNet-CBAM'],'UNet_CBAM_highres_cGAN_best.pth')}
    # QM sur CBAM (fit train 0.1°)
    tr_l=loader(Xn(tr),Yn(tr))
    cb=load(ds.CONFIGS['UNet-CBAM'],'UNet_CBAM_best.pth')
    ptr,otr=ds.predict_all(cb,tr_l,y_min,y_max); cbam_tr=agg(ptr)[:,0]; obs_tr=agg(otr)[:,0]
    qm=MonthlyQM(500); qm.fit(cbam_tr,obs_tr,months_tr)
    D['CBAM_QM']=qm.transform(D['CBAM'].copy(),months_te)
    # BCSD full à 0.1°
    Xtr_r=Xn(tr); Xte_r=Xn(te); Yfine_tr=np.clip(np.nan_to_num(np.asarray(Yall[tr],np.float32)),0,300)[:,0]
    Yc=F.avg_pool2d(torch.from_numpy(Yfine_tr[:,None]),5,5)[:,0].numpy()
    pc_tr=np.zeros((Xtr_r.shape[0],76,136),np.float32); pc_te=np.zeros((Xte_r.shape[0],76,136),np.float32)
    rg=Ridge(alpha=10.0)
    for i in range(76):
        for j in range(136):
            rg.fit(Xtr_r[:,:,i,j],Yc[:,i,j]); pc_tr[:,i,j]=rg.predict(Xtr_r[:,:,i,j]).clip(0); pc_te[:,i,j]=rg.predict(Xte_r[:,:,i,j]).clip(0)
    qmc=MonthlyQM(500); qmc.fit(pc_tr,Yc,months_tr); pc_te_qm=qmc.transform(pc_te,months_te)
    up=lambda a:F.interpolate(torch.from_numpy(a[:,None].astype(np.float32)),size=(190,340),mode='bilinear',align_corners=False)[:,0].numpy()
    ratio=np.clip(obs_tr.mean(0)/np.maximum(up(pc_tr).mean(0),0.1),0.1,10.0)
    D['BCSD_full']=np.clip(up(pc_te_qm)*ratio[None],0,None)
    np.savez_compressed(ARR,**D); print('Arrays sauvés.', flush=True)

obs=D['obs']
# ----- tables -----
perf=pd.read_csv(f'{RES}/table_perf_01deg.csv',index_col=0)
reg =pd.read_csv(f'{RES}/table_regional_01deg.csv',index_col=0)
mon =pd.read_csv(f'{RES}/table_monthly_01deg.csv',index_col=0)
imp =pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'results','channel_importance_highres.csv'))

def save(fig,name):
    fig.savefig(f'{FIG}/{name}.pdf',bbox_inches='tight'); fig.savefig(f'{FIG}/{name}.png',dpi=200,bbox_inches='tight'); plt.close(fig); print('  ->',name,flush=True)

# ===== fig5 permutation importance =====
imp10=imp.head(10).iloc[::-1]
fig,ax=plt.subplots(figsize=(7,5))
lev_color={'850':'#4477AA','700':'#EE6677','500':'#228833'}
cols=[lev_color.get(c[-3:],'#999') for c in imp10['Canal']]
ax.barh(imp10['Canal'],imp10['ΔRMSE_CBAM (mm)'],xerr=imp10['σ_CBAM'],color=cols,edgecolor='k',lw=.4)
ax.set_xlabel(r'$\Delta$RMSE (mm d$^{-1}$)'); ax.set_title('Permutation importance — UNet-CBAM (0.1°)')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=v,label=k+' hPa') for k,v in lev_color.items()],fontsize=8)
ax.grid(axis='x',alpha=.3); save(fig,'fig5_permutation_importance')

# ===== fig4 metrics comparison =====
order=['U-Net-Simple','U-Net-SE','U-Net-CBAM','U-Net-CBAM + QM','U-Net-CBAM cGAN','BCSD full']
mets=[('RMSE','RMSE (mm/j)',True),('NSE','NSE',False),('FSS_20mm','FSS$_{20}$',False),
      ('CSI_40mm','CSI$_{40}$',False),('P99_ratio','P99 ratio',False)]
fig,axes=plt.subplots(1,5,figsize=(18,3.8))
labs=[o.replace('U-Net-','').replace(' cGAN','-cGAN') for o in order]
for ax,(k,lb,low) in zip(axes,mets):
    vals=[perf.loc[o,k] if o in perf.index else 0 for o in order]
    cols=['#4477AA','#EE6677','#228833','#99DD77','#CCBB44','#777777']
    ax.bar(range(len(order)),vals,color=cols,edgecolor='k',lw=.4)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labs,rotation=45,ha='right',fontsize=7)
    ax.set_title(lb,fontweight='bold',fontsize=10); ax.grid(axis='y',alpha=.3)
    if k=='P99_ratio': ax.axhline(1,color='green',ls='--',lw=1)
fig.suptitle('Comparaison multi-métriques (test 2018–2020, 0.1°)',fontweight='bold')
fig.tight_layout(); save(fig,'fig4_metrics_comparison')

# ===== fig6 regional =====
zones=list(reg.index); zlab=['Guinean\n4–10°N','Sudanian\n10–15°N','Sahelian\n15–23°N']
fig,axes=plt.subplots(1,4,figsize=(15,3.6))
for ax,(k,lb) in zip(axes,[('RMSE','RMSE (mm/j)'),('NSE','NSE'),('r','r'),('CSI_40mm','CSI$_{40}$')]):
    ax.bar(zlab,[reg.loc[z,k] for z in zones],color=['#228833','#CCBB44','#BB5566'],edgecolor='k',lw=.4)
    ax.set_title(lb,fontweight='bold'); ax.grid(axis='y',alpha=.3); ax.tick_params(labelsize=8)
fig.suptitle('Performance régionale — UNet-CBAM (0.1°)',fontweight='bold'); fig.tight_layout(); save(fig,'fig6_regional')

# ===== fig7 monthly =====
M=[6,7,8,9,10]; ml=['Jun','Jul','Aug','Sep','Oct']
def series(cfg,k): return [mon.loc[f'{m}_{cfg}',k] for m in M]
fig,axes=plt.subplots(1,4,figsize=(16,3.6))
for ax,(k,lb) in zip(axes,[('RMSE','RMSE (mm/j)'),('FSS_20mm','FSS$_{20}$'),('P99_ratio','P99 ratio'),('NSE','NSE')]):
    for cfg,c in [('CBAM','#228833'),('CBAM+QM','#99DD77'),('cGAN','#CCBB44')]:
        ax.plot(ml,series(cfg,k),'-o',color=c,label=cfg,ms=4)
    ax.set_title(lb,fontweight='bold'); ax.grid(alpha=.3); ax.tick_params(labelsize=8)
    if k=='P99_ratio': ax.axhline(1,color='gray',ls='--',lw=.8)
axes[0].legend(fontsize=8)
fig.suptitle('Évolution mensuelle JJASO (0.1°)',fontweight='bold'); fig.tight_layout(); save(fig,'fig7_monthly')

# ===== fig2 scatter (hexbin) =====
cfgs=[('Simple','U-Net-Simple'),('CBAM','U-Net-CBAM'),('CBAM_QM','U-Net-CBAM + QM'),('cGAN','U-Net-CBAM cGAN')]
fig,axes=plt.subplots(1,4,figsize=(17,4.2))
o=obs.ravel()
for ax,(key,title) in zip(axes,cfgs):
    p=D[key].ravel(); hb=ax.hexbin(o,p,gridsize=60,bins='log',cmap='viridis',extent=(0,120,0,120))
    ax.plot([0,120],[0,120],'r--',lw=1)
    sl=np.polyfit(o,p,1)[0]; ax.plot([0,120],[0,120*sl],'k-',lw=1)
    ax.set_xlim(0,120); ax.set_ylim(0,120); ax.set_xlabel('IMERG (mm/j)')
    ax.set_title(f'{title}\nr={perf.loc[title,"r"]:.2f}, slope={sl:.2f}',fontsize=9)
axes[0].set_ylabel('Predicted (mm/j)')
fig.suptitle('Predicted vs IMERG (test, 0.1°)',fontweight='bold'); fig.tight_layout(); save(fig,'fig2_scatter')

# ===== fig8 distribution + perception-distortion =====
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,4.6))
bins=np.linspace(0.5,120,60)
a1.hist(obs[obs>0.5].ravel(),bins=bins,density=True,histtype='step',color='k',lw=2,label='IMERG')
for key,title,c in [('CBAM','CBAM','#228833'),('CBAM_QM','CBAM+QM','#99DD77'),('cGAN','cGAN','#CCBB44')]:
    d=D[key]; a1.hist(d[d>0.5].ravel(),bins=bins,density=True,histtype='step',color=c,lw=1.5,label=title)
a1.set_yscale('log'); a1.set_xlabel('Precip (mm/j)'); a1.set_ylabel('density'); a1.legend(fontsize=8)
a1.set_title('Rainy-day distribution (>0.5 mm/j)')
for o2,(title,c) in zip(order,[('Simple','#4477AA'),('SE','#EE6677'),('CBAM','#228833'),('CBAM+QM','#99DD77'),('cGAN','#CCBB44'),('BCSD','#777777')]):
    if o2 in perf.index:
        a2.scatter(perf.loc[o2,'RMSE'],perf.loc[o2,'FSS_20mm'],s=80,color=c,edgecolor='k',label=title,zorder=3)
a2.set_xlabel('RMSE (mm/j) — distortion'); a2.set_ylabel('FSS$_{20}$ — perception')
a2.set_title('Perception–distortion'); a2.grid(alpha=.3); a2.legend(fontsize=8)
fig.tight_layout(); save(fig,'fig8_distribution_tradeoff')

# ===== figE Taylor =====
def taylor_stats(p):
    pr=p.ravel(); orr=obs.ravel(); sd=pr.std()/orr.std(); cc=np.corrcoef(pr,orr)[0,1]; return sd,cc
fig=plt.figure(figsize=(7,7)); ax=fig.add_subplot(111,polar=True)
ax.set_thetamin(0); ax.set_thetamax(90); ax.set_rlim(0,1.6)
for key,title,c in [('Simple','Simple','#4477AA'),('SE','SE','#EE6677'),('CBAM','CBAM','#228833'),
                    ('CBAM_QM','CBAM+QM','#99DD77'),('cGAN','cGAN','#CCBB44'),('BCSD_full','BCSD full','#777777')]:
    sd,cc=taylor_stats(D[key]); ax.scatter(np.arccos(np.clip(cc,-1,1)),sd,s=90,color=c,edgecolor='k',label=title,zorder=3)
ax.scatter(0,1,marker='*',s=200,color='red',zorder=4,label='IMERG (ref)')
ax.set_title('Taylor diagram (0.1°)\nangle=corr, radius=norm. std',fontsize=10)
ax.legend(loc='upper right',bbox_to_anchor=(1.35,1.1),fontsize=8); save(fig,'figE_taylor_diagram')

print('TERMINÉ — figures dans',FIG,flush=True)
