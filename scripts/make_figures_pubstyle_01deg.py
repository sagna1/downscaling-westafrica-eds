# =============================================================================
# FIGURES DE PUBLICATION — STYLE D'ORIGINE, DONNÉES HONNÊTES 0.1°
# Reproduit EXACTEMENT la structure de figures_publication.py /
# figures_diagnostic.py (palette, panneaux, cartopy, annotations) mais
# alimentée par les prédictions ré-évaluées à 0.1° (corrigerOS1/results).
# Sortie -> corrigerOS1/figures/  (mêmes noms que l'article).
# =============================================================================
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter
warnings.filterwarnings('ignore')

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'results')
FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'figures'); os.makedirs(FIG, exist_ok=True)
FIGHR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'results')

LAT_MIN, LAT_MAX = 4.0, 23.0
LON_MIN, LON_MAX = -18.0, 16.0

# --- Style global (identique aux scripts d'origine) ---
plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
    'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
    'legend.fontsize':9,'figure.dpi':150,'savefig.dpi':300,
    'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False,
})

PAL = {'base':'#2166AC','qm':'#4DAC26','cgan':'#D73027','cgan_qm':'#F46D43',
       'obs':'#1A1A1A','era5':'#92C5DE'}

def precip_cmap():
    cols=[(1,1,1),(0.87,0.96,0.87),(0.50,0.80,0.50),(0.15,0.55,0.15),
          (0.10,0.45,0.80),(0.05,0.20,0.70),(0.45,0.00,0.60),(0.80,0.00,0.40)]
    return mcolors.LinearSegmentedColormap.from_list('precip',cols)
PCMAP=precip_cmap()

# =============================================================================
# DONNÉES HONNÊTES 0.1°
# =============================================================================
print('Chargement arrays 0.1°...', flush=True)
D = np.load(f'{RES}/arrays_01deg.npz')
obs = D['obs']                                  # (459,190,340) sud->nord
preds = {k: D[k] for k in ['Simple','SE','CBAM','cGAN','CBAM_QM','BCSD_full']}
H, W = obs.shape[1], obs.shape[2]
extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]
lats_asc = np.linspace(LAT_MIN, LAT_MAX, H)
lons     = np.linspace(LON_MIN, LON_MAX, W)

# mois du test 2018-2020 (JJASO, 153 j/an)
DPY=153; mseq=[]
for _ in range(21):
    for m,nd in zip([6,7,8,9,10],[30,31,31,30,31]): mseq+=[m]*nd
mseq=np.array(mseq); months_te=mseq[18*DPY:21*DPY]

perf = pd.read_csv(f'{RES}/table_perf_01deg.csv', index_col=0)

def fss(pred_mm, obs_mm, threshold, scale):
    out=[]
    for i in range(pred_mm.shape[0]):
        pb=(pred_mm[i]>=threshold).astype(float); ob=(obs_mm[i]>=threshold).astype(float)
        size=2*scale+1; fp=uniform_filter(pb,size); fo=uniform_filter(ob,size)
        num=np.mean((fp-fo)**2); den=np.mean(fp**2)+np.mean(fo**2)
        out.append(1.0 if den<1e-12 else 1.0-num/den)
    return float(np.mean(out))

def csi(p,o,t):
    P=p>=t; O=o>=t
    tp=np.sum(P&O); fp=np.sum(P&~O); fn=np.sum(~P&O)
    return float(tp/(tp+fp+fn)) if (tp+fp+fn)>0 else 0.0

def nse(p,o):
    return float(1-np.sum((p-o)**2)/(np.sum((o-o.mean())**2)+1e-12))

def save(fig,name):
    fig.savefig(f'{FIG}/{name}.png',dpi=300,bbox_inches='tight')
    fig.savefig(f'{FIG}/{name}.pdf',bbox_inches='tight'); plt.close(fig)
    print('  ->',name,flush=True)


# =============================================================================
# FIG 2 : SCATTER OBS vs PRED (2x2)  — style d'origine
# =============================================================================
print('[Fig 2] scatter...', flush=True)
fig, axes = plt.subplots(2,2,figsize=(9,8))
configs_scatter=[('(a) UNet-CBAM — Base','CBAM',PAL['base']),
                 ('(b) UNet-CBAM — cGAN','cGAN',PAL['cgan']),
                 ('(c) UNet-CBAM — Base + QM','CBAM_QM',PAL['qm']),
                 ('(d) UNet-Simple — Base','Simple','#666666')]
np.random.seed(42)
for ax,(title,key,color) in zip(axes.ravel(),configs_scatter):
    obs_f=obs.ravel(); pred_f=preds[key].ravel()
    n_pts=min(15000,len(obs_f)); idx=np.random.choice(len(obs_f),n_pts,replace=False)
    obs_s,pred_s=obs_f[idx],pred_f[idx]
    hb=ax.hexbin(obs_s,pred_s,gridsize=60,cmap='Blues',mincnt=1,bins='log',alpha=0.85)
    lim=max(obs_f.max(),pred_f.max(),1.)
    ax.plot([0,lim],[0,lim],'k--',lw=1.2,label='1:1',zorder=5)
    slope=np.polyfit(obs_s,pred_s,1)[0]; r_val=np.corrcoef(obs_s,pred_s)[0,1]
    rmse=np.sqrt(np.mean((pred_f-obs_f)**2)); p99r=np.percentile(pred_f,99)/max(np.percentile(obs_f,99),0.1)
    xfit=np.array([0,lim]); ax.plot(xfit,slope*xfit,color=color,lw=2,label=f'Slope={slope:.2f}')
    stats_txt=f'RMSE={rmse:.1f} mm\nr={r_val:.3f}  Slope={slope:.2f}\nP99={p99r:.2f}'
    ax.text(0.97,0.05,stats_txt,transform=ax.transAxes,ha='right',va='bottom',fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.3',facecolor='white',edgecolor='#CCCCCC',alpha=0.9))
    ax.set_xlim(0,min(lim,150)); ax.set_ylim(0,min(lim,150))
    ax.set_xlabel('Observed IMERG precipitation (mm/d)'); ax.set_ylabel('Predicted precipitation (mm/d)')
    ax.set_title(title,fontweight='bold'); ax.legend(fontsize=8,loc='upper left')
    plt.colorbar(hb,ax=ax,label='Nb. points (log)',pad=0.01)
plt.suptitle('Predicted vs Observed Precipitation — Test Set 2018–2020',fontsize=12,fontweight='bold',y=1.01)
fig.tight_layout(); save(fig,'fig2_scatter')


# =============================================================================
# FIG 4 : MÉTRIQUES (2x3 barres groupées) — style d'origine, 6 configs honnêtes
# =============================================================================
print('[Fig 4] metrics...', flush=True)
configs_show=[('U-Net-Simple',PAL['base'],'-','UNet-Simple (base)'),
              ('U-Net-CBAM','#1A6B3A','-','UNet-CBAM (base)'),
              ('U-Net-CBAM + QM',PAL['qm'],'--','UNet-CBAM + QM'),
              ('U-Net-CBAM cGAN',PAL['cgan'],'-','UNet-CBAM cGAN'),
              ('U-Net-CBAM cGAN+QM',PAL['cgan_qm'],'--','UNet-CBAM cGAN+QM'),
              ('BCSD full','#777777','-','BCSD full')]
metrics_show=[('RMSE','RMSE (mm/d)',True,None),('NSE','NSE',False,0.0),
              ('Slope','Slope (scatter)',False,1.0),('FSS_20mm','FSS 20mm',False,1.0),
              ('CSI_40mm','CSI 40mm',False,1.0),('P99_ratio','P99 ratio',False,1.0)]
fig,axes=plt.subplots(2,3,figsize=(13,8)); axes=axes.ravel()
x=np.arange(len(configs_show)); w=0.65
for ai,(col,label,lower_better,ref_line) in enumerate(metrics_show):
    ax=axes[ai]
    vals=[perf.loc[cn,col] if cn in perf.index else 0 for cn,_,_,_ in configs_show]
    colors=[c for _,c,_,_ in configs_show]
    hatches=['//' if '--' in ls else '' for _,_,ls,_ in configs_show]
    bars=ax.bar(x,vals,width=w,color=colors,alpha=0.85,edgecolor='#333',linewidth=0.6)
    for bar,h in zip(bars,hatches): bar.set_hatch(h)
    if ref_line is not None:
        ax.axhline(ref_line,color='#333',lw=1,ls=':',alpha=0.6,label=f'Réf = {ref_line}')
    for bar,v in zip(bars,vals):
        va='bottom' if v>=0 else 'top'; off=0.005*max(abs(z) for z in vals)
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+(off if v>=0 else -off),
                f'{v:.2f}',ha='center',va=va,fontsize=7.5,fontweight='bold')
    ax.set_title(f'({chr(97+ai)}) {label}',fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([lbl for _,_,_,lbl in configs_show],rotation=30,ha='right',fontsize=8)
    ax.grid(axis='y',alpha=0.25)
    ax.annotate('↓ better' if lower_better else '↑ better',xy=(0.02,0.95),xycoords='axes fraction',fontsize=7.5,color='#555')
plt.suptitle('Metrics Comparison — ERA5→IMERG 0.1° Downscaling\nTest set 2018–2020 (459 days)',fontsize=12,fontweight='bold')
fig.tight_layout(); save(fig,'fig4_metrics_comparison')


# =============================================================================
# FIG 5 : PERMUTATION IMPORTANCE — identique (données inchangées)
# =============================================================================
print('[Fig 5] permutation...', flush=True)
df_imp=pd.read_csv(f'{FIGHR}/channel_importance_highres.csv').sort_values('ΔRMSE_CBAM (mm)',ascending=True)
fig,ax=plt.subplots(figsize=(8,7)); y_pos=np.arange(len(df_imp))
level_pal={'850':'#D73027','700':'#FC8D59','500':'#4575B4'}
bar_colors=[]
for canal in df_imp['Canal']:
    for lv,c in level_pal.items():
        if lv in canal: bar_colors.append(c); break
ax.barh(y_pos-0.2,df_imp['ΔRMSE_CBAM (mm)'],0.35,color=bar_colors,alpha=0.9,edgecolor='#333',
        linewidth=0.5,label='UNet-CBAM',xerr=df_imp['σ_CBAM'],
        error_kw=dict(elinewidth=0.8,capsize=2,ecolor='#555'))
ax.barh(y_pos+0.2,df_imp['ΔRMSE_Simple (mm)'],0.35,color=bar_colors,alpha=0.45,edgecolor='#333',
        linewidth=0.5,linestyle='--',label='UNet-Simple',xerr=df_imp['σ_Simple'],
        error_kw=dict(elinewidth=0.8,capsize=2,ecolor='#888'))
ax.set_yticks(y_pos); ax.set_yticklabels(df_imp['Canal'],fontsize=9.5); ax.axvline(0,color='k',lw=0.8)
for tick,canal in zip(ax.get_yticklabels(),df_imp['Canal']):
    for lv,c in level_pal.items():
        if lv in canal: tick.set_color(c); break
ax.set_xlabel('ΔRMSE (mm/d) — higher = more important predictor',fontsize=10)
ax.set_title('Permutation Importance of ERA5 Channels\nUNet-CBAM vs UNet-Simple (3 repetitions, test 2018–2020)',fontweight='bold')
patches_lv=[mpatches.Patch(color=c,label=f'{lv} hPa') for lv,c in level_pal.items()]
patches_model=[mpatches.Patch(color='#555',alpha=0.9,label='UNet-CBAM'),mpatches.Patch(color='#555',alpha=0.45,label='UNet-Simple')]
ax.legend(handles=patches_lv+patches_model,loc='lower right',fontsize=8.5,framealpha=0.9)
ax.grid(axis='x',alpha=0.25); fig.tight_layout(); save(fig,'fig5_permutation_importance')


# =============================================================================
# FIG 6 : RÉGIONAL (1x4 barres groupées, 3 modèles x 3 zones) — style d'origine
# =============================================================================
print('[Fig 6] regional (calcul 3 modèles)...', flush=True)
zone_masks={'Guinean':lats_asc<10,'Sudanian':(lats_asc>=10)&(lats_asc<15),'Sahelian':lats_asc>=15}
zone_short=list(zone_masks.keys())
models_reg=[('U-Net-Simple','Simple',PAL['base']),
            ('U-Net-CBAM','CBAM','#1A6B3A'),
            ('U-Net-CBAM+QM','CBAM_QM',PAL['qm'])]
def reg_metrics(arr_key,mask):
    p=preds[arr_key][:,mask,:]; o=obs[:,mask,:]
    return dict(RMSE=float(np.sqrt(np.mean((p-o)**2))),NSE=nse(p.ravel(),o.ravel()),
                FSS_20mm=fss(p,o,20,2),CSI_40mm=csi(p,o,40))
reg_vals={m[0]:{z:reg_metrics(m[1],zone_masks[z]) for z in zone_short} for m in models_reg}
# validation CBAM vs CSV honnête
cb=reg_vals['U-Net-CBAM']
print('   CBAM Guinean RMSE=%.2f NSE=%.3f FSS=%.3f CSI=%.3f (CSV: 13.15/0.140/0.186/0.036)'%
      (cb['Guinean']['RMSE'],cb['Guinean']['NSE'],cb['Guinean']['FSS_20mm'],cb['Guinean']['CSI_40mm']))
metrics_reg=[('RMSE','RMSE (mm/d)',True),('NSE','NSE',False),('FSS_20mm','FSS 20mm',False),('CSI_40mm','CSI 40mm',False)]
fig,axes=plt.subplots(1,4,figsize=(15,5)); x=np.arange(len(zone_short)); w=0.25
for ai,(col,label,lower) in enumerate(metrics_reg):
    ax=axes[ai]
    for mi,(mname,_,mc) in enumerate(models_reg):
        vals=[reg_vals[mname][z][col] for z in zone_short]
        offset=(mi-1)*w
        bars=ax.bar(x+offset,vals,w,color=mc,alpha=0.85,edgecolor='#333',linewidth=0.5,label=mname)
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01*max(abs(z) for z in vals),
                    f'{v:.2f}',ha='center',va='bottom',fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(zone_short,fontsize=9)
    ax.set_title(f'({chr(97+ai)}) {label}',fontweight='bold'); ax.grid(axis='y',alpha=0.25)
    if ai==0: ax.legend(fontsize=8,loc='upper right')
    ax.annotate('↓ better' if lower else '↑ better',xy=(0.02,0.97),xycoords='axes fraction',fontsize=7,color='#666',va='top')
plt.suptitle('Regional Performance by Climate Zone — West Africa (test set 2018–2020)',fontsize=12,fontweight='bold')
fig.tight_layout(); save(fig,'fig6_regional')


# =============================================================================
# FIG 7 : MENSUEL (2x2 lignes, base/cGAN/QM) — style d'origine
# =============================================================================
print('[Fig 7] monthly...', flush=True)
mois_names=['Jun','Jul','Aug','Sep','Oct']; mois_idx=[6,7,8,9,10]
def monthly_metrics(key):
    rows={}
    for m in mois_idx:
        idx=np.where(months_te==m)[0]
        p=preds[key][idx]; o=obs[idx]
        rows[m]=dict(RMSE=float(np.sqrt(np.mean((p-o)**2))),NSE=nse(p.ravel(),o.ravel()),
                     P99=np.percentile(p,99)/max(np.percentile(o,99),0.1),FSS20=fss(p,o,20,2))
    return rows
mtr_base=monthly_metrics('CBAM'); mtr_cgan=monthly_metrics('cGAN'); mtr_qm=monthly_metrics('CBAM_QM')
fig,axes=plt.subplots(2,2,figsize=(12,8))
metrics_mo=[('RMSE','RMSE (mm/d)',True),('FSS20','FSS 20mm',False),('P99','P99 ratio',False),('NSE','NSE',False)]
configs_mo=[(mtr_base,PAL['base'],'o-','UNet-CBAM base'),(mtr_cgan,PAL['cgan'],'s-','UNet-CBAM cGAN'),
            (mtr_qm,PAL['qm'],'^--','UNet-CBAM + QM')]
for ai,(mk,ml,lower) in enumerate(metrics_mo):
    ax=axes.ravel()[ai]
    for mtr_dict,color,style,lbl in configs_mo:
        vals=[mtr_dict[m][mk] for m in mois_idx]
        ax.plot(mois_names,vals,style,color=color,lw=2,ms=7,markerfacecolor='white',markeredgewidth=1.5,label=lbl)
    if mk=='P99': ax.axhline(1.0,color='#2d6a4f',lw=1.2,ls=':',alpha=0.8,label='P99 ratio = 1 (perfect)')
    ax.set_title(f'({chr(97+ai)}) {ml}',fontweight='bold'); ax.grid(True,alpha=0.2); ax.set_xlabel('Month')
    if ai==0: ax.legend(fontsize=8.5,loc='upper right')
plt.suptitle('Monthly Performance Evolution — JJASO Season\nUNet-CBAM: Base vs cGAN vs QM',fontsize=12,fontweight='bold')
fig.tight_layout(); save(fig,'fig7_monthly')


# =============================================================================
# FIG 8 : DISTRIBUTION + PERCEPTION-DISTORSION — style d'origine
# =============================================================================
print('[Fig 8] distribution + tradeoff...', flush=True)
fig,axes=plt.subplots(1,2,figsize=(12,5))
ax=axes[0]
bins=np.concatenate([np.arange(0,10,1),np.arange(10,50,5),np.arange(50,150,10)])
datasets=[(obs.ravel(),PAL['obs'],'-',2.0,'IMERG obs'),
          (preds['CBAM'].ravel(),PAL['base'],'--',1.8,'UNet-CBAM base'),
          (preds['cGAN'].ravel(),PAL['cgan'],'-',1.8,'UNet-CBAM cGAN'),
          (preds['CBAM_QM'].ravel(),PAL['qm'],':',1.8,'UNet-CBAM + QM')]
for data,color,ls,lw,label in datasets:
    dp=data[data>0.5]; counts,edges=np.histogram(dp,bins=bins,density=True)
    centers=(edges[:-1]+edges[1:])/2; ax.plot(centers,counts,ls,color=color,lw=lw,label=label)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Precipitation (mm/d)')
ax.set_ylabel('Probability density'); ax.set_title('(a) Precipitation distribution\n(rainy days > 0.5 mm/d)',fontweight='bold')
ax.legend(fontsize=8.5); ax.grid(True,alpha=0.2,which='both')
# Panel B perception-distortion : points honnêtes depuis perf
ax2=axes[1]
pts=[('UNet-Simple','U-Net-Simple','o',PAL['base'],60),
     ('UNet-SE','U-Net-SE','s','#888888',60),
     ('UNet-CBAM','U-Net-CBAM','D','#1A6B3A',80),
     ('UNet-CBAM+QM','U-Net-CBAM + QM','D','#52B788',70),
     ('UNet-CBAM cGAN','U-Net-CBAM cGAN','P',PAL['cgan'],90),
     ('UNet-CBAM cGAN+QM','U-Net-CBAM cGAN+QM','P',PAL['cgan_qm'],70),
     ('BCSD full','BCSD full','X','#777777',70)]
for disp,key,mk,col,sz in pts:
    rmse=perf.loc[key,'RMSE']; fss20=perf.loc[key,'FSS_20mm']
    ax2.scatter(rmse,fss20,marker=mk,color=col,s=sz,edgecolors='#333',linewidths=0.7,zorder=5)
    short=disp.replace('UNet-','').replace(' cGAN','\ncGAN').replace('+QM','\n+QM')
    ax2.annotate(short,(rmse,fss20),xytext=(5,3),textcoords='offset points',fontsize=7,color=col)
ax2.annotate('',xy=(12.5,0.33),xytext=(9.3,0.18),
             arrowprops=dict(arrowstyle='->',color='gray',lw=1.2,connectionstyle='arc3,rad=0.15'))
ax2.text(10.9,0.24,'Tradeoff\nPerception↑\nDistortion↑',ha='center',fontsize=8,color='#555',style='italic')
ax2.set_xlabel('RMSE (mm/d)  →  distortion'); ax2.set_ylabel('FSS$_{20}$  →  perception')
ax2.set_title('(b) Perception–Distortion Tradeoff\n(Blau \\& Michaeli 2018)',fontweight='bold'); ax2.grid(True,alpha=0.2)
plt.suptitle('Precipitation Distribution and Perception–Distortion Space',fontsize=12,fontweight='bold')
fig.tight_layout(); save(fig,'fig8_distribution_tradeoff')


# =============================================================================
# FIG B : CARTES DE BIAIS (1x3 cartopy : CBAM base / cGAN / QM) — style d'origine
# =============================================================================
print('[Fig B] bias maps (cartopy)...', flush=True)
import cartopy.crs as ccrs, cartopy.feature as cfeature
PROJ=ccrs.PlateCarree()
mean_obs=obs.mean(0)
bias_base=preds['CBAM'].mean(0)-mean_obs
bias_cgan=preds['cGAN'].mean(0)-mean_obs
bias_qm  =preds['CBAM_QM'].mean(0)-mean_obs
vmax_b=min(max(abs(bias_base).max(),abs(bias_cgan).max(),abs(bias_qm).max())*0.85,12.)
panels_B=[('(a) Bias UNet-CBAM base\n(Predicted − Observed)',bias_base),
          ('(b) Bias UNet-CBAM cGAN\n(Predicted − Observed)',bias_cgan),
          ('(c) Bias UNet-CBAM + QM\n(Predicted − Observed)',bias_qm)]
fig,axes=plt.subplots(1,3,figsize=(14,5),subplot_kw={'projection':PROJ})
for ax,(title,bias) in zip(axes,panels_B):
    ax.set_extent(extent,crs=PROJ)
    im=ax.imshow(bias,extent=extent,origin='lower',transform=PROJ,cmap=plt.cm.RdBu_r,
                 vmin=-vmax_b,vmax=vmax_b,aspect='auto',zorder=2)
    ax.add_feature(cfeature.COASTLINE,linewidth=0.7,edgecolor='k',zorder=4)
    ax.add_feature(cfeature.BORDERS,linewidth=0.4,edgecolor='#444',linestyle='--',zorder=4)
    for lat_z in [10,15]:
        ax.plot([LON_MIN,LON_MAX],[lat_z,lat_z],'w--',lw=0.8,transform=PROJ,zorder=5,alpha=0.6)
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',pad=0.03,shrink=0.92,aspect=30)
    cb.set_label('mm/d (red = overestimation, blue = underestimation)',fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.contour(lons,lats_asc,bias,levels=[0],colors='k',linewidths=0.8,transform=PROJ,zorder=5)
    ax.text(0.5,-0.08,f'Mean bias={bias.mean():+.2f}  MAE={np.abs(bias).mean():.2f} mm/d',
            transform=ax.transAxes,ha='center',fontsize=8,style='italic')
    ax.set_title(title,fontweight='bold',pad=6)
plt.suptitle('Mean Spatial Bias — Test 2018–2020\n(black contour = zero line)',fontsize=12,fontweight='bold',y=1.01)
fig.tight_layout(); save(fig,'figB_bias_maps')


# =============================================================================
# FIG E : DIAGRAMME DE TAYLOR (cartésien) — style d'origine
# =============================================================================
print('[Fig E] Taylor...', flush=True)
def taylor_stats(pf,of):
    r=np.corrcoef(pf,of)[0,1]; return r,pf.std()/of.std(),np.sqrt(np.mean((pf-of)**2))/of.std()
obs_flat=obs.ravel()
models_taylor=[('UNet-Simple base','Simple','o','#4477AA',60),
               ('UNet-SE base','SE','s','#888888',60),
               ('UNet-CBAM base','CBAM','D','#1A6B3A',80),
               ('UNet-CBAM + QM','CBAM_QM','D','#52B788',70),
               ('UNet-CBAM cGAN','cGAN','P','#D73027',90),
               ('BCSD full','BCSD_full','X','#777777',70)]
taylor_data={name:(*taylor_stats(preds[key].ravel(),obs_flat),mk,col,sz)
             for name,key,mk,col,sz in models_taylor}
std_max=1.65
fig,ax=plt.subplots(figsize=(9,8)); fig.subplots_adjust(left=0.10,right=0.82,top=0.90,bottom=0.10)
ax.set_aspect('equal'); arc=np.linspace(0,np.pi/2,300)
ax.plot(std_max*np.cos(arc),std_max*np.sin(arc),'k-',lw=1.5)
for std_c in [0.5,1.0,1.25,1.5]:
    ls='--' if std_c==1.0 else '-'; col_c='#999' if std_c==1.0 else '#CCCCCC'
    ax.plot(std_c*np.cos(arc),std_c*np.sin(arc),ls,color=col_c,lw=0.9)
    ax.text(std_c,-0.06,f'{std_c:.2f}',ha='center',va='top',fontsize=8,color='#444')
for r_line in [0.0,0.2,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]:
    theta_l=np.arccos(r_line); ax.plot([0,std_max*np.cos(theta_l)],[0,std_max*np.sin(theta_l)],'k--',lw=0.4,alpha=0.25)
    xL=(std_max+0.07)*np.cos(theta_l); yL=(std_max+0.07)*np.sin(theta_l); rot=np.degrees(theta_l)-90
    ax.text(xL,yL,f'{r_line:.2f}',ha='center',va='center',fontsize=8,rotation=rot,color='#333')
for rmse_val in [0.25,0.5,0.75,1.0,1.25]:
    phi=np.linspace(0,2*np.pi,600); xc=1.0+rmse_val*np.cos(phi); yc=rmse_val*np.sin(phi)
    mask=(xc>=0)&(yc>=0)&(np.sqrt(xc**2+yc**2)<=std_max)
    if mask.sum()>3:
        ax.plot(xc[mask],yc[mask],'-',color='#AAAAAA',lw=0.6,alpha=0.5)
        idx=mask.nonzero()[0][len(mask.nonzero()[0])//3]
        ax.text(xc[idx],yc[idx],f'{rmse_val:.2f}',fontsize=7,color='#666',ha='center',
                bbox=dict(facecolor='white',edgecolor='none',pad=1))
ax.text(0.92,0.60,'Normalised RMSE',fontsize=8,color='#777',rotation=45,ha='center')
ax.plot(1,0,'*',color='k',ms=14,zorder=8,label='IMERG observations')
ax.annotate('Obs.\n(IMERG)',(1,0),xytext=(8,6),textcoords='offset points',fontsize=8,fontweight='bold')
for name,(r,std_n,rmse_n,mk,col,sz) in taylor_data.items():
    xp=std_n*r; yp=std_n*np.sin(np.arccos(np.clip(r,-1,1)))
    ax.plot(xp,yp,mk,color=col,ms=np.sqrt(sz),markeredgecolor='#333',markeredgewidth=0.7,zorder=7,label=name)
    ax.annotate(name,(xp,yp),xytext=(6,4),textcoords='offset points',fontsize=7.5,color=col,
                arrowprops=dict(arrowstyle='-',color=col,lw=0.5))
ax.set_xlim(-0.02,std_max*1.12); ax.set_ylim(-0.10,std_max*1.12)
ax.set_xlabel('Normalised Standard Deviation',fontsize=11,labelpad=8)
ax.set_ylabel('Normalised Standard Deviation',fontsize=11,labelpad=8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.set_xticks([0,0.5,1.0,1.5]); ax.set_yticks([0,0.5,1.0,1.5])
ax.text((std_max+0.18)*np.cos(np.radians(32)),(std_max+0.18)*np.sin(np.radians(32)),
        'Correlation (r)',fontsize=10,ha='center',va='center',rotation=-55)
ax.set_title('Normalised Taylor Diagram\nTest set 2018–2020 (all pixel×day values)',fontweight='bold',pad=15)
ax.legend(loc='upper right',fontsize=8.5,framealpha=0.92,ncol=1,borderpad=0.8,bbox_to_anchor=(1.0,1.0))
save(fig,'figE_taylor_diagram')

print('\nTERMINÉ — figures style publication 0.1° dans', FIG, flush=True)
