# =============================================================================
# FIGURE ANNEXE — JOURS SANS PLUIE (jours les plus secs du test)
# Champs spatiaux 0.1° : IMERG | U-Net-CBAM | U-Net-CBAM+QM | U-Net-CBAM cGAN
# pour les 3 jours les plus secs (moyenne domaine la plus faible). Style Fig. 7.
# Montre si les modèles produisent bien des champs quasi-nuls (ou hallucinent
# de la bruine). Sortie -> corrigerOS1/figures/figA1_dry_days.{pdf,png}
# =============================================================================
import os, datetime, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
import cartopy.crs as ccrs, cartopy.feature as cfeature

RES=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'results')
OUT=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'figures'); os.makedirs(OUT,exist_ok=True)
LAT_MIN,LAT_MAX,LON_MIN,LON_MAX=4.0,23.0,-18.0,16.0

D=np.load(f'{RES}/arrays_01deg.npz')
obs=D['obs']                                  # (459,190,340) sud->nord
COL_NAMES=['obs','CBAM','CBAM_QM','cGAN']
col_labels=['IMERG\n(observations)','U-Net-CBAM','U-Net-CBAM + QM','U-Net-CBAM cGAN']

# dates réelles du test 2018-2020 (JJASO)
dates=[]
for y in [2018,2019,2020]:
    for m,nd in [(6,30),(7,31),(8,31),(9,30),(10,31)]:
        for d in range(1,nd+1): dates.append(datetime.date(y,m,d))
dates=np.array(dates)
assert len(dates)==obs.shape[0], (len(dates),obs.shape[0])

# 3 jours les plus secs (moyenne domaine la plus faible)
domain_mean=obs.mean(axis=(1,2))
dry3=np.argsort(domain_mean)[:3]
dry3=dry3[np.argsort(domain_mean[dry3])]       # du plus sec au moins sec
print('Jours les plus secs (moy domaine mm/j):')
for i in dry3: print(f'  {dates[i]}  mean={domain_mean[i]:.3f}')

panels={k:D[k][dry3] for k in COL_NAMES}
dates3=dates[dry3]
n_rows,n_cols=3,len(COL_NAMES)

# échelle 0-10 mm/j (jours secs), bruine < 0.1 mm/j masquée en blanc
RAIN_MIN=0.1; vmin,vmax=0,10
cmap=plt.get_cmap('Blues').copy(); cmap.set_bad('white'); cmap.set_over('#08025c')
norm_color=mcolors.PowerNorm(gamma=0.5,vmin=vmin,vmax=vmax)
proj=ccrs.PlateCarree(); ext=[LON_MIN,LON_MAX,LAT_MIN,LAT_MAX]

fig=plt.figure(figsize=(13.4,6.6))
fig.subplots_adjust(left=0.075,right=0.925,top=0.90,bottom=0.04,hspace=0.04,wspace=0.04)
for r in range(n_rows):
    date_str=dates3[r].strftime('%d %b %Y'); o=panels['obs'][r]
    for c,(key,col_label) in enumerate(zip(COL_NAMES,col_labels)):
        data=panels[key][r]
        ax=fig.add_subplot(n_rows,n_cols,r*n_cols+c+1,projection=proj)
        ax.set_extent(ext,crs=proj)
        ax.imshow(np.ma.masked_less(data,RAIN_MIN),origin='lower',extent=ext,transform=proj,
                  cmap=cmap,norm=norm_color,interpolation='antialiased',interpolation_stage='rgba')
        ax.add_feature(cfeature.BORDERS,linewidth=0.4,edgecolor='0.3')
        ax.add_feature(cfeature.COASTLINE,linewidth=0.6)
        if r==0: ax.set_title(col_label,fontsize=8,fontweight='bold',pad=3)
        if c==0:
            ax.text(-0.16,0.5,date_str,transform=ax.transAxes,fontsize=7,ha='right',va='center',rotation=90)
            ax.text(0.02,0.04,f'domain mean={o.mean():.2f} mm/d',transform=ax.transAxes,fontsize=5.5,
                    bbox=dict(facecolor='white',alpha=0.75,pad=1.5,edgecolor='none'))
        else:
            # fraction de pixels "secs" (<1 mm/j) produite par le modèle vs obs
            dry_frac=float(np.mean(data<1.0)); mae=float(np.mean(np.abs(data-o)))
            ax.text(0.02,0.04,f'dry={dry_frac*100:.0f}%  MAE={mae:.2f}',transform=ax.transAxes,fontsize=5.5,
                    bbox=dict(facecolor='white',alpha=0.75,pad=1.5,edgecolor='none'))
cax=fig.add_axes([0.935,0.08,0.013,0.82])
sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm_color); sm.set_array([])
cb=fig.colorbar(sm,cax=cax,extend='max',ticks=[0.1,1,2,4,6,8,10])
cb.set_label('Precipitation (mm day$^{-1}$)',fontsize=8); cb.ax.tick_params(labelsize=7)
fig.suptitle('Driest test days (2018–2020) — IMERG vs U-Net-CBAM post-processing (0.1°)',
             fontsize=11,fontweight='bold',y=0.965)
fig.savefig(f'{OUT}/figA1_dry_days.pdf',dpi=300,bbox_inches='tight')
fig.savefig(f'{OUT}/figA1_dry_days.png',dpi=300,bbox_inches='tight')
plt.close(fig); print('OK -> figA1_dry_days')
