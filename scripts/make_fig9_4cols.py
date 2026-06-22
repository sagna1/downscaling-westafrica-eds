# =============================================================================
# fig9_spatial_samples_postproc SANS la colonne U-Net-CBAM cGAN+QM (non discutée)
# Re-rendu depuis le cache d'origine, style identique, 3x4 colonnes.
# Sortie -> corrigerOS1/figures/fig9_spatial_samples_postproc.{pdf,png}
# =============================================================================
import os, datetime, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
import cartopy.crs as ccrs, cartopy.feature as cfeature

CACHE=os.environ.get('SAMPLES_CACHE', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'results','_cache_samples_postproc.npz'))
OUT=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'figures'); os.makedirs(OUT,exist_ok=True)
LAT_MIN,LAT_MAX,LON_MIN,LON_MAX=4.0,23.0,-18.0,16.0

# colonnes SANS cgan_qm
COL_NAMES=['obs','base','base_qm','cgan']
col_labels=['IMERG\n(observations)','U-Net-CBAM','U-Net-CBAM + QM','U-Net-CBAM cGAN']

cache=np.load(CACHE)
panels={k:cache[k] for k in COL_NAMES}
dates3=[datetime.date.fromordinal(int(o)) for o in cache['dates_ord']]
n_rows,n_cols=3,len(COL_NAMES)

RAIN_MIN=1.0; vmin,vmax=0,50
cmap=plt.get_cmap('Blues').copy(); cmap.set_bad('white'); cmap.set_over('#08025c')
norm_color=mcolors.PowerNorm(gamma=0.5,vmin=vmin,vmax=vmax)
proj=ccrs.PlateCarree(); ext=[LON_MIN,LON_MAX,LAT_MIN,LAT_MAX]

# largeur réduite proportionnellement (4 colonnes au lieu de 5)
fig=plt.figure(figsize=(13.4,6.6))
fig.subplots_adjust(left=0.075,right=0.925,top=0.92,bottom=0.04,hspace=0.04,wspace=0.04)
for r in range(n_rows):
    date_str=dates3[r].strftime('%d %b %Y'); obs=panels['obs'][r]
    for c,(key,col_label) in enumerate(zip(COL_NAMES,col_labels)):
        data=panels[key][r]
        ax=fig.add_subplot(n_rows,n_cols,r*n_cols+c+1,projection=proj)
        ax.set_extent(ext,crs=proj)
        ax.imshow(np.ma.masked_less(data,RAIN_MIN),origin='lower',extent=ext,transform=proj,
                  cmap=cmap,norm=norm_color,interpolation='antialiased',interpolation_stage='rgba')
        ax.add_feature(cfeature.BORDERS,linewidth=0.4,edgecolor='0.3')
        ax.add_feature(cfeature.COASTLINE,linewidth=0.6)
        if r==0: ax.set_title(col_label,fontsize=8,fontweight='bold',pad=3)
        if c==0: ax.text(-0.16,0.5,date_str,transform=ax.transAxes,fontsize=7,ha='right',va='center',rotation=90)
        if c>0:
            mae=float(np.mean(np.abs(data-obs))); r_val=float(np.corrcoef(obs.ravel(),data.ravel())[0,1])
            ax.text(0.02,0.04,f'MAE={mae:.1f}  r={r_val:.2f}',transform=ax.transAxes,fontsize=5.5,
                    bbox=dict(facecolor='white',alpha=0.75,pad=1.5,edgecolor='none'))
cax=fig.add_axes([0.935,0.08,0.013,0.82])
sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm_color); sm.set_array([])
cb=fig.colorbar(sm,cax=cax,extend='max',ticks=[1,5,10,20,30,40,50])
cb.set_label('Precipitation (mm day$^{-1}$)',fontsize=8); cb.ax.tick_params(labelsize=7)
fig.savefig(f'{OUT}/fig9_spatial_samples_postproc.pdf',dpi=300,bbox_inches='tight')
fig.savefig(f'{OUT}/fig9_spatial_samples_postproc.png',dpi=300,bbox_inches='tight')
plt.close(fig); print('OK -> fig9_spatial_samples_postproc (4 colonnes, sans cGAN+QM)')
