# =============================================================================
# fig1_domain — reconstruction fidèle avec l'ENCART AFRIQUE DÉPLACÉ À DROITE.
# Panneau (a) : précip moyenne IMERG (JJASO 2000-2020, 0.1°) + zones + villes +
#               encart locator Afrique (haut-droite). Panneau (b) : schéma 2.5x.
# Sortie -> corrigerOS1/figures/fig1_domain.pdf
# =============================================================================
import os, sys, warnings, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import cartopy.crs as ccrs, cartopy.feature as cfeature
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'src'))
import downscaling_highres as ds

OUT=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'figures'); os.makedirs(OUT,exist_ok=True)
LAT_MIN,LAT_MAX,LON_MIN,LON_MAX=4.0,23.0,-18.0,16.0

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.dpi':300,
                     'savefig.bbox':'tight'})

# ---- moyenne IMERG plein-période (en mémoire, chunks) -> 0.1° ----
Y=np.load(ds.Y_PATH, mmap_mode='r')              # (3213,1,380,680)
H,W=Y.shape[-2],Y.shape[-1]; acc=np.zeros((H,W),np.float64); n=0
for s in range(0,Y.shape[0],300):
    a=np.asarray(Y[s:s+300],np.float32)
    if a.ndim==4: a=a[:,0]
    a=np.clip(np.nan_to_num(a),0,300); acc+=a.sum(0); n+=a.shape[0]
mean05=(acc/n).astype(np.float32)
mean01=mean05.reshape(H//2,2,W//2,2).mean((1,3))  # (190,340)
# oriente row0=sud comme le cache obs (origin='lower')
o=np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'results','arrays_01deg.npz'))['obs'].mean(0)
if np.corrcoef(mean01[::-1].ravel(),o.ravel())[0,1] > np.corrcoef(mean01.ravel(),o.ravel())[0,1]:
    mean01=mean01[::-1]
print('mean01 prêt, glob=%.2f mm/j'%mean01.mean(), flush=True)

# colormap précip blanc->vert->bleu
cols=[(1,1,1),(0.90,0.96,0.86),(0.65,0.86,0.60),(0.30,0.70,0.45),
      (0.12,0.55,0.55),(0.10,0.40,0.75),(0.05,0.20,0.65)]
PCMAP=mcolors.LinearSegmentedColormap.from_list('precip',cols)
proj=ccrs.PlateCarree(); extent=[LON_MIN,LON_MAX,LAT_MIN,LAT_MAX]

fig=plt.figure(figsize=(14,6))
gs=gridspec.GridSpec(1,2,width_ratios=[3,1.2],wspace=0.08)

# ================= Panneau (a) =================
ax1=fig.add_subplot(gs[0],projection=proj)
ax1.set_extent([LON_MIN-3,LON_MAX+5,LAT_MIN-2,LAT_MAX+2],crs=proj)
ax1.add_feature(cfeature.OCEAN,facecolor='#EAF2F8',zorder=0)
ax1.add_feature(cfeature.LAND,facecolor='#F7F5EF',zorder=0)
im=ax1.imshow(mean01,origin='lower',extent=extent,transform=proj,cmap=PCMAP,
              vmin=0,vmax=14,zorder=1,interpolation='bilinear')
ax1.add_feature(cfeature.COASTLINE,linewidth=0.9,edgecolor='#444',zorder=4)
ax1.add_feature(cfeature.BORDERS,linewidth=0.5,edgecolor='#777',linestyle='--',zorder=4)

# séparateurs de zones + labels (à gauche)
for lat_z in (10,15):
    ax1.plot([LON_MIN,LON_MAX],[lat_z,lat_z],'--',color='#333',lw=0.9,transform=proj,zorder=5)
zlabels=[('Guinean (4–10°N)',7,'#1A6B3A'),('Sudanian (10–15°N)',12.5,'#B8860B'),
         ('Sahelian (15–23°N)',19,'#B5651D')]
for txt,lat_c,col in zlabels:
    ax1.text(LON_MIN+0.4,lat_c,txt,fontsize=8.5,va='center',ha='left',color=col,
             fontweight='bold',transform=proj,zorder=6,
             bbox=dict(facecolor='white',alpha=0.7,edgecolor='none',pad=1.5))

# domaine d'étude
ax1.add_patch(patches.Rectangle((LON_MIN,LAT_MIN),LON_MAX-LON_MIN,LAT_MAX-LAT_MIN,
              linewidth=2.2,edgecolor='#CC0000',facecolor='none',transform=proj,zorder=5))

# villes
cities={'Dakar':(-17.4,14.7),'Bamako':(-8.0,12.6),'Niamey':(2.1,13.5),'Abidjan':(-4.0,5.3),
        'Accra':(-0.2,5.6),'Lagos':(3.4,6.5),'Conakry':(-13.7,9.5),'Ouaga.':(-1.5,12.4)}
for city,(lon,lat) in cities.items():
    ax1.plot(lon,lat,'o',color='#222',ms=3,transform=proj,zorder=7)
    ax1.text(lon+0.4,lat+0.3,city,fontsize=6.5,transform=proj,zorder=7,color='#222')

gl=ax1.gridlines(draw_labels=True,linewidth=0.4,color='gray',alpha=0.5,linestyle='--',zorder=3)
gl.top_labels=False; gl.right_labels=False
gl.xlocator=mticker.FixedLocator([-15,-10,-5,0,5,10,15])
gl.ylocator=mticker.FixedLocator([0,5,10,15,20])
gl.xlabel_style={'size':8}; gl.ylabel_style={'size':8}

# ---- ENCART AFRIQUE : déplacé complètement à DROITE (haut-droite du panneau a) ----
ax_ins=fig.add_axes([0.505,0.595,0.10,0.185],projection=proj)
ax_ins.set_extent([-20,55,-38,40],crs=proj)
ax_ins.add_feature(cfeature.LAND,facecolor='#E8E4DA',zorder=1)
ax_ins.add_feature(cfeature.OCEAN,facecolor='#CFE3F2',zorder=1)
ax_ins.add_feature(cfeature.COASTLINE,linewidth=0.5,edgecolor='#555',zorder=2)
ax_ins.add_patch(patches.Rectangle((LON_MIN,LAT_MIN),LON_MAX-LON_MIN,LAT_MAX-LAT_MIN,
                 linewidth=1.3,edgecolor='#CC0000',facecolor='#CC0000',alpha=0.45,
                 transform=proj,zorder=3))
ax_ins.set_xticks([]); ax_ins.set_yticks([])
for sp in ax_ins.spines.values(): sp.set_edgecolor('#333'); sp.set_linewidth(0.8)

cb=fig.colorbar(im,ax=ax1,orientation='vertical',pad=0.02,shrink=0.85,aspect=28)
cb.set_label('Mean precipitation (mm d$^{-1}$)',fontsize=9); cb.ax.tick_params(labelsize=8)
ax1.set_title('(a) Study Domain — West Africa (ERA5 → IMERG 0.1°, JJASO 2000–2020)',
              fontweight='bold',pad=10,fontsize=11)

# ================= Panneau (b) : schéma 2.5x (inchangé) =================
ax2=fig.add_subplot(gs[1]); ERA5_W,IMERG_W=5,2
for i in range(2):
    for j in range(2):
        ax2.add_patch(patches.Rectangle((j*ERA5_W,i*ERA5_W),ERA5_W,ERA5_W,linewidth=2.0,
                      edgecolor='#2166AC',facecolor='#D6EAF5',alpha=0.85,zorder=2))
for i in range(5):
    for j in range(5):
        ax2.add_patch(patches.Rectangle((j*IMERG_W,i*IMERG_W),IMERG_W,IMERG_W,linewidth=0.5,
                      edgecolor='#D73027',facecolor='none',alpha=0.9,zorder=3))
ax2.add_patch(patches.Rectangle((0,0),ERA5_W,ERA5_W,linewidth=2.5,edgecolor='#2166AC',
              facecolor='#A9D4EE',alpha=0.9,zorder=2))
ax2.annotate('',xy=(ERA5_W,-0.8),xytext=(0,-0.8),arrowprops=dict(arrowstyle='<->',color='#2166AC',lw=1.8))
ax2.text(ERA5_W/2,-1.7,'1 ERA5 cell\n0.25° ≈ 28 km',ha='center',va='top',fontsize=8,color='#2166AC',fontweight='bold')
ax2.annotate('',xy=(ERA5_W,10.8),xytext=(0,10.8),arrowprops=dict(arrowstyle='<->',color='#D73027',lw=1.8))
ax2.text(ERA5_W/2,11.4,'2.5 IMERG cells\n0.1° ≈ 11 km each',ha='center',va='bottom',fontsize=8,color='#D73027',fontweight='bold')
ax2.text(ERA5_W/2,ERA5_W/2,'2.5×',ha='center',va='center',fontsize=22,fontweight='bold',color='#333',alpha=0.35,zorder=4)
ax2.set_xlim(-0.5,10.5); ax2.set_ylim(-4.5,13.0); ax2.set_aspect('equal')
ax2.set_xticks([]); ax2.set_yticks([]); ax2.spines[:].set_visible(False)
p1=patches.Patch(facecolor='#D6EAF5',edgecolor='#2166AC',linewidth=1.5,label='ERA5 (0.25°)')
p2=patches.Patch(facecolor='none',edgecolor='#D73027',linewidth=1.0,label='IMERG (0.1°)')
ax2.legend(handles=[p1,p2],loc='lower right',fontsize=8.5,framealpha=0.9,edgecolor='#ccc')
ax2.set_title('(b) Downscaling factor: 2.5×\n(1 ERA5 cell = 2.5 IMERG cells)',fontweight='bold',pad=10,fontsize=10)

fig.savefig(f'{OUT}/fig1_domain.pdf',bbox_inches='tight')
fig.savefig('/tmp/fig1_check.png',dpi=150,bbox_inches='tight')
plt.close(fig); print('OK -> fig1_domain.pdf (encart à droite)')
