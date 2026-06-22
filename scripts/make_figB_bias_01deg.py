# =============================================================================
# figB_bias_maps à 0.1° (corrigerOS1)
# Carte du biais moyen (prediction - observation) pour les 3 U-Nets, à la vraie
# résolution IMERG 0.1° (190x340). Bleu = sous-estimation, rouge = surestimation.
# Sortie -> corrigerOS1/figures/figB_bias_maps.{pdf,png}
# =============================================================================
import os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = '/home/dsagna/corrigerOS1/results'
FIG = '/home/dsagna/corrigerOS1/figures'; os.makedirs(FIG, exist_ok=True)

D = np.load(f'{RES}/arrays_01deg.npz')
obs = D['obs']                                   # (459,190,340)
extent = (-18, 16, 4, 23)                         # lon W->E, lat S->N ; row0 = 4°N

panels = [('Simple', 'U-Net-Simple'),
          ('SE',     'U-Net-SE'),
          ('CBAM',   'U-Net-CBAM')]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
vmax = 6.0
im = None
for ax, (key, title) in zip(axes, panels):
    bias = (D[key] - obs).mean(axis=0)            # (190,340) biais temporel moyen
    im = ax.imshow(bias, origin='lower', extent=extent, cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, aspect='auto')
    # contour zéro
    lon = np.linspace(extent[0], extent[1], bias.shape[1])
    lat = np.linspace(extent[2], extent[3], bias.shape[0])
    ax.contour(lon, lat, bias, levels=[0], colors='k', linewidths=0.6)
    # repères de zones climatiques
    ax.axhline(10, color='grey', ls='--', lw=0.6)
    ax.axhline(15, color='grey', ls='--', lw=0.6)
    dom = bias.mean(); sd = bias.std()
    ax.set_title(f'{title}\nmean bias = {dom:+.2f}, std = {sd:.2f} mm d$^{{-1}}$',
                 fontsize=10)
    ax.set_xlabel('Longitude (°E)')
axes[0].set_ylabel('Latitude (°N)')
cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
cb.set_label('Mean bias (mm d$^{-1}$)')
fig.suptitle('Spatial distribution of mean bias — test 2018–2020 (0.1°)',
             fontweight='bold')

for ext, dpi in [('pdf', None), ('png', 200)]:
    kw = {} if dpi is None else {'dpi': dpi}
    fig.savefig(f'{FIG}/figB_bias_maps.{ext}', bbox_inches='tight', **kw)
print('OK -> figB_bias_maps.pdf/.png')

# Récap chiffres pour le texte de l'article
print('\n--- chiffres pour la section biais ---')
for key, title in panels + [('cGAN', 'cGAN'), ('CBAM_QM', 'CBAM+QM')]:
    bm = (D[key] - obs).mean(axis=0)
    print(f'{title:14s} domain-mean bias {bm.mean():+.3f} | std {bm.std():.3f}')
print(f'mean obs = {obs.mean():.3f} mm/d ; cGAN |bias|/obs = '
      f'{abs((D["cGAN"]-obs).mean())/obs.mean()*100:.1f}%')
