# =============================================================================
# QM POST-PROCESSING — 3 MODÈLES HAUTE RÉSOLUTION (ERA5 → IMERG 0.05°)
# Applique le Quantile Mapping mensuel sur les checkpoints highres
# Sortie : figures_highres/full_table_highres.csv + figures
# =============================================================================

import os
import sys
import warnings
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from downscaling_highres import (
    load_and_preprocess, UNet, CONFIGS,
    DownscalingDataset, denorm_y,
    compute_metrics, compute_categorical, fss,
    CKPT_DIR, FIG_DIR, device,
    H_IMERG, W_IMERG, BATCH_SIZE,
)

THRESHOLDS = (5, 20, 40)


# =============================================================================
# QUANTILE MAPPING MENSUEL (poolé sur tous les pixels du mois)
# =============================================================================

class MonthlyQM:
    def __init__(self, n_quantiles=500):
        self.n_quantiles = n_quantiles
        self.q_levels    = np.linspace(0, 1, n_quantiles)
        self.maps        = {}

    def fit(self, pred_tr_mm, obs_tr_mm, months_tr):
        """pred_tr_mm, obs_tr_mm : (N_tr, H, W) en mm/j"""
        for m in [6, 7, 8, 9, 10]:
            idx = np.where(months_tr == m)[0]
            if len(idx) == 0:
                continue
            p_flat = pred_tr_mm[idx].ravel()
            o_flat = obs_tr_mm[idx].ravel()
            self.maps[m] = (
                np.quantile(p_flat, self.q_levels),
                np.quantile(o_flat, self.q_levels),
            )
            print(f"    Mois {m} : {len(idx)} jours, "
                  f"pred_mean={p_flat.mean():.2f}  obs_mean={o_flat.mean():.2f} mm/j",
                  flush=True)

    def transform(self, pred_mm, months):
        """pred_mm : (N, H, W) → retourne (N, H, W) corrigé"""
        out = np.zeros_like(pred_mm)
        for m in [6, 7, 8, 9, 10]:
            idx = np.where(months == m)[0]
            if len(idx) == 0 or m not in self.maps:
                out[idx] = pred_mm[idx]
                continue
            q_pred, q_obs = self.maps[m]
            flat      = pred_mm[idx].ravel()
            corrected = np.interp(flat, q_pred, q_obs)
            out[idx]  = corrected.reshape(pred_mm[idx].shape)
        return np.clip(out, 0, None)


# =============================================================================
# PRÉDICTIONS
# =============================================================================

@torch.no_grad()
def predict(model, loader, y_min, y_max):
    model.eval()
    preds, targets = [], []
    for Xb, Yb in loader:
        p = torch.clamp(model(Xb.to(device)), 0., 1.).cpu().numpy()
        preds.append(p)
        targets.append(Yb.numpy())
    preds   = np.concatenate(preds,   axis=0)   # (N, 1, H, W)
    targets = np.concatenate(targets, axis=0)
    return denorm_y(preds, y_min, y_max), denorm_y(targets, y_min, y_max)


# =============================================================================
# ÉVALUATION
# =============================================================================

def evaluate(name, preds_mm, targets_mm):
    m   = compute_metrics(preds_mm, targets_mm)
    cat = compute_categorical(preds_mm, targets_mm, THRESHOLDS)
    f10 = fss(preds_mm[:, 0], targets_mm[:, 0], threshold=10, scale=3)
    f20 = fss(preds_mm[:, 0], targets_mm[:, 0], threshold=20, scale=5)
    p99r = np.percentile(preds_mm.ravel(), 99) / max(np.percentile(targets_mm.ravel(), 99), 0.1)
    print(f"\n  {name}")
    print(f"    RMSE={m['RMSE']:.3f}  NSE={m['NSE']:.4f}  r={m['r']:.4f}  "
          f"Slope={m['Slope']:.3f}  P99={p99r:.2f}", flush=True)
    print(f"    FSS_10={f10:.3f}  FSS_20={f20:.3f}  "
          f"CSI_40={cat[40]['CSI']:.3f}  POD_40={cat[40]['POD']:.3f}", flush=True)
    return {**m, 'P99_ratio': p99r,
            'CSI_40mm': cat[40]['CSI'], 'POD_40mm': cat[40]['POD'],
            'CSI_20mm': cat[20]['CSI'], 'POD_20mm': cat[20]['POD'],
            'CSI_5mm':  cat[5]['CSI'],  'POD_5mm':  cat[5]['POD'],
            'FSS_10mm': f10, 'FSS_20mm': f20}


# =============================================================================
# FIGURES
# =============================================================================

COLORS = {
    'UNet-Simple':    '#4477AA',
    'UNet-Simple+QM': '#AACCEE',
    'UNet-SE':        '#EE6677',
    'UNet-SE+QM':     '#FFAAAA',
    'UNet-CBAM':      '#228833',
    'UNet-CBAM+QM':   '#99DD77',
}


def plot_qm_impact(all_results):
    base_names = ['UNet-Simple', 'UNet-SE', 'UNet-CBAM']
    metrics_k  = ['RMSE', 'Slope', 'P99_ratio', 'FSS_20mm']
    metrics_lb = ['RMSE (mm/j)', 'Pente scatter', 'P99 ratio', 'FSS 20mm']

    fig, axes = plt.subplots(len(base_names), len(metrics_k),
                             figsize=(14, 9), sharey='col')
    bar_labels = ['Base', 'Base+QM']
    bar_colors = ['#4477AA', '#AACCEE']

    for ri, bname in enumerate(base_names):
        for ci, (mk, ml) in enumerate(zip(metrics_k, metrics_lb)):
            ax = axes[ri, ci]
            vals = [
                all_results.get(bname,          {}).get(mk, 0),
                all_results.get(f'{bname}+QM',  {}).get(mk, 0),
            ]
            bars = ax.bar(bar_labels, vals, color=bar_colors,
                          alpha=0.85, edgecolor='k', linewidth=0.5)
            if ci == 0:
                ax.set_ylabel(bname.replace('UNet-', ''), fontweight='bold')
            if ri == 0:
                ax.set_title(ml, fontweight='bold')
            if mk == 'P99_ratio':
                ax.axhline(1.0, color='green', lw=1.2, ls='--', alpha=0.7)
            ax.grid(axis='y', alpha=0.3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01 * max(vals + [1e-6]),
                        f'{v:.3f}', ha='center', va='bottom', fontsize=8)
            ax.tick_params(axis='x', labelsize=8)

    plt.suptitle('Impact du QM — 3 variantes UNet (ERA5→IMERG 0.05°)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'qm_impact_highres.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Sauvé : {path}", flush=True)


def plot_radar(all_results):
    metrics = ['NSE', 'FSS_20mm', 'CSI_40mm', 'P99_score', 'Slope']
    labels  = ['NSE', 'FSS 20mm', 'CSI 40mm', 'P99 score', 'Pente']
    ranges  = {
        'NSE':      (0.0, 0.35),
        'FSS_20mm': (0.0, 0.50),
        'CSI_40mm': (0.0, 0.10),
        'P99_score':(0.0, 1.0),
        'Slope':    (0.0, 0.80),
    }

    def norm(val, mn, mx):
        return max(0, min(1, (val - mn) / (mx - mn + 1e-9)))

    n_met  = len(metrics)
    angles = [2 * np.pi * i / n_met for i in range(n_met)] + [0]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    for name, res in all_results.items():
        p99_score = 1.0 - abs(res.get('P99_ratio', 1.0) - 1.0)
        vals = [
            norm(res.get('NSE',      0), *ranges['NSE']),
            norm(res.get('FSS_20mm', 0), *ranges['FSS_20mm']),
            norm(res.get('CSI_40mm', 0), *ranges['CSI_40mm']),
            norm(p99_score,              *ranges['P99_score']),
            norm(res.get('Slope',    0), *ranges['Slope']),
        ]
        vals += [vals[0]]
        c  = COLORS.get(name, '#999999')
        ls = '--' if '+QM' in name else '-'
        ax.plot(angles, vals, color=c, lw=2, ls=ls, label=name)
        ax.fill(angles, vals, color=c, alpha=0.05)

    ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1.15), fontsize=9)
    ax.set_title('Profil de performance — 6 configurations\n(ERA5 0.25° → IMERG 0.05°)',
                 fontsize=11, fontweight='bold', pad=20)
    path = os.path.join(FIG_DIR, 'radar_highres.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Sauvé : {path}", flush=True)


def make_table(all_results):
    order = [
        'UNet-Simple', 'UNet-Simple+QM',
        'UNet-SE',     'UNet-SE+QM',
        'UNet-CBAM',   'UNet-CBAM+QM',
    ]
    rows = []
    for name in order:
        if name not in all_results:
            continue
        r = all_results[name]
        rows.append({
            'Modèle':    name,
            'RMSE':      round(r.get('RMSE',     0), 3),
            'NSE':       round(r.get('NSE',      0), 4),
            'r':         round(r.get('r',        0), 4),
            'Slope':     round(r.get('Slope',    0), 3),
            'P99_ratio': round(r.get('P99_ratio',0), 2),
            'FSS_10mm':  round(r.get('FSS_10mm', 0), 3),
            'FSS_20mm':  round(r.get('FSS_20mm', 0), 3),
            'CSI_20mm':  round(r.get('CSI_20mm', 0), 3),
            'CSI_40mm':  round(r.get('CSI_40mm', 0), 3),
            'POD_40mm':  round(r.get('POD_40mm', 0), 3),
        })
    df = pd.DataFrame(rows).set_index('Modèle')
    print("\n" + "="*90)
    print("  TABLEAU COMPLET — 6 CONFIGURATIONS HAUTE RÉSOLUTION")
    print("="*90)
    print(df.to_string())
    path = os.path.join(FIG_DIR, 'full_table_highres.csv')
    df.to_csv(path)
    print(f"\n  Sauvé : {path}", flush=True)
    return df


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

if __name__ == '__main__':
    print("="*65)
    print("  QM Post-processing — Haute résolution ERA5→IMERG 0.05°")
    print("="*65, flush=True)

    # --- Données ---
    (X_tr, Y_tr, X_val, Y_val, X_te, Y_te,
     y_min, y_max, dpyr,
     months_tr, months_val, months_te,
     Y_raw_tr, Y_norm_tr) = load_and_preprocess()

    n_in  = X_tr.shape[1]   # 18
    n_out = Y_tr.shape[1]   # 1

    # DataLoaders avec upsample bilinéaire dans collate_fn
    tr_set = DownscalingDataset(X_tr, Y_tr, augment=False)
    te_set = DownscalingDataset(X_te, Y_te, augment=False)

    tr_loader = DataLoader(tr_set, batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=DownscalingDataset.collate_upsample,
                           num_workers=2, pin_memory=True)
    te_loader = DataLoader(te_set, batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=DownscalingDataset.collate_upsample,
                           num_workers=2, pin_memory=True)

    # Observations mm/j sur le train set pour fitter le QM
    # Y_raw_tr : (N_tr, H_IMERG, W_IMERG) mm/j
    Y_obs_tr = Y_raw_tr    # (N_tr, 380, 680) déjà en mm/j

    print(f"  Grille IMERG : {H_IMERG}×{W_IMERG}  |  Train {len(X_tr)} / Test {len(X_te)}")
    print(f"  Y_obs_tr : mean={Y_obs_tr.mean():.2f} max={Y_obs_tr.max():.1f} mm/j\n",
          flush=True)

    all_results = {}

    # =========================================================================
    # BOUCLE SUR LES 3 MODÈLES
    # =========================================================================

    for mname, cfg in CONFIGS.items():
        ckpt_path = os.path.join(CKPT_DIR, f"{mname.replace('-','_')}_best.pth")

        print(f"\n{'#'*55}")
        print(f"  {mname}")
        print(f"{'#'*55}", flush=True)

        if not os.path.exists(ckpt_path):
            print(f"  SKIP : {ckpt_path} introuvable", flush=True)
            continue

        # Charger le modèle
        model = UNet(n_in, n_out, **cfg).to(device)
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Chargé (val_loss={ckpt['val_loss']:.5f})", flush=True)

        # Prédictions train et test
        print("  Prédictions train...", flush=True)
        preds_tr_mm, _ = predict(model, tr_loader, y_min, y_max)
        print("  Prédictions test...",  flush=True)
        preds_te_mm, targets_mm = predict(model, te_loader, y_min, y_max)

        # Évaluation base
        res_base = evaluate(mname, preds_te_mm, targets_mm)
        all_results[mname] = res_base

        # Fitter le QM sur les prédictions train vs observations train
        print(f"\n  Fitting QM (n_quantiles=500)...", flush=True)
        qm = MonthlyQM(n_quantiles=500)
        qm.fit(preds_tr_mm[:, 0], Y_obs_tr, months_tr)

        # Appliquer QM sur le test
        preds_qm_mm = preds_te_mm.copy()
        preds_qm_mm[:, 0] = qm.transform(preds_te_mm[:, 0], months_te)

        qm_name = f'{mname}+QM'
        res_qm  = evaluate(qm_name, preds_qm_mm, targets_mm)
        all_results[qm_name] = res_qm

        # Résumé du gain QM
        print(f"\n  Gain QM ({mname}):", flush=True)
        print(f"    RMSE  : {res_base['RMSE']:.3f} → {res_qm['RMSE']:.3f} "
              f"({100*(res_qm['RMSE']-res_base['RMSE'])/res_base['RMSE']:+.1f}%)", flush=True)
        print(f"    Slope : {res_base['Slope']:.3f} → {res_qm['Slope']:.3f}", flush=True)
        print(f"    P99   : {res_base['P99_ratio']:.2f} → {res_qm['P99_ratio']:.2f}", flush=True)
        print(f"    FSS20 : {res_base['FSS_20mm']:.3f} → {res_qm['FSS_20mm']:.3f}", flush=True)
        print(f"    CSI40 : {res_base['CSI_40mm']:.3f} → {res_qm['CSI_40mm']:.3f}", flush=True)

        del model
        torch.cuda.empty_cache()

    # =========================================================================
    # TABLEAU ET FIGURES
    # =========================================================================

    print("\n[Figures]", flush=True)
    plot_qm_impact(all_results)
    plot_radar(all_results)
    df = make_table(all_results)

    print("\n" + "="*65)
    print("  MEILLEUR PAR MÉTRIQUE")
    print("="*65, flush=True)
    for metric, lower_is_better in [('RMSE', True), ('NSE', False),
                                     ('FSS_20mm', False), ('CSI_40mm', False),
                                     ('P99_ratio', False), ('Slope', False)]:
        if metric not in df.columns:
            continue
        best = df[metric].idxmin() if lower_is_better else df[metric].idxmax()
        print(f"  {metric:15s} → {best:30s} ({df[metric][best]:.4f})", flush=True)

    print("\n  Pipeline QM haute résolution terminé.", flush=True)
