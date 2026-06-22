# Deep-learning statistical downscaling of daily precipitation over West Africa

Replication code and materials for the manuscript submitted to *Environmental Data Science*
(Cambridge University Press).

We downscale daily precipitation from the ERA5 reanalysis (0.25°) to the IMERG product at its
**native 0.1° resolution** (factor 2.5×) over West Africa, comparing U-Net architectures
(plain / Squeeze-and-Excitation / CBAM attention) and two post-processing strategies
(monthly quantile mapping and a conditional GAN) against a BCSD baseline.

## Repository structure

```
src/        Core library
  downscaling_highres.py   U-Net models, dataset, training, metrics (FSS/CSI/NSE…), prediction
  qm_highres.py            Monthly quantile-mapping post-processing
scripts/    Reproduction scripts
  reeval_01deg.py            Deterministic re-evaluation at 0.1° (+ 0.05° fidelity check)
  reeval_full_01deg.py       Global, regional and monthly performance tables
  reeval_categorical_01deg.py POD/FAR/CSI categorical table
  make_figures_pubstyle_01deg.py  Quantitative figures (scatter, metrics, Taylor, …)
  make_fig1_domain.py        Study domain map
  make_figB_bias_01deg.py    Bias maps
  make_fig9_4cols.py         Post-processing spatial samples
  make_figdry_01deg.py       Appendix: three driest days
  make_figwet_01deg.py       Appendix: three wettest days
results/    Pre-computed metric tables (CSV) reproduced by the scripts
figures/    Figures of the manuscript (PDF) produced by the scripts
```

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Data and model weights

The **raw predictor and target data are openly available** from the original providers:

- **ERA5** reanalysis — Copernicus Climate Change Service (C3S) Climate Data Store.
- **IMERG** Final Run (V06) precipitation — NASA GES DISC.

The **pre-processed arrays** used in this study (`X_nc_data.npy`, `Y_nc_data.npy`), the **cached
model outputs** (`arrays_01deg.npz`) and the **trained model weights** (`checkpoints_highres/`) are
too large for GitHub and are archived on **Zenodo**:

> **DOI: [10.5281/zenodo.20801025](https://doi.org/10.5281/zenodo.20801025)**

The small files needed to run the scripts (`results/norm_stats.npz`, the channel-wise
normalisation statistics, and `results/channel_importance_highres.csv`, the permutation-importance
table) are **already included in this repository**.

All paths in the scripts resolve **automatically** relative to the repository, so no editing is
required. From Zenodo you only need to place the large files yourself:

- put `arrays_01deg.npz` in `results/` (enables the figure and table scripts);
- unzip `weights.zip` into `checkpoints_highres/` (only needed to re-run the networks).

Optional environment variables let you point elsewhere without editing the code:

| Variable | Default | Purpose |
|----------|---------|---------|
| `CKPT_DIR` | `<repo>/checkpoints_highres` | location of the trained `*.pth` weights |
| `X_PATH` / `Y_PATH` | NFS paths in `src/downscaling_highres.py` | raw ERA5 / IMERG `.npy` arrays |

## Reproducing the results

```bash
# from the repository root — no PYTHONPATH needed, paths are auto-resolved

# Figures (use only results/arrays_01deg.npz, no GPU)  ->  figures/*.pdf
python scripts/make_figures_pubstyle_01deg.py
python scripts/make_figdry_01deg.py
python scripts/make_figwet_01deg.py

# Metric tables (need the weights from weights.zip + raw arrays)  ->  results/*.csv
python scripts/reeval_full_01deg.py
python scripts/reeval_categorical_01deg.py
```

Model outputs are cached in `arrays_01deg.npz` (from Zenodo) so the figure scripts run
without re-running the networks.

## Citation

If you use this code, please cite the article and this repository (see `CITATION.cff`).

## License

Code is released under the MIT License (see `LICENSE`). The archived data and model weights on
Zenodo are released under CC-BY-4.0.
