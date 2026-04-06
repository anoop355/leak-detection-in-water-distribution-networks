# Repository Guide

This guide explains how the research workspace maps into a publishable
GitHub repository.

---

## What Is Tracked

These are the files and folders that represent the project:

* Core training and evaluation scripts at the repository root
* `ekf_wdn_project/` — standalone EKF Python implementation (retained for reference)
* `EKFplusSTGCN/` — EKF + ST-GCN investigation experiments and results
* `stgcn_placement_bundles/` — trained ST-GCN sensor-placement model bundles
* `stgcn_placement_results/` — ST-GCN sensor placement evaluation outputs
* `test_data_results/evaluation/` — final TCN evaluation outputs
* `multileak_tcn_bundleV6.pt` — final trained TCN model bundle
* `base.inp`, `base2.inp`, `base3.inp` — EPANET network input files
* `first_project.net`, `first_project_1.net`, `first_project_2.net` — EPANET project files
* `EPANET_Patterns_Final/` — final demand pattern files used across all scenarios
* `sensor_placements.csv` — GA sensor placement configurations and fitness scores
* `docs/` — repository guide and script index

---

## What Is Excluded

The `.gitignore` excludes folders that are either regenerated from scripts,
too large for a source repository, or local-environment specific. Examples:

* `stgcn_dataset/`
* `training_cases/`
* `training_cases_output/`
* `test_dataset/`
* Generated raw datasets and scratch output directories
* `.venv/`

---

## Main Result Folders

| Folder | Contents |
|--------|----------|
| `test_data_results/evaluation/` | Final TCN evaluation outputs |
| `stgcn_placement_results/` | ST-GCN sensor placement study outputs |
| `EKFplusSTGCN/results/` | EKF + ST-GCN investigation outputs (not primary pipeline) |

---

## EKF Folder

The repository keeps only Python source files from `ekf_wdn_project/`.

* This keeps the EKF implementation visible for reference
* It avoids mixing the code folder with generated data and output files
* Supporting EKF + ST-GCN investigation results are retained in `EKFplusSTGCN/`
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
