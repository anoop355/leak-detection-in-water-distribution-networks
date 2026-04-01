# Repository Guide

This guide explains how the current research workspace maps into a publishable GitHub repository.

## What Should Be Tracked

These are the files and folders that best represent the project:

- Core training and evaluation scripts at the repository root
- `ekf_wdn_project/` for the standalone EKF Python implementation
- `EKFplusSTGCN/` for EKF-STGCN integration experiments
- `stgcn_placement_bundles/` for trained ST-GCN sensor-placement models
- `stgcn_placement_results/` for the ST-GCN sensor-placement outputs
- `test_data_results/evaluation/` for the final TCN outputs
- The final TCN bundle at the repository root
- Network definition files such as `base3.inp` and related `.net` assets
- Configuration files such as `sensor_placements.csv`
- `EPANET_Patterns_Final/` for final pattern assets
- Documentation that explains final scripts and result folders

## What Is Currently Ignored

The `.gitignore` excludes folders that are either:

- regenerated from scripts
- very large and impractical for a source repository
- local-environment specific
- result-heavy rather than source-heavy

Examples include:

- `stgcn_dataset/`
- `training_cases/`
- `training_cases_output/`
- `test_dataset/`
- generated raw datasets and scratch output directories
- `.venv/`

## Suggested Public-Facing Structure

If you decide to refactor later, this is a good target structure:

```text
.
|-- README.md
|-- requirements.txt
|-- docs/
|-- models/
|-- data/
|-- stgcn/
|-- tcn/
|-- ekf/
|-- pipelines/
`-- scripts/
```

For now, keeping the research scripts in place is the safest choice because many of them rely on relative paths to the current directory layout.

## Recommended First GitHub Release Content

For the first public version, include:

- the final ST-GCN training and evaluation scripts
- the final TCN training and evaluation scripts
- the standalone EKF Python implementation
- the EKF + ST-GCN pipeline script
- the ST-GCN sensor-placement bundle collection
- the main result folders for the final implementations
- the final TCN bundle
- sample hydraulic network files
- final pattern assets
- documentation on how datasets are generated or obtained
- a script index so readers can find final code and result folders quickly

## Main Result Folders

The main result folders that should remain visible in the repository are:

- `test_data_results/evaluation/` for the final TCN implementation
- `stgcn_placement_results/` for the ST-GCN placement study
- `EKFplusSTGCN/results/` for the hybrid EKF-STGCN outputs

## Code-Only EKF Folder

The repository keeps only Python source files from `ekf_wdn_project/`.

- This keeps the EKF implementation visible
- It avoids mixing the code folder with generated data and output files
- Supporting hybrid EKF-STGCN results are instead retained in `EKFplusSTGCN/`

## Before Pushing Publicly

Check for:

- hardcoded personal file paths
- sensitive or accidental local files
- oversized datasets
- folders that repeat raw/generated data already reproducible from scripts

## Helpful Git Commands

```bash
git init
git add .
git commit -m "Initial commit for leak localisation and detection system"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
