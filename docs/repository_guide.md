# Repository Guide

This guide explains how the current research workspace maps into a publishable GitHub repository.

## What Should Be Tracked

These are the files and folders that best represent the project:

- Core training and evaluation scripts at the repository root
- `ekf_wdn_project/` for the EKF implementation
- `EKFplusSTGCN/` for EKF-STGCN integration experiments
- `stgcn_placement_bundles/` for trained ST-GCN sensor-placement models
- The final TCN bundle at the repository root
- Network definition files such as `base3.inp`, `Hanoi.inp`, and related `.net` assets
- Configuration files such as `sensor_placements.csv`
- `EPANET_Patterns_Final/` for final pattern assets
- Documentation that explains final scripts versus experimental variants

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
- result and plot directories
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
- the EKF implementation
- the EKF + ST-GCN pipeline script
- the ST-GCN sensor-placement bundle collection
- the final TCN bundle
- sample hydraulic network files
- final pattern assets
- documentation on how datasets are generated or obtained
- a script index so experimental variants remain understandable

## Final vs Experimental Scripts

Because this repository preserves research history, not every staged script is part of the final pipeline.

- Final scripts should be highlighted in `README.md`
- Experimental and numbered variants should remain documented rather than hidden
- `docs/script_index.md` should be the main navigation aid for readers who want to distinguish final scripts from historical ones

## Before Pushing Publicly

Check for:

- hardcoded personal file paths
- sensitive or accidental local files
- oversized datasets
- duplicated experiment folders that are not needed for reproducibility

## Helpful Git Commands

```bash
git init
git add .
git commit -m "Initial commit for leak localisation and detection system"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
