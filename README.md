# Leak Localisation and Detection System

This repository packages the main components of a water distribution network leak localisation and detection project into one research codebase:

- `ST-GCN` models for leak detection and localisation
- `TCN` baselines for temporal leak detection/localisation
- `EKF` state estimation and reconstruction for partially observed sensing
- `EKF + ST-GCN` pipeline evaluation for hybrid reconstruction and classification

The current workspace already contains trained model bundles, evaluation scripts, EPANET/WNTR assets, and supporting utilities. This repository layer documents and organizes those pieces for GitHub publication.

## Final Pipeline

The repository contains multiple experimental scripts, but the main final pipeline for the project is:

### Final TCN

- Training: `train_tcn_detection_localisation5.py`
- Evaluation: `evaluate_model2.py`
- Final bundle: `multileak_tcn_bundleV6.pt`

### Final ST-GCN

- Training: `train_stgcn_sensor_placement.py`
- Evaluation: `evaluate_stgcn_sensor_placement.py`
- Final model family and placement sweep: `stgcn_placement_bundles/`

### Final EKF + ST-GCN Integration

- EKF-STGCN experiments and evaluation: `EKFplusSTGCN/`

### Final Data / Scenario Generation

- ST-GCN dataset generation: `generate_stgcn_dataset_v2.py`
- Test scenario generation: `generate_test_dataset2.py`, `generate_test_set.py`
- Multi-leak TCN scenario generation:
  - `generate_one_leak_training_data.py`
  - `generate_two_leaks_training_data.py`
  - `generate_three_leaks_training_data.py`
  - `generate_three_leaks_training_data2.py`
- GA pipeline used in the workflow: `ga_pipeline2.py`

Older numbered scripts are retained in the repository as experimental history, ablations, and intermediate model iterations.

## Project Scope

The repository is centered on three technical tracks:

1. `ST-GCN`: training and evaluation of spatiotemporal graph models, including the 10-sensor setup and sensor placement studies.
2. `TCN`: temporal convolution baselines for leak detection and localisation.
3. `EKF`: extended Kalman filter estimation for reconstructing unmonitored hydraulic states and supporting hybrid inference.

## Repository Map

The repository is organized conceptually into five groups:

- Final pipelines and supporting scripts at the repository root
- `stgcn_placement_bundles/` for ST-GCN sensor-placement model bundles
- `EKFplusSTGCN/` for hybrid EKF-STGCN experiments
- `ekf_wdn_project/` for the standalone EKF implementation
- `EPANET_Patterns_Final/` for final pattern assets used with the hydraulic model

For a guided list of final scripts versus experimental variants, see `docs/script_index.md`.

## Key Components

### ST-GCN

- Final training for sensor-placement experiments: `train_stgcn_sensor_placement.py`
- Final evaluation for sensor-placement experiments: `evaluate_stgcn_sensor_placement.py`
- Additional ST-GCN variants and intermediate experiments are retained at the repository root
- Placement bundles: `stgcn_placement_bundles/`

### TCN

- Final TCN training entry point: `train_tcn_detection_localisation5.py`
- Final TCN evaluation entry point: `evaluate_model2.py`
- Additional TCN variants and intermediate experiments are retained at the repository root
- Final TCN bundle: `multileak_tcn_bundleV6.pt`

### EKF

- Core EKF implementation: `ekf_wdn_project/ekf.py`
- Hydraulic model interface: `ekf_wdn_project/hydraulic_interface.py`
- Jacobians and configuration: `ekf_wdn_project/jacobians.py`, `ekf_wdn_project/config.py`
- Batch and focused evaluation runners: `ekf_wdn_project/run_ekf_batch_eval.py`, `ekf_wdn_project/run_ekf_focused_eval.py`

### Hybrid EKF + ST-GCN

- End-to-end pipeline: `pipeline_ekf_stgcn.py`
- Supporting integration experiments: `EKFplusSTGCN/`

## Repository Layout

This project currently keeps the original research scripts at the repository root to avoid breaking hardcoded paths during publication. The repository is documented as follows:

- [`README.md`](README.md): project overview and setup
- [`requirements.txt`](requirements.txt): Python dependencies
- [`docs/repository_guide.md`](docs/repository_guide.md): file map and publication guidance
- [`docs/script_index.md`](docs/script_index.md): final scripts and experimental variants map

## Recommended Environment

Python `3.10` or `3.11` is recommended.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Typical Workflows

### 1. Train or evaluate final ST-GCN models

```bash
python train_stgcn_sensor_placement.py
python evaluate_stgcn_sensor_placement.py
```

### 2. Train or evaluate the final TCN model

```bash
python train_tcn_detection_localisation5.py
python evaluate_model2.py --bundle multileak_tcn_bundleV6.pt
```

### 3. Run EKF estimation

```bash
python ekf_wdn_project/run_ekf_batch_eval.py
```

### 4. Run the hybrid EKF + ST-GCN pipeline

```bash
python pipeline_ekf_stgcn.py
```

## Data and Large Outputs

The working directory contains large generated datasets and experiment outputs. Those have been excluded from version control through `.gitignore` so the repository stays GitHub-friendly.

If you want to share data publicly, the best approach is to:

- keep code and trained lightweight bundles in this repository
- publish large datasets separately via Google Drive, Zenodo, or GitHub Releases
- link the external dataset location from this README

## Suggested GitHub Repository Name

One strong option is:

`leak-localisation-detection-system`

Other good options:

- `wdn-leak-localisation-detection`
- `hybrid-ekf-stgcn-leak-detection`
- `water-network-leak-detection`

## Next Publication Steps

1. Review the `.gitignore` and confirm which datasets or results you want public.
2. Initialize git and commit the curated repository contents.
3. Create an empty GitHub repository in your account.
4. Add the remote and push this local repository.

## Notes

- Some scripts use hardcoded local paths or dataset folders and may need small cleanup before wider public release.
- The current layout preserves your working environment first; it does not yet refactor everything into a package structure.
- Experimental and numbered variants are intentionally retained for research traceability; use `docs/script_index.md` to distinguish final scripts from older iterations.
