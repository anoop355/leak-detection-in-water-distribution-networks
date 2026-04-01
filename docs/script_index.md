# Script Index

This index is intended to make the repository easy to navigate even though multiple experimental and numbered script variants are retained.

## Final Scripts

### Final TCN

- Training: `train_tcn_detection_localisation5.py`
- Evaluation: `evaluate_model2.py`
- Final bundle: `multileak_tcn_bundleV6.pt`

### Final ST-GCN

- Training: `train_stgcn_sensor_placement.py`
- Evaluation: `evaluate_stgcn_sensor_placement.py`
- Model bundles: `stgcn_placement_bundles/`

### Final EKF + ST-GCN

- Training / integration experiments: `EKFplusSTGCN/train_stgcn_ekf.py`
- Evaluation / pipeline scripts: files under `EKFplusSTGCN/`

### Final Dataset and Scenario Generation

- `generate_stgcn_dataset_v2.py`
- `generate_test_dataset2.py`
- `generate_test_set.py`
- `generate_one_leak_training_data.py`
- `generate_two_leaks_training_data.py`
- `generate_three_leaks_training_data.py`
- `generate_three_leaks_training_data2.py`
- `ga_pipeline2.py`

## Experimental Variants Kept for Traceability

These scripts are retained to document model iteration, ablations, and earlier experiments:

### TCN variants

- `train_tcn_detection_localisation.py`
- `train_tcn_detection_localisation1.py`
- `train_tcn_detection_localisation2.py`
- `train_tcn_detection_localisation3.py`
- `train_tcn_detection_localisation4.py`
- `train_tcn_sensor_placement.py`
- `evaluate_tcnmodels.py`
- `evaluate_tcn_v2_test_dataset.py`

### ST-GCN variants

- `train_stgtcn_detection_localisation.py`
- `train_stgtcn_detection_localisation2.py`
- `train_stgtcn_detection_localisation3.py`
- `train_stgtcn_detection_localisation4.py`
- `train_stgtcn_detection_localisation5.py`
- `train_stgtcn_detection_localisation6.py`
- `train_stgtcn_detection_localisation7.py`
- `train_stgcn_v8.py`
- `evaluate_stgcn_model_v1.py`
- `evaluate_stgcn_model_v2.py`
- `evaluate_sensor_placement_models.py`

### Other model experiments

- `train_stgcn_autoencoder.py`
- `train_stgcn_autoencoder_v2.py`
- `train_stgcn_autoencoder_v3.py`
- `train_stgcn_regressor.py`
- `evaluate_stgcn_autoencoder.py`
- `evaluate_stgcn_autoencoder_v2.py`
- `evaluate_stgcn_autoencoder_v3.py`
- `evaluate_stgcn_regressor.py`
- `evaluate_v3_focused.py`
- `evaluate_single_leak.py`
- `evaluate_GCNTCN_model.py`

## Supporting Project Areas

### Standalone EKF

- `ekf_wdn_project/` contains the standalone Extended Kalman Filter implementation and related utilities.

### Hydraulic Pattern Assets

- `EPANET_Patterns_Final/` contains final pattern files used with the hydraulic model.

### Network Assets

- Root `.inp` and `.net` files are retained as supporting hydraulic network assets.

## Reading Order for New Users

If you are new to the repository, start with:

1. `README.md`
2. `docs/repository_guide.md`
3. The final scripts listed above
4. Experimental variants only if you want to trace model evolution
