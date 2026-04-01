# Script Index

This index is intended to make the repository easy to navigate for someone seeing the project for the first time.

## Final TCN Path

- Training: `train_tcn_detection_localisation5.py`
- Evaluation: `evaluate_model2.py`
- Final bundle: `multileak_tcn_bundleV6.pt`
- Main result folder: `test_data_results/evaluation/`

## Final ST-GCN Path

- Training: `train_stgcn_sensor_placement.py`
- Evaluation: `evaluate_stgcn_sensor_placement.py`
- Model bundles: `stgcn_placement_bundles/`
- Main result folder: `stgcn_placement_results/`

## Final EKF + ST-GCN Path

- Training / integration experiments: `EKFplusSTGCN/train_stgcn_ekf.py`
- Evaluation / pipeline scripts: files under `EKFplusSTGCN/`
- Main result folder: `EKFplusSTGCN/results/`

## Dataset And Scenario Generation

- `generate_stgcn_dataset_v2.py`
- `generate_test_dataset2.py`
- `generate_test_set.py`
- `generate_one_leak_training_data.py`
- `generate_two_leaks_training_data.py`
- `generate_three_leaks_training_data.py`
- `generate_three_leaks_training_data2.py`
- `ga_pipeline2.py`

## Standalone EKF Code

- `ekf_wdn_project/` contains the standalone EKF Python source files used in the broader workflow.

## Supporting Assets

- `EPANET_Patterns_Final/` contains the final pattern files used with the hydraulic model.
- Root `.inp` and `.net` files are retained as supporting network assets.
- `sensor_placements.csv` defines the placement study configurations.

## Main Result Locations

- Final TCN results: `test_data_results/evaluation/`
- Final ST-GCN placement results: `stgcn_placement_results/`
- Final EKF-STGCN hybrid results: `EKFplusSTGCN/results/`

## Reading Order for New Users

If you are new to the repository, start with:

1. `README.md`
2. `docs/repository_guide.md`
3. This `docs/script_index.md`
4. The final scripts and result folders listed above
