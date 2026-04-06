# Script Index

This index maps every script in the repository to its role and development
stage. It is intended to make the repository easy to navigate for someone
seeing the project for the first time.

---

## Where to Start

If you are new to the repository, read in this order:

1. `README.md` — project overview and setup
2. `docs/repository_guide.md` — file map and folder structure
3. This `docs/script_index.md` — role and status of every script

---

## Final ST-GCN Pipeline (Primary Model)

These are the definitive scripts for the ST-GCN single-leak detection
and localisation model, including the sensor placement study.

| Script | Role |
|--------|------|
| `generate_stgcn_dataset_v2.py` | Generates hydraulic simulation scenarios for ST-GCN training |
| `train_stgcn_sensor_placement.py` | Trains ST-GCN models across all sensor budget configurations |
| `ga_pipeline2.py` | Runs the Genetic Algorithm sensor placement optimisation |
| `generate_test_dataset2.py` | Generates held-out test scenarios for ST-GCN evaluation; used with `evaluate_stgcn_sensor_placement.py` |
| `evaluate_stgcn_sensor_placement.py` | Evaluates trained ST-GCN models across all sensor placements |
| `predict_from_inp.py` | Accepts a user-supplied `.inp` file and returns leak detection and localisation predictions |

**Model bundles:** `stgcn_placement_bundles/`
**Results:** `stgcn_placement_results/`

---

## Final TCN Pipeline (Multi-Leak Model)

These are the definitive scripts for the TCN multi-leak detection and
localisation model.

| Script | Role |
|--------|------|
| `generate_one_leak_training_data.py` | Generates single-leak training scenarios |
| `generate_two_leaks_training_data.py` | Generates two-leak training scenarios |
| `generate_three_leaks_training_data2.py` | Generates three-leak training scenarios (augmented, final version) |
| `generate_test_set.py` | Generates the held-out test dataset for TCN evaluation |
| `train_tcn_detection_localisation5.py` | Final TCN training script |
| `evaluate_model2.py` | Final TCN evaluation script |

**Final bundle:** `multileak_tcn_bundleV6.pt`
**Results:** `test_data_results/evaluation/`

---

## EKF (Investigated Alternative — Not Primary Pipeline)

The Extended Kalman Filter was investigated as a state estimation component
to reconstruct hydraulic states at unmonitored nodes. Structural
observability limitations prevented successful integration. These scripts
are retained for reference.

| Script | Role |
|--------|------|
| `ekf_wdn_project/ekf.py` | Core EKF implementation |
| `ekf_wdn_project/hydraulic_interface.py` | Hydraulic model interface for EKF |
| `ekf_wdn_project/jacobians.py` | Measurement Jacobian definitions |
| `ekf_wdn_project/config.py` | EKF configuration parameters |
| `ekf_wdn_project/run_ekf_batch_eval.py` | Batch EKF evaluation runner |
| `ekf_wdn_project/run_ekf_focused_eval.py` | Focused EKF evaluation runner |
| `pipeline_ekf_stgcn.py` | End-to-end EKF + ST-GCN integration pipeline (investigative) |
| `EKFplusSTGCN/` | EKF + ST-GCN experiment scripts and results |

---

## Experimental Script Variants (Development History)

The repository root contains numbered variants of several scripts. These
represent iterative development stages and are retained for reproducibility.
They should not be used in place of the final scripts listed above.

| Script | Status |
|--------|--------|
| `train_tcn_detection_localisation.py` | Early prototype — superseded |
| `train_tcn_detection_localisation1.py` | Experimental — superseded |
| `train_tcn_detection_localisation2.py` | Experimental — superseded |
| `train_tcn_detection_localisation3.py` | Experimental — superseded |
| `train_tcn_detection_localisation4.py` | Experimental — superseded |
| `train_tcn_detection_localisation5.py` | **FINAL** |
| `generate_stgcn_dataset.py` | Early version — superseded by `_v2.py` |
| `generate_stgcn_dataset_v2.py` | **FINAL** |
| `generate_test_dataset.py` | Early version — superseded |
| `generate_test_dataset2.py` | **FINAL** (ST-GCN test scenarios) |
| `generate_test_set.py` | **FINAL** (TCN test scenarios) |
| `generate_three_leaks_training_data.py` | Early version — superseded |
| `generate_three_leaks_training_data2.py` | **FINAL** (augmented data) |
| `ga_pipeline.py` | Early GA version — superseded |
| `ga_pipeline2.py` | **FINAL** |
| `evaluate_model.py` | Early evaluation — superseded |
| `evaluate_model2.py` | **FINAL** |
| `evaluate_stgcn_model_v1.py` | Early ST-GCN evaluation — superseded |
| `evaluate_stgcn_sensor_placement.py` | **FINAL** |

---

## Diagnostic Scripts

These scripts were used during model investigation to trace the source of
a specific failure mode: the systematic misclassification of Pipe 1 leaks
as Pipe 4 leaks in the ST-GCN model (12 affected scenarios under the
S10-A sensor configuration). They are retained for reproducibility.

| Script | Role |
|--------|------|
| `debug_s10a_softmax.py` | Loads the S10-A bundle and runs inference on the 12 misclassified scenarios; extracts raw pipe head logits, softmax probabilities across all six pipe classes (Pipe 1–5 + no-pipe), and detect head softmax. Output saved to `stgcn_placement_results/S10-A/debug_pipe1_misclassified_softmax.csv` |
| `debug_s10a_internals.py` | Runs three internal probing analyses using a modified probe model (`SingleLeakSTGCNv4Probe`) that exposes intermediate tensors: (1) layer-by-layer cosine similarity, (2) PCA of the embedding space, and (3) temporal attention weights. All outputs saved to `stgcn_placement_results/S10-A/debug_internals/` |

---

## Supporting Scripts

| Script | Role |
|--------|------|
| `evaluate_sensor_placement_models.py` | Supporting sensor placement evaluation |
| `evaluate_tcn_models.py` | Evaluates multiple TCN variants |
| `evaluate_tcn_v2_test_dataset.py` | TCN evaluation on v2 test dataset |
| `train_tcn_sensor_placement.py` | Trains TCN under sensor placement configurations |
| `generate_no_leak_training_scenarios.py` | Generates no-leak baseline scenarios |

---

## Hydraulic Network Assets

These files define the water distribution network used in all simulations.
They are a core component of the framework — all hydraulic scenario
generation depends on these inputs.

| Item | Description |
|------|-------------|
| `base.inp` | Primary EPANET network input file |
| `base2.inp` | Variant network input file used in multi-start scenario generation |
| `base3.inp` | Variant network input file used in extended scenario generation |
| `first_project.net`, `first_project_1.net`, `first_project_2.net` | EPANET project files |
| `EPANET_Patterns_Final/` | Final demand pattern files applied across all simulation scenarios |

---

## Supporting Data Files

| Item | Description |
|------|-------------|
| `sensor_placements.csv` | GA sensor placement study configurations and fitness scores |
| `multileak_tcn_bundleV6.pt` | Final trained TCN model bundle |
| `stgcn_placement_bundles/` | Trained ST-GCN model bundles for all sensor budgets |
