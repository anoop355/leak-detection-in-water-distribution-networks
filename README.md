# Leak Detection and Localisation in Water Distribution Networks

This repository contains the full research codebase for a simulation-based
study of data-driven leak detection and localisation in water distribution
networks (WDNs). The work is modelled on similar infrastructure to the Water
and Sewerage Authority (WASA) of Trinidad and Tobago.

The framework combines hydraulic simulation (EPANET/WNTR) with deep
learning to detect and localise leaks in a five-pipe, five-junction
network monitored by ten sensor channels (P2–P6, Q1a–Q5a).

---

## Models

### ST-GCN (Primary Model — Single-Leak Detection and Localisation)
A Spatio-Temporal Graph Convolutional Network trained to detect and
localise single leaks across five pipes. A sensor placement study
using a Genetic Algorithm (GA) was conducted to identify optimal
sensor configurations across varying budget levels.

### TCN (Multi-Leak Detection and Localisation)
A Temporal Convolutional Network trained to detect and localise
simultaneous leaks (one, two, or three concurrent leaks).

### EKF (Investigated Alternative)
An Extended Kalman Filter was investigated as a state estimation
component to reconstruct hydraulic states at unmonitored nodes.
Structural observability limitations on pipes D4, D5, and D6,
which lie in the null space of the measurement Jacobian 
caused reconstruction failure. The EKF is retained in the 
repository for reference and is not part of the primary pipeline.

---

## Repository Structure

```
.
├── base.inp / base2.inp / base3.inp      # EPANET network input files
├── EPANET_Patterns_Final/                # Demand pattern files
├── multileak_tcn_bundleV6.pt             # Final trained TCN model bundle
├── stgcn_placement_bundles/              # Trained ST-GCN models (all sensor budgets)
├── stgcn_placement_results/              # ST-GCN sensor placement evaluation results
├── test_data_results/evaluation/         # TCN evaluation outputs
├── EKFplusSTGCN/                         # EKF investigation scripts and results
├── ekf_wdn_project/                      # EKF core implementation
├── docs/                                 # Script index and repository guide
├── sensor_placements.csv                 # GA sensor placement results summary
├── requirements.txt                      # Python dependencies
└── README.md
```

> **Note on generated datasets:** The `stgcn_dataset/` and `test_dataset/`
> folders are not tracked in this repository. They are produced on demand
> by the dataset generation scripts listed below and may be excluded from
> version control due to their size.

---

## Final Pipeline

### ST-GCN

| Step | Script |
|------|--------|
| Dataset generation | `generate_stgcn_dataset_v2.py` |
| Training (10-sensor S10 model) | `train_stgcn_detection_localisation_s10.py` |
| Sensor placement sweep (GA) | `ga_pipeline2.py` |
| Training (all budget configurations) | `train_stgcn_sensor_placement.py` |
| Test dataset generation | `generate_test_dataset2.py` |
| Evaluation (S10 model) | `evaluate_stgcn_s10.py` |
| Evaluation (placement sweep) | `evaluate_stgcn_sensor_placement.py` |
| Inference from `.inp` file | `predict_from_inp.py` |

**Model bundles:** `stgcn_placement_bundles/`
**Results:** `stgcn_placement_results/`

### TCN

| Step | Script |
|------|--------|
| Data generation (1-leak) | `generate_one_leak_training_data.py` |
| Data generation (2-leak) | `generate_two_leaks_training_data.py` |
| Data generation (3-leak) | `generate_three_leaks_training_data2.py` |
| Test set generation | `generate_test_set.py` |
| Training (final) | `train_tcn_detection_localisation5.py` |
| Evaluation (final) | `evaluate_model2.py` |
| Final bundle | `multileak_tcn_bundleV6.pt` |

**Results:** `test_data_results/evaluation/`

### EKF (Investigated — not primary pipeline)

| Step | Script |
|------|--------|
| Core EKF | `ekf_wdn_project/ekf.py` |
| Hydraulic interface | `ekf_wdn_project/hydraulic_interface.py` |
| Batch evaluation | `ekf_wdn_project/run_ekf_batch_eval.py` |

---

## Environment Setup

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/anoop355/leak-detection-in-water-distribution-networks.git
cd leak-detection-in-water-distribution-networks
pip install -r requirements.txt
```

> **Note on paths:** Several training and evaluation scripts were developed
> in Google Colab and contain hardcoded paths beginning with `/content/` or
> `/content/drive/MyDrive/`. These paths must be updated to match your local
> directory structure before running. Each script contains a clearly marked
> `# CONFIG` or `# SETTINGS` block at the top where these paths are set.

---

## Typical Workflows

### 1. Generate the ST-GCN training dataset
```bash
python generate_stgcn_dataset_v2.py
```

### 2. Train the ST-GCN S10 model
```bash
python train_stgcn_detection_localisation_s10.py
```

### 3. Generate the ST-GCN test dataset and evaluate
```bash
python generate_test_dataset2.py
python evaluate_stgcn_s10.py
```

### 4. Run inference on a custom `.inp` file
```bash
python predict_from_inp.py
```

### 5. Train or evaluate the final TCN model
```bash
python train_tcn_detection_localisation5.py
python evaluate_model2.py
```

---

## Note on Experimental Scripts

The repository root contains multiple numbered variants of training and
evaluation scripts (e.g. `train_tcn_detection_localisation1.py` through
`_5.py`). These represent iterative development stages and are retained
for reproducibility. The final scripts to use are identified in the
tables above and in `docs/script_index.md`.

---

## Documentation

- `docs/script_index.md` — maps every script to its role and development stage
- `docs/repository_guide.md` — file map and publication guidance

---

## Project Context

This project was completed as ECNG 3020 (Engineering Project) at the
Department of Electrical and Computer Engineering, University of the
West Indies, St. Augustine. Supervised by Prof. Arvind Singh.
