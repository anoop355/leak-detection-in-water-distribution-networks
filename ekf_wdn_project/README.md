# EKF-Based Sparse-Sensor Hydraulic State Estimation

## Implementation Brief for Codex

### Objective

Implement a working Python system that uses a calibrated EPANET/WNTR network model and an Extended Kalman Filter to estimate the hydraulic state of a sparsely monitored water distribution network at 15-minute intervals.

The system must use:

* a calibrated `.inp` model
* sparse real measurements from:

  * pressure at **node 4**
  * flow in **pipe 1a**
  * flow in **pipe 3a**
* an EKF to estimate:

  * pressures/heads at unmonitored demand nodes
  * demands at demand nodes
  * flows in unmonitored pipes, recovered from the updated hydraulic model

This is a **functional engineering implementation**, not a novelty-focused research prototype.

---

# 1. Network and modeling assumptions

### Base network file

Use the provided calibrated EPANET input file:

* `base3.inp`

### Physical demand nodes

The real demand nodes are:

* 2
* 3
* 4
* 5
* 6

### Hydraulic helper leak nodes

The model also contains leak split/helper nodes:

* L1
* L2
* L3
* L4
* L5

These should remain in the hydraulic model, but they are **not part of the EKF state vector**.

### Real measurement locations

Measured in real life:

* pressure at node **4**
* flow in pipe **1a**
* flow in pipe **3a**

### Sampling interval

* **15 minutes**

---

# 2. EKF design

## 2.1 State vector

Use the following 10-state vector:

[
x_k =
\begin{bmatrix}
H_2\
H_3\
H_4\
H_5\
H_6\
D_2\
D_3\
D_4\
D_5\
D_6
\end{bmatrix}
]

Where:

* (H_i) = hydraulic head at node (i)
* (D_i) = demand at node (i)

Do **not** include pipe flows in the state vector.

Pipe flows are to be recovered by re-running the hydraulic model using the updated EKF demand estimates.

## 2.2 Measurement vector

Use:

[
y_k =
\begin{bmatrix}
P_4^{(m)}\
Q_{1a}^{(m)}\
Q_{3a}^{(m)}
\end{bmatrix}
]

Where:

* (P_4^{(m)}) = measured pressure at node 4
* (Q_{1a}^{(m)}) = measured flow in pipe 1a
* (Q_{3a}^{(m)}) = measured flow in pipe 3a

## 2.3 Pressure-head relation

The state stores **head**, not gauge pressure.

Pressure at node 4 must be computed as:

[
P_4 = H_4 - z_4
]

where node 4 elevation is taken from the `.inp` file.

## 2.4 Process model

Use the hydraulic model as the nonlinear predictor.

At each timestep:

* inject the current estimated demands (D_2 \ldots D_6) into the EPANET/WNTR model
* simulate one hydraulic step
* extract predicted heads at nodes 2–6
* keep demand states as a random-walk process

Demand evolution model:

[
D_{i,k+1} = D_{i,k} + \eta_{i,k}
]

This means demands are allowed to drift between timesteps.

## 2.5 Measurement function

The predicted measurement vector must be:

[
\hat{y}*k =
\begin{bmatrix}
\hat{P}*4\
\hat{Q}*{1a}\
\hat{Q}*{3a}
\end{bmatrix}
=============

\begin{bmatrix}
\hat{H}*4 - z_4\
\hat{Q}*{1a}\
\hat{Q}_{3a}
\end{bmatrix}
]

## 2.6 Residual

Residual must be computed as:

[
r_k = y_k - \hat{y}_k
]

## 2.7 EKF update

Implement standard EKF prediction and correction using numerical Jacobians.

Do not derive symbolic Jacobians manually.

Use finite-difference numerical Jacobians for:

* state transition Jacobian (F_k)
* measurement Jacobian (H_k)

---

# 3. Initial conditions

## 3.1 Initial demands

Initialize demands from the base demands in the `.inp` file:

* node 2: 1.5
* node 3: 1.0
* node 4: 0.5
* node 5: 1.0
* node 6: 1.0

So:

[
D_0 =
\begin{bmatrix}
1.5\
1.0\
0.5\
1.0\
1.0
\end{bmatrix}
]

If later needed, this can be expanded to include demand-pattern multipliers at each 15-minute step.

## 3.2 Initial heads

Initialize heads by running the hydraulic model at the first timestep and reading simulated heads at nodes 2–6.

So initial state is:

[
x_0 =
\begin{bmatrix}
H_2(0)\
H_3(0)\
H_4(0)\
H_5(0)\
H_6(0)\
1.5\
1.0\
0.5\
1.0\
1.0
\end{bmatrix}
]

## 3.3 Initial covariance (P_0)

Use:

[
P_0=\text{diag}(1,1,1,1,1,0.25,0.25,0.25,0.25,0.25)
]

Interpretation:

* initial head uncertainty: 1
* initial demand uncertainty: 0.25

---

# 4. Noise covariance assumptions

## 4.1 Process noise (Q)

Use this diagonal starting matrix:

[
Q=\text{diag}(0.01,0.01,0.01,0.01,0.01,0.0025,0.0025,0.0025,0.0025,0.0025)
]

Interpretation:

* small process noise on heads
* moderate process noise on demands

This should be configurable from a settings file.

## 4.2 Measurement noise (R)

Use a diagonal matrix:

[
R=
\begin{bmatrix}
\sigma_{P4}^2 & 0 & 0\
0 & \sigma_{Q1a}^2 & 0\
0 & 0 & \sigma_{Q3a}^2
\end{bmatrix}
]

Default starting assumptions:

* pressure std dev = 0.5
* flow std dev = 5% of nominal model flow magnitude in each corresponding pipe

Implementation requirement:

* compute nominal (Q_{1a}) and (Q_{3a}) from a baseline model run
* use 5% of those as the default flow standard deviations

Make (Q), (R), and (P_0) easy to tune.

---

# 5. Required implementation behavior

## 5.1 Runtime loop

For each 15-minute timestep:

1. read incoming measurements:

   * pressure at node 4
   * flow at pipe 1a
   * flow at pipe 3a

2. prediction step:

   * use previous state estimate
   * inject current estimated demands into the model
   * run one hydraulic step
   * extract predicted heads and measured-equivalent outputs

3. compute residual:

   * measured minus predicted

4. compute numerical Jacobians

5. EKF update:

   * update state estimate
   * update covariance

6. re-run hydraulic model using updated demand estimates

7. save full estimated hydraulic outputs:

   * heads at nodes 2–6
   * pressures at nodes 2–6
   * heads/pressures at helper nodes if available
   * flows in all pipes, especially:

     * 1a, 1b
     * 2a, 2b
     * 3a, 3b
     * 4a, 4b
     * 5a, 5b

## 5.2 Output requirements

Save outputs for every timestep to CSV files.

Required outputs:

* `state_estimates.csv`
* `predicted_measurements.csv`
* `residuals.csv`
* `all_node_heads.csv`
* `all_node_pressures.csv`
* `all_pipe_flows.csv`

## 5.3 Diagnostics

Generate plots for:

* measured vs predicted pressure at node 4
* measured vs predicted flow in pipe 1a
* measured vs predicted flow in pipe 3a
* residual time series for all 3 measurements
* estimated demands (D_2) to (D_6)
* estimated heads (H_2) to (H_6)

## 5.4 Logging

Add clear logging for:

* timestep index
* simulation success/failure
* residual magnitude
* covariance issues
* Jacobian failures
* negative or unrealistic demand estimates

---

# 6. Constraints and safeguards

## 6.1 Numerical stability

* guard against singular matrix inversion in EKF update
* use regularization if needed
* clip or constrain negative demands if they occur
* handle hydraulic simulation failures gracefully

## 6.2 Modularity

Organize code into clear modules:

* `load_model.py`
* `ekf.py`
* `hydraulic_interface.py`
* `jacobians.py`
* `run_estimator.py`
* `config.py`
* `plot_results.py`

## 6.3 Reproducibility

* include one main script that runs the full estimator
* include a config file with tunable parameters
* make the code easy to rerun on another measurement dataset

---

# 7. Measurement data interface

Assume measurement input comes from a CSV with columns like:

* `timestamp`
* `P4`
* `Q1a`
* `Q3a`

Sampling interval:

* 15 minutes

Codex should build the system so this file can be swapped easily for real data later.

---

# 8. First success criteria

The first working version is successful if it can:

1. load the `.inp` model
2. initialize the EKF correctly
3. process a 15-minute measurement time series
4. produce stable state estimates over time
5. recover unmonitored heads and pipe flows from the updated hydraulic model
6. save residuals and estimated demand trajectories

No leak-classification logic is required in this first implementation unless explicitly added later.

---

# 9. Nice-to-have features

If straightforward, also include:

* innovation covariance tracking
* normalized residual plotting
* optional simple bad-data rejection for extreme measurements
* optional switching between pressure or head outputs for reporting
* optional demand lower-bound constraint at zero

---

# 10. Do not do these things in version 1

* do not add deep learning
* do not add explicit leak states to the EKF
* do not rewrite the `.inp` file every timestep unless absolutely necessary
* do not use EPANET GUI in the workflow
* do not overcomplicate the state vector with all pipe flows as EKF states

---

# 11. Final design summary

Implement a Python-based EKF hydraulic state estimator for the provided EPANET model using sparse measurements at:

* node 4 pressure
* pipe 1a flow
* pipe 3a flow

Use a 10-state EKF with:

* heads at nodes 2–6
* demands at nodes 2–6

Use WNTR/EPANET as the nonlinear prediction engine, numerical Jacobians for EKF linearization, and re-run hydraulics after each update to recover full-network flows and pressures.

