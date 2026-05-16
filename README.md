# T-rate: Bayesian Inference of Stress Evolution From Seismicity Rate Observations

<p align="center">
  <img src="figures/T_rate_logo.png" alt="T-rate logo" width="300">
</p>

**T-rate** is a Python/PyMC tool for inferring transient stress evolution from changes in seismicity rate. It was developed for seismicity-rate observations governed by rate-and-state friction and is demonstrated here using synthetic experiments and sensitivity tests.

The code was developed for:

Jiang, Y., Trugman, D. T., & González, P. J. (2026).  
*Bayesian inference of complex stress evolution in rate-and-state governed faults constrained by seismicity rate observations*.  
Journal of Geophysical Research: Solid Earth.

---

## 1. Overview

T-rate estimates the timing and amplitude of transient stress-rate histories from observed changes in seismicity rate. The goal of the inversion is to find the stress-rate history that best explains the observed seismicity rate changes.

In the default implementation, each transient source is represented by a **trapezoidal stress-rate function**. For each ramp function, the source parameters are:

$$
L,\quad t_1,\quad \Delta t_{12},\quad \Delta t_{23},\quad \Delta t_{34}.
$$

Here, $L$ defines the peak stress-rate amplitude, $t_1$ defines the onset time, and $\Delta t_{12}$, $\Delta t_{23}$, and $\Delta t_{34}$ define the durations of the ramp-up, plateau, and ramp-down stages. The transition times are:

$$
t_2 = t_1 + \Delta t_{12},
$$

$$
t_3 = t_2 + \Delta t_{23},
$$

$$
t_4 = t_3 + \Delta t_{34}.
$$

The global constitutive/loading parameters are:

$$
\dot{\tau}_r,\quad A\sigma_0.
$$

Here, $\dot{\tau}_r$ is the background stressing rate and $A\sigma_0$ is the effective frictional stress scale.

The characteristic time is calculated internally as:

$$
t_a = \frac{A\sigma_0}{\dot{\tau}_r}.
$$

For a model with $N$ ramp functions, the total number of estimated model parameters is:

$$
2 + 5N.
$$

where the two global parameters are $\dot{\tau}_r$ and $A\sigma_0$, and each ramp contributes five source parameters.

![Figure 1. Schematic of the T-rate model parameters.](figures/Figure_parameter_schematic.png)

*Figure 1. Stress history and seismicity-rate response for a trapezoidal stress-rate function. (a) Prescribed stress-rate history (blue) and the resulting cumulative stress history (red). The characteristic transition times, t1–t4, are marked in green and define the onset, ramp-up, plateau, ramp-down, and termination stages of the transient stressing episode. (b) Predicted seismicity-rate ratio, R/r, derived from the stress history shown in panel (a).*

The physical meaning of the key parameters is summarized below. The table uses manuscript-style symbols; the corresponding Python variable names are given where helpful.

| Group | Symbol | Python variable | Parameter name | Physical definition |
|:---:|:---:|:---:|:---:|:---:|
| Source | $N$ | `n_ramps` | Number of ramp functions | Number of transient stress-rate source functions |
| Source | $L$ | `L` | Stress-rate amplitude | Peak amplitude of<br>the transient Coulomb stress rate |
| Source | $t_1$ | `t1` | Onset time | Start time of<br>transient stress loading |
| Source | $\Delta t_{12}$ | `delta_t12`<br>or equivalent prior variable | Ramp duration | Duration of stress-rate ramp-up |
| Source | $\Delta t_{23}$ | `delta_t23`<br>or equivalent prior variable | Plateau duration | Duration of constant stress-rate plateau |
| Source | $\Delta t_{34}$ | `delta_t34`<br>or equivalent prior variable | Ramp duration | Duration of stress-rate ramp-down |
| Constitutive | $\dot{\tau}_r$ | `tau_rate_bg` | Background stress rate | Steady-state tectonic shear stressing rate |
| Constitutive | $A\sigma_0$ | `Asigma_0` | Frictional stress scale | Controls seismicity sensitivity to<br>stress changes |
| Constitutive | $t_a$ | `ta` | Decay time | Characteristic aftershock decay time, calculated as $A\sigma_0 / \dot{\tau}_r$ |
| Constitutive | $R$ | `R` | Seismicity rate | Non-earthquake-triggered seismicity rate |
| Constitutive | $r$ | `r` | Seismicity rate | Background seismicity rate |

---

## 2. Repository Structure

```text
T-rate/
├── Trate.py
├── run_T_rate_inversion_example.py
├── example_input/
│   └── example_input.txt
├── example_output/
├── figures/
│   ├── T_rate_logo.png
│   ├── Figure_parameter_schematic.png
│   └── Figure_control_experiment_results.png
├── README.md
├── requirements.txt
└── LICENSE
```

`Trate.py` contains the reusable T-rate core functions, including the forward model, likelihood functions, posterior extraction, and plotting utilities. For most users, `Trate.py` should be treated as a library file.

The `run_T_rate_inversion_example.py` file is the user-facing driver script. New experiments can be created by copying and editing this script.

---

## 3. Installation and Quick Start

Create a separate environment:

```bash
conda create -n Trate_env python=3.11
conda activate Trate_env
pip install -r requirements.txt
```

The `requirements.txt` file should include:

```text
numpy
pandas
matplotlib
seaborn
pymc
pytensor
arviz
scipy
```

Run the control example:

```bash
python run_T_rate_inversion_example.py
```

The example inversion uses a synthetic input file included in `example_input/`:

```text
example_input/example_input.txt
```

New users do not need to prepare their own catalog before running this example. The example inversion should run without modification if the repository is downloaded completely.

After a successful run, output files should be created in `example_output/`. It should contain files such as:

```text
parameters.txt
trace_plot.png
lambda_modeled.png
S_modeled.png
```

---

## 4. First-Time User Checklist

For a first test, users should:

1. Create and activate the `Trate_env` environment.
2. Run `python run_T_rate_inversion_example.py` without changing anything.
3. Confirm that output files are created in `example_output/`.
4. Open `lambda_modeled.png`, `S_modeled.png`, and `trace_plot.png`.

---

## 5. What Users Should Modify

For a new application or sensitivity test, users should copy one existing driver script and modify the copy.

For example:

```text
run_T_rate_inversion_example.py
```

There are two main places to modify.

### 5.1 `run_inversion()`

This function controls the Bayesian inversion setup, including:

- prior distributions and prior bounds
- number of MCMC draws and tuning steps
- sampler choice
- likelihood choice
- number of chains and CPU cores
- output directory

Typical parameters to modify include $L$, $t_1$, $\Delta t_{12}$, $\Delta t_{23}$, $\Delta t_{34}$, $A\sigma_0$, and $\dot{\tau}_r$.

In the README and manuscript notation, the Python variable `Asigma_0` corresponds to $A\sigma_0$, and `tau_rate_bg` corresponds to $\dot{\tau}_r$.

For a first trial, choose prior bounds that are broad enough to include the expected transient timing and amplitude. The prior bounds can be narrowed later in sensitivity tests if needed.

### 5.2 Main execution block

The main execution block is:

```python
if __name__ == "__main__":
```

This part controls experiment-level settings, including:

- input data file
- time-bin width `time_bin`
- background seismicity rate $r$
- number of ramp functions $N$, commonly defined in the code as `n_ramps`
- experiment ID
- likelihood option
- output folder

---

## 6. Input Data Format

The example driver script reads:

```text
example_input/example_input.txt
```

This file is a plain text file with **two columns**:

```text
time    earthquake_count
```

where:

- `time` is the middle time of each time bin
- `earthquake_count` is the number of earthquakes or events in that time bin

Example:

```text
time    earthquake_count
0.5     2
1.5     5
2.5     12
3.5     20
4.5     8
5.5     4
```

If the time-bin width is 1 day, the first row represents the bin from 0 to 1 day, with middle time 0.5 day and 2 earthquakes in that bin.

The example above includes column names for explanation. If the driver script reads the file using `np.loadtxt` without `skiprows`, the actual `.txt` input file should contain only numeric values, for example:

```text
0.5     2
1.5     5
2.5     12
3.5     20
4.5     8
5.5     4
```

The input file should not include extra columns unless the driver script has been modified to read them.

The same format can also be used for other seismicity rate datasets. For example, in laboratory acoustic emission experiments, the second column can be the number of acoustic emission events in each time bin.

---

## 7. Time Units and Background Rate

T-rate is unit-flexible. The time unit can be seconds, minutes, days, years, or normalized laboratory time. However, the same time unit must be used consistently for:

- time
- time-bin width `time_bin`
- source timing parameters $t_1$, $\Delta t_{12}$, $\Delta t_{23}$, and $\Delta t_{34}$
- background seismicity rate $r$
- background stressing rate $\dot{\tau}_r$

The background seismicity rate $r$ is the long-term or reference seismicity rate. For detailed steps on estimating $R$ and $r$, users should refer to Section 3.2 and the associated sensitivity tests and discussions in Section 4.2 of Jiang, Trugman, and González (2026).

---

## 8. Choosing Key Parameters

### Number of ramp functions

Start with:

```python
n_ramps = 1
```

Increase `n_ramps` only if the data require more than one transient episode. More ramp functions increase model flexibility but also increase the number of parameters.

### Timing parameters

The timing parameters are $t_1$, $\Delta t_{12}$, $\Delta t_{23}$, and $\Delta t_{34}$. Choose prior bounds that include the plausible time window of the transient event and physically reasonable durations for the ramp-up, plateau, and ramp-down stages.

### $A\sigma_0$

$A\sigma_0$ is the effective rate-and-state constitutive parameter. In the code, this parameter is named `Asigma_0`. As a practical starting value, users can try:

$$
A\sigma_0 = 1500~\mathrm{Pa}.
$$

This value is motivated by Sirorattanakul and Avouac (2026, Science Advances).

### $\dot{\tau}_r$

$\dot{\tau}_r$ is the background stressing rate. In the code, this parameter is named `tau_rate_bg`.

A practical first-order estimate is:

$$
\dot{\tau}_r = \text{shear modulus} \times \text{strain rate}.
$$

For crustal applications, users may start with a shear modulus of 30 GPa. The strain rate can be obtained from reliable sources such as GPS-based strain rate models (such as Global Strain Rate Model, Kreemer et al., 2014, G3), InSAR-based strain rate models, or other geodetic/experimental constraints.

---

## 9. Output Files

Each run produces output files in `example_output/`. The exact output folder name can be modified in the driver script.

Typical outputs include:

```text
parameters.txt
parameters_p5_p95.txt
lambda_results.txt
lambda_optimal.txt
dS_results.txt
S_results.txt
S_optimal.txt
trace_plot.png
prior_predictive_check_log.png
joint_posterior_probability_kde.png
lambda_modeled.png
S_modeled.png
```

![Figure 2. Example output from the control experiment.](figures/Figure_control_experiment_results.png)

*Figure 2. Input data and inversion results for the synthetic example experiment. (a) Observed event-rate ratio, R/r, and modeled conditional intensity. (b) Inferred stressing-rate history and cumulative stress history. (c) Joint posterior probability distribution of the model parameters.*

The posterior parameter file is organized as:

```text
[t1..., t2..., t3..., t4..., L..., Asigma_0, tau_rate_bg]
```

where `Asigma_0` corresponds to $A\sigma_0$, `tau_rate_bg` corresponds to $\dot{\tau}_r$, and `t2`, `t3`, and `t4` are calculated from `t1`, `Δt12`, `Δt23`, and `Δt34`.

For one ramp:

```text
t1  t2  t3  t4  L  Asigma_0  tau_rate_bg
```

For two ramps:

```text
t1_1  t1_2  t2_1  t2_2  t3_1  t3_2  t4_1  t4_2  L_1  L_2  Asigma_0  tau_rate_bg
```

---

## 10. Sensitivity Tests

The repository includes one example driver script:

```text
run_T_rate_inversion_example.py
```

Users can copy and modify this script to create their own sensitivity tests. For example, users may create separate driver scripts for prior sensitivity, sampler sensitivity, or source-function sensitivity tests.

---

## 11. Potential Applications Beyond the Synthetic Examples

This repository is primarily designed to demonstrate T-rate using synthetic experiments and sensitivity tests. In the current implementation, the stress source is represented by a trapezoidal stress-rate function, but this function is flexible and can approximate several simpler source types. For example, it can be specified as an impulse-like function to represent an instantaneous stress jump, such as a coseismic Coulomb stress change, or as a boxcar function to represent a period of approximately constant stressing rate, such as a simplified slow-slip episode. The same framework may therefore be useful for seismicity-rate data from laboratory acoustic emission experiments, earthquake swarms, induced seismicity, and slow-slip-related seismicity or tremor. For these applications, users should carefully define the background seismicity rate $r$, choose a consistent time unit, and set physically reasonable priors based on their experiment, catalog, or independent geodetic constraints. The inferred stress history should be interpreted as an effective stressing history that explains the observed seismicity-rate modulation under the assumed rate-and-state model.

---

## 12. Citation

If you use T-rate in your research, please cite:

Jiang, Y., Trugman, D. T., & González, P. J. (2026).  
*Bayesian inference of complex stress evolution in rate-and-state governed faults constrained by seismicity rate observations*.  
Journal of Geophysical Research: Solid Earth.

---

## 13. License

This repository is released under the MIT License. See the `LICENSE` file for details.
