#!/usr/bin/env python

"""
User-facing driver script for T-rate seismicity rate inversion.

For most applications, users should edit only
this file and keep Trate.py unchanged. The reusable forward-model, plotting,
and posterior-processing functions are imported from Trate.py.

Reference
---------
Jiang, Y., Trugman, D. T., & González, P. J. (2026)
Bayesian inference of complex stress evolution in rate-and-state governed faults
constrained by seismicity rate observations
Journal of Geophysical Research: Solid Earth
"""

# Keep Trate.py and this driver script in the same folder.
# Import the reusable library as a module so module-level variables used by
# plotting routines can be updated explicitly.
import Trate as tr

# Import public names from Trate.py so the original run_inversion() and main
# block can be reused without changing their executable code.
from Trate import *

# ====================================================================
# USER-DEFINED SETTINGS FOR NEW APPLICATIONS
# --------------------------------------------------------------------
# Most users should only edit the two sections below:
#
#   1) run_inversion(...)
#      Define the inversion setup:
#        - prior ranges for ramp timing parameters t1, delta_t1_t2, delta_t2_t3, and delta_t3_t4
#        - prior for ramp amplitude L
#        - prior for Asigma_0
#        - prior for background stressing rate tau_rate_bg
#        - likelihood choice: Poisson or Gaussian
#        - sampler choice, draws, tuning steps, chains, and cores
#        - prior predictive sample size
#
#   2) if __name__ == "__main__":
#      Define the data and experiment setup:
#        - input file path and data columns
#        - time-bin width, time_bin
#        - background seismicity rate, r
#        - number of trapezoidal ramps, n_ramps
#        - number of MCMC chains, n_chains
#        - experiment IDs and likelihood/noise flags
#        - output folders and post-processing choices
#
# ====================================================================

def run_inversion(t_data_np, R_r_observed_np, n_ramps, flag_noise_poisson, r, time_bin, n_chains):
    """
    Build and sample the binned seismicity rate inversion model.
    """

    # --------------------------------------------------------------------
    # 1) PyMC Model Definition
    # --------------------------------------------------------------------
    with pm.Model() as model:
        # ------------------------------------------------------------
        # User-defined prior ranges for the trapezoidal stress-rate ramp
        # ------------------------------------------------------------
        # For each ramp:
        #   t1 = start time of stress-rate increase
        #   t2 = end time of increase / start of constant stress rate
        #   t3 = end time of constant stress rate / start of decrease
        #   t4 = end time of stress-rate decrease
        #   L  = peak stress-rate amplitude
        #
        # The code samples t1 and positive time intervals
        # (delta_t1_t2, delta_t2_t3, delta_t3_t4), then reconstructs
        # t2, t3, and t4. This guarantees t1 < t2 < t3 < t4.
        #
        # For a new data set, adjust the lower/upper bounds below so that
        # they cover the expected timing of the stress-rate transient.
        # The division by time_bin keeps the prior ranges consistent with
        # the time unit used by the binned observations.
        if n_ramps == 1:
            t1_raw    = pm.Uniform('t1_raw', lower=np.array([10]) / time_bin, upper=np.array([40]) / time_bin, shape=n_ramps)
            dt1t2_raw = pm.Uniform('delta_t1_t2_raw', lower=np.array([20]) / time_bin, upper=np.array([40]) / time_bin, shape=n_ramps)
            t2_raw    = pm.Deterministic('t2_raw', t1_raw + dt1t2_raw)

            dt2t3_raw = pm.Uniform('delta_t2_t3_raw', lower=np.array([10]) / time_bin, upper=np.array([30]) / time_bin, shape=n_ramps)
            t3_raw    = pm.Deterministic('t3_raw', t2_raw + dt2t3_raw)

            dt3t4_raw = pm.Uniform('delta_t3_t4_raw', lower=np.array([5]) / time_bin, upper=np.array([20]) / time_bin, shape=n_ramps)
            t4_raw    = pm.Deterministic('t4_raw', t3_raw + dt3t4_raw)

            L_raw = pm.Lognormal('L_raw', mu=np.log(np.array([6500]) * time_bin), sigma=[1.0], shape=n_ramps)

        # --- one permutation from t1, apply to all per-ramp vectors ---
        idx = pt.argsort(t1_raw)

        # expose the ordered parameters under the original names
        t1 = pm.Deterministic('t1', t1_raw[idx])
        t2_ = pm.Deterministic('t2', t2_raw[idx])
        t3_ = pm.Deterministic('t3', t3_raw[idx])
        t4_ = pm.Deterministic('t4', t4_raw[idx])
        L   = pm.Deterministic('L',  L_raw[idx])

        # (Optional) keep ordered deltas if you need to inspect them
        dt1t2 = pm.Deterministic('delta_t1_t2', dt1t2_raw[idx])
        dt2t3 = pm.Deterministic('delta_t2_t3', dt2t3_raw[idx])
        dt3t4 = pm.Deterministic('delta_t3_t4', dt3t4_raw[idx])

        # --------------------------------------------------------------------
        # User-defined scalar priors
        # --------------------------------------------------------------------
        # 1. Prior for tau_rate_bg
        mean_value = 0.00135e6 / 365
        sigma_value = (0.00165e6 / 365 - mean_value) / 2
        tau_rate_bg = pm.TruncatedNormal('tau_rate_bg', mu=(mean_value) * time_bin, sigma=(sigma_value*5) * time_bin, lower=1e-10)

        # 2. Prior for A*sigma.
        Asigma_0 = pm.Lognormal('Asigma_0', mu=np.log(6.5e3), sigma=1.0)

        # 3. Deterministically calculate ta.
        ta = pm.Deterministic('ta', Asigma_0 / tau_rate_bg)   

        # --------------------------------------------------------------------
        # 1.2) Forward Model (Symbolic Calculation)
        # --------------------------------------------------------------------
        # Compute the predicted log_R_r using the PyTensor-compatible function
        log_R_r_pred = compute_log_R_r_theano(t_data_np, ta, tau_rate_bg, L, t1, t2_, t3_, t4_, n_ramps)  # (n_time,), e.g., (100,)
        R_r_pred      = pt.exp(log_R_r_pred)
        R_r_pred_safe = pt.clip(R_r_pred, 1e-6, np.inf)   # μ must be >0

        # --------------------------------------------------------------------
        # 1.3) Likelihood Definition
        # --------------------------------------------------------------------
        if flag_noise_poisson:
            # Use Poisson distribution for noise
            R_r_pred_scaled = R_r_pred_safe
            R_r_obs_scaled  = R_r_observed_np 
            R_obs_scaled    = np.asarray(np.round(R_r_obs_scaled * r), dtype="int64") 
            pm.Poisson("obs", mu=R_r_pred_scaled * r, observed=R_obs_scaled) 
        else:
            # Use Gaussian distribution for noise
            R_r_pred_scaled = R_r_pred_safe
            R_r_obs_scaled  = R_r_observed_np  
            tsig = pm.HalfNormal("tsig", sigma=1.0)
            pm.Normal("obs", mu=R_r_pred_scaled, sigma=tsig, observed=R_r_obs_scaled)
                
       # --------------------------------------------------------------------
        # 1.4) User-defined sampling configuration
        # --------------------------------------------------------------------
        # Choose the sampler and the number of draws/tuning steps here.
        step = pm.Slice() 
        
        # Sample from the prior predictive distribution.
        prior = pm.sample_prior_predictive(1000, var_names=["t1", "t2", "t3", "t4", "delta_t1_t2", "delta_t2_t3", "delta_t3_t4", "L", "tau_rate_bg", "Asigma_0", "ta"], random_seed=42)

        # Run MCMC sampling.
        trace = pm.sample(10000, tune=10000, step=step, chains=n_chains, cores=n_chains, random_seed=42) 

    # --------------------------------------------------------------------
    # 3) Return Model and Sampling Results
    # --------------------------------------------------------------------
    return model, trace, prior

# --------------------------------------------------------------------
# 10) Main execution block: user-defined data and experiment settings
# --------------------------------------------------------------------
# This is the second main place users should edit for a new application.
# Set the input data file, time-bin width, background rate, number of ramps,
# and likelihood/noise options here.
if __name__ == "__main__":
    """
    Run the configured example inversion.

    This block loads the synthetic input time series, compiles the vectorized
    forward model, runs the PyMC inversion, and writes posterior summaries,
    figures, and text outputs.
    """
    print_startup_banner()

    # --- USER-DEFINED EXPERIMENT CONFIGURATION ---
    # Choose the likelihood/noise model for this example.
    flag_noise_poisson = 1  # 1 = Poisson noise model; 0 = Gaussian noise model

    # Main user-defined values for a new data set.
    n_ramps = 1   # Number of trapezoidal stress-rate ramps.
    n_chains = 4  # Number of MCMC chains.
    r = 0.02      # Background seismicity rate.
    time_bin = 1  # Time-bin width used to construct the observed R/r time series.

    # Output folder for the example run.
    ex_ID = "example"
    os.makedirs('example_output/' + str(ex_ID), exist_ok=True)

    start_time = time.time()
    
    # --- LOAD INPUT DATA ---
    # Expected format: two columns, [time, observed R/r].
    # The first column is the middle time of each time bin.
    # The second column is the observed R/r or event-rate quantity used by the inversion.
    t_R_r = np.loadtxt('example_input/example_input.txt') 
    t_data = t_R_r[:, 0]          
    R_r_observed = t_R_r[:, 1]    

    # --- PRECOMPILE FORWARD MODEL ---
    # Precompile the vectorized PyTensor forward model for posterior prediction.
    t_data_shared = pt.vector('t_data')
    t1_vec = pt.matrix('t1')
    t2_vec = pt.matrix('t2')
    t3_vec = pt.matrix('t3')
    t4_vec = pt.matrix('t4')
    L_vec = pt.matrix('L')
    tau_rate_bg_vec = pt.vector('tau_rate_bg')
    ta_vec = pt.vector('ta')

    log_R_r_vec_tt = compute_log_R_r_vectorized(
        t_data_shared,
        ta_vec,
        tau_rate_bg_vec,
        L_vec,
        t1_vec,
        t2_vec,
        t3_vec,
        t4_vec,
        n_ramps
    )

    global log_R_r_fn_vectorized
    log_R_r_fn_vectorized = function(
        [t_data_shared, ta_vec, tau_rate_bg_vec, L_vec, t1_vec, t2_vec, t3_vec, t4_vec],
        log_R_r_vec_tt
    )
    tr.log_R_r_fn_vectorized = log_R_r_fn_vectorized

    # --- RUN INVERSION ---
    model, trace, prior = run_inversion(
        t_data,
        R_r_observed,
        n_ramps,
        flag_noise_poisson,
        r,
        time_bin,
        n_chains
    )

    # --- POSTERIOR SUMMARY AND PLOTS ---
    print("\n" * 3)
    print("\n[1/6] Summarizing posterior samples...")
    selected_vars = ['t1', 't2', 't3', 't4', 'L', 'Asigma_0', 'tau_rate_bg']
    posterior_summary = summarize_last_fraction_posterior(
        trace=trace,
        selected_vars=selected_vars,
        ex_ID=ex_ID,
        fraction=0.10,
        hdi_prob=0.95,
        output_dir="example_output",
        save_to_file=True
    )
    posterior_summary_file = 'example_output/' + str(ex_ID) + '/posterior_summary.txt'
    print(f"[1/6] Posterior summary saved to {posterior_summary_file}")

    print("\n[2/6] Plotting prior predictive check...")
    plot_prior_predictive_check(prior, n_ramps, ex_ID)
    print("[2/6] Prior predictive check saved.")

    print("\n[3/6] Plotting trace plot...")
    plot_trace(trace, ex_ID)
    print("[3/6] Trace plot saved.")

    print("\n[4/6] Extracting and saving posterior samples...")
    parameters = extract_and_save_posterior_samples(trace, n_ramps, ex_ID)
    print(f"[4/6] Posterior samples extracted.")

    print("\n[5/6] Plotting joint posterior KDE distributions...")
    plot_joint_posterior_seaborn_kde(trace, n_ramps, ex_ID)
    print(f"[5/6] Joint posterior KDE plot saved.")

    print("\n[6/6] Plotting posterior predictions against observations...")
    plot_predicted_R_r_and_S_t(t_data, R_r_observed, ex_ID, n_ramps, parameters, n_chains)
    print("[6/6] Posterior prediction plots and output files saved.")

    # --- TIMING AND COMPLETION MESSAGE ---
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nElapsed time: {elapsed_time:.4f} seconds")

    message = "#### Example inversion is done. ####"
    border = "#" * len(message)
    print(border)
    print(message)
    print(border)