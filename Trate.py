#!/usr/bin/env python

"""
T-rate seismicity rate inversion example.

This script implements a PyMC/PyTensor Bayesian inversion for estimating a
trapezoidal stress-rate history from seismicity rate observations. 

Reference
---------
Jiang, Y., Trugman, D. T., & González, P. J. (2026)
Bayesian inference of complex stress evolution in rate-and-state governed faults
constrained by seismicity rate observations
Journal of Geophysical Research: Solid Earth

Model summary
-------------
The stress-rate history is represented by one or more trapezoidal ramps. For
each ramp, L is the peak stressing-rate amplitude, t1 is the onset time, and
delta_t1_t2, delta_t2_t3, and delta_t3_t4 define the three ramp durations.
The model evaluates the cumulative stress S(t), the
integral of exp(S / Asigma_0), and the predicted seismicity rate ratio R/r.

Parameterization
----------------
The current inversion samples Asigma_0 and tau_rate_bg directly. The relaxation
time ta is treated as a deterministic internal variable through

    ta = Asigma_0 / tau_rate_bg
"""

import pymc as pm
import numpy as np
import pytensor.tensor as pt
from pytensor import function
import arviz as az
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import seaborn as sns
import multiprocessing as mp
from tqdm.auto import tqdm

import os
os.environ["PYTENSOR_FLAGS"] = "exception_verbosity=high,optimizer=fast_run"
    
# --------------------------------------------------------------------
# 1) Core PyTensor mathematical utilities
# --------------------------------------------------------------------

def erfi(x):
    """
    Evaluate the imaginary error function erfi(x) in a PyTensor graph.
    
    PyTensor does not provide erfi directly, so the identity
    
        erfi(x) = -i erf(i x)
    
    is used and the real part is returned.
    """
    # PyTensor constant for the imaginary unit.
    i = pt.constant(1j)  
    
    # Return the real-valued erfi(x).
    return pt.real(-i * pt.erf(i * x))

# --------------------------------------------------------------------
# 2) Exact-integration helpers for one or more trapezoidal ramps
# --------------------------------------------------------------------
# These helpers evaluate the integral of exp(S / Asigma_0) exactly on
# each interval between ramp breakpoints. On such an interval, the total
# stress is locally quadratic in time.
# --------------------------------------------------------------------

def _safe_exp_theano(x):
    """
    Numerical helper for stable exponential evaluation.

    This function evaluates exp(x) after clipping x to avoid numerical
    overflow or underflow in PyTensor. It is used only for numerical
    stability and does not change the mathematical model.
    """
    return pt.exp(pt.clip(x, -700.0, 700.0))


def _S_single_broadcast_theano(t, L, t1, t2, t3, t4):
    """
    Cumulative stress S(t) for one trapezoidal stress-rate ramp.
    
    The implementation supports scalar or broadcastable PyTensor inputs, so it can
    be used both during model construction and during vectorized posterior
    post-processing.
    """
    zero = pt.zeros_like(t)

    val1 = (L / (2.0 * (t2 - t1))) * (t - t1) ** 2
    val2 = (L * (t2 - t1)) / 2.0 + L * (t - t2)
    val3 = (
        (L * (t2 - t1)) / 2.0
        + L * (t3 - t2)
        + L * (t - t3)
        - (L / (2.0 * (t4 - t3))) * (t - t3) ** 2
    )
    val4 = (L * (t2 - t1)) / 2.0 + L * (t3 - t2) + (L * (t4 - t3)) / 2.0

    return pt.switch(
        pt.lt(t, t1),
        zero,
        pt.switch(
            pt.lt(t, t2),
            val1,
            pt.switch(
                pt.lt(t, t3),
                val2,
                pt.switch(
                    pt.lt(t, t4),
                    val3,
                    val4
                )
            )
        )
    )


def _dS_dt_single_broadcast_theano(t, L, t1, t2, t3, t4):
    """
    Stress rate dS/dt for one trapezoidal ramp, with PyTensor broadcasting.
    """
    zero = pt.zeros_like(t)
    k1 = L / (t2 - t1)
    k2 = L / (t4 - t3)

    val1 = k1 * (t - t1)
    val2 = L
    val3 = L - k2 * (t - t3)

    return pt.switch(
        pt.lt(t, t1),
        zero,
        pt.switch(
            pt.lt(t, t2),
            val1,
            pt.switch(
                pt.lt(t, t3),
                val2,
                pt.switch(
                    pt.lt(t, t4),
                    val3,
                    zero
                )
            )
        )
    )


def _d2S_dt2_single_broadcast_theano(t, L, t1, t2, t3, t4):
    """
    Numerical helper for exact integration of exp(S / Asigma_0).

    This function returns the local curvature of S(t). It is used only
    to express S(t) as a local quadratic function within each integration
    interval. It is not an additional physical model variable and is not
    part of the user-facing manuscript parameterization.
    """
    zero = pt.zeros_like(t)
    k1 = L / (t2 - t1)
    k2 = L / (t4 - t3)

    val1 = pt.ones_like(t) * k1
    val2 = zero
    val3 = -pt.ones_like(t) * k2

    return pt.switch(
        pt.lt(t, t1),
        zero,
        pt.switch(
            pt.lt(t, t2),
            val1,
            pt.switch(
                pt.lt(t, t3),
                val2,
                pt.switch(
                    pt.lt(t, t4),
                    val3,
                    zero
                )
            )
        )
    )

def _log_erfc_stable_theano(q):
    """
    Numerical helper for stable evaluation of log(erfc(q)).

    This function is used in the negative-quadratic branch of the exact
    integral of exp(S / Asigma_0). For large positive q, direct evaluation
    of log(1 - erf(q)) can underflow, so an asymptotic expansion is used.
    The expansion coefficients are numerical constants only; they are not
    model parameters and do not need to be discussed in the manuscript.
    """
    tiny = np.finfo("float64").tiny

    # Direct evaluation is stable for moderate q.
    erfc_direct = 1.0 - pt.erf(q)
    log_direct = pt.log(pt.clip(erfc_direct, tiny, np.inf))

    # Asymptotic branch avoids underflow for large positive q.
    q_safe = pt.maximum(q, 1e-12)
    inv_q2 = 1.0 / (q_safe**2)

    corr = (
        1.0
        - 0.5 * inv_q2
        + 0.75 * inv_q2**2
        - 1.875 * inv_q2**3
        + 6.5625 * inv_q2**4
        - 29.53125 * inv_q2**5
    )

    corr = pt.clip(corr, tiny, np.inf)

    log_asymp = (
        -q_safe**2
        - pt.log(q_safe)
        - 0.5 * pt.log(np.pi)
        + pt.log(corr)
    )

    return pt.switch(
        pt.gt(q, 5.0),
        log_asymp,
        log_direct
    )


def _logdiffexp_theano(log_hi, log_lo):
    """
    Numerical helper for stable log-space subtraction.

    This function computes log(exp(log_hi) - exp(log_lo)) when the
    difference is mathematically non-negative but may be numerically small.
    It is used only to stabilize the exact integral calculation and is not
    part of the physical rate-and-state model.
    """
    diff = log_lo - log_hi

    # Roundoff can make log_lo slightly larger than log_hi; clip it back.
    # Equal logs correctly produce log(0) = -inf.
    diff = pt.minimum(diff, 0.0)

    return log_hi + pt.log1p(-pt.exp(diff))


def _integral_local_quadratic_theano(u, left, mid, S_mid, V_mid, A_mid, Asigma_0):
    """
    Exact local integral of exp(S / Asigma_0).

    Within each interval bounded by ramp transition times, the total stress
    can be written locally as

        S(t) = S_mid + V_mid (t - mid) + 0.5 A_mid (t - mid)^2.

    This local quadratic representation is used only to evaluate the
    integral analytically and stably. A_mid is the local curvature of S(t),
    not an additional physical parameter. The function handles positive-
    quadratic, negative-quadratic, linear, and constant local forms.
    """

    eps = pt.constant(1e-12)

    x = u - mid
    x0 = left - mid

    c0 = S_mid / Asigma_0
    c1 = V_mid / Asigma_0
    c2 = 0.5 * A_mid / Asigma_0

    # Keep inactive switch branches finite in the PyTensor graph.
    c2_pos = pt.maximum(c2, eps)          # for c2 > 0
    c2_neg = -pt.maximum(-c2, eps)        # for c2 < 0
    c1_safe = pt.switch(pt.gt(pt.abs(c1), eps), c1, pt.ones_like(c1))

    # ------------------------------------------------------------
    # Positive quadratic branch: c2 > 0
    # Integral of exp(c0 + c1*x + c2*x^2)
    # Uses erfi.
    # ------------------------------------------------------------
    quad_pos = (
        _safe_exp_theano(c0 - (c1 ** 2) / (4.0 * c2_pos))
        * (pt.sqrt(np.pi) / (2.0 * pt.sqrt(c2_pos)))
        * (
            erfi(pt.sqrt(c2_pos) * (x + c1 / (2.0 * c2_pos)))
            - erfi(pt.sqrt(c2_pos) * (x0 + c1 / (2.0 * c2_pos)))
        )
    )

    a_neg = -c2_neg  # positive

    z = pt.sqrt(a_neg) * (x - c1 / (2.0 * a_neg))
    z0 = pt.sqrt(a_neg) * (x0 - c1 / (2.0 * a_neg))

    q = -z
    q0 = -z0

    log_erfc_q = _log_erfc_stable_theano(q)
    log_erfc_q0 = _log_erfc_stable_theano(q0)

    # Since u >= left, mathematically z >= z0, so:
    # erf(z) - erf(z0) = erfc(-z) - erfc(-z0)
    #                   = erfc(q) - erfc(q0)
    log_erf_diff = _logdiffexp_theano(log_erfc_q, log_erfc_q0)

    log_prefactor = (
        c0
        + (c1 ** 2) / (4.0 * a_neg)
        + 0.5 * pt.log(np.pi)
        - pt.log(2.0)
        - 0.5 * pt.log(a_neg)
    )

    quad_neg_raw = _safe_exp_theano(log_prefactor + log_erf_diff)

    # If u == left, the integral must be exactly zero.
    quad_neg = pt.switch(
        pt.le(x, x0),
        pt.zeros_like(quad_neg_raw),
        quad_neg_raw
    )

    # ------------------------------------------------------------
    # Linear branch: c2 approximately 0, c1 != 0
    # ------------------------------------------------------------
    linear = (
        _safe_exp_theano(c0)
        * (_safe_exp_theano(c1 * x) - _safe_exp_theano(c1 * x0))
        / c1_safe
    )

    # ------------------------------------------------------------
    # Constant branch: c2 approximately 0, c1 approximately 0
    # ------------------------------------------------------------
    constant = _safe_exp_theano(c0) * (x - x0)

    return pt.switch(
        pt.gt(c2, eps),
        quad_pos,
        pt.switch(
            pt.lt(c2, -eps),
            quad_neg,
            pt.switch(
                pt.gt(pt.abs(c1), eps),
                linear,
                constant
            )
        )
    )


# --------------------------------------------------------------------
# 3) Single-ramp PyTensor stress-history functions
# --------------------------------------------------------------------

def S_of_t_single_theano(t, L, t1, t2, t3, t4):
    """
    PyTensor cumulative stress S(t) for a single trapezoidal stress-rate ramp.
    
    The four times t1, t2, t3, and t4 define the ramp shape:
    
        t < t1          : no stress change
        t1 <= t < t2   : stress rate increases linearly
        t2 <= t < t3   : stress rate is constant at L
        t3 <= t < t4   : stress rate decreases linearly
        t >= t4        : cumulative stress remains constant
    
    The returned S(t) is the time integral of that stress-rate history.
    """

    # Expand dimensions for broadcasting (if vectorized)
    t_exp = t if L.ndim == 0 else t[None, :]  # [n_time] or [1, n_time]
    L_exp = L if L.ndim == 0 else L[:, None]  # [scalar] or [n_samples, 1]
    t1_exp = t1 if t1.ndim == 0 else t1[:, None]  # [scalar] or [n_samples, 1]
    t2_exp = t2 if t2.ndim == 0 else t2[:, None]  # [scalar] or [n_samples, 1]
    t3_exp = t3 if t3.ndim == 0 else t3[:, None]  # [scalar] or [n_samples, 1]
    t4_exp = t4 if t4.ndim == 0 else t4[:, None]  # [scalar] or [n_samples, 1]

    # Define the piecewise expressions for each time interval
    val0 = pt.zeros_like(t_exp)  # Before t1 [n_time] or [n_samples, n_time]
    val1 = (L_exp / (2.0 * (t2_exp - t1_exp))) * (t_exp - t1_exp)**2  # Quadratic increase [n_time] or [n_samples, n_time]
    val2 = (L_exp * (t2_exp - t1_exp)) / 2.0 + L_exp * (t_exp - t2_exp)  # Linear increase [n_time] or [n_samples, n_time]
    val3 = (L_exp * (t2_exp - t1_exp)) / 2.0 + L_exp * (t3_exp - t2_exp) + L_exp * (t_exp - t3_exp) - (L_exp / (2.0 * (t4_exp - t3_exp))) * (t_exp - t3_exp)**2  # Quadratic decrease [n_time] or [n_samples, n_time]
    val4 = (L_exp * (t2_exp - t1_exp)) / 2.0 + L_exp * (t3_exp - t2_exp) + L_exp * (t4_exp - t3_exp) / 2.0  # Constant after t4 [n_time] or [n_samples, n_time]

    # Build nested switch conditions for the piecewise function
    S = pt.switch(
        pt.lt(t_exp, t1_exp),  # t < t1
        val0,                  # Stress is 0 before t1
        pt.switch(
            pt.lt(t_exp, t2_exp),  # t1 <= t < t2
            val1,                 # Quadratic increase
            pt.switch(
                pt.lt(t_exp, t3_exp),  # t2 <= t < t3
                val2,                 # Linear increase
                pt.switch(
                    pt.lt(t_exp, t4_exp),  # t3 <= t < t4
                    val3,                 # Quadratic decrease
                    val4                  # Constant after t4
                )
            )
        )
    )

    # Return the cumulative-stress time series.
    return S  # [n_time] (scalar case) or [n_samples, n_time] (vector case)

# --------------------------------------------------------------------
# 4) Generic multi-ramp PyTensor forward-model functions
# --------------------------------------------------------------------

def _S_total_scalar_theano(t, Larr, t1arr, t2arr, t3arr, t4arr, n_ramps):
    """
    Sum cumulative stress over all ramps for the symbolic model branch.
    
    Inputs are one-dimensional PyTensor vectors with length n_ramps. The output has
    one value for each time in t.
    """
    S_total = _S_single_broadcast_theano(
        t, Larr[0], t1arr[0], t2arr[0], t3arr[0], t4arr[0]
    )

    for i in range(1, n_ramps):
        S_total = S_total + _S_single_broadcast_theano(
            t, Larr[i], t1arr[i], t2arr[i], t3arr[i], t4arr[i]
        )

    return S_total


def _S_total_vector_theano(t, Larr, t1arr, t2arr, t3arr, t4arr, n_ramps):
    """
    Sum cumulative stress over all ramps for vectorized posterior samples.
    
    Inputs contain one row per posterior sample and one column per ramp. The output
    has shape (n_samples, n_time).
    """
    S_total = S_of_t_single_theano(
        t, Larr[:, 0], t1arr[:, 0], t2arr[:, 0], t3arr[:, 0], t4arr[:, 0]
    )

    for i in range(1, n_ramps):
        S_total = S_total + S_of_t_single_theano(
            t, Larr[:, i], t1arr[:, i], t2arr[:, i], t3arr[:, i], t4arr[:, i]
        )

    return S_total


def integral_expS_n_scalar_theano(
    t,
    Larr, t1arr, t2arr, t3arr, t4arr,
    tau_rate_bg, ta,
    n_ramps
):
    """
    Exact integral of exp(S_total / Asigma_0) for the symbolic model branch.
    
    The breakpoints from all ramps are sorted to form intervals. On each interval,
    S_total(t) is locally quadratic, so the integral can be evaluated analytically.
    
    Parameters
    ----------
    t : PyTensor vector
        Time grid.
    Larr, t1arr, t2arr, t3arr, t4arr : PyTensor vectors
        Ramp parameters with length n_ramps.
    tau_rate_bg, ta : PyTensor scalars
        Background stressing rate and deterministic relaxation time.
    n_ramps : int
        Number of ramps.
    
    Returns
    -------
    PyTensor vector
        Integral I(t) on the input time grid.
    """
    Asigma_0 = tau_rate_bg * ta

    # Sorted union of all 4*n_ramps breakpoints
    bounds = pt.sort(
        pt.concatenate([t1arr, t2arr, t3arr, t4arr], axis=0)
    )  # shape: (4*n_ramps,)

    I = pt.zeros_like(t)
    cum = pt.constant(0.0)

    # There are (4*n_ramps - 1) finite intervals
    for k in range(4 * n_ramps - 1):
        left = bounds[k]
        right = bounds[k + 1]
        mid = 0.5 * (left + right)

        S_mid = _S_single_broadcast_theano(
            mid, Larr[0], t1arr[0], t2arr[0], t3arr[0], t4arr[0]
        )
        V_mid = _dS_dt_single_broadcast_theano(
            mid, Larr[0], t1arr[0], t2arr[0], t3arr[0], t4arr[0]
        )
        A_mid = _d2S_dt2_single_broadcast_theano(
            mid, Larr[0], t1arr[0], t2arr[0], t3arr[0], t4arr[0]
        )

        for i in range(1, n_ramps):
            S_mid = S_mid + _S_single_broadcast_theano(
                mid, Larr[i], t1arr[i], t2arr[i], t3arr[i], t4arr[i]
            )
            V_mid = V_mid + _dS_dt_single_broadcast_theano(
                mid, Larr[i], t1arr[i], t2arr[i], t3arr[i], t4arr[i]
            )
            A_mid = A_mid + _d2S_dt2_single_broadcast_theano(
                mid, Larr[i], t1arr[i], t2arr[i], t3arr[i], t4arr[i]
            )

        I_seg_t = _integral_local_quadratic_theano(
            t, left, mid, S_mid, V_mid, A_mid, Asigma_0
        )
        I_seg_right = _integral_local_quadratic_theano(
            right, left, mid, S_mid, V_mid, A_mid, Asigma_0
        )

        mask = pt.bitwise_and(pt.ge(t, left), pt.lt(t, right))
        I = pt.switch(mask, cum + I_seg_t, I)

        cum = cum + I_seg_right

    # After the final breakpoint, total stress is constant
    last = bounds[-1]
    S_last = _S_single_broadcast_theano(
        last, Larr[0], t1arr[0], t2arr[0], t3arr[0], t4arr[0]
    )
    for i in range(1, n_ramps):
        S_last = S_last + _S_single_broadcast_theano(
            last, Larr[i], t1arr[i], t2arr[i], t3arr[i], t4arr[i]
        )

    I = pt.switch(
        pt.ge(t, last),
        cum + _safe_exp_theano(S_last / Asigma_0) * (t - last),
        I
    )

    return pt.switch(pt.isnan(I) | pt.isinf(I), pt.constant(np.nan), I)


def integral_expS_n_vector_theano(
    t,
    Larr, t1arr, t2arr, t3arr, t4arr,
    tau_rate_bg, ta,
    n_ramps
):
    """
    Exact integral of exp(S_total / Asigma_0) for vectorized posterior samples.
    
    This is the posterior-sample version of integral_expS_n_scalar_theano. Each row
    corresponds to one sampled parameter set.
    """
    t_exp = t  
    Asigma_0 = (tau_rate_bg * ta)[:, None] 

    # Sorted union of all 4*n_ramps breakpoints for each sample
    bounds = pt.sort(
        pt.concatenate([t1arr, t2arr, t3arr, t4arr], axis=1),
        axis=1
    )  

    I = pt.zeros((Larr.shape[0], t.shape[1]))
    cum = pt.zeros((Larr.shape[0], 1))

    for k in range(4 * n_ramps - 1):
        left = bounds[:, k][:, None]       
        right = bounds[:, k + 1][:, None]  
        mid = 0.5 * (left + right)

        S_mid = _S_single_broadcast_theano(
            mid,
            Larr[:, 0][:, None],
            t1arr[:, 0][:, None],
            t2arr[:, 0][:, None],
            t3arr[:, 0][:, None],
            t4arr[:, 0][:, None],
        )
        V_mid = _dS_dt_single_broadcast_theano(
            mid,
            Larr[:, 0][:, None],
            t1arr[:, 0][:, None],
            t2arr[:, 0][:, None],
            t3arr[:, 0][:, None],
            t4arr[:, 0][:, None],
        )
        A_mid = _d2S_dt2_single_broadcast_theano(
            mid,
            Larr[:, 0][:, None],
            t1arr[:, 0][:, None],
            t2arr[:, 0][:, None],
            t3arr[:, 0][:, None],
            t4arr[:, 0][:, None],
        )

        for i in range(1, n_ramps):
            S_mid = S_mid + _S_single_broadcast_theano(
                mid,
                Larr[:, i][:, None],
                t1arr[:, i][:, None],
                t2arr[:, i][:, None],
                t3arr[:, i][:, None],
                t4arr[:, i][:, None],
            )
            V_mid = V_mid + _dS_dt_single_broadcast_theano(
                mid,
                Larr[:, i][:, None],
                t1arr[:, i][:, None],
                t2arr[:, i][:, None],
                t3arr[:, i][:, None],
                t4arr[:, i][:, None],
            )
            A_mid = A_mid + _d2S_dt2_single_broadcast_theano(
                mid,
                Larr[:, i][:, None],
                t1arr[:, i][:, None],
                t2arr[:, i][:, None],
                t3arr[:, i][:, None],
                t4arr[:, i][:, None],
            )

        I_seg_t = _integral_local_quadratic_theano(
            t_exp, left, mid, S_mid, V_mid, A_mid, Asigma_0
        ) 

        I_seg_right = _integral_local_quadratic_theano(
            right, left, mid, S_mid, V_mid, A_mid, Asigma_0
        )  

        mask = pt.bitwise_and(pt.ge(t_exp, left), pt.lt(t_exp, right))
        I = pt.switch(mask, cum + I_seg_t, I)

        cum = cum + I_seg_right

    # After the final breakpoint, total stress is constant
    last = bounds[:, -1][:, None]  

    S_last = _S_single_broadcast_theano(
        last,
        Larr[:, 0][:, None],
        t1arr[:, 0][:, None],
        t2arr[:, 0][:, None],
        t3arr[:, 0][:, None],
        t4arr[:, 0][:, None],
    )
    for i in range(1, n_ramps):
        S_last = S_last + _S_single_broadcast_theano(
            last,
            Larr[:, i][:, None],
            t1arr[:, i][:, None],
            t2arr[:, i][:, None],
            t3arr[:, i][:, None],
            t4arr[:, i][:, None],
        )

    I = pt.switch(
        pt.ge(t_exp, last),
        cum + _safe_exp_theano(S_last / Asigma_0) * (t_exp - last),
        I
    )

    return pt.switch(pt.isnan(I) | pt.isinf(I), pt.constant(np.nan), I)


def compute_log_R_r_theano(t_data, ta, tau_rate_bg, Larr, t1arr, t2arr, t3arr, t4arr, n_ramps):
    """
    Compute the predicted log seismicity rate ratio, log(R/r).
    
    The calculation follows the manuscript formulation using the total cumulative
    stress S_total(t) and the raw integral
    
        I(t) = ∫ exp(S_total(t') / Asigma_0) dt'.
    
    The returned value is
    
        log(R/r) = S_total / Asigma_0 + log(ta) - log(I + ta),
    
    with log(R/r) set to zero before the earliest ramp start time.
    
    The same function supports both the symbolic PyMC model branch and the
    vectorized posterior-prediction branch.
    """
    is_symbolic = Larr.ndim == 1

    if is_symbolic:
        S_total = _S_total_scalar_theano(
            t_data, Larr, t1arr, t2arr, t3arr, t4arr, n_ramps
        )

        I_total = integral_expS_n_scalar_theano(
            t_data,
            Larr, t1arr, t2arr, t3arr, t4arr,
            tau_rate_bg, ta,
            n_ramps
        )

        Asigma_0 = tau_rate_bg * ta
        S_over_Asigma0 = pt.clip(S_total / Asigma_0, -700, 700)

        # Manuscript formula using the raw integral I(t):
        # log(R/r) = S/Asigma_0 + log(ta) - log(I + ta)
        log_R_r = S_over_Asigma0 + pt.log(ta) - pt.log(I_total + ta)

        t1_min = pt.min(t1arr)
        outside_mask = pt.le(t_data, t1_min)

        return pt.set_subtensor(log_R_r[outside_mask], 0.0)

    else:
        S_total = _S_total_vector_theano(
            t_data, Larr, t1arr, t2arr, t3arr, t4arr, n_ramps
        )

        I_total = integral_expS_n_vector_theano(
            t_data[None, :],
            Larr, t1arr, t2arr, t3arr, t4arr,
            tau_rate_bg, ta,
            n_ramps
        )

        Asigma_0 = (tau_rate_bg * ta)[:, None]
        ta_exp = ta[:, None]
        S_over_Asigma0 = pt.clip(S_total / Asigma_0, -700, 700)

        # Manuscript formula using the raw integral I(t):
        # log(R/r) = S/Asigma_0 + log(ta) - log(I + ta)
        log_R_r = S_over_Asigma0 + pt.log(ta_exp) - pt.log(I_total + ta_exp)

        t1_min = pt.min(t1arr, axis=1)[:, None]
        outside_mask = pt.le(t_data, t1_min)

        return pt.set_subtensor(log_R_r[outside_mask], 0.0)


def compute_log_R_r_vectorized(t_data, ta, tau_rate_bg, Larr, t1arr, t2arr, t3arr, t4arr, n_ramps):
    """
    Convenience wrapper for vectorized posterior prediction of log(R/r).
    """
    return compute_log_R_r_theano(
        t_data, ta, tau_rate_bg, Larr, t1arr, t2arr, t3arr, t4arr, n_ramps
    )


# --------------------------------------------------------------------
# 5) Posterior extraction and visualization
# --------------------------------------------------------------------

def _add_delta_time_variables_to_posterior_dataset(dataset):
    """
    Add duration variables derived from t1, t2, t3, and t4.

    This keeps plots and text summaries in the user-facing parameterization:

        t1, delta_t1_t2, delta_t2_t3, delta_t3_t4, L, Asigma_0, tau_rate_bg

    while the internal forward model can still use t1, t2, t3, and t4.
    """
    if all(var in dataset for var in ["t1", "t2", "t3", "t4"]):
        dataset["delta_t1_t2"] = dataset["t2"] - dataset["t1"]
        dataset["delta_t2_t3"] = dataset["t3"] - dataset["t2"]
        dataset["delta_t3_t4"] = dataset["t4"] - dataset["t3"]
    return dataset

def extract_and_save_posterior_samples(trace, n_ramps, ex_ID):
    """
    Extract posterior samples and arrange them in the saved parameter format.
    """
    
    # Flatten chains and draws into one posterior-sample dimension.
    posterior_samples = {
        't1': trace.posterior['t1'].values.reshape(-1, n_ramps),
        't2': trace.posterior['t2'].values.reshape(-1, n_ramps),
        't3': trace.posterior['t3'].values.reshape(-1, n_ramps),
        't4': trace.posterior['t4'].values.reshape(-1, n_ramps),
        'L': trace.posterior['L'].values.reshape(-1, n_ramps),
        'tau_rate_bg': trace.posterior['tau_rate_bg'].values.flatten()
    }

    # Save duration parameters instead of the absolute transition times t2--t4.
    posterior_samples['delta_t1_t2'] = posterior_samples['t2'] - posterior_samples['t1']
    posterior_samples['delta_t2_t3'] = posterior_samples['t3'] - posterior_samples['t2']
    posterior_samples['delta_t3_t4'] = posterior_samples['t4'] - posterior_samples['t3']
    posterior_samples['Asigma_0'] = trace.posterior['Asigma_0'].values.flatten()

    # Arrange ramp and scalar parameters in the saved output format.
    # Column order:
    #   [t1..., delta_t1_t2..., delta_t2_t3..., delta_t3_t4..., L..., Asigma_0, tau_rate_bg]
    ramp_params = [posterior_samples[key] for key in ['t1', 'delta_t1_t2', 'delta_t2_t3', 'delta_t3_t4', 'L']]
    ramp_columns = np.column_stack([ramp[:, i] for ramp in ramp_params for i in range(n_ramps)])
    scalar_columns = np.column_stack([posterior_samples['Asigma_0'], posterior_samples['tau_rate_bg']])
    posterior_array = np.hstack([ramp_columns, scalar_columns]) 

    return posterior_array


def plot_prior_predictive_check(prior, n_ramps, ex_ID):
    """
    Plot log-transformed prior distributions for quick prior checking.
    """

    # --------------------------------------------------------------------
    # 1) Log Transformation of Prior Variables
    # --------------------------------------------------------------------
    # Copy the InferenceData object to avoid modifying the original data
    prior_log = prior.copy() 
    prior_log.prior = _add_delta_time_variables_to_posterior_dataset(prior_log.prior)
    
    # Log-transform positive parameters for easier visual inspection.
    prior_log.prior["log_L"] = np.log(prior_log.prior["L"]) 
    prior_log.prior["log_tau_rate_bg"] = np.log(prior_log.prior["tau_rate_bg"])
    if "Asigma_0" in prior_log.prior:
        prior_log.prior["log_Asigma_0"] = np.log(prior_log.prior["Asigma_0"])
    else:
        prior_log.prior["log_Asigma_0"] = np.log(prior_log.prior["ta"] * prior_log.prior["tau_rate_bg"])

    # --------------------------------------------------------------------
    # 2) Create Density Plot for Prior Predictive Check
    # -------------------------------------------------------------------- 

    plt.figure(figsize=(12, 8))
    
    # Plot density of the log-transformed priors
    az.plot_density(
        prior_log, 
        group="prior", 
        var_names=["t1", "delta_t1_t2", "delta_t2_t3", "delta_t3_t4", "log_L", "log_Asigma_0", "log_tau_rate_bg"], 
        shade=0.2 
    )
    
    # Set the super title for the figure
    plt.suptitle('Prior Predictive Check', fontsize=33, y=0.98)

    # --------------------------------------------------------------------
    # 3) Customize Subplot Titles and Axis Labels
    # --------------------------------------------------------------------
    # Get the list of matplotlib axes (e.g., 7 axes for 7 variables)
    axes = plt.gcf().get_axes()  # [list of axes]
    for i, ax in enumerate(axes):
        title = ax.get_title()  # [scalar] (string, e.g., "t1")
        
        if title:  # Only process if the title exists
            # For ramp-related variables, append the ramp index
            if title in ["t1", "delta_t1_t2", "delta_t2_t3", "delta_t3_t4", "log_L"]:
                ramp_index = i % n_ramps 
                ax.set_title(f"{title}_{ramp_index}", fontsize=25) 
            else:
                ax.set_title(title, fontsize=25)
        
        ax.tick_params(axis='x', labelsize=25)
        ax.tick_params(axis='y', labelsize=25)

    # --------------------------------------------------------------------
    # 4) Adjust Layout and Save Figure
    # --------------------------------------------------------------------
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.savefig('example_output/' + str(ex_ID) + '/prior_predictive_check_log.png', dpi=600, bbox_inches='tight')
    
    plt.close()

def summarize_last_fraction_posterior(
    trace,
    selected_vars,
    ex_ID=None,
    fraction=0.10,
    hdi_prob=0.95,
    output_dir="example_output",
    save_to_file=True
):
    """
    Summarize the last fraction of posterior draws from each chain.

    Parameters
    ----------
    trace : arviz.InferenceData
        PyMC posterior trace.
    selected_vars : list of str
        Variables to summarize. If this list contains t2, t3, or t4, they
        are replaced in the output by delta_t1_t2, delta_t2_t3, and
        delta_t3_t4.
    ex_ID : str or int, optional
        Experiment ID used for the output folder.
    fraction : float
        Fraction of posterior draws to use from the end of each chain.
        Default is 0.10, meaning the last 10%.
    hdi_prob : float
        HDI probability. Default is 0.95.
    output_dir : str
        Base output directory.
    save_to_file : bool
        If True, save the summary to posterior_summary.txt.

    Returns
    -------
    posterior_summary : pandas.DataFrame
        Posterior summary based on the last fraction of posterior draws.
        The first column is posterior median instead of mean.
    """

    if not (0 < fraction <= 1):
        raise ValueError("fraction must be between 0 and 1.")

    # Select the last fraction of posterior draws from each chain.
    n_draws = trace.posterior.sizes["draw"]
    start_draw = int((1.0 - fraction) * n_draws)

    trace_subset = trace.copy()
    trace_subset.posterior = trace.posterior.isel(draw=slice(start_draw, None)).copy()
    trace_subset.posterior = _add_delta_time_variables_to_posterior_dataset(trace_subset.posterior)

    # Replace absolute transition-time variables with duration variables in the summary.
    replacement_map = {
        "t2": "delta_t1_t2",
        "t3": "delta_t2_t3",
        "t4": "delta_t3_t4"
    }
    summary_vars = []
    for var in selected_vars:
        mapped_var = replacement_map.get(var, var)
        if mapped_var not in summary_vars:
            summary_vars.append(mapped_var)

    # Default ArviZ summary includes mean, sd, HDI, ESS, r_hat, etc.
    posterior_summary = az.summary(
        trace_subset,
        var_names=summary_vars,
        hdi_prob=hdi_prob
    )

    # Compute posterior median and use it as the first column.
    posterior_summary_median = az.summary(
        trace_subset,
        var_names=summary_vars,
        hdi_prob=hdi_prob,
        stat_focus="median"
    )

    posterior_summary.insert(0, "median", posterior_summary_median["median"])

    # Remove mean so the first main estimate is the median.
    if "mean" in posterior_summary.columns:
        posterior_summary = posterior_summary.drop(columns=["mean"])

    print(posterior_summary)

    if save_to_file:
        if ex_ID is None:
            raise ValueError("ex_ID must be provided when save_to_file=True.")

        output_file = f"{output_dir}/{ex_ID}/posterior_summary.txt"

        with open(output_file, "w") as f:
            f.write(posterior_summary.to_string())

    return posterior_summary


def plot_joint_posterior_seaborn_kde(trace, n_ramps, ex_ID):
    """
    Create a pairwise KDE plot for the posterior parameter distribution.

    The plot includes t1, the three ramp-duration parameters, log(L),
    log(Asigma_0), and log(tau_rate_bg).

    To reduce computation time, only the last 10% of posterior draws from
    each chain are used for the joint posterior KDE plot.
    """
    # --------------------------------------------------------------------
    # 1) Convert Trace to DataFrame
    # --------------------------------------------------------------------
    tau_rate_bg_values = trace.posterior['tau_rate_bg'].values.flatten()
    Asigma_0_values = trace.posterior['Asigma_0'].values.flatten()

    trace_df = pd.DataFrame({
        "log_Asigma_0": np.log(Asigma_0_values),
        "log_tau_rate_bg": np.log(tau_rate_bg_values)
    })

    # --------------------------------------------------------------------
    # 2) Add Individual Ramp Parameters
    # --------------------------------------------------------------------
    for i in range(n_ramps):
        t1_values = trace.posterior['t1'].values[:, :, i].flatten()
        t2_values = trace.posterior['t2'].values[:, :, i].flatten()
        t3_values = trace.posterior['t3'].values[:, :, i].flatten()
        t4_values = trace.posterior['t4'].values[:, :, i].flatten()

        trace_df[f"log_L_{i}"] = np.log(trace.posterior['L'].values[:, :, i].flatten())
        trace_df[f"t1_{i}"] = t1_values
        trace_df[f"delta_t1_t2_{i}"] = t2_values - t1_values
        trace_df[f"delta_t2_t3_{i}"] = t3_values - t2_values
        trace_df[f"delta_t3_t4_{i}"] = t4_values - t3_values

    # --------------------------------------------------------------------
    # 3) Keep Only the Bottom 10% of Draws From Each Chain
    # --------------------------------------------------------------------
    n_chains = trace.posterior.sizes["chain"]
    n_draws = trace.posterior.sizes["draw"]

    parameters_shape = len(trace_df)
    chain_length = parameters_shape // n_chains

    mask = np.zeros(parameters_shape, dtype=bool)

    for i in range(n_chains):
        start_idx = i * chain_length + int(0.9 * chain_length)
        end_idx = (i + 1) * chain_length
        mask[start_idx:end_idx] = True

    trace_df = trace_df[mask].reset_index(drop=True)

    # --------------------------------------------------------------------
    # 4) Define Variables for PairGrid Plot
    # --------------------------------------------------------------------
    base_vars = ["log_Asigma_0", "log_tau_rate_bg"]

    ramp_vars = [f"t1_{i}" for i in range(n_ramps)] + \
                [f"delta_t1_t2_{i}" for i in range(n_ramps)] + \
                [f"delta_t2_t3_{i}" for i in range(n_ramps)] + \
                [f"delta_t3_t4_{i}" for i in range(n_ramps)] + \
                [f"log_L_{i}" for i in range(n_ramps)]

    selected_vars = ramp_vars + base_vars
    selected_df = trace_df[selected_vars]

    # --------------------------------------------------------------------
    # 5) Initialize Seaborn PairGrid
    # --------------------------------------------------------------------
    g = sns.PairGrid(selected_df, diag_sharey=False)

    # Map KDE plots for lower triangle and diagonal
    g.map_lower(sns.kdeplot, cmap="Blues", fill=True)
    g.map_diag(sns.kdeplot, fill=True)

    # --------------------------------------------------------------------
    # 6) Set Axis Limits for Each Subplot
    # --------------------------------------------------------------------
    for i, row_vars in enumerate(selected_vars):
        for j, col_vars in enumerate(selected_vars):
            ax = g.axes[i, j]

            # Hide upper triangle plots
            if j > i:
                if ax is not None:
                    ax.set_visible(False)
                continue

            if ax is not None:
                if i != j:
                    x_vals = selected_df[col_vars]
                    y_vals = selected_df[row_vars]
                    xmin, xmax = x_vals.min(), x_vals.max()
                    ymin, ymax = y_vals.min(), y_vals.max()
                else:
                    x_vals = selected_df[row_vars]
                    xmin, xmax = x_vals.min(), x_vals.max()
                    ymin, ymax = 0, None

                ax.set_xlim([xmin, xmax])
                if ymin is not None and ymax is not None:
                    ax.set_ylim([ymin, ymax])

    # --------------------------------------------------------------------
    # 7) Customize Font Sizes and Save Plot
    # --------------------------------------------------------------------
    for ax_row in g.axes:
        for ax in ax_row:
            if ax is not None:
                ax.xaxis.label.set_size(20)
                ax.yaxis.label.set_size(20)
                ax.tick_params(axis='x', labelsize=15)
                ax.tick_params(axis='y', labelsize=15)

    g.fig.subplots_adjust(left=0.2, bottom=0.2)
    plt.suptitle('Joint Posterior Probability (KDE)', y=1.02, fontsize=25)

    plt.savefig(
        'example_output/' + str(ex_ID) + '/joint_posterior_probability_kde.png',
        dpi=600,
        bbox_inches='tight'
    )

    plt.close()



def plot_trace(trace, ex_ID):
    """
    Save ArviZ trace plots for the main sampled parameters.
    """
    
    # Add derived duration variables for plotting without modifying the original trace.
    trace_for_plot = trace.copy()
    trace_for_plot.posterior = trace.posterior.copy()
    trace_for_plot.posterior = _add_delta_time_variables_to_posterior_dataset(trace_for_plot.posterior)

    # Plot the main sampled/derived parameters.
    trace_var_names = ["t1", "delta_t1_t2", "delta_t2_t3", "delta_t3_t4", "L"]
    if "Asigma_0" in trace_for_plot.posterior:
        trace_var_names.append("Asigma_0")
    elif "ta" in trace_for_plot.posterior:
        trace_var_names.append("ta")
    trace_var_names.append("tau_rate_bg")

    az.plot_trace(trace_for_plot,
                  var_names=trace_var_names,
                  figsize=(8, 10),
                  compact=True)

    # Get the list of axes from the current figure
    axes = plt.gcf().get_axes() 

    # Customize y-axis labels and titles
    for i, ax in enumerate(axes):
        if i % 2 == 1:  # Odd indices correspond to density plots
            ax.set_ylabel('')  
        else:  # Even indices correspond to trace plots
            ax.set_ylabel(ax.get_title()) 

        ax.set_title('') 
        ax.tick_params(axis='both', labelsize=10) 
        ax.xaxis.label.set_size(12)  
        ax.yaxis.label.set_size(12)  

    # Adjust layout and save the figure 
    plt.subplots_adjust(hspace=0.5) 
    plt.suptitle('Trace', fontsize=12, y=0.92) 
    plt.savefig('example_output/' + str(ex_ID) + '/trace_plot.png', dpi=600, bbox_inches='tight') 
    plt.close() 

def plot_predicted_R_r_and_S_t(t_data, R_r_observed, ex_ID, n_ramps, parameters, n_chains):
    """
    Plot posterior predictions for R/r, stress S(t), and stress rate.
    
    The function compares predicted and observed R/r and saves prediction arrays and
    summary curves for later inspection.
    """
    
    parameters_shape = parameters.shape[0]
    chain_length = parameters_shape // n_chains
    mask = np.zeros(parameters_shape, dtype=int)

    # Set mask to 1 for the bottom 10% of each chain
    for i in range(n_chains):
        start_idx = i * chain_length + int(0.9 * chain_length)  
        end_idx = (i + 1) * chain_length  
        mask[start_idx:end_idx] = 1
    mask = mask == 1
    parameters_p5_p95 = parameters[mask]

    # --- EXTRACT PARAMETERS (Full Set) ---
    t1 = parameters[:, 0:n_ramps]                           # [n_samples, n_ramps]
    delta_t1_t2 = parameters[:, n_ramps:2*n_ramps]          # [n_samples, n_ramps]
    delta_t2_t3 = parameters[:, 2*n_ramps:3*n_ramps]        # [n_samples, n_ramps]
    delta_t3_t4 = parameters[:, 3*n_ramps:4*n_ramps]        # [n_samples, n_ramps]
    L  = parameters[:, 4*n_ramps:5*n_ramps]                 # [n_samples, n_ramps]
    t2 = t1 + delta_t1_t2                                   # [n_samples, n_ramps]
    t3 = t2 + delta_t2_t3                                   # [n_samples, n_ramps]
    t4 = t3 + delta_t3_t4                                   # [n_samples, n_ramps]
    Asigma_0 = parameters[:, 5*n_ramps]                     # [n_samples]
    tau_rate_bg = parameters[:, 5*n_ramps+1]                # [n_samples]
    ta = Asigma_0 / tau_rate_bg                             # [n_samples]

    # --- EVALUATE PREDICTED R/r ---
    log_R_r_all = log_R_r_fn_vectorized(
        t_data, ta, tau_rate_bg, L, t1, t2, t3, t4
    )  # [n_samples, n_times]
    R_r_results = np.exp(log_R_r_all[mask])  # [n_masked_samples, n_times]

    # --- VECTORIZE dS and S CALCULATIONS ---
    n_masked = len(parameters_p5_p95)          # [scalar]
    n_times  = len(t_data)                     # [scalar]

    # Expand t_data and parameters to match masked samples
    t_data_exp = np.tile(t_data, (n_masked, 1))  # [n_masked_samples, n_times]
    params_exp = parameters_p5_p95[:, None, :]   # [n_masked_samples, 1, 5*n_ramps + 2]

    # Initialise dS and S arrays
    dS = np.zeros((n_masked, n_times))  # [n_masked_samples, n_times]
    S  = np.zeros((n_masked, n_times))  # [n_masked_samples, n_times]

    for i in range(n_ramps):
        # shape => (n_masked_samples, 1)
        t1_exp_i = params_exp[:, :, i]
        delta_t1_t2_exp_i = params_exp[:, :, n_ramps + i]
        delta_t2_t3_exp_i = params_exp[:, :, 2*n_ramps + i]
        delta_t3_t4_exp_i = params_exp[:, :, 3*n_ramps + i]
        L_exp_i  = params_exp[:, :, 4*n_ramps + i]
        t2_exp_i = t1_exp_i + delta_t1_t2_exp_i
        t3_exp_i = t2_exp_i + delta_t2_t3_exp_i
        t4_exp_i = t3_exp_i + delta_t3_t4_exp_i

        # Build masks => shape (n_masked_samples, n_times)
        mask1_i = (t_data_exp >= t1_exp_i) & (t_data_exp < t2_exp_i)
        mask2_i = (t_data_exp >= t2_exp_i) & (t_data_exp < t3_exp_i)
        mask3_i = (t_data_exp >= t3_exp_i) & (t_data_exp < t4_exp_i)
        mask4_i = (t_data_exp >= t4_exp_i)

        # shape => (n_masked_samples, 1)
        k1_i = L_exp_i / (t2_exp_i - t1_exp_i)
        k2_i = L_exp_i / (t4_exp_i - t3_exp_i)

        # Expand to (n_masked_samples, n_times)
        k1_i_expanded  = np.repeat(k1_i,  n_times, axis=1)
        k2_i_expanded  = np.repeat(k2_i,  n_times, axis=1)
        t1_i_expanded  = np.repeat(t1_exp_i, n_times, axis=1)
        t2_i_expanded  = np.repeat(t2_exp_i, n_times, axis=1)
        t3_i_expanded  = np.repeat(t3_exp_i, n_times, axis=1)
        t4_i_expanded  = np.repeat(t4_exp_i, n_times, axis=1)
        L_i_expanded   = np.repeat(L_exp_i,  n_times, axis=1)

        # --- dS increments ---
        dS[mask1_i] += (k1_i_expanded[mask1_i]
                        * (t_data_exp[mask1_i] - t1_i_expanded[mask1_i]))
        dS[mask2_i] += L_i_expanded[mask2_i]
        dS[mask3_i] += (L_i_expanded[mask3_i]
                        - k2_i_expanded[mask3_i]
                          * (t_data_exp[mask3_i] - t3_i_expanded[mask3_i]))

        # --- S increments (area under triangular/trapezoidal ramp) ---
        # First compute the total partial areas in shape (n_masked_samples, 1).
        part_t2_i = (L_exp_i * (t2_exp_i - t1_exp_i)) / 2.0
        part_t3_i = part_t2_i + L_exp_i * (t3_exp_i - t2_exp_i)
        part_t4_i = part_t3_i + (L_exp_i * (t4_exp_i - t3_exp_i)) / 2.0

        # Expand them to (n_masked_samples, n_times) so we can index with mask.
        part_t2_i_expanded = np.repeat(part_t2_i, n_times, axis=1)
        part_t3_i_expanded = np.repeat(part_t3_i, n_times, axis=1)
        part_t4_i_expanded = np.repeat(part_t4_i, n_times, axis=1)

        # For time between t1_i and t2_i
        S[mask1_i] += (
            (L_i_expanded[mask1_i] /
             (2.0 * (t2_i_expanded[mask1_i] - t1_i_expanded[mask1_i])))
            * (t_data_exp[mask1_i] - t1_i_expanded[mask1_i])**2
        )
        # For time between t2_i and t3_i
        S[mask2_i] += (
            part_t2_i_expanded[mask2_i]
            + L_i_expanded[mask2_i]
              * (t_data_exp[mask2_i] - t2_i_expanded[mask2_i])
        )
        # For time between t3_i and t4_i
        S[mask3_i] += (
            part_t3_i_expanded[mask3_i]
            + L_i_expanded[mask3_i]
              * (t_data_exp[mask3_i] - t3_i_expanded[mask3_i])
            - (L_i_expanded[mask3_i]
               / (2.0 * (t4_i_expanded[mask3_i] - t3_i_expanded[mask3_i])))
              * (t_data_exp[mask3_i] - t3_i_expanded[mask3_i])**2
        )
        # For time >= t4_i
        S[mask4_i] += part_t4_i_expanded[mask4_i]

    # Convert to MPa
    dS /= 1e6  # [n_masked_samples, n_times]
    S  /= 1e6  # [n_masked_samples, n_times]

    # R/r best fit + envelopes
    R_r_optimal = R_r_results[0]  # [n_times]
    R_r_min, R_r_max = np.min(R_r_results, axis=0), np.max(R_r_results, axis=0)  # [n_times]
    R_r_median = np.median(R_r_results, axis=0)  # [n_times]

    # S best fit + envelopes
    S_optimal = S[0]  # [n_times]
    S_min, S_max = np.min(S, axis=0), np.max(S, axis=0)  # [n_times]
    S_median = np.median(S, axis=0)  # [n_times]

    # dS best fit + envelopes
    dS_optimal = dS[0]  # [n_times]
    dS_min, dS_max = np.min(dS, axis=0), np.max(dS, axis=0)  # [n_times]
    dS_median = np.median(dS, axis=0)  # [n_times]

    # --- PLOT R/r ---
    plt.figure(figsize=(12, 6))
    plt.fill_between(t_data, R_r_min, R_r_max, alpha=0.3, color='blue', label='R/r') 
    plt.plot(t_data, R_r_median, 'b-', label='Median R/r') 
    plt.plot(t_data, R_r_observed, 'ro', markersize=5, label='Observed R/r')
    plt.xlabel('Time (days)', fontsize=14)
    plt.ylabel('R/r', fontsize=14)
    plt.title('Modelled R/r vs. Time', fontsize=16)
    plt.legend(loc='upper left', fontsize=16)
    plt.grid(True)
    plt.savefig(f'example_output/{ex_ID}/R_obs_vs_mod.png', dpi=600, bbox_inches='tight')
    plt.close()

    # --- PLOT dS & S ---
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()  # Primary y-axis for dS (Stressing Rate)

    # dS Plot (Stressing Rate)
    ax1.fill_between(t_data, dS_min, dS_max, alpha=0.3, color='blue', label='Stressing Rate') 
    ax1.plot(t_data, dS_median, 'b-', label='Median Stressing Rate')
    ax1.set_xlabel('Time (days)', fontsize=16)
    ax1.set_ylabel('Stressing Rate (MPa/day)', color='blue', fontsize=16)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)
    lines1, labels1 = ax1.get_legend_handles_labels()

    # S Plot (Stress)
    ax2 = ax1.twinx()  # Secondary y-axis for S (Stress)
    ax2.fill_between(t_data, S_min, S_max, alpha=0.3, color='red', label='Stress')
    ax2.plot(t_data, S_median, 'r-', label='Median Stress')
    ax2.set_ylabel('Stress (MPa)', color='red', fontsize=16)
    ax2.tick_params(axis='y', labelcolor='red')
    lines2, labels2 = ax2.get_legend_handles_labels()

    # Legend and Title
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=16)
    plt.title('Modeled Stress vs. Time', fontsize=16)
    plt.savefig(f'example_output/{ex_ID}/S_modeled.png', dpi=600, bbox_inches='tight')
    plt.close()

    # --- SAVE OUTPUTS ---
    out_data = np.column_stack((t_data, R_r_observed, R_r_median))
    np.savetxt(f'example_output/{ex_ID}/Inversion_t_obsRr_modRr.txt', out_data, fmt='%10.5f', comments='')
    np.savetxt(f'example_output/{ex_ID}/parameters.txt', parameters, fmt='%10.5f', comments='')
    np.savetxt(f'example_output/{ex_ID}/R_r_results.txt', R_r_results, fmt='%10.5f', comments='')
    np.savetxt(f'example_output/{ex_ID}/dS_results.txt', dS, fmt='%10.5f', comments='')
    np.savetxt(f'example_output/{ex_ID}/S_results.txt', S, fmt='%10.5f', comments='')
    np.savetxt(f'example_output/{ex_ID}/R_r_optimal.txt', R_r_median, fmt='%10.5f', comments='')
    np.savetxt(f'example_output/{ex_ID}/S_optimal.txt', S_median, fmt='%10.5f', comments='')
    np.savetxt(f'example_output/{ex_ID}/dS_optimal.txt', dS_median, fmt='%10.5f', comments='')

def print_startup_banner():
    print("#################################################################")
    print("T-rate: Trapezoidal-source function seismicity rate modeling tool")
    print("Software for Bayesian inference of transient stress evolution")
    print("from seismicity rate observations.")
    print(" ")
    print("by Yu Jiang, Daniel T. Trugman, & Pablo J. González")
    print("University of Nevada, Reno")
    print("https://github.com/jiangyuinsar/T-rate")
    print("Last update: May 16, 2026")
    print("#################################################################")
    print(" ")
    print(" ")

# --------------------------------------------------------------------
# End of reusable T-rate library code.
# User-facing inversion settings are kept in run_T_rate_inversion_example.py.
# --------------------------------------------------------------------
