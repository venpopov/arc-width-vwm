import numpy as np
from numba import njit
from numba.extending import get_cython_function_address
import ctypes
from scipy.stats import gamma, beta

##---------------------------------------------------------------------------
#                            Core building blocks
#---------------------------------------------------------------------------
addr = get_cython_function_address("scipy.special.cython_special", "i0")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
i0_fast = functype(addr)

@njit
def von_mises_kernel(kappa: float, grid_size: int = 360) -> np.ndarray:
    """Compute von Mises kernel centered at zero with proper normalization.
    
    Parameters
    ----------
    kappa : float
        Concentration parameter of the von Mises distribution
    grid_size : int, optional
        Size of circular grid (default 360)
    
    Returns
    -------
    np.ndarray
        Von Mises probability density function values on the circular grid
        Normalized such that the discrete sum approximates 1
    
    Notes
    -----
    The von Mises distribution PDF is defined as:
    f(θ; μ, κ) = exp(κ * cos(θ - μ)) / (2π * I₀(κ))
    where I₀(κ) is the modified Bessel function of order 0.
    
    For discrete approximation on a grid, we normalize such that
    the sum of PDF values multiplied by the angular step equals 1.
    """
    # Generate angular grid from -180 to 179 degrees, convert to radians
    # x = np.arange(grid_size)
    # diff_rad = np.deg2rad(np.mod(x + 180, 360) - 180)  # Range: [-π, π)
    diff_rad = np.arange(grid_size) * 2 * np.pi / grid_size # Range: [0, 2π)
    
    # Compute unnormalized von Mises values
    unnormalized = np.exp(kappa * np.cos(diff_rad))
    
    # Compute normalization constant: 2π * I₀(κ)
    # Using the modified Bessel function I₀ for proper normalization
    normalization = 2.0 * np.pi * i0_fast(kappa)
    
    # Normalize to get proper probability density
    pdf = unnormalized / normalization
    
    return pdf

@njit
def activation_signal_fast(
    x_grid: np.ndarray,
    memory_items: np.ndarray,
    vm_kernel: np.ndarray,
    c: float,
    a: float,
    s: float,
    target_idx: int
) -> np.ndarray:
    """Fast activation signal computation using precomputed kernel.
    
    Parameters
    ----------
    x_grid : np.ndarray
        Feature grid (degrees)
    memory_items : np.ndarray
        Stored feature values (degrees)
    vm_kernel : np.ndarray
        Precomputed von Mises kernel
    c : float
        Context weight scaling factor
    a : float
        Attentional weight
    s : float
        Context discrimination sharpness
    target_idx : int
        Index of target item in memory
    
    Returns
    -------
    np.ndarray
        Combined activation signal across feature grid
    """
    n_items = len(memory_items)
    S_x = np.zeros_like(x_grid, dtype=np.float64)
    
    for i in range(n_items):
        weight = c * np.exp(-s * abs(i - target_idx)) + a
        shift = int(memory_items[i])
        shifted = np.roll(vm_kernel, shift-1)
        S_x += weight * shifted
    
    return S_x

@njit
def wrap_angle_batch(angles: np.ndarray, bound: float) -> np.ndarray:
    """Wrap multiple angles to [-bound, bound] range.
    
    Parameters
    ----------
    angles : np.ndarray
        Input angles
    bound : float
        Boundary value
    
    Returns
    -------
    np.ndarray
        Wrapped angles in [-bound, bound]
    """
    result = angles.copy()
    for i in range(len(result)):
        while result[i] > bound:
            result[i] -= 2 * bound
        while result[i] < -bound:
            result[i] += 2 * bound
    return result

@njit
def sdm_arc_v7_batch(
    memory_items_batch: np.ndarray,
    c: float,
    kappa: float,
    up_down: float,      
    step_size: float,    
    lambda_poisson: float,
    initial_arc: float = 0.3 * np.pi, 
    a: float = 0.0,
    target_idx: int = 0,
    seeds: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    SDM v7 Simulator (Python Numba Version)
    
    Key Features:
    1. Remove spatial_distances, use target_idx to distinguish weights.
    2. Sequential Loop: The arc width of trial t depends on the result of trial t-1.
    """
    n_trials = memory_items_batch.shape[0]
    n_items = memory_items_batch.shape[1]
    
    # Output containers
    points = np.empty(n_trials, dtype=np.float64)
    arcs = np.empty(n_trials, dtype=np.float64)
    
    # Pre-computed constants
    grid_radians = np.arange(1, 361) * (np.pi / 180.0)
    vm_kernel = von_mises_kernel(kappa)
    
    # Staircase process step definitions
    radians_up = step_size * up_down
    radians_down = step_size * (1.0 - up_down)
    
    # === v7 Core Variable: Cross-trial state ===
    # This variable will be continuously updated, carrying the previous state to the next trial
    current_arc = initial_arc 
    
    # Note: Using range instead of prange because sequential execution is required to pass current_arc
    for trial in range(n_trials):
        if seeds is not None:
            np.random.seed(seeds[trial])
            
        # 1. Determine sample count (Poisson)
        n_samples = 2 + np.random.poisson(lambda_poisson)
        
        # 2. Compute activation (Activation)
        # Not using spatial distance, but using target_idx logic
        act = np.zeros(360, dtype=np.float64)
        
        for item in range(n_items):
            # Weight logic: if it's the target item, weight is C+A, otherwise C (or adjust according to your specific model)
            # Here assuming the item pointed by target_idx has additional attention A
            weight = c + a if item == target_idx else c
            
            # Get memory color (1-360) and convert to index (0-359)
            mem_val = memory_items_batch[trial, item]
            shift_idx = int(mem_val) - 1
            
            # Circular convolution/addition (Circular Shift and Add)
            for i in range(360):
                # Equivalent to binhf::shift in R
                template_idx = (i - shift_idx) % 360
                act[i] += weight * vm_kernel[template_idx]
        
        # 3. Compute probability (Softmax)
        # Subtract maximum to prevent numerical overflow
        exp_act = np.exp(act - np.max(act))
        prob_dist = exp_act / np.sum(exp_act)
        
        # 4. Sampling (Sampling)
        # Use cumulative distribution function for inverse transform sampling
        cumsum = np.cumsum(prob_dist)
        sample_indices = np.empty(n_samples, dtype=np.int64)
        
        for i in range(n_samples):
            r = np.random.random()
            idx = np.searchsorted(cumsum, r)
            if idx >= 360: idx = 359
            sample_indices[i] = idx
            
        sampled_features = grid_radians[sample_indices]
        
        # 5. Select Point Response
        # Randomly select one sample as Point
        point_idx_in_samples = np.random.randint(0, n_samples)
        point_radian = sampled_features[point_idx_in_samples]
        
        # Record Point (convert to degrees)
        points[trial] = point_radian * (180.0 / np.pi)
        
        # 6. Update Arc Width (Staircase Process)
        # v7 core logic: Use remaining samples to adjust current_arc
        
        # A. Calculate original differences from Point to all samples
        # Note: Here calculating "Point - Samples", resulting in a vector
        raw_diffs = point_radian - sampled_features
        
        # B. Batch wrapping
        # [Key]: Must use np.pi as bound because sampled_features is in radians
        # If using 180.0, the calculation will be completely wrong
        wrapped_diffs = wrap_angle_batch(raw_diffs, np.pi)
        
        # C. Take absolute values to get distances
        all_dists = np.abs(wrapped_diffs)
        
        # D. Prepare update sequence (excluding Point itself)
        other_indices = np.arange(n_samples)
        mask = other_indices != point_idx_in_samples
        sequence_indices = other_indices[mask]
        
        np.random.shuffle(sequence_indices)
        
        # E. Staircase update of Arc
        # Although distances have been computed in batch, updating Arc must be sequential
        for s_idx in sequence_indices:
            # Directly get distance from pre-calculated array, no need to compute wrap again
            dist = all_dists[s_idx]
            
            if dist < current_arc:
                current_arc = max(0.0, current_arc - radians_down)
            elif dist > current_arc:
                current_arc = min(np.pi, current_arc + radians_up)
                
        arcs[trial] = current_arc * (180.0 / np.pi)
        
    return points, arcs

# Register the SDM v7 batch simulator
BATCH_SIMULATOR = {
    "SDM_v7": sdm_arc_v7_batch
}

##---------------------------------------------------------------------------
#                            Prior
#---------------------------------------------------------------------------

# UpDown: Range [0, 1], controls convergence target.
# Mean of 0.5 means Arc will converge to make the probability of Point falling within Arc equal to 50%.
# Higher precision means the distribution is more concentrated around the mean.
_common_UpDown = {
    "prior_UpDown": 0.5,       # Mean
    "precision_UpDown": 10.0   # Precision (alpha + beta)
}
# StepSize: Range > 0 (in radians).
# Mean of 0.1 radians (approximately 5.7 degrees). This is a typical psychophysical staircase step size.
_common_StepSize = {
    "prior_StepSize": 0.1,     # Mean (in radians)
    "var_StepSize": 0.05**2    # Variance (allows it to fluctuate between 0.05 and 0.2)
}
_common_C = {"prior_C": 4.0, "var_C": 3.0**2}
_common_PHit = {"prior_PHit": 0.5, "precision_PHit": 10}
_common_Lambda = {"prior_Lambda": 4.0, "lambda_var_ratio": 1/3}
_common_kappa = {"prior_Kappa": 4, "kappa_var_ratio": 1/3}

MODEL_PRIOR_DEFAULTS = {
    "SDM_v7": {
        **_common_C,
        **_common_kappa,
        **_common_Lambda,
        **_common_UpDown,
        **_common_StepSize
    },
}

def gamma_params_from_mean_var(mean, var):
    """Return (a, scale) for gamma distribution with given mean and variance."""
    if mean <= 0 or var <= 0:
        raise ValueError(f"Gamma mean/var must be >0: mean={mean}, var={var}")
    a = mean**2 / var
    scale = var / mean
    return a, scale

def beta_params_from_mean_precision(mean, precision):
    """
    precision = alpha + beta
    mean = alpha / (alpha + beta)
    """
    alpha = mean * precision
    beta_param = (1 - mean) * precision
    return alpha, beta_param

def sdm_v7_prior():
    cfg = MODEL_PRIOR_DEFAULTS["SDM_v7"].copy()
    
    # Compute variance terms from ratios
    var_Kappa = (cfg["prior_Kappa"] * cfg["kappa_var_ratio"])**2
    var_Lambda = (cfg["prior_Lambda"] * cfg["lambda_var_ratio"])**2

    a_C, scale_C = gamma_params_from_mean_var(cfg["prior_C"], cfg["var_C"])
    a_Kappa, scale_Kappa = gamma_params_from_mean_var(cfg["prior_Kappa"], var_Kappa)
    a_Lambda, scale_Lambda = gamma_params_from_mean_var(cfg["prior_Lambda"], var_Lambda)

    a_Step, scale_Step = gamma_params_from_mean_var(cfg["prior_StepSize"], cfg["var_StepSize"])
    a_UpDown, b_UpDown = beta_params_from_mean_precision(
        cfg["prior_UpDown"], 
        cfg["precision_UpDown"]
    )

    return {
        "c": np.float32(gamma.rvs(a=a_C, scale=scale_C)),
        "kappa": np.float32(gamma.rvs(a=a_Kappa, scale=scale_Kappa)),
        "lambda_poisson": np.float32(gamma.rvs(a=a_Lambda, scale=scale_Lambda)),
        "up_down": np.float32(beta.rvs(a=a_UpDown, b=b_UpDown)),
        "step_size": np.float32(gamma.rvs(a=a_Step, scale=scale_Step)),
    }

PRIOR_SAMPLER = {
    "SDM_v7": sdm_v7_prior
}
