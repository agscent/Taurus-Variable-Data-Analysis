"""
Robust baseline correction module.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import warnings


def exponential_func(x, a, b, c):
    """Exponential function: y = a * exp(b * x) + c"""
    return a * np.exp(b * x) + c


def smooth_baseline_data(raw_data, method='savgol', window=11):
    """
    Smooth noisy baseline data before fitting.
    
    Critical for handling high-frequency oscillations in raw data.
    """
    if method == 'savgol':
        # Savitzky-Golay filter preserves shape while smoothing
        if len(raw_data) < window:
            window = len(raw_data) if len(raw_data) % 2 == 1 else len(raw_data) - 1
        if window < 5:
            return raw_data
        return savgol_filter(raw_data, window_length=window, polyorder=3)
    
    elif method == 'median':
        # Median filter removes spikes
        from scipy.signal import medfilt
        kernel = min(5, len(raw_data))
        if kernel % 2 == 0:
            kernel += 1
        return medfilt(raw_data, kernel_size=kernel)
    
    elif method == 'rolling':
        # Rolling mean
        return pd.Series(raw_data).rolling(window=min(window, len(raw_data)), 
                                          center=True, min_periods=1).mean().values
    
    return raw_data


def fit_baseline_robust(X_baseline, Y_baseline, fit_type='linear', smooth=True):
    """
    Fit baseline with optional smoothing and robust error handling.
    
    Returns:
        fitted_params: Parameters of the fit
        fit_func: Function to evaluate fit at any X value
        fit_info: Dict with quality metrics
    """
    # Step 1: Smooth the baseline data if requested
    if smooth and len(Y_baseline) > 10:
        Y_smooth = smooth_baseline_data(Y_baseline, method='savgol', window=11)
    else:
        Y_smooth = Y_baseline
    
    # Normalize X for better numerical stability
    X_norm = (X_baseline - X_baseline.min()) / (X_baseline.max() - X_baseline.min() + 1e-10)
    
    # Step 2: Fit based on type
    if fit_type == 'constant':
        # Use median (robust to outliers)
        baseline_val = np.median(Y_smooth)
        fit_func = lambda x: np.full_like(x, baseline_val, dtype=float)
        fitted_params = [baseline_val]
        
    elif fit_type == 'linear':
        # Robust linear regression using Theil-Sen estimator (resistant to outliers)
        from scipy.stats import theilslopes
        try:
            slope, intercept, _, _ = theilslopes(Y_smooth, X_norm)
        except:
            # Fallback to standard linear regression
            slope, intercept, _, _, _ = stats.linregress(X_norm, Y_smooth)
        
        # Create fit function that works with original X scale
        def fit_func(x_orig):
            x_n = (x_orig - X_baseline.min()) / (X_baseline.max() - X_baseline.min() + 1e-10)
            return slope * x_n + intercept
        
        fitted_params = [slope, intercept]
    
    elif fit_type == 'exponential':
        # Exponential fit with robust initial guess
        try:
            Y_range = Y_smooth.max() - Y_smooth.min()
            p0 = [Y_range * 0.1, -1.0, np.median(Y_smooth)]
            
            # Bounded fit
            bounds = (
                [-np.inf, -10, -np.inf],  # Lower bounds
                [np.inf, 10, np.inf]       # Upper bounds
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                popt, _ = curve_fit(
                    lambda x, a, b, c: a * np.exp(b * x) + c,
                    X_norm, Y_smooth, 
                    p0=p0, 
                    bounds=bounds,
                    maxfev=5000
                )
            
            # Create fit function
            def fit_func(x_orig):
                x_n = (x_orig - X_baseline.min()) / (X_baseline.max() - X_baseline.min() + 1e-10)
                return popt[0] * np.exp(popt[1] * x_n) + popt[2]
            
            fitted_params = popt
            
        except:
            # Fallback to linear if exponential fails
            return fit_baseline_robust(X_baseline, Y_baseline, fit_type='linear', smooth=smooth)
    
    else:
        raise ValueError(f"Unknown fit_type: {fit_type}")
    
    # Step 3: Calculate fit quality on ORIGINAL (unsmoothed) data
    fitted_baseline = fit_func(X_baseline)
    residuals = Y_baseline - fitted_baseline  # Use original Y, not smoothed
    rmse = np.sqrt(np.mean(residuals**2))
    baseline_sd = np.std(residuals)
    
    fit_info = {
        'rmse': float(rmse),
        'baseline_sd': float(baseline_sd),
        'params': fitted_params,
        'smoothed': smooth
    }
    
    return fitted_params, fit_func, fit_info


def correct_baseline_robust(sensor_data: pd.Series, baseline_start_idx: int, 
                            baseline_end_idx: int, fit_type='linear', 
                            smooth_baseline=True):
    """
    Improved baseline correction with smoothing and robust fitting.
    
    Returns: Corrected data (pd.Series)
    """
    X_full = sensor_data.index.values
    baseline_segment = sensor_data.loc[baseline_start_idx:baseline_end_idx]
    X_baseline = baseline_segment.index.values
    Y_baseline = baseline_segment.values
    
    if len(X_baseline) < 3:
        # Not enough points - just subtract median
        return sensor_data - np.median(Y_baseline)
    
    # Fit baseline with robust method
    _, fit_func, _ = fit_baseline_robust(X_baseline, Y_baseline, 
                                         fit_type=fit_type, 
                                         smooth=smooth_baseline)
    
    # Subtract fitted baseline from entire time series
    fitted_curve = fit_func(X_full)
    corrected = sensor_data - fitted_curve
    
    return corrected


# ===== CORRECTION MODALITIES (Main Export) =====
CORRECTION_MODALITIES_ROBUST = {
    "Constant_Median": lambda data, start, end: correct_baseline_robust(
        data, start, end, fit_type='constant', smooth_baseline=False
    ),
    "Linear_Fit": lambda data, start, end: correct_baseline_robust(
        data, start, end, fit_type='linear', smooth_baseline=True
    ),
    "Exponential_Fit": lambda data, start, end: correct_baseline_robust(
        data, start, end, fit_type='exponential', smooth_baseline=True
    ),
}


def verify_baseline_correction(corrected_data: pd.Series, baseline_start_idx: int, 
                               baseline_end_idx: int, tolerance=0.05):
    """
    Verify that baseline correction actually worked.
    
    Returns:
        is_good: Boolean - True if baseline is acceptably flat
        metrics: Dict with quality metrics
    """
    baseline_corrected = corrected_data.loc[baseline_start_idx:baseline_end_idx]
    
    mean_val = baseline_corrected.mean()
    std_val = baseline_corrected.std()
    median_val = baseline_corrected.median()
    
    # Good correction should have:
    # 1. Mean close to zero
    # 2. Low standard deviation relative to signal range
    signal_range = corrected_data.max() - corrected_data.min()
    
    is_good = (
        abs(mean_val) < tolerance * signal_range and
        abs(median_val) < tolerance * signal_range and
        std_val < 0.2 * signal_range
    )
    
    metrics = {
        'baseline_mean': float(mean_val),
        'baseline_median': float(median_val),
        'baseline_std': float(std_val),
        'signal_range': float(signal_range),
        'is_acceptable': is_good
    }
    
    return is_good, metrics


# ===== EXAMPLE USAGE (Only runs if executed directly) =====
if __name__ == "__main__":
    # Simulate problematic sensor data (like your T1)
    t = np.linspace(0, 100, 1000)
    
    # Baseline with drift + high-frequency noise
    baseline_drift = 0.002 * t  # Linear drift
    high_freq_noise = 0.05 * np.sin(2 * np.pi * 10 * t)  # Fast oscillations
    baseline = 100 + baseline_drift + high_freq_noise + np.random.normal(0, 0.01, len(t))
    
    # Add a breath response later
    breath_response = np.zeros_like(t)
    breath_response[300:700] = 2.0 * np.exp(-(t[300:700] - t[300])/10)
    
    full_signal = baseline + breath_response
    
    # Create pandas Series
    sensor_data = pd.Series(full_signal, index=range(len(full_signal)))
    
    # Correct baseline
    corrected = correct_baseline_robust(
        sensor_data, 
        baseline_start_idx=0, 
        baseline_end_idx=299,
        fit_type='linear',
        smooth_baseline=True
    )
    
    # Verify
    is_good, metrics = verify_baseline_correction(corrected, 0, 299)
    
    print("Baseline Correction Verification:")
    print(f"  Mean: {metrics['baseline_mean']:.6f} (should be ~0)")
    print(f"  Std: {metrics['baseline_std']:.6f}")
    print(f"  Acceptable: {metrics['is_acceptable']}")
    
    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    axes[0].plot(sensor_data.values, 'k-', label='Raw', alpha=0.7)
    axes[0].axvspan(0, 299, color='green', alpha=0.2)
    axes[0].set_title('Raw Data')
    axes[0].legend()
    
    axes[1].plot(corrected.values, 'b-', label='Corrected', alpha=0.7)
    axes[1].axvspan(0, 299, color='green', alpha=0.2)
    axes[1].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_title('Corrected Data (Baseline Should Be Flat at 0)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()