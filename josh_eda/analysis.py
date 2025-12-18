from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import warnings
from data_cleaning import check_stuck_at_zero_live, check_temperature_quality_live 

# Suppress all RuntimeWarnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

root = Path("historical_reference_data")
results = []

def determine_box_type(cleaned_file: Path) -> str:
    """
    Determine if this is a Taurus or Variable Box dataset based on the first column name.
    """
    try:
        df = pd.read_excel(cleaned_file, nrows=0)
    except Exception as e:
        print(f"ERROR: Could not read header from {cleaned_file.name}: {e}")
        return None
    
    if len(df.columns) == 0:
        print(f"WARNING: File {cleaned_file.name} has no columns.")
        return None
    
    first_col_name = str(df.columns[0])

    if first_col_name == 'Seq':
        return "Taurus"
    
    if first_col_name == 'seq_order':
        return "Variable"
    
    print(f"WARNING: Could not detect box type for {cleaned_file.name}. First column: '{first_col_name}'")
    return None

def identify_sensor_columns(df: pd.DataFrame, box_type: str) -> list:
    """
    Robustly identify sensor columns by checking common patterns and fallbacks.
    """
    if box_type == "Taurus":
        # Try standard pattern first
        cols = [c for c in df.columns if c.startswith("T") and len(c) > 1 and c[1:].isdigit()]
        if cols:
            return cols
    else:  # Variable Box
        # Try V pattern
        cols = [c for c in df.columns if c.startswith("V") and len(c) > 1 and c[1:].isdigit()]
        if cols:
            return cols
        
        # Try d pattern (d1, d2, etc.)
        cols = [c for c in df.columns if c.startswith("d") and len(c) > 1 and c[1:].isdigit()]
        if cols:
            return cols
        
        # Try generic "sensor" or "channel" pattern
        cols = [c for c in df.columns if 'sensor' in c.lower() or 'channel' in c.lower()]
        if cols:
            return cols
        
        # Last resort: try numeric columns (excluding time-like columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_patterns = ['time', 'seq', 'name', 'state', 'index', 'id', 'temp']
        cols = [c for c in numeric_cols if not any(p in c.lower() for p in exclude_patterns)]
        if cols:
            return cols
    
    return []

def get_taurus_baseline_indices(df: pd.DataFrame, filename: str = "") -> tuple:
    """
    Finds the start (first OPEN) and end (first close) indices for Taurus baseline.
    Falls back to Open Purge → Close Purge if OPEN/CLOSE not found.
    """
    open_indices = df[df['Name'] == 'OPEN'].index
    close_indices = df[df['Name'] == 'CLOSE'].index
    
    if len(open_indices) > 0 and len(close_indices) > 0:
        first_open_idx = open_indices[0]
        first_close_after_open = close_indices[close_indices > first_open_idx]
        
        if len(first_close_after_open) > 0:
            first_close_idx = first_close_after_open[0]
            return first_open_idx, first_close_idx
    
    # Fall back to Open Purge → Close Purge pattern
    open_purge_indices = df[df['Name'].str.contains('open purge', case=False, na=False)].index
    close_purge_indices = df[df['Name'].str.contains('close purge', case=False, na=False)].index
    
    if len(open_purge_indices) > 0 and len(close_purge_indices) > 0:
        first_open_purge = open_purge_indices[0]
        close_after_open = close_purge_indices[close_purge_indices > first_open_purge]
        if len(close_after_open) > 0:
            first_close_purge = close_after_open[0]
            return first_open_purge, first_close_purge
    
    # Final fallback: use full dataset
    print(f"Warning: Could not find OPEN/CLOSE or Open Purge/Close Purge in {filename}, using full dataset.")
    return df.index[0], df.index[-1]

def get_variable_baseline_indices(df: pd.DataFrame, filename: str = "") -> tuple:
    """
    Finds the start ('Initialize system') and end (row before first 'breath') 
    indices for Variable Box baseline.
    
    Variable box structure:
    - Row 0: Initialize system
    - Rows 1-N: cyl air (baseline)
    - Rows N+1-M: breath (response)
    - Rows M+1+: cyl air (post-breath, NOT baseline)
    """
    # Check if 'Name' column exists
    if 'Name' not in df.columns:
        # Try alternate column names
        name_candidates = ['seq_name', 'sequence_name', 'State', 'state']
        name_col = None
        for candidate in name_candidates:
            if candidate in df.columns:
                name_col = candidate
                break
        
        if name_col is None:
            print(f"Warning: No sequence name column in {filename}, using first 30% as baseline.")
            baseline_end = int(len(df) * 0.3)
            return df.index[0], baseline_end
    else:
        name_col = 'Name'
    
    # Look for breath markers (case-insensitive)
    breath_indices = df[df[name_col].str.contains('breath', case=False, na=False)].index
    
    if len(breath_indices) == 0:
        # No breath found - try other patterns
        sample_indices = df[df[name_col].str.contains('sample|expose|test', case=False, na=False)].index
        if len(sample_indices) > 0:
            breath_indices = sample_indices
    
    if len(breath_indices) == 0:
        print(f"Warning: No breath/sample found in {filename}, using first 30% as baseline.")
        baseline_end = int(len(df) * 0.3)
        return df.index[0], baseline_end
    
    # Baseline ends just before first breath
    first_breath_idx = breath_indices[0]
    last_baseline_idx = first_breath_idx - 1
    
    # Start from beginning (includes "Initialize system" and "cyl air")
    first_baseline_idx = df.index[0]
    
    if last_baseline_idx < first_baseline_idx:
        print(f"Warning: Breath at start of {filename}, using first 30% as baseline.")
        baseline_end = int(len(df) * 0.3)
        return df.index[0], baseline_end
    
    return first_baseline_idx, last_baseline_idx


def check_fit_quality(normalised_df: pd.DataFrame, start_idx: int, end_idx: int) -> float:
    """
    Calculates the mean Standard Deviation of the normalized baseline segment.
    """
    baseline_residuals = normalised_df.loc[start_idx:end_idx]
    if baseline_residuals.empty or len(baseline_residuals) == 0:
        return float('inf')
    return baseline_residuals.std().mean()

def linear_baseline_correction(sensor_data: pd.Series, baseline_start_idx: int, baseline_end_idx: int) -> pd.Series:
    """
    Fits a linear curve to the baseline and subtracts it from the entire time series.
    """
    X_full = sensor_data.index.values
    baseline_segment = sensor_data.loc[baseline_start_idx:baseline_end_idx]
    X_baseline = baseline_segment.index.values
    Y_baseline = baseline_segment.values
    
    if len(X_baseline) < 2:
        return sensor_data - sensor_data.median()

    slope, intercept, _, _, _ = stats.linregress(X_baseline, Y_baseline)
    fitted_curve = slope * X_full + intercept
    return sensor_data - fitted_curve

def exponential_func(x, a, b, c):
    """Exponential function: y = a * exp(b * x) + c"""
    return a * np.exp(b * x) + c

def robust_exp_fit(t, y):
    """
    Perform a robust exponential fit with:
    - NaN filtering
    - bounded curve_fit
    - fallback on failure
    - RMSE calculation
    """
    # Clean NaNs
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]

    if len(t) < 5:
        return None, {"error": "Insufficient points for exponential fit"}

    # Initial guess
    p0 = [y.max() - y.min(), -0.1, y.min()]

    # Reasonable bounds for stability
    bounds = (
        [-np.inf, -5.0, -np.inf],   # a, b, c lower bounds
        [ np.inf,  5.0,  np.inf],   # a, b, c upper bounds
    )

    try:
        popt, pcov = curve_fit(
            exponential_func,
            t,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=8000
        )
    except Exception as e:
        return None, {"error": f"exp_fit_failed: {str(e)}"}

    # Compute RMSE
    residuals = y - exponential_func(t, *popt)
    rmse = np.sqrt(np.mean(residuals**2))

    return popt, {"rmse": float(rmse), "params": popt}

def exponential_baseline_correction(sensor_data: pd.Series, baseline_start_idx: int, baseline_end_idx: int) -> pd.Series:
    """
    Fits an exponential curve to baseline. Falls back gracefully if fit fails.
    """
    X_full = sensor_data.index.values
    baseline_segment = sensor_data.loc[baseline_start_idx:baseline_end_idx]
    X_baseline = baseline_segment.index.values
    Y_baseline = baseline_segment.values
    
    if len(X_baseline) < 3:
        return sensor_data - sensor_data.median()

    try:
        # Normalize X for better numerical stability
        X_norm = (X_baseline - X_baseline.min()) / (X_baseline.max() - X_baseline.min() + 1e-10)
        X_full_norm = (X_full - X_baseline.min()) / (X_baseline.max() - X_baseline.min() + 1e-10)
        
        # Better initial guess
        Y_range = Y_baseline.max() - Y_baseline.min()
        p0 = (Y_range * 0.1, -1.0, Y_baseline.mean())
        
        # Suppress warnings during curve_fit
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            popt, _ = curve_fit(exponential_func, X_norm, Y_baseline, p0=p0, maxfev=3000, ftol=1e-3)
        
        fitted_curve = exponential_func(X_full_norm, *popt)
        return sensor_data - fitted_curve
    
    except (RuntimeError, ValueError):
        # Silently fall back to linear
        return linear_baseline_correction(sensor_data, baseline_start_idx, baseline_end_idx)

CORRECTION_MODALITIES = {
    "Constant_Median": lambda data, start, end: data - data.loc[start:end].median(),
    "Linear_Fit": linear_baseline_correction,
    "Exponential_Fit": exponential_baseline_correction,
}

def fit_baseline(x, y, mode="exp"):
    """
    Compute baseline using one of:
    - flat
    - linear
    - exp (robust exponential)
    Returns:
        baseline_curve (np.array), fit_info (dict)
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if mode == "flat":
        baseline = np.full_like(y, np.median(y))
        return baseline, {"type": "flat", "median": float(np.median(y))}

    elif mode == "linear":
        p = np.polyfit(x, y, 1)
        baseline = np.polyval(p, x)
        return baseline, {"type": "linear", "slope": float(p[0]), "intercept": float(p[1])}

    elif mode == "exp":
        popt, info = robust_exp_fit(x, y)
        if popt is None:
            # fallback → linear
            p = np.polyfit(x, y, 1)
            baseline = np.polyval(p, x)
            info["fallback"] = "linear"
            return baseline, info

        baseline = exponential_func(x, *popt)
        info["type"] = "exp"
        return baseline, info

    else:
        raise ValueError(f"Unknown baseline mode: {mode}")
    

def subtract_baseline(full_series, baseline_curve):
    """
    Baseline subtraction: full_signal - fitted_baseline
    """
    full_series = np.asarray(full_series)
    corrected = full_series - baseline_curve
    return corrected


def normalise_signal(signal, method="zscore"):
    signal = np.asarray(signal)

    if method == "zscore":
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    elif method == "minmax":
        return (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)

    elif method == "deltaR":
        # delta R / R
        baseline = signal[0]
        return (signal - baseline) / (baseline + 1e-8)

    else:
        raise ValueError(f"Unknown normalization type: {method}")


def analyze_box(df: pd.DataFrame, cleaned_file: Path, df_faults: pd.DataFrame, box_type: str) -> dict:
    """
    Unified analysis for both Taurus and Variable boxes.
    """
    sensor_cols = identify_sensor_columns(df, box_type)
    
    if not sensor_cols:
        print(f"ERROR: No sensor columns identified in {cleaned_file.name}")
        return None
    
    data = df[sensor_cols].copy()
    
    if box_type == "Taurus":
        baseline_start_idx, baseline_end_idx = get_taurus_baseline_indices(df, cleaned_file.name)
    else:
        baseline_start_idx, baseline_end_idx = get_variable_baseline_indices(df, cleaned_file.name)
    
    best_fit_type = "Constant_Median"
    min_sd = float('inf')
    best_normalised_data = None
    all_fit_quality = {}
    
    for fit_name, correction_func in CORRECTION_MODALITIES.items():
        normalised_temp = pd.DataFrame(index=data.index)
        
        for col in sensor_cols:
            normalised_temp[col] = correction_func(data[col], baseline_start_idx, baseline_end_idx)
        
        current_sd = check_fit_quality(normalised_temp, baseline_start_idx, baseline_end_idx)
        all_fit_quality[f"{fit_name}_SD"] = round(current_sd, 6)
        
        if current_sd < min_sd:
            min_sd = current_sd
            best_fit_type = fit_name
            best_normalised_data = normalised_temp
    
    normalised = best_normalised_data if best_normalised_data is not None else pd.DataFrame()
    response_strength = normalised.max() - normalised.min()
    
    total_faults = len(df_faults)
    num_channels = len(sensor_cols)
    
    if box_type == "Taurus":
        is_calib = any(x in cleaned_file.parts for x in ["Box_A_B_Test", "BOX_A", "BOX_B"])
        sample_type = "Pregnancy Chip" if is_calib else "Box A/B Test"
    else:
        is_calib = any(x in cleaned_file.parts for x in ["calibration", "Calibration", "CAL"])
        sample_type = "Calibration" if is_calib else "Standard"
    
    results_dict = {
        "sample": cleaned_file.stem.replace("_CLEANED", ""),
        "box_type": box_type,
        "type": sample_type,
        "num_channels": num_channels,
        "dead_channels": total_faults,
        "usable_channels": num_channels - total_faults,
        "data_quality_%": round((num_channels - total_faults) / num_channels * 100, 1),
        "mean_response_kΩ": round(response_strength.mean() / 1000, 3),
        "selected_baseline_fit": best_fit_type, 
        **all_fit_quality
    }
    
    return results_dict

# ===== MAIN EXECUTION LOOP =====
if __name__ == "__main__":

    output_filename = "analysis_summary.csv" 

    for cleaned_file in sorted(root.rglob("*_CLEANED.xlsx")):
        # Skip temp files created by Excel
        if cleaned_file.name.startswith('~$'):
            continue
        
        box_type = determine_box_type(cleaned_file)
        
        if box_type is None:
            continue
        
        try:
            df = pd.read_excel(cleaned_file)
        except Exception as e:
            print(f"ERROR: Could not read {cleaned_file.name}: {e}")
            continue
        
        # Identify sensor columns using the robust detection function
        sensor_cols = identify_sensor_columns(df, box_type)
        
        if not sensor_cols:
            print(f"ERROR: No sensor columns found in {cleaned_file.name}")
            continue

        df_stuck_at_zero = check_stuck_at_zero_live(df, sensor_cols)
        df_temp_quality = check_temperature_quality_live(df, sensor_cols)
        
        df_faults = pd.concat([df_stuck_at_zero, df_temp_quality], ignore_index=True)
        df_faults = df_faults.drop_duplicates(subset=['Channel', 'Fault_Type'])

        result = analyze_box(df, cleaned_file, df_faults, box_type)
        
        if result:
            results.append(result)

    if results:
        df_summary = pd.DataFrame(results)
        df_summary.to_csv(output_filename, index=False)
        print(f"\n--- Analysis complete. Processed {len(results)} files ---")
        print(f"--- Summary exported to {output_filename} ---")
    else:
        print("\n--- Analysis complete, but no files were processed successfully. ---")