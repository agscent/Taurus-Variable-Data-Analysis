from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

root = Path("historical_reference_data")
results = []

def determine_box_type(cleaned_file: Path) -> str:
    """
    Determine if this is a Taurus or Variable Box dataset based on header markers.
    Uses the same primary detection logic as data_cleaning.py.
    """
    try:
        df_test = pd.read_excel(cleaned_file, header=None)
    except Exception as e:
        print(f"ERROR: Could not read {cleaned_file.name}: {e}")
        return None
    
    # 1. Check for the TAURUS Box header column 'Seq' (at or near column 0)
    if df_test.iloc[:, 0].astype(str).str.contains('Seq', case=True, na=False).any():
        return "Taurus"
    
    # 2. Check for the VARIABLE Box header marker 'seq_order' 
    # (Checking rows 8-10, as the header is typically around row 10 / index 9)
    header_rows = df_test.iloc[8:10].astype(str).values.flatten()
    if any('seq_order' in str(cell).lower() for cell in header_rows):
        return "Variable"
    
    # If no format is definitively detected
    print(f"WARNING: Could not detect box type for {cleaned_file.name}")
    return None

def calculate_taurus_baseline(df: pd.DataFrame, sensor_cols: list) -> pd.Series:
    """
    For Taurus: Calculate baseline from first OPEN to first CLOSED.
    Baseline is the median of all sensor values in this range.
    """
    # Find first OPEN and first CLOSED
    open_indices = df[df['Name'] == 'OPEN'].index
    closed_indices = df[df['Name'] == 'CLOSED'].index
    
    if len(open_indices) == 0 or len(closed_indices) == 0:
        # Fallback: use entire dataset median
        print("Warning: Could not find OPEN or CLOSED states, using full dataset median")
        return df[sensor_cols].median()
    
    first_open_idx = open_indices[0]
    first_closed_idx = closed_indices[0]
    
    # Ensure we have a valid range
    if first_open_idx >= first_closed_idx:
        first_closed_idx = closed_indices[closed_indices > first_open_idx]
        if len(first_closed_idx) == 0:
            print("Warning: No CLOSED state after first OPEN, using first CLOSED only")
            return df.loc[closed_indices[0], sensor_cols]
        first_closed_idx = first_closed_idx[0]
    
    # Extract baseline segment (first OPEN to first CLOSED inclusive)
    baseline_range = df.loc[first_open_idx:first_closed_idx, sensor_cols]
    baseline = baseline_range.median()
    
    return baseline

def calculate_variable_baseline(df: pd.DataFrame, sensor_cols: list) -> pd.Series:
    """
    For Variable Box: Calculate baseline from first 'Initialize system' 
    to the row before first 'breath' (use cyl air row before breath).
    Baseline is the median of all sensor values in this range.
    """
    # Find first 'Initialize system' and first 'breath'
    init_indices = df[df['Name'].str.contains('Initialize system', case=False, na=False)].index
    breath_indices = df[df['Name'].str.contains('breath', case=False, na=False)].index
    
    if len(init_indices) == 0:
        print("Warning: Could not find 'Initialize system' state, using full dataset median")
        return df[sensor_cols].median()
    
    first_init_idx = init_indices[0]
    
    if len(breath_indices) == 0:
        print("Warning: Could not find 'breath' state, using from Initialize system to end")
        baseline_range = df.loc[first_init_idx:, sensor_cols]
    else:
        first_breath_idx = breath_indices[0]
        # Use data from first init up to (but not including) first breath
        # The last row before breath should be included
        last_baseline_idx = first_breath_idx - 1
        
        if last_baseline_idx < first_init_idx:
            print("Warning: 'breath' found before 'Initialize system', using Initialize system only")
            baseline_range = df.loc[first_init_idx:first_init_idx, sensor_cols]
        else:
            baseline_range = df.loc[first_init_idx:last_baseline_idx, sensor_cols]
    
    baseline = baseline_range.median()
    
    return baseline

def taurus_box_analysis(df: pd.DataFrame, cleaned_file: Path, df_faults: pd.DataFrame) -> dict:
    """
    Performs core feature extraction and data quality checks for Taurus chip.
    28 channels, baseline calculated from first OPEN to first CLOSED.
    """
    # Identify sensor columns
    sensor_cols = [c for c in df.columns if c.startswith("T") and len(c) > 1 and c[1:].isdigit()]
    data = df[sensor_cols].copy()
    
    # Calculate baseline using Taurus-specific method
    baseline = calculate_taurus_baseline(df, sensor_cols)
    
    # Apply baseline subtraction to the entire sensor data 
    normalised = data - baseline
    
    # Calculate response strength (max-min difference from the normalized data)
    response_strength = normalised.max() - normalised.min()
    
    total_faults = len(df_faults)
    
    # Determine run type (calibration vs. cow breath)
    is_calibration = any(x in cleaned_file.parts for x in ["Box_A_B_Test", "BOX_A", "BOX_B"])
    
    return {
        "sample": cleaned_file.stem.replace("_CLEANED", ""),
        "box_type": "Taurus",
        "type": "Calibration" if is_calibration else "Cow Breath",
        "dead_channels": total_faults,
        "mean_response_kΩ": response_strength.mean() / 1000,
        "usable_channels": 28 - total_faults,
        "data_quality_%": round((28 - total_faults) / 28 * 100, 1)
    }

def variable_box_analysis(df: pd.DataFrame, cleaned_file: Path, df_faults: pd.DataFrame) -> dict:
    """
    Performs core feature extraction and data quality checks for Variable Box.
    Baseline calculated from first 'Initialize system' to row before first 'breath'.
    """
    # Identify sensor columns (adjust naming pattern as needed for Variable Box)
    sensor_cols = [c for c in df.columns if c.startswith("V") and len(c) > 1 and c[1:].isdigit()]
    
    if not sensor_cols:
        # Fallback pattern if different naming convention
        sensor_cols = [c for c in df.columns if 'sensor' in c.lower() or 'channel' in c.lower()]
    
    if not sensor_cols:
        print(f"Warning: Could not identify sensor columns in {cleaned_file.name}")
        return None
    
    data = df[sensor_cols].copy()
    
    # Calculate baseline using Variable Box-specific method
    baseline = calculate_variable_baseline(df, sensor_cols)
    
    # Apply baseline subtraction to the entire sensor data 
    normalised = data - baseline
    
    # Calculate response strength (max-min difference from the normalized data)
    response_strength = normalised.max() - normalised.min()
    
    total_faults = len(df_faults)
    num_channels = len(sensor_cols)
    
    # Determine run type
    is_calibration = any(x in cleaned_file.parts for x in ["calibration", "Calibration", "CAL"])
    
    return {
        "sample": cleaned_file.stem.replace("_CLEANED", ""),
        "box_type": "Variable",
        "type": "Calibration" if is_calibration else "Test",
        "dead_channels": total_faults,
        "mean_response_kΩ": response_strength.mean() / 1000,
        "usable_channels": num_channels - total_faults,
        "data_quality_%": round((num_channels - total_faults) / num_channels * 100, 1)
    }

for cleaned_file in sorted(root.rglob("*_CLEANED.xlsx")):
    faults_file = cleaned_file.with_name(cleaned_file.stem.replace("_CLEANED", "") + "_FAULTS.csv")
    
    if not faults_file.exists():
        print(f"Warning: No faults file for {cleaned_file.name}")
        continue
    
    # Determine box type
    box_type = determine_box_type(cleaned_file)
    
    if box_type is None:
        print(f"Warning: Could not determine box type for {cleaned_file.name}, skipping")
        continue
    
    # Load data
    df = pd.read_excel(cleaned_file)
    df_faults = pd.read_csv(faults_file)
    
    # Call appropriate analysis function
    if box_type == "Taurus":
        result = taurus_box_analysis(df, cleaned_file, df_faults)
    elif box_type == "Variable":
        result = variable_box_analysis(df, cleaned_file, df_faults)
    else:
        result = None
    
    if result is not None:
        results.append(result)
        print(f"Processed {result['sample']} ({result['box_type']})")

# Convert results to DataFrame for analysis
if results:
    df_results = pd.DataFrame(results)
    print("\n" + "="*80)
    print(df_results.to_string(index=False))