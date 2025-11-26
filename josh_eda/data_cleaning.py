from pathlib import Path
import pandas as pd
import numpy as np

# ====================================================================
# MASTER DISPATCHER FUNCTION
# ====================================================================

def clean_file(raw_path: str) -> tuple[Path, list[dict]]:
    """
    Detects the file format (Taurus or Variable Box) and calls the appropriate 
    cleaning function.
    
    Returns: A tuple containing (Path to cleaned file, list of structured faults).
    """
    raw_path = Path(raw_path)
    
    # Load the file once using pd.read_excel with no header to inspect content
    try:
        df_test = pd.read_excel(raw_path, header=None)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not read {raw_path.name} as an Excel file. Error: {e}")
        return raw_path, []

    # 1. Check for the TAURUS Box header column 'Seq' (at or near column 0)
    if df_test.iloc[:, 0].astype(str).str.contains('Seq', case=True, na=False).any():
        print(f"Detected TAURUS Box (T1-T28) format.")
        return clean_taurus_file(raw_path)
    
    # 2. Check for the VARIABLE Box header marker 'seq_order' 
    # (Checking rows 8-10, as the header is typically around row 10 / index 9)
    header_rows = df_test.iloc[8:10].astype(str).values.flatten()
    if any('seq_order' in str(cell).lower() for cell in header_rows):
        print(f"Detected VARIABLE Box (D1-D64) format.")
        return clean_variable_box_file(raw_path)

    # If no format is definitively detected, log and return empty.
    print(f"Format detection failed for {raw_path.name}. Could not find 'Seq' or 'seq_order' header markers.")
    return raw_path, []


# ====================================================================
# TAURUS BOX CLEANER (T1-T28) - Refactored
# ====================================================================

def clean_taurus_file(raw_path: Path) -> tuple[Path, list[dict]]:
    """Cleans a Taurus (T1-T28) sensor file, which uses a complex header."""
    df_full = pd.read_excel(raw_path, header=None)

    # Find real header (row containing "Seq")
    header_row_idx = df_full[df_full.iloc[:, 0] == "Seq"].index[0]
    header = df_full.iloc[header_row_idx]
    data = df_full.iloc[header_row_idx + 1 :]
    df = pd.DataFrame(data.values, columns=header.values).reset_index(drop=True)

    # Clean column names and map to standard names
    df.columns = df.columns.str.replace(r"\s*\(.*\)", "", regex=True).str.strip()
    rename_map = {
        "Time": "Time_ms", "Temp": "Temperature_C", "Humidity": "Humidity_percent",
        "Pump Rate": "Pump_Rate", "V1 pos": "V1_position", "V2 pos": "V2_position",
        "Name": "Name" 
    }
    df = df.rename(columns=rename_map)
    
    # Coerce to numeric, skipping 'Name' column
    for col in df.columns:
        if col == "Name":
            continue
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sensor columns filter
    sensor_cols = [c for c in df.columns if c.startswith("T") and c[1:].isdigit()]
    
    # Run fault detection and saving
    return _apply_fault_checks_and_save(df, raw_path, sensor_cols)


# ====================================================================
# VARIABLE BOX CLEANER (D1-D64) - NEW (Now uses pd.read_excel)
# ====================================================================

def clean_variable_box_file(raw_path: Path) -> tuple[Path, list[dict]]:
    """Cleans a Variable Box (D1-D64) sensor file, which is a standard XLSX."""
    
    HEADER_ROW_IDX = 9 
    
    # Read the data, directly using the header row index (10th row)
    df = pd.read_excel(raw_path, header=HEADER_ROW_IDX)

    # Clean column names
    df.columns = df.columns.str.replace(r'\s*\([^)]*\)', '', regex=True).str.strip().str.lower().str.replace(' ', '_')
    
    # Map to standard names expected by the analysis script
    rename_map = {
        "seq_time": "Time_ms", 
        "temperature": "Temperature_C", 
        "humidity": "Humidity_percent",
        "seq_name": "Name" # The event name column
    }
    df = df.rename(columns=rename_map)
    
    # Coerce to numeric, skipping critical string/boolean/non-data columns
    non_numeric_cols = ["Name", "uv_led", "record_data", "date", "seq_order"] 
    for col in df.columns:
        if col in non_numeric_cols:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce") 

    # Sensor columns filter (D1 to D64, exclude D-Norm)
    # The D-Norm columns are intentionally excluded here.
    sensor_cols = [c for c in df.columns if c.startswith("d") and c[1:].isdigit() and not c.startswith('d-norm')]
    
    # Run fault detection and saving
    return _apply_fault_checks_and_save(df, raw_path, sensor_cols)


# ====================================================================
# SHARED FAULT CHECK AND SAVE LOGIC
# ====================================================================

def _apply_fault_checks_and_save(df: pd.DataFrame, raw_path: Path, sensor_cols: list[str]) -> tuple[Path, list[dict]]:
    """
    Applies the two active fault checks and saves the cleaned file.
    """
    fault_log = []

    # ------------------------------------------------------------------
    # Temperature Stabilization Quality Check 
    # ------------------------------------------------------------------
    # Increased tolerance for stability check 
    STABILITY_MAX_STD_C = 5.0  # Maximum acceptable standard deviation for a "stable" run
    STABILIZATION_TARGET_C =60.0  # Target mean temperature for heated runs
    HIGH_TEMP_RUN_THRESHOLD = 30.0  # If mean is above this, we assume it was a heated run

    if 'Temperature_C' in df.columns and not df['Temperature_C'].empty:
        temp_series = pd.to_numeric(df['Temperature_C'], errors='coerce').dropna()
        if temp_series.empty:
            for col in sensor_cols:
                fault_log.append((col, "DATA_QUALITY_FAIL: MISSING_TEMP_DATA"))
            # return fault_log # Early exit if no temp data

        mean_temp = temp_series.mean()
        std_temp = temp_series.std()
        
        # 1. Check for stability (applies to all runs, room temp or heated)
        if std_temp > STABILITY_MAX_STD_C:
            for col in sensor_cols:
                fault_log.append((col, f"DATA_QUALITY_FAIL: T_UNSTABLE (STD: {std_temp:.2f}C > {STABILITY_MAX_STD_C:.1f}C)"))

        # 2. Check for target mean (only applies if the run was attempting to be heated)
        if mean_temp >= HIGH_TEMP_RUN_THRESHOLD:
            if mean_temp < STABILIZATION_TARGET_C:
                for col in sensor_cols:
                    if not any("T_UNSTABLE" in d for d in fault_log):
                         fault_log.append((col, f"DATA_QUALITY_FAIL: T_STAB_TOO_LOW (Mean: {mean_temp:.1f}C < Target: {STABILIZATION_TARGET_C:.1f}C)"))

    else:
        for col in sensor_cols:
            fault_log.append((col, "DATA_QUALITY_FAIL: MISSING_TEMP_DATA"))

    # ------------------------------------------------------------------
    # Channel stuck at 0 Ω (if 99% of readings are 0)
    # ------------------------------------------------------------------
    ZERO_READING_THRESHOLD = 0.99 

    for col in sensor_cols:
        values = df[col]
        if values.empty:
            continue
            
        values = pd.to_numeric(values, errors='coerce')
        
        zero_count = (values == 0).sum()
        total_count = len(values)
        
        if total_count > 0 and (zero_count / total_count) >= ZERO_READING_THRESHOLD:
            fault_log.append((col, "CRITICAL: STUCK_AT_ZERO_99PCT"))
        
    # Save cleaned file
    output_dir = raw_path.parent / "cleaned"
    output_dir.mkdir(exist_ok=True)
    cleaned_path = output_dir / f"{raw_path.stem}_CLEANED.xlsx"
    df.to_excel(cleaned_path, index=False)

    # Structure fault data for return
    structured_faults = []
    for channel, fault_type in fault_log:
        structured_faults.append({
            "Sample_Name": raw_path.stem,
            "Channel": channel,
            "Fault_Type": fault_type
        })
    
    print(f"Cleaned: {cleaned_path.name}")
    if structured_faults:
        unique_channels = set(d['Channel'] for d in structured_faults)
        print(f"   Faulty channels detected: {len(unique_channels)} → will be aggregated in final report.")
    
    return cleaned_path, structured_faults

    # ====================================================================
    # ❌ COMMENTED OUT: SIGNAL-BASED RUN TYPE CLASSIFICATION
    # This section determines if the run is a 'calibration' or 'cow breath' sample
    # based on the magnitude of sensor swings. It is currently disabled.
    # ====================================================================
    """
    large_swing_channels = 0
    channel_stats = {} 

    for col in sensor_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.empty:
            continue
        mean_val = values.mean()
        std_val = values.std()
        range_val = values.max() - values.min()
        channel_stats[col] = (mean_val, std_val, range_val)

        if range_val > 200_000:           
            large_swing_channels += 1

    is_resistor_calibration_run = large_swing_channels >= 15
    """

    # ====================================================================
    # COMMENTED OUT: APPLY CHANNEL-SPECIFIC FAULT RULES (1-4)
    # This section contains all the individual sensor fault checks 
    # ====================================================================
    """
    # NOTE: The variables 'channel_stats' and 'is_resistor_calibration_run' 
    # must be uncommented in the section above if you wish to re-enable this section.

    for col in sensor_cols:
        # Skip if channel stats weren't calculated (e.g., if signal classification is commented out)
        if col not in channel_stats: 
            # We skip explicit channel checks if their dependencies are missing
            continue 

        mean_val, std_val, range_val = channel_stats[col]

        # 1. Known 1 MΩ overflow bug → resistance collapses to 0 Ω
        if (df[col] == 0).any() and mean_val < 1000:
            fault_log.append((col, "CRITICAL: 1MΩ_OVERFLOW_BUG (reports 0 Ω)"))

        # 2. Physically impossible resistance values
        if (df[col] < 50).any() or (df[col] > 5_000_000).any():
            fault_log.append((col, "CRITICAL: IMPOSSIBLE_PHYSICAL_RANGE"))

        # 3. Channel stuck at open-circuit default value (101 Ω)
        if (df[col].round(1) == 101).mean() > 0.90:
            fault_log.append((col, "CRITICAL: STUCK_AT_101 (open circuit)"))

        # 4. Insufficient response — context-dependent evaluation
        if is_resistor_calibration_run:
            # Strict mode: this is a resistor calibration run
            if range_val < 100:
                fault_log.append((col, "CALIBRATION_FAILURE: range <100 Ω (missing resistor step)"))
            elif std_val < 10 and mean_val > 5_000:
                fault_log.append((col, "CALIBRATION_FAILURE: flat during resistor sweep"))
        else:
            # Lenient mode: this is a real cow breath sample
            if range_val < 10:
                fault_log.append((col, "SUSPECT: flatline/weak response in breath sample (range <10 Ω)"))
    """