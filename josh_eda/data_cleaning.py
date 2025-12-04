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

    # TAURUS Box header column 'Seq' (at or near column 0)
    if df_test.iloc[:, 0].astype(str).str.contains('Seq', case=True, na=False).any():
        print(f"Detected TAURUS Box (T1-T28) format.")
        # Assuming clean_taurus_file still only needs raw_path
        return clean_taurus_file(raw_path)
    
    # VARIABLE Box header marker 'seq_order' 
    #    Find the index where 'seq_order' appears in the first column
    header_row_index = None
    
    # Check the common header locations: (Rows 9, 10, and 11)
    for i in range(8, 11): 
        # Check the first cell (column 0) of the row
        cell_value = str(df_test.iloc[i, 0]).lower()
        
        # Check if 'seq_order' is in the cell's content
        if 'seq_order' in cell_value:
            header_row_index = i
            break 

    if header_row_index is not None:
        # Pass the detected header index to the cleaning function
        return clean_variable_box_file(raw_path, header_row_index)

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

def clean_variable_box_file(raw_path: Path, header_row_idx: int) -> tuple[Path, list[dict]]:
    """Cleans a Variable Box (D1-D64) sensor file, which is a standard XLSX.""" 
    
    # Read the data, directly using the header row index (10th row)
    df = pd.read_excel(raw_path, header=header_row_idx)

    # Clean column names
    df.columns = df.columns.str.replace(r'\s*\([^)]*\)', '', regex=True).str.strip().str.lower().str.replace(' ', '_')
    # Convert column names to lowercase for case-insensitive processing
    df.columns = df.columns.str.lower()
    
    # Map to standard names expected by the analysis script
    rename_map = {
        "seq_time": "Sequence_Time_ms", 
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

    # We use the 'date' column to create the highly accurate, zero-indexed elapsed time axis.
    # Calculate Time_ms series using temporary structures
    # Calculate Date_Time series from the 'date' column
    date_time_series = pd.to_datetime(df['date'], errors='coerce')
    t0 = date_time_series.min()
    
    # Calculate elapsed time in milliseconds. Use 'Int64' to allow for NaN values.
    # NaT - t0 correctly yields NaN for any row with an unusable date.
    time_ms_series = ((date_time_series - t0).dt.total_seconds() * 1000)

    # Insert 'Time_ms' and reorder ALL columns to eliminate fragmentation warnings.
    
    # Get the list of existing columns after dropping 'date'.
    cols = df.columns.tolist()
    
    # Find the index for insertion (after 'Sequence_Time_ms').
    insert_after_col = 'Sequence_Time_ms'
    try:
        insert_idx = cols.index(insert_after_col) + 1
    except ValueError:
        insert_idx = 0 # Fallback to front if duration column is missing
        
    # Insert the new column name after Sequence_Time_ms
    cols.insert(insert_idx, 'Time_ms')
    
    # Use .assign() to efficiently add the new Time_ms column, 
    # then immediately select the columns in the desired order [cols].
    df = df.assign(Time_ms=time_ms_series)[cols]


    # Run fault detection and saving
    return _apply_fault_checks_and_save(df, raw_path, sensor_cols)


# ====================================================================
# SHARED FAULT CHECK AND SAVE LOGIC
# ====================================================================

def check_stuck_at_zero_live(df: pd.DataFrame, sensor_cols: list) -> pd.DataFrame:
    """
    Analyzes the sensor data (df) for the CRITICAL: STUCK_AT_ZERO_99PCT fault.
    Returns a DataFrame of faults with columns ['Channel', 'Fault_Type'].
    """
    fault_log = []
    ZERO_READING_THRESHOLD = 0.99 
    
    for col in sensor_cols:
        values = df[col]
        if values.empty:
            continue
            
        values = pd.to_numeric(values, errors='coerce').dropna()
        total_count = len(values)
        
        if total_count == 0:
            continue

        # Check: Channel stuck at 0 Ω (if 99% of readings are 0)
        zero_count = (values == 0).sum()
        if (zero_count / total_count) >= ZERO_READING_THRESHOLD:
            fault_log.append({'Channel': col, 'Fault_Type': "CRITICAL: STUCK_AT_ZERO_99PCT"})

    return pd.DataFrame(fault_log, columns=['Channel', 'Fault_Type'])


def check_temperature_quality_live(df: pd.DataFrame, sensor_cols: list) -> pd.DataFrame:
    """
    Analyzes the temperature column for stability and target adherence.
    Returns a DataFrame of faults with columns ['Channel', 'Fault_Type'].
    The fault is applied to all sensor channels if the temperature check fails.
    """
    fault_log = []
    
    # Constants from your provided logic
    STABILITY_MAX_STD_C = 5.0      # Maximum acceptable standard deviation
    STABILIZATION_TARGET_C = 60.0  # Target mean temperature for heated runs
    HIGH_TEMP_RUN_THRESHOLD = 30.0 # Threshold to assume a run was intended to be heated

    # Check for the presence of the temperature column
    if 'Temperature_C' not in df.columns or df['Temperature_C'].empty:
        for col in sensor_cols:
            fault_log.append({'Channel': col, 'Fault_Type': "DATA_QUALITY_FAIL: MISSING_TEMP_DATA"})
        return pd.DataFrame(fault_log, columns=['Channel', 'Fault_Type'])

    temp_series = pd.to_numeric(df['Temperature_C'], errors='coerce').dropna()
    
    if temp_series.empty:
        for col in sensor_cols:
            fault_log.append({'Channel': col, 'Fault_Type': "DATA_QUALITY_FAIL: MISSING_TEMP_DATA"})
        return pd.DataFrame(fault_log, columns=['Channel', 'Fault_Type'])

    mean_temp = temp_series.mean()
    std_temp = temp_series.std()
    
    # 1. Check for stability (applies to all runs)
    if std_temp > STABILITY_MAX_STD_C:
        for col in sensor_cols:
            fault_log.append({'Channel': col, 'Fault_Type': f"DATA_QUALITY_FAIL: T_UNSTABLE (STD: {std_temp:.2f}C > {STABILITY_MAX_STD_C:.1f}C)"})

    # 2. Check for target mean (only if the run was attempting to be heated)
    is_unstable = any("T_UNSTABLE" in f['Fault_Type'] for f in fault_log)
    
    if mean_temp >= HIGH_TEMP_RUN_THRESHOLD and not is_unstable:
        if mean_temp < STABILIZATION_TARGET_C:
            for col in sensor_cols:
                # Only add if not already marked unstable to avoid redundant errors
                fault_log.append({'Channel': col, 'Fault_Type': f"DATA_QUALITY_FAIL: T_STAB_TOO_LOW (Mean: {mean_temp:.1f}C < Target: {STABILIZATION_TARGET_C:.1f}C)"})

    # We only return the unique faults to avoid excessive log entries, 
    # though the analysis will use unique channel count anyway.
    return pd.DataFrame(fault_log, columns=['Channel', 'Fault_Type'])


def _apply_fault_checks_and_save(df: pd.DataFrame, raw_path: Path, sensor_cols: list[str]) -> tuple[Path, list[dict]]:
    """
    Applies the two active fault checks (Stuck-at-Zero and Temperature Quality) 
    by calling external functions, and then saves the cleaned file.
    """
    
    # 1. Execute Fault Checks using modular functions
    df_stuck_at_zero = check_stuck_at_zero_live(df, sensor_cols)
    df_temp_quality = check_temperature_quality_live(df, sensor_cols)
    
    # 2. Combine the fault results into a single DataFrame
    df_faults = pd.concat([df_stuck_at_zero, df_temp_quality], ignore_index=True)
    
    # Drop duplicates in case a channel has the same fault type logged twice
    df_faults = df_faults.drop_duplicates(subset=['Channel', 'Fault_Type'])
    
    # 3. Save cleaned file
    output_dir = raw_path.parent / "cleaned"
    output_dir.mkdir(exist_ok=True)
    cleaned_path = output_dir / f"{raw_path.stem}_CLEANED.xlsx"
    df.to_excel(cleaned_path, index=False)

    # 4. Structure fault data for return
    sample_name = raw_path.stem
    structured_faults = []
    
    if not df_faults.empty:
        # Convert the DataFrame rows into the required list of dicts format
        structured_faults = df_faults.apply(
            lambda row: {
                "Sample_Name": sample_name,
                "Channel": row['Channel'],
                "Fault_Type": row['Fault_Type']
            }, axis=1
        ).tolist()
    
    print(f"Cleaned: {cleaned_path.name}")
    if structured_faults:
        # Calculate unique channels from the combined fault DataFrame
        unique_channels = df_faults['Channel'].nunique() 
        print(f"   Faulty channels detected: {unique_channels} → will be aggregated in final report.")
    
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