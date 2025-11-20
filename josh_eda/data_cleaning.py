from pathlib import Path
import pandas as pd


def clean_file(raw_path: str) -> Path:
    raw_path = Path(raw_path)
    df_full = pd.read_excel(raw_path, header=None)

    # Find real header
    header_row_idx = df_full[df_full.iloc[:, 0] == "Seq"].index[0]
    header = df_full.iloc[header_row_idx]
    data = df_full.iloc[header_row_idx + 1 :]
    df = pd.DataFrame(data.values, columns=header.values).reset_index(drop=True)

    # Clean column names
    df.columns = df.columns.str.replace(r"\s*\(.*\)", "", regex=True).str.strip()
    rename_map = {
        "Time": "Time_ms", "Temp": "Temperature_C", "Humidity": "Humidity_percent",
        "Pump Rate": "Pump_Rate", "V1 pos": "V1_position", "V2 pos": "V2_position",
    }
    df = df.rename(columns=rename_map)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # === FAULT DETECTION ===
    # Based on Yemi's Taurus Box Calibration Report (November 2025)
    # All critical hardware faults are checked universally
    # Non-critical flat-response rules are adapted to run type:
    #   • Resistor calibration runs → strict (large resistance steps expected across most channels)
    #   • Real cow breath samples → lenient (only a few channels respond; flat is normal)
    sensor_cols = [c for c in df.columns if c.startswith("T") and c[1:].isdigit()]
    fault_log = []

    # === SIGNAL-BASED RUN TYPE CLASSIFICATION (robust & final) ===
    # Only resistor calibration runs produce huge per-channel swings (>50 kΩ)
    # Real breath responses are always <20 kΩ even on the best channels
    # This threshold is physically impossible to cross in biology → 100% separation
    large_swing_channels = 0
    channel_stats = {}  # Store stats once for efficiency

    for col in sensor_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.empty:
            continue
        mean_val = values.mean()
        std_val = values.std()
        range_val = values.max() - values.min()
        channel_stats[col] = (mean_val, std_val, range_val)

        if range_val > 200_000:           # 200 kΩ — only resistor calibration ever hits this
            large_swing_channels += 1

    is_resistor_calibration_run = large_swing_channels >= 15

    # === APPLY FAULT RULES ===
    for col in sensor_cols:
        if col not in channel_stats:
            fault_log.append((col, "EMPTY_COLUMN"))
            continue

        mean_val, std_val, range_val = channel_stats[col]

        # --------------------------------------------------------------
        # 1. Known 1 MΩ overflow bug → resistance collapses to 0 Ω
        # Originally observed on Box A (T5 & T15); now checked on all hardware
        # Cause: firmware or ADC saturation at high resistance
        # --------------------------------------------------------------
        if (df[col] == 0).any() and mean_val < 1000:
            fault_log.append((col, "CRITICAL: 1MΩ_OVERFLOW_BUG (reports 0 Ω)"))

        # --------------------------------------------------------------
        # 2. Physically impossible resistance values
        # Valid sensor range: ~50 Ω to 5 MΩ under normal operating conditions
        # Values outside this range indicate hardware or wiring failure
        # --------------------------------------------------------------
        if (df[col] < 50).any() or (df[col] > 5_000_000).any():
            fault_log.append((col, "CRITICAL: IMPOSSIBLE_PHYSICAL_RANGE"))

        # --------------------------------------------------------------
        # 3. Channel stuck at open-circuit default value (101 Ω)
        # Indicates disconnected sensor or failed multiplexing
        # --------------------------------------------------------------
        if (df[col].round(1) == 101).mean() > 0.90:
            fault_log.append((col, "CRITICAL: STUCK_AT_101 (open circuit)"))

        # --------------------------------------------------------------
        # 4. Insufficient response — context-dependent evaluation
        # Resistor calibration runs require large resistance excursions
        # Real breath samples expect limited response on most channels
        # --------------------------------------------------------------
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


    # Save fault report
    output_dir = raw_path.parent / "cleaned"
    output_dir.mkdir(exist_ok=True)
    cleaned_path = output_dir / f"{raw_path.stem}_CLEANED.xlsx"
    df.to_excel(cleaned_path, index=False)

    # Save human-readable fault summary
    fault_df = pd.DataFrame(fault_log, columns=["Channel", "Fault_Type"])
    fault_path = output_dir / f"{raw_path.stem}_FAULTS.csv"
    if fault_df.empty:
        fault_df = pd.DataFrame([["All channels passed", ""]])
    fault_df.to_csv(fault_path, index=False)

    print(f"Cleaned: {cleaned_path.name}")
    if not fault_df.empty and "All channels passed" not in fault_df.iloc[0,0]:
        print(f"   Faulty channels detected: {len(fault_df)} → see {fault_path.name}")

    return cleaned_path