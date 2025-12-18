import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy.signal import welch

def classify_channel_quality(raw_data: pd.Series, corrected_data: pd.Series,
                             baseline_start_idx: int, baseline_end_idx: int,
                             signal_start_idx: int = None) -> Dict:
    """
    Comprehensive channel quality assessment.
    
    Categories:
    - EXCELLENT: Clean baseline, good SNR, no artifacts
    - GOOD: Acceptable baseline, usable for analysis  
    - MARGINAL: Noisy but might contain useful signal
    - POOR: High noise, questionable utility
    - DEAD: No response or saturated
    
    Returns dict with classification and detailed metrics.
    """
    
    # ===== 1. BASELINE QUALITY =====
    baseline_corrected = corrected_data.loc[baseline_start_idx:baseline_end_idx]
    baseline_mean = baseline_corrected.mean()
    baseline_std = baseline_corrected.std()
    baseline_median = baseline_corrected.median()
    
    # Check if baseline is flat near zero
    baseline_centered = abs(baseline_mean) < 0.02 and abs(baseline_median) < 0.02
    
    # ===== 2. SIGNAL RANGE ANALYSIS =====
    signal_range = raw_data.max() - raw_data.min()
    corrected_range = corrected_data.max() - corrected_data.min()
    
    # Relative noise level
    noise_to_signal_ratio = baseline_std / (signal_range + 1e-10)
    
    # ===== 3. RESISTANCE RANGE CHECK =====
    mean_resistance = raw_data.mean()
    
    # Andrea's criteria: 1kΩ to 1MΩ is good
    in_good_resistance_range = 1e3 <= mean_resistance <= 1e6
    too_low = mean_resistance < 1e3
    too_high = mean_resistance > 1e6
    
    # ===== 4. HIGH-FREQUENCY NOISE DETECTION =====
    # Use power spectral density to detect oscillatory noise
    baseline_raw = raw_data.loc[baseline_start_idx:baseline_end_idx].values
    
    if len(baseline_raw) > 50:
        try:
            freqs, psd = welch(baseline_raw, fs=1.0, nperseg=min(len(baseline_raw)//4, 256))
            # Check if high-frequency components dominate
            total_power = np.sum(psd)
            high_freq_power = np.sum(psd[freqs > 0.1])  # Freq > 0.1 cycles/sample
            high_freq_ratio = high_freq_power / (total_power + 1e-10)
        except:
            high_freq_ratio = 0.0
    else:
        high_freq_ratio = 0.0
    
    has_high_freq_noise = high_freq_ratio > 0.6  # >60% power in high frequencies
    
    # ===== 5. DRIFT DETECTION =====
    # Check if there's a persistent trend even after correction
    if signal_start_idx is not None and signal_start_idx < len(corrected_data):
        post_baseline = corrected_data.loc[signal_start_idx:]
        
        # Fit linear trend to post-baseline region
        if len(post_baseline) > 10:
            x = np.arange(len(post_baseline))
            try:
                slope = np.polyfit(x, post_baseline.values, 1)[0]
                drift_rate = abs(slope)
            except:
                drift_rate = 0.0
        else:
            drift_rate = 0.0
    else:
        drift_rate = 0.0
    
    has_significant_drift = drift_rate > 0.001  # Drift > 0.001 Ω/sample
    
    # ===== 6. DEAD CHANNEL DETECTION =====
    is_dead = signal_range < 0.1  # Less than 0.1Ω total variation
    is_saturated = signal_range > 1e4  # More than 10kΩ variation (likely bad)
    
    # ===== 7. OVERALL CLASSIFICATION =====
    
    if is_dead:
        quality = "DEAD"
        usability = "EXCLUDE"
        reason = "No measurable signal variation"
        
    elif is_saturated:
        quality = "DEAD"
        usability = "EXCLUDE"
        reason = "Sensor saturated or unstable"
        
    elif too_low:
        quality = "POOR"
        usability = "EXCLUDE"
        reason = f"Resistance too low ({mean_resistance:.0f}Ω < 1kΩ) - not sensitive"
        
    elif too_high:
        quality = "POOR"
        usability = "EXCLUDE"
        reason = f"Resistance too high ({mean_resistance:.2e}Ω > 1MΩ) - not sensitive"
        
    elif has_high_freq_noise and noise_to_signal_ratio > 0.3:
        quality = "POOR"
        usability = "MARGINAL"
        reason = "Persistent high-frequency noise contaminates signal"
        
    elif has_significant_drift and not baseline_centered:
        quality = "POOR"
        usability = "MARGINAL"
        reason = "Significant drift persists after baseline correction"
        
    elif not baseline_centered and noise_to_signal_ratio > 0.2:
        quality = "MARGINAL"
        usability = "CAUTION"
        reason = "Baseline correction incomplete, moderate noise"
        
    elif baseline_centered and noise_to_signal_ratio < 0.05:
        quality = "EXCELLENT"
        usability = "INCLUDE"
        reason = "Clean baseline, low noise, good SNR"
        
    elif baseline_centered and noise_to_signal_ratio < 0.15:
        quality = "GOOD"
        usability = "INCLUDE"
        reason = "Acceptable baseline and noise levels"
        
    else:
        quality = "MARGINAL"
        usability = "CAUTION"
        reason = "Borderline quality - review carefully"
    
    # ===== 8. RETURN COMPREHENSIVE METRICS =====
    return {
        'quality': quality,
        'usability': usability,
        'reason': reason,
        'metrics': {
            'baseline_mean': float(baseline_mean),
            'baseline_std': float(baseline_std),
            'baseline_centered': baseline_centered,
            'signal_range': float(signal_range),
            'noise_to_signal': float(noise_to_signal_ratio),
            'mean_resistance': float(mean_resistance),
            'in_good_range': in_good_resistance_range,
            'high_freq_noise_ratio': float(high_freq_ratio),
            'has_high_freq_noise': has_high_freq_noise,
            'drift_rate': float(drift_rate),
            'has_drift': has_significant_drift,
        }
    }


def generate_quality_report(df: pd.DataFrame, sensor_cols: list,
                            baseline_start_idx: int, baseline_end_idx: int,
                            correction_func) -> pd.DataFrame:
    """
    Generate a quality report for all channels.
    
    Returns DataFrame with quality classifications and recommendations.
    """
    results = []
    
    for sensor_col in sensor_cols:
        raw_data = df[sensor_col]
        
        # Apply baseline correction
        try:
            corrected_data = correction_func(raw_data, baseline_start_idx, baseline_end_idx)
            
            # Classify quality
            quality_info = classify_channel_quality(
                raw_data, corrected_data,
                baseline_start_idx, baseline_end_idx,
                signal_start_idx=baseline_end_idx + 1
            )
            
            results.append({
                'Channel': sensor_col,
                'Quality': quality_info['quality'],
                'Usability': quality_info['usability'],
                'Reason': quality_info['reason'],
                'Baseline_Std': quality_info['metrics']['baseline_std'],
                'Noise_to_Signal': quality_info['metrics']['noise_to_signal'],
                'Mean_Resistance': quality_info['metrics']['mean_resistance'],
                'High_Freq_Noise': quality_info['metrics']['high_freq_noise_ratio'],
                'Has_Drift': quality_info['metrics']['has_drift'],
            })
            
        except Exception as e:
            results.append({
                'Channel': sensor_col,
                'Quality': 'ERROR',
                'Usability': 'EXCLUDE',
                'Reason': f'Processing failed: {str(e)}',
                'Baseline_Std': np.nan,
                'Noise_to_Signal': np.nan,
                'Mean_Resistance': np.nan,
                'High_Freq_Noise': np.nan,
                'Has_Drift': np.nan,
            })
    
    return pd.DataFrame(results)


# ===== EXAMPLE USAGE =====
if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    # Import your baseline correction
    sys.path.append('.')
    from robust_baseline import CORRECTION_MODALITIES_ROBUST
    from analysis import (
        identify_sensor_columns,
        get_taurus_baseline_indices,
        get_variable_baseline_indices
    )
    
    # Load data
    cleaned_file = Path("historical_reference_data/Box_A_B_Test/cleaned") / \
                   "UNKNOWN950425_051942_0001VRP_BOX_A_TEST_CLEANED.xlsx"
    
    df = pd.read_excel(cleaned_file)
    
    # Detect box type and get baseline
    first_col = str(df.columns[0])
    box_type = "Taurus" if first_col == 'Seq' else "Variable"
    sensor_cols = identify_sensor_columns(df, box_type)
    
    if box_type == "Taurus":
        baseline_start, baseline_end = get_taurus_baseline_indices(df, cleaned_file.name)
    else:
        baseline_start, baseline_end = get_variable_baseline_indices(df, cleaned_file.name)
    
    # Generate quality report
    quality_df = generate_quality_report(
        df, sensor_cols,
        baseline_start, baseline_end,
        CORRECTION_MODALITIES_ROBUST["Linear_Fit"]
    )
    
    # Display results
    print("\n" + "="*80)
    print("CHANNEL QUALITY REPORT")
    print("="*80)
    
    for category in ['EXCELLENT', 'GOOD', 'MARGINAL', 'POOR', 'DEAD', 'ERROR']:
        channels = quality_df[quality_df['Quality'] == category]
        if len(channels) > 0:
            print(f"\n{category} ({len(channels)} channels):")
            for _, row in channels.iterrows():
                print(f"  {row['Channel']:6s} | {row['Usability']:8s} | {row['Reason']}")
    
    print("\n" + "="*80)
    
    # Summary statistics
    print("\nSUMMARY:")
    print(f"  INCLUDE:  {len(quality_df[quality_df['Usability'] == 'INCLUDE'])} channels")
    print(f"  CAUTION:  {len(quality_df[quality_df['Usability'] == 'CAUTION'])} channels")
    print(f"  MARGINAL: {len(quality_df[quality_df['Usability'] == 'MARGINAL'])} channels")
    print(f"  EXCLUDE:  {len(quality_df[quality_df['Usability'] == 'EXCLUDE'])} channels")
    
    # Save report
    output_file = "channel_quality_report.csv"
    quality_df.to_csv(output_file, index=False)
    print(f"\n✓ Full report saved to: {output_file}")