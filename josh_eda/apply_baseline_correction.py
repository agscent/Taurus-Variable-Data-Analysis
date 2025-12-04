"""
Apply baseline correction to INCLUDE channels only.
Creates new datasets with corrected data for downstream analysis.
"""

import sys
import importlib

# CRITICAL: Force reload modules to prevent using cached old code
for module_name in ['analysis', 'robust_baseline', 'channel_quality_classifier']:
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

from analysis import (
    identify_sensor_columns,
    get_taurus_baseline_indices,
    get_variable_baseline_indices,
    determine_box_type
)

from robust_baseline import CORRECTION_MODALITIES_ROBUST

from channel_quality_classifier import classify_channel_quality


def apply_baseline_correction_to_sample(df: pd.DataFrame, sensor_cols: list,
                                       baseline_start_idx: int, baseline_end_idx: int,
                                       quality_threshold: str = 'INCLUDE',
                                       fit_method: str = 'Linear_Fit'):
    """
    Apply baseline correction only to high-quality channels.
    
    Args:
        df: Input dataframe
        sensor_cols: List of sensor column names
        baseline_start_idx: Start of baseline region
        baseline_end_idx: End of baseline region
        quality_threshold: 'INCLUDE' (EXCELLENT+GOOD) or 'CAUTION' (includes MARGINAL)
        fit_method: 'Constant_Median', 'Linear_Fit', or 'Exponential_Fit'
    
    Returns:
        corrected_df: DataFrame with corrected channels
        quality_info: Dict with channel quality classifications
    """
    
    # Copy original dataframe (preserve non-sensor columns)
    corrected_df = df.copy()
    
    # Track quality for each channel
    quality_info = {
        'included_channels': [],
        'excluded_channels': [],
        'channel_details': {}
    }
    
    correction_func = CORRECTION_MODALITIES_ROBUST[fit_method]
    
    for sensor_col in sensor_cols:
        raw_data = df[sensor_col]
        
        try:
            # Apply baseline correction
            corrected_data = correction_func(raw_data, baseline_start_idx, baseline_end_idx)
            
            # Classify channel quality
            quality = classify_channel_quality(
                raw_data, corrected_data,
                baseline_start_idx, baseline_end_idx,
                signal_start_idx=baseline_end_idx + 1
            )
            
            usability = quality['usability']
            
            # Store quality info
            quality_info['channel_details'][sensor_col] = {
                'quality': quality['quality'],
                'usability': usability,
                'reason': quality['reason'],
                'metrics': quality['metrics']
            }
            
            # Decide whether to include channel
            if quality_threshold == 'INCLUDE':
                include = (usability == 'INCLUDE')
            elif quality_threshold == 'CAUTION':
                include = (usability in ['INCLUDE', 'CAUTION'])
            elif quality_threshold == 'ALL':
                include = True  # Include all channels regardless of quality
            else:
                raise ValueError(f"Unknown quality_threshold: {quality_threshold}")
            
            if include:
                # Replace with corrected data
                corrected_df[sensor_col] = corrected_data
                quality_info['included_channels'].append(sensor_col)
            else:
                # Mark as excluded (set to NaN or keep original - your choice)
                # Option 1: Set to NaN to clearly mark as excluded
                corrected_df[sensor_col] = np.nan
                
                # Option 2: Keep original data but flag it
                # corrected_df[sensor_col] = raw_data  
                
                quality_info['excluded_channels'].append(sensor_col)
        
        except Exception as e:
            # If correction fails, mark as excluded
            corrected_df[sensor_col] = np.nan
            quality_info['excluded_channels'].append(sensor_col)
            quality_info['channel_details'][sensor_col] = {
                'quality': 'ERROR',
                'usability': 'EXCLUDE',
                'reason': f'Processing failed: {str(e)}',
                'metrics': {}
            }
    
    return corrected_df, quality_info


def batch_apply_baseline_correction(root_dir: Path = Path("historical_reference_data"),
                                    output_dir: Path = Path("baseline_corrected_data"),
                                    quality_threshold: str = 'INCLUDE',
                                    fit_method: str = 'Linear_Fit',
                                    save_quality_reports: bool = True):
    """
    Apply baseline correction to all samples, saving only high-quality channels.
    
    Args:
        root_dir: Directory containing cleaned data
        output_dir: Directory to save corrected data
        quality_threshold: 'INCLUDE', 'CAUTION', or 'ALL'
        fit_method: Baseline correction method to use
        save_quality_reports: Whether to save per-sample quality reports
    """
    
    output_dir.mkdir(exist_ok=True)
    
    if save_quality_reports:
        reports_dir = output_dir / "quality_reports"
        reports_dir.mkdir(exist_ok=True)
    
    # Find all cleaned files
    cleaned_files = list(root_dir.rglob("*_CLEANED.xlsx"))
    cleaned_files = [f for f in cleaned_files if not f.name.startswith('~$')]
    
    if not cleaned_files:
        print(f"❌ No cleaned files found in {root_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"APPLYING BASELINE CORRECTION")
    print(f"{'='*80}")
    print(f"Method: {fit_method}")
    print(f"Quality threshold: {quality_threshold}")
    print(f"Files to process: {len(cleaned_files)}\n")
    
    summary_stats = []
    
    for cleaned_file in tqdm(cleaned_files, desc="Processing"):
        try:
            # Load data
            df = pd.read_excel(cleaned_file)
            
            # Determine box type
            box_type = determine_box_type(cleaned_file)
            if box_type is None:
                print(f"  ⚠ Skipping {cleaned_file.name}: Could not determine box type")
                continue
            
            # Get sensor columns
            sensor_cols = identify_sensor_columns(df, box_type)
            if not sensor_cols:
                print(f"  ⚠ Skipping {cleaned_file.name}: No sensor columns found")
                continue
            
            # Get baseline indices
            if box_type == "Taurus":
                baseline_start, baseline_end = get_taurus_baseline_indices(df, cleaned_file.name)
            else:  # Variable box
                baseline_start, baseline_end = get_variable_baseline_indices(df, cleaned_file.name)
            
            # Debug: Print baseline info
            print(f"  {cleaned_file.name[:60]:60s} | {box_type:8s} | baseline: {baseline_start:3d}-{baseline_end:3d} / {len(df):3d} rows")
            
            # Apply baseline correction
            corrected_df, quality_info = apply_baseline_correction_to_sample(
                df, sensor_cols,
                baseline_start, baseline_end,
                quality_threshold=quality_threshold,
                fit_method=fit_method
            )
            
            # Save corrected data
            output_file = output_dir / cleaned_file.name.replace('_CLEANED.xlsx', '_CORRECTED.xlsx')
            corrected_df.to_excel(output_file, index=False)
            
            # Save quality report for this sample
            if save_quality_reports:
                quality_report = []
                for channel, info in quality_info['channel_details'].items():
                    quality_report.append({
                        'Channel': channel,
                        'Quality': info['quality'],
                        'Usability': info['usability'],
                        'Included': channel in quality_info['included_channels'],
                        'Reason': info['reason'],
                        **info['metrics']
                    })
                
                quality_df = pd.DataFrame(quality_report)
                report_file = reports_dir / cleaned_file.name.replace('_CLEANED.xlsx', '_quality.csv')
                quality_df.to_csv(report_file, index=False)
            
            # Track summary stats
            summary_stats.append({
                'Sample': cleaned_file.stem.replace('_CLEANED', ''),
                'Box_Type': box_type,
                'Total_Channels': len(sensor_cols),
                'Included_Channels': len(quality_info['included_channels']),
                'Excluded_Channels': len(quality_info['excluded_channels']),
                'Inclusion_Rate_%': round(
                    len(quality_info['included_channels']) / len(sensor_cols) * 100, 1
                )
            })
            
        except Exception as e:
            print(f"  ❌ Error processing {cleaned_file.name}: {e}")
            continue
    
    # Save summary statistics
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        summary_file = output_dir / "CORRECTION_SUMMARY.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\n{'='*80}")
        print("BASELINE CORRECTION COMPLETE")
        print(f"{'='*80}")
        
        total_channels = summary_df['Total_Channels'].sum()
        total_included = summary_df['Included_Channels'].sum()
        total_excluded = summary_df['Excluded_Channels'].sum()
        
        print(f"\n✓ Processed: {len(summary_df)} samples")
        print(f"✓ Total channels: {total_channels}")
        print(f"✓ Included channels: {total_included} ({total_included/total_channels*100:.1f}%)")
        print(f"✗ Excluded channels: {total_excluded} ({total_excluded/total_channels*100:.1f}%)")
        
        print(f"\n📂 Corrected data saved to: {output_dir.resolve()}")
        print(f"📊 Summary saved to: {summary_file}")
        
        if save_quality_reports:
            print(f"📋 Quality reports saved to: {reports_dir.resolve()}")
        
        # Show samples with most exclusions
        print("\n⚠ SAMPLES WITH MOST EXCLUSIONS:")
        worst_samples = summary_df.nsmallest(5, 'Inclusion_Rate_%')
        for _, row in worst_samples.iterrows():
            print(f"  {row['Sample'][:60]:60s} | {row['Inclusion_Rate_%']:5.1f}% included | "
                  f"{row['Excluded_Channels']}/{row['Total_Channels']} excluded")
        
        return summary_df
    
    return None


def compare_original_vs_corrected(original_file: Path, corrected_file: Path,
                                  sensor_col: str, output_dir: Path):
    """
    Create side-by-side comparison plot of original vs corrected data for a single channel.
    Useful for quality control and verification.
    """
    import matplotlib.pyplot as plt
    
    df_orig = pd.read_excel(original_file)
    df_corr = pd.read_excel(corrected_file)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Original vs Corrected: {sensor_col}\n{original_file.stem}', 
                 fontsize=14, fontweight='bold')
    
    # Original
    axes[0].plot(df_orig.index, df_orig[sensor_col], 'b-', linewidth=1, alpha=0.8)
    axes[0].set_title('Original Data', fontweight='bold')
    axes[0].set_ylabel('Resistance (Ω)')
    axes[0].set_xlabel('Index')
    axes[0].grid(True, alpha=0.3)
    
    # Corrected
    if not df_corr[sensor_col].isna().all():
        axes[1].plot(df_corr.index, df_corr[sensor_col], 'g-', linewidth=1, alpha=0.8)
        axes[1].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[1].set_title('Baseline Corrected', fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'Channel Excluded\n(Low Quality)', 
                    transform=axes[1].transAxes, ha='center', va='center',
                    fontsize=12, color='red')
        axes[1].set_title('Baseline Corrected (EXCLUDED)', fontweight='bold', color='red')
    
    axes[1].set_ylabel('Corrected Resistance (Ω)')
    axes[1].set_xlabel('Index')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / f"comparison_{sensor_col}_{original_file.stem}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply baseline correction to sensor data')
    parser.add_argument('--quality', type=str, default='INCLUDE',
                       choices=['INCLUDE', 'CAUTION', 'ALL'],
                       help='Quality threshold for including channels')
    parser.add_argument('--method', type=str, default='Linear_Fit',
                       choices=['Constant_Median', 'Linear_Fit', 'Exponential_Fit'],
                       help='Baseline correction method')
    parser.add_argument('--input', type=str, default='historical_reference_data',
                       help='Input directory with cleaned data')
    parser.add_argument('--output', type=str, default='baseline_corrected_data',
                       help='Output directory for corrected data')
    
    args = parser.parse_args()
    
    # Run batch correction
    summary = batch_apply_baseline_correction(
        root_dir=Path(args.input),
        output_dir=Path(args.output),
        quality_threshold=args.quality,
        fit_method=args.method,
        save_quality_reports=True
    )
    
    print("\n✅ DONE!")
    print("\nNext steps:")
    print("1. Review corrected data files in:", args.output)
    print("2. Check quality_reports/ for per-sample channel classifications")
    print("3. Proceed with peak detection and kinetics analysis on corrected data")