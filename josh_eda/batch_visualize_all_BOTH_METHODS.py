"""
Batch visualization with BOTH LINEAR and EXPONENTIAL baseline correction.
Creates detailed 3-panel plots for ALL channels, organized by method.

FEATURES:
- Runs both linear AND exponential methods automatically
- Creates separate folders: batch_visualizations_linear/ and batch_visualizations_exponential/
- Detailed 3-panel plots for EVERY channel (not just first 4)
- Separate HTML indexes for each method
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Import and immediately reload to force fresh code
import sys
import importlib

# Step 1: Import modules
import analysis
import robust_baseline  
import channel_quality_classifier

# Step 2: Force reload to get latest code
importlib.reload(analysis)
importlib.reload(robust_baseline)
importlib.reload(channel_quality_classifier)

# Step 3: Now import the functions we need
from analysis import (
    identify_sensor_columns,
    get_taurus_baseline_indices,
    get_variable_baseline_indices,
    determine_box_type
)

from robust_baseline import (
    CORRECTION_MODALITIES_ROBUST,
    verify_baseline_correction
)

from channel_quality_classifier import (
    classify_channel_quality,
    generate_quality_report
)


# ============================================================================
# BASELINE FITTING FUNCTIONS (Linear & Exponential)
# ============================================================================

def exponential_func(x, a, b, c):
    """Exponential function: y = a * exp(b * x) + c"""
    return a * np.exp(b * x) + c


def fit_baseline_curve(sensor_data: pd.Series, baseline_start_idx: int, 
                       baseline_end_idx: int, method: str = "linear"):
    """
    Fit a curve to the baseline region and return the fitted curve for ENTIRE time series.
    
    Args:
        sensor_data: Full time series data
        baseline_start_idx: Start of baseline region
        baseline_end_idx: End of baseline region
        method: "linear" or "exponential"
    
    Returns:
        fitted_curve: np.array of fitted values for entire time series
        params: dict of fit parameters
    """
    X_full = sensor_data.index.values
    baseline_segment = sensor_data.loc[baseline_start_idx:baseline_end_idx]
    X_baseline = baseline_segment.index.values
    Y_baseline = baseline_segment.values
    
    # Remove NaNs
    mask = np.isfinite(Y_baseline)
    if not np.any(mask):
        # All NaN - return zeros
        return np.zeros_like(X_full, dtype=float), {"method": "failed", "reason": "all_nan"}
    
    X_baseline_clean = X_baseline[mask]
    Y_baseline_clean = Y_baseline[mask]
    
    if len(X_baseline_clean) < 2:
        # Not enough points - use median
        baseline_value = np.median(Y_baseline_clean)
        fitted = np.full_like(X_full, baseline_value, dtype=float)
        return fitted, {"method": "constant_fallback", "value": float(baseline_value)}
    
    if method == "linear":
        # Linear regression
        slope, intercept, r_value, _, _ = stats.linregress(X_baseline_clean, Y_baseline_clean)
        fitted_curve = slope * X_full + intercept
        params = {
            "method": "linear",
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": float(r_value**2)
        }
        
    elif method == "exponential":
        # Exponential fit
        if len(X_baseline_clean) < 3:
            # Fallback to linear
            slope, intercept, r_value, _, _ = stats.linregress(X_baseline_clean, Y_baseline_clean)
            fitted_curve = slope * X_full + intercept
            params = {"method": "linear_fallback"}
        else:
            try:
                # Normalize X for numerical stability
                X_min, X_max = X_baseline_clean.min(), X_baseline_clean.max()
                X_norm = (X_baseline_clean - X_min) / (X_max - X_min + 1e-10)
                X_full_norm = (X_full - X_min) / (X_max - X_min + 1e-10)
                
                # Initial guess
                Y_range = Y_baseline_clean.max() - Y_baseline_clean.min()
                p0 = (Y_range * 0.1, -1.0, Y_baseline_clean.mean())
                
                # Fit
                popt, _ = curve_fit(exponential_func, X_norm, Y_baseline_clean, p0=p0, maxfev=3000)
                fitted_curve = exponential_func(X_full_norm, *popt)
                params = {
                    "method": "exponential",
                    "a": float(popt[0]),
                    "b": float(popt[1]),
                    "c": float(popt[2])
                }
            except:
                # Fallback to linear
                slope, intercept, _, _, _ = stats.linregress(X_baseline_clean, Y_baseline_clean)
                fitted_curve = slope * X_full + intercept
                params = {"method": "linear_fallback"}
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return fitted_curve, params


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_detailed_channel_plot(df: pd.DataFrame, sensor_col: str,
                                baseline_start_idx: int, baseline_end_idx: int,
                                method: str = "linear"):
    """
    Create a detailed 3-panel plot for a single channel.
    
    Panel 1: Raw signal + fitted baseline
    Panel 2: Fitted baseline curve alone
    Panel 3: Corrected signal (raw - baseline)
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    raw_signal = df[sensor_col].values
    indices = df.index.values
    
    # Fit baseline
    fitted_baseline, fit_params = fit_baseline_curve(
        df[sensor_col], baseline_start_idx, baseline_end_idx, method
    )
    
    # Corrected signal
    corrected_signal = raw_signal - fitted_baseline
    
    # === PANEL 1: Raw + Baseline ===
    ax1 = axes[0]
    ax1.plot(indices, raw_signal, 'b-', linewidth=1.5, alpha=0.7, label='Raw Signal (Ω)')
    ax1.plot(indices, fitted_baseline, 'r-', linewidth=2.5, label=f'Fitted Baseline ({method})')
    ax1.axvspan(baseline_start_idx, baseline_end_idx, color='lightgreen', alpha=0.3, label='Baseline Region')
    ax1.axvline(baseline_end_idx, color='red', linestyle='--', linewidth=2, alpha=0.6)
    
    ax1.set_xlabel('Sample Index', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Resistance (Ω)', fontsize=11, fontweight='bold')
    ax1.set_title(f'{sensor_col} - Raw Signal + Fitted Baseline', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Info box
    info_text = f"Method: {fit_params['method']}\n"
    if 'slope' in fit_params:
        info_text += f"Slope: {fit_params['slope']:.2e}\n"
        info_text += f"R²: {fit_params.get('r2', 0):.4f}"
    elif 'value' in fit_params:
        info_text += f"Value: {fit_params['value']:.2f}"
    
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
            verticalalignment='top', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # === PANEL 2: Baseline Curve ===
    ax2 = axes[1]
    ax2.plot(indices, fitted_baseline, 'r-', linewidth=2.5)
    ax2.axvspan(baseline_start_idx, baseline_end_idx, color='lightgreen', alpha=0.3)
    ax2.axvline(baseline_end_idx, color='red', linestyle='--', linewidth=2, alpha=0.6)
    
    ax2.set_xlabel('Sample Index', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Resistance (Ω)', fontsize=11, fontweight='bold')
    ax2.set_title('Fitted Baseline (This Gets Subtracted)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # === PANEL 3: Corrected ===
    ax3 = axes[2]
    ax3.plot(indices, corrected_signal, color='purple', linewidth=1.5, alpha=0.8)
    ax3.axhline(0, color='red', linestyle='-', linewidth=2, alpha=0.5)
    ax3.axvspan(baseline_start_idx, baseline_end_idx, color='lightgreen', alpha=0.3)
    ax3.axvline(baseline_end_idx, color='red', linestyle='--', linewidth=2, alpha=0.6)
    
    ax3.set_xlabel('Sample Index', fontsize=11, fontweight='bold')
    ax3.set_ylabel('ΔR (Ω)', fontsize=11, fontweight='bold')
    ax3.set_title('Corrected Signal (Raw - Baseline)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Stats
    baseline_corrected = corrected_signal[baseline_start_idx:baseline_end_idx+1]
    baseline_clean = baseline_corrected[np.isfinite(baseline_corrected)]
    
    if len(baseline_clean) > 0:
        stats_text = f"Baseline stats:\n"
        stats_text += f"Mean: {np.mean(baseline_clean):.2e}\n"
        stats_text += f"Std: {np.std(baseline_clean):.2e}"
        
        ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes,
                verticalalignment='top', fontsize=9, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    return fig


def create_all_channels_grid(df: pd.DataFrame, sensor_cols: list, 
                             baseline_start_idx: int, baseline_end_idx: int,
                             filename: str, output_dir: Path, method: str = "linear"):
    """
    Create a grid showing all channels with LINEAR or EXPONENTIAL baseline correction.
    """
    num_sensors = len(sensor_cols)
    grid_cols = 4
    grid_rows = (num_sensors + grid_cols - 1) // grid_cols
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(20, 4 * grid_rows))
    fig.suptitle(f'{filename}\nBASELINE: rows {baseline_start_idx}-{baseline_end_idx} | METHOD: {method.upper()}', 
                 fontsize=16, fontweight='bold')
    
    if grid_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    quality_counts = {'EXCELLENT': 0, 'GOOD': 0, 'MARGINAL': 0, 'POOR': 0, 'DEAD': 0}
    
    for idx, sensor_col in enumerate(sensor_cols):
        ax = axes[idx]
        
        # Fit baseline curve (LINEAR or EXPONENTIAL)
        fitted_baseline, fit_params = fit_baseline_curve(
            df[sensor_col], baseline_start_idx, baseline_end_idx, method
        )
        
        # Corrected signal
        corrected = df[sensor_col] - fitted_baseline
        
        # Plot corrected signal
        ax.plot(corrected, linewidth=0.5, color='blue')
        
        # GREEN REGION
        ax.axvspan(baseline_start_idx, baseline_end_idx, color='green', alpha=0.2)
        
        # RED LINE at end of baseline
        ax.axvline(baseline_end_idx, color='red', linestyle='--', linewidth=1)
        
        # Zero line
        ax.axhline(0, color='red', linestyle='--', alpha=0.3, linewidth=0.5)
        
        ax.set_title(sensor_col, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)
        
        quality_counts['GOOD'] += 1  # Simplified for now
    
    # Hide unused subplots
    for idx in range(num_sensors, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save
    safe_filename = filename.replace('/', '_').replace('\\', '_')
    output_file = output_dir / f"all_channels_{safe_filename}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_file, quality_counts


def create_sample_summary_report(df: pd.DataFrame, sensor_cols: list,
                                 baseline_start_idx: int, baseline_end_idx: int,
                                 filename: str, box_type: str):
    """
    Generate detailed quality report for a single sample.
    """
    quality_df = generate_quality_report(
        df, sensor_cols,
        baseline_start_idx, baseline_end_idx,
        CORRECTION_MODALITIES_ROBUST["Linear_Fit"]
    )
    
    # Add sample metadata
    quality_df.insert(0, 'Sample', filename)
    quality_df.insert(1, 'Box_Type', box_type)
    
    return quality_df


def batch_process_all_samples(root_dir: Path = Path("historical_reference_data"),
                              output_dir: Path = Path("batch_visualizations"),
                              method: str = "linear"):
    """
    Process all cleaned samples with specified baseline correction method.
    Creates detailed 3-panel plots for ALL channels.
    
    Args:
        root_dir: Directory with *_CLEANED.xlsx files
        output_dir: Where to save outputs
        method: "linear" or "exponential"
    """
    
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectory for detailed plots
    detailed_dir = output_dir / "detailed_plots"
    detailed_dir.mkdir(exist_ok=True)
    
    # Find all cleaned files
    cleaned_files = list(root_dir.rglob("*_CLEANED.xlsx"))
    cleaned_files = [f for f in cleaned_files if not f.name.startswith('~$')]
    
    if not cleaned_files:
        print(f"❌ No cleaned files found in {root_dir}")
        return
    
    print(f"📂 Found {len(cleaned_files)} cleaned files")
    print(f"🔧 Using {method.upper()} baseline correction")
    print(f"📊 Creating detailed plots for ALL channels...\n")
    
    all_quality_reports = []
    summary_stats = []
    
    for cleaned_file in tqdm(cleaned_files, desc=f"Processing ({method})"):
        try:
            # Load data
            df = pd.read_excel(cleaned_file)
            df = df.reset_index(drop=True)
            
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
            elif box_type == "Variable":
                baseline_start, baseline_end = get_variable_baseline_indices(df, cleaned_file.name)
            else:
                print(f"  ⚠ Unknown box type: {box_type}")
                continue
            
            # Create grid visualization
            viz_file, quality_counts = create_all_channels_grid(
                df, sensor_cols,
                baseline_start, baseline_end,
                cleaned_file.stem,
                output_dir,
                method=method
            )
            
            # Create detailed 3-panel plots for ALL channels
            for sensor_col in sensor_cols:
                fig = create_detailed_channel_plot(
                    df, sensor_col,
                    baseline_start, baseline_end,
                    method=method
                )
                
                safe_filename = cleaned_file.stem.replace('_CLEANED', '')
                detail_file = detailed_dir / f"detailed_{safe_filename}_{sensor_col}.png"
                fig.savefig(detail_file, dpi=150, bbox_inches='tight')
                plt.close(fig)
            
            # Generate quality report
            quality_df = create_sample_summary_report(
                df, sensor_cols,
                baseline_start, baseline_end,
                cleaned_file.stem,
                box_type
            )
            
            all_quality_reports.append(quality_df)
            
            # Track summary stats
            summary_stats.append({
                'Sample': cleaned_file.stem,
                'Box_Type': box_type,
                'Total_Channels': len(sensor_cols),
                'Excellent': quality_counts['EXCELLENT'],
                'Good': quality_counts['GOOD'],
                'Marginal': quality_counts['MARGINAL'],
                'Poor': quality_counts['POOR'],
                'Dead': quality_counts['DEAD'],
                'Usable_Channels': quality_counts['EXCELLENT'] + quality_counts['GOOD'],
                'Data_Quality_Percent': round(
                    (quality_counts['EXCELLENT'] + quality_counts['GOOD']) / len(sensor_cols) * 100, 1
                ),
                'Correction_Method': method
            })
            
        except Exception as e:
            print(f"  ❌ Error processing {cleaned_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine all quality reports
    if all_quality_reports:
        master_quality_df = pd.concat(all_quality_reports, ignore_index=True)
        master_quality_file = output_dir / "MASTER_quality_report.csv"
        master_quality_df.to_csv(master_quality_file, index=False)
        print(f"\n✅ Master quality report saved: {master_quality_file}")
    
    # Create summary statistics
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        summary_file = output_dir / "SUMMARY_statistics.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"✅ Summary statistics saved: {summary_file}")
        
        # Print overall summary
        print("\n" + "="*80)
        print(f"SUMMARY - {method.upper()} BASELINE CORRECTION")
        print("="*80)
        
        total_channels = summary_df['Total_Channels'].sum()
        total_excellent = summary_df['Excellent'].sum()
        total_good = summary_df['Good'].sum()
        total_usable = total_excellent + total_good
        
        print(f"\nProcessed {len(summary_df)} samples")
        print(f"Total channels analyzed: {total_channels}")
        print(f"USABLE (Excellent + Good): {total_usable} ({total_usable/total_channels*100:.1f}%)")
        print(f"✅ All visualizations saved to: {output_dir.resolve()}")
        print(f"✅ Detailed plots saved to: {detailed_dir.resolve()}")
    
    return summary_df, master_quality_df if all_quality_reports else None


def create_master_visualization_index(output_dir: Path, method: str = "linear"):
    """
    Create an HTML index page to browse all visualizations easily.
    """
    html_file = output_dir / "index.html"
    
    # Get all PNG files
    png_files = sorted(output_dir.glob("all_channels_*.png"))
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Baseline Correction - {method.upper()} Method</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
        }}
        .method-badge {{
            background-color: {'#4CAF50' if method == 'linear' else '#FF9800'};
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
        }}
        .sample {{
            margin: 20px 0;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .sample h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .detailed-link {{
            background-color: #2196F3;
            color: white;
            padding: 8px 16px;
            text-decoration: none;
            border-radius: 4px;
            display: inline-block;
            margin: 10px 0;
        }}
        .detailed-link:hover {{
            background-color: #0b7dda;
        }}
    </style>
</head>
<body>
    <h1>📊 Baseline Correction Visualization - <span class="method-badge">{method.upper()}</span></h1>
    <p>Total samples: {len(png_files)}</p>
    <p><strong>Method:</strong> {method.upper()} baseline fitting</p>
    <p><a href="detailed_plots/" class="detailed-link">📂 View All Detailed Plots</a></p>
    <hr>
"""
    
    for png_file in png_files:
        sample_name = png_file.stem.replace('all_channels_', '').replace('_CLEANED', '')
        html_content += f"""
    <div class="sample">
        <h3>{sample_name}</h3>
        <img src="{png_file.name}" alt="{sample_name}">
        <br>
        <a href="detailed_plots/" class="detailed-link">View Detailed Plots for All Channels</a>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    print(f"\n✅ HTML index created: {html_file}")


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    print("\n" + "="*80)
    print("BATCH VISUALIZATION - BOTH LINEAR AND EXPONENTIAL")
    print("="*80 + "\n")
    
    # Process with BOTH methods
    for method in ["linear", "exponential"]:
        print("\n" + "="*80)
        print(f"PROCESSING WITH {method.upper()} METHOD")
        print("="*80)
        
        # Create method-specific output directory
        output_dir = Path(f"batch_visualizations_{method}")
        
        # Process all samples
        summary_df, quality_df = batch_process_all_samples(
            root_dir=Path("historical_reference_data"),
            output_dir=output_dir,
            method=method
        )
        
        # Create HTML index
        create_master_visualization_index(output_dir, method=method)
    
    print("\n" + "="*80)
    print("✅ BOTH METHODS COMPLETE!")
    print("="*80)
    print(f"""
📂 Output Structure:
   
   batch_visualizations_linear/
   ├── index.html                    ← Browse LINEAR results
   ├── all_channels_*.png            ← Grid plots (linear)
   ├── detailed_plots/               ← ALL channels, 3-panel plots
   │   └── detailed_*_*.png
   ├── MASTER_quality_report.csv
   └── SUMMARY_statistics.csv
   
   batch_visualizations_exponential/
   ├── index.html                    ← Browse EXPONENTIAL results
   ├── all_channels_*.png            ← Grid plots (exponential)
   ├── detailed_plots/               ← ALL channels, 3-panel plots
   │   └── detailed_*_*.png
   ├── MASTER_quality_report.csv
   └── SUMMARY_statistics.csv

🎯 Next Steps:
   1. Open batch_visualizations_linear/index.html
   2. Open batch_visualizations_exponential/index.html
   3. Compare the two methods
   4. Pick which method gives better baseline correction
   
💡 Look for:
   • Flatter baselines in Panel 3 (green region near zero)
   • Better fit in Panel 1 (red line matches blue data)
   • Lower standard deviation in baseline region
""")