import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# Import your correction functions and helpers from analysis.py
# Adjust the import path as needed
from analysis import (
    CORRECTION_MODALITIES, 
    linear_baseline_correction,
    exponential_baseline_correction,
    exponential_func,
    get_taurus_baseline_indices,
    get_variable_baseline_indices,
    identify_sensor_columns,
    fit_baseline, 
    subtract_baseline
)

def extract_fitted_curve(sensor_data: pd.Series, baseline_start_idx: int, baseline_end_idx: int, fit_type: str) -> pd.Series:
    """
    Extracts just the fitted curve (without subtracting it from raw data) for visualization.
    Returns a pd.Series with the same index as sensor_data for proper alignment.
    """
    baseline_segment = sensor_data.loc[baseline_start_idx:baseline_end_idx]
    X_baseline = baseline_segment.index.values
    Y_baseline = baseline_segment.values
    
    if fit_type == "Constant_Median":
        median_val = np.median(Y_baseline)
        fitted_curve = pd.Series(np.full_like(sensor_data.values, median_val, dtype=float), index=sensor_data.index)
        return fitted_curve
    
    elif fit_type == "Linear_Fit":
        if len(X_baseline) < 2:
            fitted_curve = pd.Series(np.full_like(sensor_data.values, np.median(Y_baseline), dtype=float), index=sensor_data.index)
            return fitted_curve
        slope, intercept, _, _, _ = stats.linregress(X_baseline, Y_baseline)
        fitted_values = slope * sensor_data.index.values + intercept
        fitted_curve = pd.Series(fitted_values, index=sensor_data.index)
        return fitted_curve
    
    elif fit_type == "Exponential_Fit":
        if len(X_baseline) < 3:
            fitted_curve = pd.Series(np.full_like(sensor_data.values, np.median(Y_baseline), dtype=float), index=sensor_data.index)
            return fitted_curve
        try:
            X_norm = (X_baseline - X_baseline.min()) / (X_baseline.max() - X_baseline.min() + 1e-10)
            X_full_norm = (sensor_data.index.values - X_baseline.min()) / (X_baseline.max() - X_baseline.min() + 1e-10)
            Y_range = Y_baseline.max() - Y_baseline.min()
            p0 = (Y_range * 0.1, -1.0, Y_baseline.mean())
            # Try exponential first
            try:
                from scipy.optimize import curve_fit
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    popt, _ = curve_fit(exponential_func, X_norm, Y_baseline, p0=p0, maxfev=3000, ftol=1e-3)
                fitted_values = exponential_func(X_full_norm, *popt)
                fitted_curve = pd.Series(fitted_values, index=sensor_data.index)
                return fitted_curve
            except:
                # Fallback to polynomial if exponential fails
                fitted_values = np.polyval(np.polyfit(X_baseline, Y_baseline, 2), sensor_data.index.values)
                fitted_curve = pd.Series(fitted_values, index=sensor_data.index)
                return fitted_curve
        except:
            fitted_curve = pd.Series(np.full_like(sensor_data.values, np.median(Y_baseline), dtype=float), index=sensor_data.index)
            return fitted_curve
    
    return pd.Series(np.full_like(sensor_data.values, np.median(Y_baseline), dtype=float), index=sensor_data.index)


def plot_single_sensor_comparison(df: pd.DataFrame, sensor_col: str, baseline_start_idx: int, 
                                   baseline_end_idx: int, filename: str = ""):
    """
    Plots a single sensor showing raw data, fitted curves, and corrected data for all fit types.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Sensor {sensor_col} Baseline Analysis\n{filename}', fontsize=14, fontweight='bold')
    
    raw_data = df[sensor_col]
    
    fit_types = list(CORRECTION_MODALITIES.keys())
    
    # Top-left: Raw data with all fitted curves overlaid
    ax = axes[0, 0]
    ax.plot(raw_data.index, raw_data.values, 'k-', linewidth=2, label='Raw Data', zorder=3)
    ax.axvspan(baseline_start_idx, baseline_end_idx, color='green', alpha=0.15, label='Baseline Region')
    
    colors = ['red', 'blue', 'orange']
    for i, fit_type in enumerate(fit_types):
        fitted_curve = extract_fitted_curve(raw_data, baseline_start_idx, baseline_end_idx, fit_type)
        ax.plot(raw_data.index, fitted_curve, '--', linewidth=2, color=colors[i], label=f'{fit_type} Fit', alpha=0.8)
    
    ax.set_title('Raw Data with Fitted Curves', fontweight='bold')
    ax.set_ylabel('Resistance (Ω)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Top-right: Residuals (corrected data) for each fit type
    ax = axes[0, 1]
    for i, fit_type in enumerate(fit_types):
        corrected = CORRECTION_MODALITIES[fit_type](raw_data, baseline_start_idx, baseline_end_idx)
        ax.plot(corrected.index, corrected.values, linewidth=1.5, color=colors[i], label=fit_type, alpha=0.8)
    
    ax.axvspan(baseline_start_idx, baseline_end_idx, color='green', alpha=0.15)
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_title('Corrected Data (Residuals)', fontweight='bold')
    ax.set_ylabel('Corrected Resistance (Ω)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Bottom-left: Baseline region zoomed in - raw data and fits
    ax = axes[1, 0]
    baseline_slice = raw_data.loc[baseline_start_idx:baseline_end_idx]
    ax.plot(baseline_slice.index, baseline_slice.values, 'ko-', linewidth=2, markersize=4, label='Raw Data', zorder=3)
    
    for i, fit_type in enumerate(fit_types):
        fitted_curve = extract_fitted_curve(raw_data, baseline_start_idx, baseline_end_idx, fit_type)
        baseline_fit_slice = fitted_curve.loc[baseline_start_idx:baseline_end_idx]
        ax.plot(baseline_fit_slice.index, baseline_fit_slice.values, 's--', linewidth=2, color=colors[i], label=fit_type, markersize=3)
    
    ax.set_title('Baseline Region (Zoomed)', fontweight='bold')
    ax.set_ylabel('Resistance (Ω)')
    ax.set_xlabel('Index')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Bottom-right: Baseline SD comparison (fit quality)
    ax = axes[1, 1]
    baseline_sds = []
    for fit_type in fit_types:
        corrected = CORRECTION_MODALITIES[fit_type](raw_data, baseline_start_idx, baseline_end_idx)
        baseline_residuals = corrected.loc[baseline_start_idx:baseline_end_idx]
        sd = baseline_residuals.std()
        baseline_sds.append(sd)
    
    bars = ax.bar(fit_types, baseline_sds, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_title('Baseline Fit Quality (Lower SD = Better)', fontweight='bold')
    ax.set_ylabel('Std Dev of Baseline Residuals (Ω)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2e}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_all_sensors_grid(df: pd.DataFrame, sensor_cols: list, baseline_start_idx: int, 
                          baseline_end_idx: int, filename: str = "", fit_type: str = "Linear_Fit"):
    """
    Creates a grid showing corrected data for all sensors using a single fit type.
    Useful for quick overview of data quality across all channels.
    """
    num_sensors = len(sensor_cols)
    grid_cols = 4
    grid_rows = (num_sensors + grid_cols - 1) // grid_cols
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(16, 3 * grid_rows))
    fig.suptitle(f'All Sensors - {fit_type} Correction\n{filename}', fontsize=14, fontweight='bold')
    
    # Flatten axes array for easier iteration
    axes = axes.flatten()
    
    for idx, sensor_col in enumerate(sensor_cols):
        ax = axes[idx]
        raw_data = df[sensor_col]
        corrected = CORRECTION_MODALITIES[fit_type](raw_data, baseline_start_idx, baseline_end_idx)
        
        ax.plot(corrected.index, corrected.values, linewidth=1, color='steelblue')
        ax.axvspan(baseline_start_idx, baseline_end_idx, color='green', alpha=0.15)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title(f'{sensor_col}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Calculate and display baseline SD
        baseline_residuals = corrected.loc[baseline_start_idx:baseline_end_idx]
        sd = baseline_residuals.std()
        ax.text(0.98, 0.98, f'SD: {sd:.2e}', transform=ax.transAxes, 
                ha='right', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=8)
    
    # Hide unused subplots
    for idx in range(num_sensors, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_baseline_modes(time, raw_signal):
    for mode in ["flat", "linear", "exp"]:
        baseline, info = fit_baseline(time, raw_signal, mode)
        corrected = subtract_baseline(raw_signal, baseline)

        plt.plot(time, corrected, label=f"{mode} (RMSE={info.get('rmse')})")

    plt.legend()
    plt.show()


# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    # Load a cleaned file
    cleaned_file = Path("historical_reference_data/Box_A_B_Test/cleaned") / "UNKNOWN950425_051942_0001VRP_BOX_A_TEST_CLEANED.xlsx"
    
    try:
        if not cleaned_file.exists():
            print(f"❌ File not found: {cleaned_file}")
            print(f"Please update the cleaned_file path in the script.")
            exit(1)
        
        print(f"📂 Loading file: {cleaned_file}")
        df = pd.read_excel(cleaned_file)
        print(f"✓ Loaded successfully. Shape: {df.shape}")
        
        # Determine box type and sensor columns
        first_col = str(df.columns[0])
        if first_col == 'Seq':
            box_type = "Taurus"
        else:
            box_type = "Variable"
        
        # Use the robust sensor column detection
        sensor_cols = identify_sensor_columns(df, box_type)
        
        # Get baseline indices based on box type
        if box_type == "Taurus":
            baseline_start_idx, baseline_end_idx = get_taurus_baseline_indices(df, cleaned_file.name)
        else:
            baseline_start_idx, baseline_end_idx = get_variable_baseline_indices(df, cleaned_file.name)
        
        print(f"\n📊 Analysis Summary:")
        print(f"  • Box Type: {box_type}")
        print(f"  • Sensor Columns: {sensor_cols}")
        print(f"  • Baseline Region: {baseline_start_idx} to {baseline_end_idx}")
        
        if not sensor_cols:
            print("❌ No sensor columns found!")
            exit(1)
        
        # Create output directory
        output_dir = Path("visualizations")
        output_dir.mkdir(exist_ok=True)
        
        # Plot detailed comparison for the first sensor
        print(f"\n📈 Creating single sensor comparison plot...")
        fig1 = plot_single_sensor_comparison(df, sensor_cols[0], baseline_start_idx, 
                                             baseline_end_idx, cleaned_file.name)
        output_file_1 = output_dir / f"sensor_comparison_{sensor_cols[0]}.png"
        plt.savefig(output_file_1, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_file_1.resolve()}")
        plt.close(fig1)
        
        # Plot grid of all sensors
        print(f"📈 Creating all sensors grid plot...")
        fig2 = plot_all_sensors_grid(df, sensor_cols, baseline_start_idx, 
                                    baseline_end_idx, cleaned_file.name, fit_type="Linear_Fit")
        output_file_2 = output_dir / f"all_sensors_grid_{Path(cleaned_file.name).stem}.png"
        plt.savefig(output_file_2, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_file_2.resolve()}")
        plt.close(fig2)
        
        print(f"\n✅ All plots saved to: {output_dir.resolve()}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

