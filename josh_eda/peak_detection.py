"""
Peak Detection and Analysis for Baseline-Corrected Sensor Data

This script:
1. Loads baseline-corrected data from baseline_corrected_data/
2. Identifies peaks (positive and negative)
3. Calculates peak properties (height, width, area, time to peak)
4. Visualizes detected peaks
5. Exports peak data for further analysis

Usage:
    python peak_detection.py
    
Or with options:
    python peak_detection.py --prominence 50 --width 3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks, peak_widths, peak_prominences
from tqdm import tqdm
import warnings
import sys
warnings.filterwarnings('ignore')


# ============================================================================
# PEAK DETECTION FUNCTIONS
# ============================================================================

def detect_peaks(signal: np.array, 
                prominence: float = 50,
                width: int = 3,
                distance: int = 5,
                detect_negative: bool = True):
    """
    Detect peaks in a baseline-corrected signal.
    
    Args:
        signal: 1D array of signal values (already baseline-corrected)
        prominence: Minimum prominence (vertical distance) of peaks
        width: Minimum width of peaks (in samples)
        distance: Minimum distance between peaks (in samples)
        detect_negative: If True, also detect negative peaks (troughs)
    
    Returns:
        dict with peak information for positive and negative peaks
    """
    results = {
        'positive_peaks': {},
        'negative_peaks': {}
    }
    
    # Detect POSITIVE peaks (increases in resistance)
    pos_peaks, pos_properties = find_peaks(
        signal,
        prominence=prominence,
        width=width,
        distance=distance
    )
    
    if len(pos_peaks) > 0:
        # Get peak prominences and widths
        prominences = peak_prominences(signal, pos_peaks)[0]
        widths_data = peak_widths(signal, pos_peaks, rel_height=0.5)
        
        results['positive_peaks'] = {
            'indices': pos_peaks,
            'heights': signal[pos_peaks],
            'prominences': prominences,
            'widths': widths_data[0],
            'width_heights': widths_data[1],
            'left_ips': widths_data[2],
            'right_ips': widths_data[3]
        }
    
    # Detect NEGATIVE peaks (decreases in resistance) if requested
    if detect_negative:
        neg_peaks, neg_properties = find_peaks(
            -signal,  # Invert signal to find troughs
            prominence=prominence,
            width=width,
            distance=distance
        )
        
        if len(neg_peaks) > 0:
            prominences = peak_prominences(-signal, neg_peaks)[0]
            widths_data = peak_widths(-signal, neg_peaks, rel_height=0.5)
            
            results['negative_peaks'] = {
                'indices': neg_peaks,
                'heights': signal[neg_peaks],  # Original (negative) values
                'prominences': prominences,
                'widths': widths_data[0],
                'width_heights': -widths_data[1],  # Convert back
                'left_ips': widths_data[2],
                'right_ips': widths_data[3]
            }
    
    return results


def calculate_peak_features(signal: np.array, peak_info: dict, 
                           baseline_end_idx: int):
    """
    Calculate additional features for detected peaks.
    
    Args:
        signal: Full signal array
        peak_info: Peak information from detect_peaks()
        baseline_end_idx: Index where baseline ends (breath starts)
    
    Returns:
        DataFrame with peak features
    """
    features = []
    
    for peak_type in ['positive_peaks', 'negative_peaks']:
        if not peak_info[peak_type]:
            continue
        
        peaks = peak_info[peak_type]
        
        for i, peak_idx in enumerate(peaks['indices']):
            # Time to peak (from end of baseline)
            time_to_peak = peak_idx - baseline_end_idx
            
            # Peak area (simple trapezoidal approximation)
            left_idx = int(peaks['left_ips'][i])
            right_idx = int(peaks['right_ips'][i])
            peak_area = np.trapz(signal[left_idx:right_idx+1])
            
            # Absolute peak area (for negative peaks)
            abs_peak_area = np.abs(peak_area)
            
            feature = {
                'peak_type': 'positive' if peak_type == 'positive_peaks' else 'negative',
                'peak_index': peak_idx,
                'peak_height': peaks['heights'][i],
                'peak_prominence': peaks['prominences'][i],
                'peak_width': peaks['widths'][i],
                'time_to_peak': time_to_peak,
                'peak_area': peak_area,
                'abs_peak_area': abs_peak_area,
                'left_bound': left_idx,
                'right_bound': right_idx,
                'relative_time': time_to_peak / len(signal) * 100  # % of total time
            }
            
            features.append(feature)
    
    return pd.DataFrame(features)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_peaks_detailed(signal: np.array, peak_info: dict, 
                       baseline_end_idx: int, channel_name: str,
                       sample_name: str, output_file: Path = None):
    """
    Create detailed visualization of detected peaks.
    
    Shows:
    - Full signal with baseline region marked
    - Detected positive peaks (red)
    - Detected negative peaks (blue)
    - Peak widths and prominences
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    
    indices = np.arange(len(signal))
    
    # Plot signal
    ax.plot(indices, signal, 'k-', linewidth=1, label='Corrected Signal', alpha=0.7)
    
    # Mark baseline region
    ax.axvspan(0, baseline_end_idx, color='lightgreen', alpha=0.2, label='Baseline')
    ax.axvline(baseline_end_idx, color='red', linestyle='--', linewidth=2, 
              alpha=0.5, label='Response Starts')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.3)
    
    # Plot POSITIVE peaks
    if peak_info['positive_peaks']:
        pos_peaks = peak_info['positive_peaks']
        ax.plot(pos_peaks['indices'], pos_peaks['heights'], 
               'r^', markersize=10, label=f'Positive Peaks (n={len(pos_peaks["indices"])})')
        
        # Plot peak widths
        for i in range(len(pos_peaks['indices'])):
            ax.hlines(
                y=pos_peaks['width_heights'][i],
                xmin=pos_peaks['left_ips'][i],
                xmax=pos_peaks['right_ips'][i],
                color='red', alpha=0.5, linewidth=2
            )
    
    # Plot NEGATIVE peaks
    if peak_info['negative_peaks']:
        neg_peaks = peak_info['negative_peaks']
        ax.plot(neg_peaks['indices'], neg_peaks['heights'], 
               'bv', markersize=10, label=f'Negative Peaks (n={len(neg_peaks["indices"])})')
        
        # Plot peak widths
        for i in range(len(neg_peaks['indices'])):
            ax.hlines(
                y=neg_peaks['width_heights'][i],
                xmin=neg_peaks['left_ips'][i],
                xmax=neg_peaks['right_ips'][i],
                color='blue', alpha=0.5, linewidth=2
            )
    
    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('ΔR (Ω)', fontsize=12, fontweight='bold')
    ax.set_title(f'{sample_name} - {channel_name}\nPeak Detection Results', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig


def create_peak_summary_plot(all_peaks_df: pd.DataFrame, output_dir: Path):
    """
    Create summary plots showing peak statistics across all samples/channels.
    """
    if len(all_peaks_df) == 0:
        print("  ⚠️ No peaks detected to summarize")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Peak Detection Summary - All Samples', fontsize=16, fontweight='bold')
    
    # 1. Peak height distribution
    ax = axes[0, 0]
    pos_peaks = all_peaks_df[all_peaks_df['peak_type'] == 'positive']
    neg_peaks = all_peaks_df[all_peaks_df['peak_type'] == 'negative']
    
    if len(pos_peaks) > 0:
        ax.hist(pos_peaks['peak_height'], bins=30, alpha=0.6, color='red', label='Positive')
    if len(neg_peaks) > 0:
        ax.hist(neg_peaks['peak_height'], bins=30, alpha=0.6, color='blue', label='Negative')
    ax.set_xlabel('Peak Height (Ω)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Peak Height Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Peak prominence distribution
    ax = axes[0, 1]
    if len(pos_peaks) > 0:
        ax.hist(pos_peaks['peak_prominence'], bins=30, alpha=0.6, color='red', label='Positive')
    if len(neg_peaks) > 0:
        ax.hist(neg_peaks['peak_prominence'], bins=30, alpha=0.6, color='blue', label='Negative')
    ax.set_xlabel('Peak Prominence (Ω)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Peak Prominence Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Time to peak distribution
    ax = axes[0, 2]
    if len(pos_peaks) > 0:
        ax.hist(pos_peaks['time_to_peak'], bins=30, alpha=0.6, color='red', label='Positive')
    if len(neg_peaks) > 0:
        ax.hist(neg_peaks['time_to_peak'], bins=30, alpha=0.6, color='blue', label='Negative')
    ax.set_xlabel('Time to Peak (samples)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Time to Peak Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Peak width distribution
    ax = axes[1, 0]
    if len(pos_peaks) > 0:
        ax.hist(pos_peaks['peak_width'], bins=30, alpha=0.6, color='red', label='Positive')
    if len(neg_peaks) > 0:
        ax.hist(neg_peaks['peak_width'], bins=30, alpha=0.6, color='blue', label='Negative')
    ax.set_xlabel('Peak Width (samples)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Peak Width Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Peak area distribution
    ax = axes[1, 1]
    if len(pos_peaks) > 0:
        ax.hist(pos_peaks['abs_peak_area'], bins=30, alpha=0.6, color='red', label='Positive')
    if len(neg_peaks) > 0:
        ax.hist(neg_peaks['abs_peak_area'], bins=30, alpha=0.6, color='blue', label='Negative')
    ax.set_xlabel('Peak Area (Ω·samples)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Peak Area Distribution (Absolute)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Peaks per sample
    ax = axes[1, 2]
    peaks_per_sample = all_peaks_df.groupby('sample').size()
    ax.bar(range(len(peaks_per_sample)), peaks_per_sample.values, color='green', alpha=0.7)
    ax.set_xlabel('Sample Index', fontweight='bold')
    ax.set_ylabel('Number of Peaks', fontweight='bold')
    ax.set_title('Peaks Detected per Sample')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_file = output_dir / "peak_detection_summary.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✅ Summary plot saved: {output_file}")


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_all_corrected_data(
    input_dir: Path = Path("baseline_corrected_data"),
    output_dir: Path = Path("peak_detection_results"),
    prominence: float = 50,
    width: int = 3,
    distance: int = 5,
    create_plots: bool = True,
    plot_limit: int = 100
):
    """
    Process all baseline-corrected files and detect peaks.
    
    Args:
        input_dir: Directory with *_CORRECTED.xlsx files
        output_dir: Where to save results
        prominence: Minimum peak prominence
        width: Minimum peak width (samples)
        distance: Minimum distance between peaks (samples)
        create_plots: Whether to create individual peak plots
        plot_limit: Maximum number of channel plots to create (to avoid too many files)
    """
    
    output_dir.mkdir(exist_ok=True)
    
    if create_plots:
        plots_dir = output_dir / "peak_plots"
        plots_dir.mkdir(exist_ok=True)
    
    # Find all corrected files
    corrected_files = list(input_dir.glob("*_CORRECTED.xlsx"))
    
    if not corrected_files:
        print(f"❌ No corrected files found in {input_dir}")
        return
    
    print(f"\n{'='*80}")
    print("PEAK DETECTION ANALYSIS")
    print(f"{'='*80}\n")
    print(f"📂 Found {len(corrected_files)} corrected files")
    print(f"🔧 Parameters:")
    print(f"   • Prominence: {prominence} Ω")
    print(f"   • Width: {width} samples")
    print(f"   • Distance: {distance} samples")
    print(f"   • Create plots: {create_plots}")
    print()
    
    all_peaks = []
    plot_count = 0
    
    for corrected_file in tqdm(corrected_files, desc="Processing samples"):
        try:
            # Load corrected data
            df = pd.read_excel(corrected_file)
            sample_name = corrected_file.stem.replace('_CORRECTED', '')
            
            # Identify sensor columns (exclude metadata columns and NaN-only columns)
            sensor_cols = []
            for col in df.columns:
                if col in ['Unnamed: 0', 'index']:
                    continue
                # Try to convert to numeric and check if any non-NaN values exist
                try:
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    if numeric_col.notna().any():
                        sensor_cols.append(col)
                except:
                    continue
            
            if not sensor_cols:
                continue
            
            # Determine baseline end (first non-baseline row)
            # Assume baseline is already marked in the visualization step
            # For now, use a heuristic: baseline is first 30% or detect from data
            baseline_end_idx = int(len(df) * 0.3)  # Heuristic
            
            sample_peaks = []
            
            for channel in sensor_cols:
                # Get signal and convert to numeric, coercing errors to NaN
                signal = pd.to_numeric(df[channel], errors='coerce').values
                
                # Check if signal is all NaN or has insufficient data
                if np.all(np.isnan(signal)) or np.sum(~np.isnan(signal)) < 10:
                    continue  # Skip if all NaN or too few points
                
                # Detect peaks
                peak_info = detect_peaks(
                    signal, 
                    prominence=prominence,
                    width=width,
                    distance=distance,
                    detect_negative=True
                )
                
                # Skip if no peaks detected
                if not peak_info['positive_peaks'] and not peak_info['negative_peaks']:
                    continue
                
                # Calculate peak features
                peak_features = calculate_peak_features(signal, peak_info, baseline_end_idx)
                
                if len(peak_features) == 0:
                    continue
                
                # Add metadata
                peak_features['sample'] = sample_name
                peak_features['channel'] = channel
                
                sample_peaks.append(peak_features)
                
                # Create plot if requested
                if create_plots and plot_count < plot_limit:
                    plot_file = plots_dir / f"peaks_{sample_name}_{channel}.png"
                    plot_peaks_detailed(
                        signal, peak_info, baseline_end_idx,
                        channel, sample_name, plot_file
                    )
                    plot_count += 1
            
            # Combine peaks for this sample
            if sample_peaks:
                sample_peaks_df = pd.concat(sample_peaks, ignore_index=True)
                all_peaks.append(sample_peaks_df)
                
                # Save per-sample peak report
                sample_report = output_dir / f"{sample_name}_peaks.csv"
                sample_peaks_df.to_csv(sample_report, index=False)
        
        except Exception as e:
            print(f"  ❌ Error processing {corrected_file.name}: {e}")
            import traceback
            if "--verbose" in sys.argv:
                traceback.print_exc()
            continue
    
    # Combine all peaks
    if all_peaks:
        all_peaks_df = pd.concat(all_peaks, ignore_index=True)
        
        # Save master peak report
        master_file = output_dir / "MASTER_peak_report.csv"
        all_peaks_df.to_csv(master_file, index=False)
        
        print(f"\n✅ Master peak report saved: {master_file}")
        
        # Create summary statistics
        print(f"\n{'='*80}")
        print("PEAK DETECTION SUMMARY")
        print(f"{'='*80}\n")
        
        total_peaks = len(all_peaks_df)
        pos_peaks = len(all_peaks_df[all_peaks_df['peak_type'] == 'positive'])
        neg_peaks = len(all_peaks_df[all_peaks_df['peak_type'] == 'negative'])
        
        print(f"Total peaks detected: {total_peaks}")
        print(f"  • Positive peaks: {pos_peaks} ({pos_peaks/total_peaks*100:.1f}%)")
        print(f"  • Negative peaks: {neg_peaks} ({neg_peaks/total_peaks*100:.1f}%)")
        print(f"\nPeak Statistics:")
        print(f"  • Mean height: {all_peaks_df['peak_height'].mean():.2f} Ω")
        print(f"  • Mean prominence: {all_peaks_df['peak_prominence'].mean():.2f} Ω")
        print(f"  • Mean width: {all_peaks_df['peak_width'].mean():.2f} samples")
        print(f"  • Mean time to peak: {all_peaks_df['time_to_peak'].mean():.2f} samples")
        
        # Create summary plots
        print(f"\n📊 Creating summary visualizations...")
        create_peak_summary_plot(all_peaks_df, output_dir)
        
        # Print top peaks
        print(f"\n🏆 TOP 10 LARGEST PEAKS (by prominence):")
        top_peaks = all_peaks_df.nlargest(10, 'peak_prominence')
        for idx, row in top_peaks.iterrows():
            print(f"  {row['sample'][:40]:40s} | {row['channel']:8s} | "
                  f"{row['peak_type']:8s} | {row['peak_prominence']:8.2f} Ω")
        
        print(f"\n✅ All results saved to: {output_dir.resolve()}")
        
        if create_plots:
            print(f"✅ Peak plots saved to: {plots_dir.resolve()}")
            if plot_count >= plot_limit:
                print(f"   ⚠️ Plot limit reached ({plot_limit}). Only first {plot_limit} channels plotted.")
        
        return all_peaks_df
    
    else:
        print("\n⚠️ No peaks detected in any samples")
        return None


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Detect and analyze peaks in baseline-corrected sensor data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python peak_detection.py                           # Use defaults
  python peak_detection.py --prominence 100          # Higher threshold
  python peak_detection.py --no-plots                # Skip individual plots
  python peak_detection.py --plot-limit 50           # Limit plot count
        """
    )
    
    parser.add_argument('--input-dir', type=str, default='baseline_corrected_data',
                       help='Input directory with corrected data')
    parser.add_argument('--output-dir', type=str, default='peak_detection_results',
                       help='Output directory for results')
    parser.add_argument('--prominence', type=float, default=50,
                       help='Minimum peak prominence (Ω)')
    parser.add_argument('--width', type=int, default=3,
                       help='Minimum peak width (samples)')
    parser.add_argument('--distance', type=int, default=5,
                       help='Minimum distance between peaks (samples)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip creating individual peak plots')
    parser.add_argument('--plot-limit', type=int, default=100,
                       help='Maximum number of channel plots to create')
    
    args = parser.parse_args()
    
    # Run peak detection
    all_peaks_df = process_all_corrected_data(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        prominence=args.prominence,
        width=args.width,
        distance=args.distance,
        create_plots=not args.no_plots,
        plot_limit=args.plot_limit
    )
    
    print(f"\n{'='*80}")
    print("✅ PEAK DETECTION COMPLETE!")
    print(f"{'='*80}\n")