"""
Quick Peak Visualization Script
Shows the top N detected peaks with detailed plots.

Usage:
    python visualize_peaks.py
    python visualize_peaks.py --top 10
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_and_visualize_peaks(master_report: Path, 
                             corrected_data_dir: Path,
                             top_n: int = 5,
                             output_dir: Path = Path("peak_visualizations")):
    """
    Load peak data and create visualizations for top peaks.
    """
    
    # Load peak report
    if not master_report.exists():
        print(f"❌ Peak report not found: {master_report}")
        print("   Run peak_detection.py first!")
        return
    
    peaks_df = pd.read_csv(master_report)
    
    if len(peaks_df) == 0:
        print("❌ No peaks found in report")
        return
    
    print(f"\n{'='*80}")
    print(f"VISUALIZING TOP {top_n} PEAKS")
    print(f"{'='*80}\n")
    
    print(f"Total peaks in report: {len(peaks_df)}")
    print(f"Samples: {peaks_df['sample'].nunique()}")
    print(f"Channels: {peaks_df['channel'].nunique()}")
    
    # Sort by prominence to get top peaks
    top_peaks = peaks_df.nlargest(top_n, 'peak_prominence')
    
    print(f"\n📊 Top {top_n} peaks by prominence:")
    for idx, row in top_peaks.iterrows():
        print(f"  {idx+1}. {row['sample'][:40]:40s} | {row['channel']:10s} | "
              f"{row['peak_type']:8s} | {row['peak_prominence']:8.2f} Ω")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Visualize each top peak
    print(f"\n🎨 Creating visualizations...")
    
    for i, (idx, peak_row) in enumerate(top_peaks.iterrows(), 1):
        try:
            # Load the corrected data file
            sample_file = corrected_data_dir / f"{peak_row['sample']}_CORRECTED.xlsx"
            
            if not sample_file.exists():
                print(f"  ⚠️  Skipping peak {i}: File not found: {sample_file.name}")
                continue
            
            df = pd.read_excel(sample_file)
            
            # Get the channel data
            channel = peak_row['channel']
            if channel not in df.columns:
                print(f"  ⚠️  Skipping peak {i}: Channel {channel} not found")
                continue
            
            # Convert to numeric
            signal = pd.to_numeric(df[channel], errors='coerce').values
            
            # Create visualization
            fig, axes = plt.subplots(2, 1, figsize=(16, 10))
            
            indices = np.arange(len(signal))
            
            # === TOP PANEL: Full signal with peak marked ===
            ax1 = axes[0]
            ax1.plot(indices, signal, 'b-', linewidth=1.5, alpha=0.7, label='Corrected Signal')
            
            # Mark the baseline region (heuristic: first 30%)
            baseline_end = int(len(signal) * 0.3)
            ax1.axvspan(0, baseline_end, color='lightgreen', alpha=0.2, label='Baseline')
            ax1.axvline(baseline_end, color='red', linestyle='--', linewidth=2, alpha=0.5)
            
            # Mark the peak
            peak_idx = int(peak_row['peak_index'])
            peak_height = peak_row['peak_height']
            
            marker = '^' if peak_row['peak_type'] == 'positive' else 'v'
            color = 'red' if peak_row['peak_type'] == 'positive' else 'blue'
            
            ax1.plot(peak_idx, peak_height, marker, markersize=15, color=color,
                    label=f"{peak_row['peak_type'].capitalize()} Peak", zorder=5)
            
            # Mark peak width
            left_bound = int(peak_row['left_bound'])
            right_bound = int(peak_row['right_bound'])
            ax1.axvspan(left_bound, right_bound, color=color, alpha=0.1)
            
            # Add prominence line
            baseline_at_peak = signal[peak_idx] - peak_row['peak_prominence']
            ax1.plot([peak_idx, peak_idx], [baseline_at_peak, peak_height], 
                    'k-', linewidth=3, alpha=0.5, label=f'Prominence: {peak_row["peak_prominence"]:.1f} Ω')
            
            ax1.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
            ax1.set_ylabel('ΔR (Ω)', fontsize=12, fontweight='bold')
            
            # Create comprehensive title with all key info
            title = f'Peak #{i} - CHANNEL: {channel}\n'
            title += f'Sample: {peak_row["sample"]}\n'
            title += f'Type: {peak_row["peak_type"].upper()} | '
            title += f'Prominence: {peak_row["peak_prominence"]:.1f} Ω | '
            title += f'Height: {peak_row["peak_height"]:.1f} Ω'
            
            ax1.set_title(title, fontsize=13, fontweight='bold', pad=15)
            ax1.legend(loc='best', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.3)
            
            # === BOTTOM PANEL: Zoomed view of peak ===
            ax2 = axes[1]
            
            # Zoom window: ±20 samples around peak (or adjust based on width)
            zoom_window = max(int(peak_row['peak_width']) * 2, 20)
            zoom_start = max(0, peak_idx - zoom_window)
            zoom_end = min(len(signal), peak_idx + zoom_window)
            
            zoom_indices = indices[zoom_start:zoom_end]
            zoom_signal = signal[zoom_start:zoom_end]
            
            ax2.plot(zoom_indices, zoom_signal, 'b-', linewidth=2, label='Signal (zoomed)')
            ax2.plot(peak_idx, peak_height, marker, markersize=20, color=color, zorder=5)
            
            # Mark peak boundaries
            ax2.axvline(left_bound, color='green', linestyle='--', linewidth=2, 
                       alpha=0.7, label='Peak Boundaries')
            ax2.axvline(right_bound, color='green', linestyle='--', linewidth=2, alpha=0.7)
            ax2.axvspan(left_bound, right_bound, color=color, alpha=0.1)
            
            # Add prominence
            ax2.plot([peak_idx, peak_idx], [baseline_at_peak, peak_height], 
                    'k-', linewidth=4, alpha=0.5)
            
            ax2.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
            ax2.set_ylabel('ΔR (Ω)', fontsize=12, fontweight='bold')
            ax2.set_title(f'Zoomed View - Channel: {channel}', fontsize=13, fontweight='bold')
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.3)
            
            # Add statistics box
            stats_text = f"""CHANNEL: {channel}
Sample: {peak_row['sample'][:30]}

Peak Statistics:
Height: {peak_row['peak_height']:.2f} Ω
Prominence: {peak_row['peak_prominence']:.2f} Ω
Width: {peak_row['peak_width']:.2f} samples
Time to peak: {peak_row['time_to_peak']:.1f} samples
Area: {peak_row['abs_peak_area']:.2f} Ω·samples"""
            
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                    verticalalignment='top', fontsize=10, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', 
                             alpha=0.8, edgecolor='black'))
            
            plt.tight_layout()
            
            # Save figure
            safe_sample = peak_row['sample'].replace('/', '_').replace('\\', '_')
            output_file = output_dir / f"peak_{i:02d}_{safe_sample}_{channel}.png"
            fig.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  ✅ Peak {i}: {output_file.name}")
            
        except Exception as e:
            print(f"  ❌ Error visualizing peak {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✅ All visualizations saved to: {output_dir.resolve()}")
    print(f"\n💡 Tip: Open the images to see:")
    print(f"   • Top panel: Full signal with peak location")
    print(f"   • Bottom panel: Zoomed view with peak details")
    
    # Create summary comparison
    create_summary_comparison(top_peaks, output_dir)


def create_summary_comparison(top_peaks_df: pd.DataFrame, output_dir: Path):
    """
    Create a summary figure showing all top peaks side-by-side.
    """
    n_peaks = len(top_peaks_df)
    
    fig, axes = plt.subplots(1, min(n_peaks, 5), figsize=(4*min(n_peaks, 5), 4))
    
    if n_peaks == 1:
        axes = [axes]
    elif n_peaks > 1:
        axes = axes.flatten()
    
    fig.suptitle(f'Top {min(n_peaks, 5)} Peaks Comparison', fontsize=16, fontweight='bold')
    
    for i, (idx, peak_row) in enumerate(top_peaks_df.head(5).iterrows()):
        ax = axes[i]
        
        # Simple bar showing prominence
        color = 'red' if peak_row['peak_type'] == 'positive' else 'blue'
        ax.bar(0, peak_row['peak_prominence'], color=color, alpha=0.7)
        
        ax.set_ylabel('Prominence (Ω)', fontweight='bold')
        title_text = f"Peak {i+1}\n{peak_row['channel']}\n({peak_row['sample'][:20]})"
        ax.set_title(title_text, fontsize=9, fontweight='bold')
        ax.set_xticks([])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add text
        ax.text(0, peak_row['peak_prominence']/2, 
               f"{peak_row['peak_prominence']:.1f} Ω\n{peak_row['peak_type']}", 
               ha='center', va='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    
    summary_file = output_dir / "peaks_summary_comparison.png"
    fig.savefig(summary_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✅ Summary comparison: {summary_file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize top detected peaks',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--top', type=int, default=5,
                       help='Number of top peaks to visualize (default: 5)')
    parser.add_argument('--peak-report', type=str, 
                       default='peak_detection_results/MASTER_peak_report.csv',
                       help='Path to peak report CSV')
    parser.add_argument('--data-dir', type=str,
                       default='baseline_corrected_data',
                       help='Directory with corrected data')
    parser.add_argument('--output-dir', type=str,
                       default='peak_visualizations',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Run visualization
    load_and_visualize_peaks(
        master_report=Path(args.peak_report),
        corrected_data_dir=Path(args.data_dir),
        top_n=args.top,
        output_dir=Path(args.output_dir)
    )
    
    print(f"\n{'='*80}")
    print("✅ VISUALIZATION COMPLETE!")
    print(f"{'='*80}\n")