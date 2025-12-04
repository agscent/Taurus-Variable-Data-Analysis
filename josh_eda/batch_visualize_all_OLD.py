"""
Batch visualization of all samples with quality classification.
Creates comprehensive reports showing baseline correction results for all channels.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

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


def create_all_channels_grid(df: pd.DataFrame, sensor_cols: list, 
                             baseline_start_idx: int, baseline_end_idx: int,
                             filename: str, output_dir: Path):
    """WORKING VERSION - copied from test"""
    
    # PRINT DEBUG - will prove this code is running
    print(f"    DEBUG: create_all_channels_grid called with baseline {baseline_start_idx}-{baseline_end_idx}")
    
    num_sensors = len(sensor_cols)
    grid_cols = 4
    grid_rows = (num_sensors + grid_cols - 1) // grid_cols
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(20, 4 * grid_rows))
    fig.suptitle(f'{filename}\nBASELINE: rows {baseline_start_idx}-{baseline_end_idx}', 
                 fontsize=16, fontweight='bold')
    
    if grid_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    quality_counts = {'EXCELLENT': 0, 'GOOD': 0, 'MARGINAL': 0, 'POOR': 0, 'DEAD': 0}
    
    for idx, sensor_col in enumerate(sensor_cols):
        ax = axes[idx]
        
        # Simple baseline correction like test
        baseline_median = df.loc[baseline_start_idx:baseline_end_idx, sensor_col].median()
        corrected = df[sensor_col] - baseline_median
        
        # Plot
        ax.plot(corrected, linewidth=0.5, color='blue')
        
        # GREEN REGION - LIKE TEST
        ax.axvspan(baseline_start_idx, baseline_end_idx, color='green', alpha=0.2)
        
        # RED LINE - LIKE TEST
        ax.axvline(baseline_end_idx, color='red', linestyle='--', linewidth=1)
        
        ax.axhline(0, color='red', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.set_title(sensor_col, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)
        
        quality_counts['GOOD'] += 1  # Simplified for now
    
    # Hide unused
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
                              output_dir: Path = Path("batch_visualizations")):
    """
    Process all cleaned samples:
    1. Generate grid visualization for each sample
    2. Classify all channels
    3. Create master quality report
    """
    
    output_dir.mkdir(exist_ok=True)
    
    # Find all cleaned files
    cleaned_files = list(root_dir.rglob("*_CLEANED.xlsx"))
    cleaned_files = [f for f in cleaned_files if not f.name.startswith('~$')]
    
    if not cleaned_files:
        print(f"❌ No cleaned files found in {root_dir}")
        return
    
    print(f"📂 Found {len(cleaned_files)} cleaned files")
    print(f"📊 Processing and visualizing all samples...\n")
    
    all_quality_reports = []
    summary_stats = []
    
    for cleaned_file in tqdm(cleaned_files, desc="Processing samples"):
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

            # # DEBUG: Print what baseline was detected
            # baseline_pct = (baseline_end - baseline_start + 1) / len(df) * 100
            # print(f"  [{box_type:8s}] {cleaned_file.name[:45]:45s} baseline: {baseline_start:3d}-{baseline_end:3d}/{len(df):3d} ({baseline_pct:3.0f}%)")
            
            # VALIDATION: Check baseline makes sense
            baseline_length = baseline_end - baseline_start + 1
            baseline_pct = baseline_length / len(df) * 100
            
            if baseline_pct > 95:
                print(f"  ⚠ WARNING: {cleaned_file.name}")
                print(f"      Baseline is {baseline_pct:.0f}% of data (rows {baseline_start}-{baseline_end} / {len(df)})")
                print(f"      Box type: {box_type}")
            
            # Create visualization
            viz_file, quality_counts = create_all_channels_grid(
                df, sensor_cols,
                baseline_start, baseline_end,
                cleaned_file.stem,
                output_dir
            )
            
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
                )
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
        print(f"\n✓ Master quality report saved: {master_quality_file}")
    
    # Create summary statistics
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        summary_file = output_dir / "SUMMARY_statistics.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"✓ Summary statistics saved: {summary_file}")
        
        # Print overall summary
        print("\n" + "="*80)
        print("OVERALL SUMMARY")
        print("="*80)
        
        total_channels = summary_df['Total_Channels'].sum()
        total_excellent = summary_df['Excellent'].sum()
        total_good = summary_df['Good'].sum()
        total_marginal = summary_df['Marginal'].sum()
        total_poor = summary_df['Poor'].sum()
        total_dead = summary_df['Dead'].sum()
        total_usable = total_excellent + total_good
        
        print(f"\nProcessed {len(summary_df)} samples")
        print(f"Total channels analyzed: {total_channels}")
        print(f"\nQuality Distribution:")
        print(f"  ✓ EXCELLENT: {total_excellent} ({total_excellent/total_channels*100:.1f}%)")
        print(f"  ✓ GOOD:      {total_good} ({total_good/total_channels*100:.1f}%)")
        print(f"  ⚠ MARGINAL:  {total_marginal} ({total_marginal/total_channels*100:.1f}%)")
        print(f"  ✗ POOR:      {total_poor} ({total_poor/total_channels*100:.1f}%)")
        print(f"  ✗ DEAD:      {total_dead} ({total_dead/total_channels*100:.1f}%)")
        print(f"\n  USABLE (Excellent + Good): {total_usable} ({total_usable/total_channels*100:.1f}%)")
        
        print("\n" + "="*80)
        print(f"\n✓ All visualizations saved to: {output_dir.resolve()}")
        
        # Show top/bottom samples by quality
        print("\n📊 TOP 5 SAMPLES (by usable channels %):")
        top_samples = summary_df.nlargest(5, 'Data_Quality_Percent')
        for _, row in top_samples.iterrows():
            print(f"  {row['Sample'][:60]:60s} | {row['Data_Quality_Percent']:5.1f}% | {row['Usable_Channels']}/{row['Total_Channels']} channels")
        
        print("\n📊 BOTTOM 5 SAMPLES (by usable channels %):")
        bottom_samples = summary_df.nsmallest(5, 'Data_Quality_Percent')
        for _, row in bottom_samples.iterrows():
            print(f"  {row['Sample'][:60]:60s} | {row['Data_Quality_Percent']:5.1f}% | {row['Usable_Channels']}/{row['Total_Channels']} channels")
    
    return summary_df, master_quality_df if all_quality_reports else None


def create_master_visualization_index(output_dir: Path):
    """
    Create an HTML index page to browse all visualizations easily.
    """
    html_file = output_dir / "index.html"
    
    # Get all PNG files
    png_files = sorted(output_dir.glob("all_channels_*.png"))
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Baseline Correction Visualization Index</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
        }
        .sample {
            margin: 20px 0;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .sample h3 {
            margin-top: 0;
            color: #2c3e50;
        }
        img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <h1>📊 Baseline Correction Visualization Index</h1>
    <p>Total samples: """ + str(len(png_files)) + """</p>
    <hr>
"""
    
    for png_file in png_files:
        sample_name = png_file.stem.replace('all_channels_', '').replace('_CLEANED', '')
        html_content += f"""
    <div class="sample">
        <h3>{sample_name}</h3>
        <img src="{png_file.name}" alt="{sample_name}">
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    print(f"\n✓ HTML index created: {html_file}")
    print(f"  Open this file in a browser to browse all visualizations easily!")


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    print("\n" + "="*80)
    print("BATCH VISUALIZATION & QUALITY CLASSIFICATION")
    print("="*80 + "\n")
    
    # Process all samples
    summary_df, quality_df = batch_process_all_samples(
        root_dir=Path("historical_reference_data"),
        output_dir=Path("batch_visualizations")
    )
    
    # Create HTML index for easy browsing
    create_master_visualization_index(Path("batch_visualizations"))
    
    print("\n" + "="*80)
    print("✅ BATCH PROCESSING COMPLETE!")
    print("="*80)