"""
Master pipeline script - runs the complete baseline correction workflow.
MODIFIED: Now runs BOTH linear and exponential methods for visualization.

Steps:
1. Visualize all samples with BOTH linear and exponential methods
2. Generate quality reports
3. Apply baseline correction to INCLUDE channels

Usage:
    python run_full_pipeline.py
    
Or with options:
    python run_full_pipeline.py --quality CAUTION --method Linear_Fit
"""

import sys
from pathlib import Path

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                    BASELINE CORRECTION PIPELINE                            ║
║                                                                            ║
║  Step 1: Visualize with BOTH linear and exponential methods               ║
║  Step 2: Generate comprehensive quality reports                           ║
║  Step 3: Apply baseline correction to high-quality channels               ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

# Parse command-line arguments
import argparse

parser = argparse.ArgumentParser(description='Run complete baseline correction pipeline')
parser.add_argument('--quality', type=str, default='INCLUDE',
                   choices=['INCLUDE', 'CAUTION', 'ALL'],
                   help='Quality threshold: INCLUDE (EXCELLENT+GOOD), CAUTION (includes MARGINAL), ALL')
parser.add_argument('--method', type=str, default='Linear_Fit',
                   choices=['Constant_Median', 'Linear_Fit', 'Exponential_Fit'],
                   help='Baseline correction method')
parser.add_argument('--skip-viz', action='store_true',
                   help='Skip visualization step (if already done)')
parser.add_argument('--input', type=str, default='historical_reference_data',
                   help='Input directory with cleaned data')

args = parser.parse_args()

print(f"Configuration:")
print(f"  • Quality threshold: {args.quality}")
print(f"  • Correction method: {args.method}")
print(f"  • Input directory: {args.input}")
print(f"  • Skip visualization: {args.skip_viz}")
print()

# ============================================================================
# STEP 1: VISUALIZE ALL SAMPLES WITH BOTH METHODS
# ============================================================================

if not args.skip_viz:
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║ STEP 1: VISUALIZING ALL SAMPLES (LINEAR & EXPONENTIAL)                    ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print()
    
    try:
        from batch_visualize_all_BOTH_METHODS import (
            batch_process_all_samples,
            create_master_visualization_index
        )
        
        # Process with BOTH methods
        for viz_method in ["linear", "exponential"]:
            print(f"\n{'='*80}")
            print(f"VISUALIZING WITH {viz_method.upper()} METHOD")
            print(f"{'='*80}\n")
            
            output_dir = Path(f"batch_visualizations_{viz_method}")
            
            summary_df, quality_df = batch_process_all_samples(
                root_dir=Path(args.input),
                output_dir=output_dir,
                method=viz_method
            )
            
            create_master_visualization_index(output_dir, method=viz_method)
            
            print(f"\n✓ {viz_method.upper()} visualizations saved to {output_dir}/")
        
        print("\n" + "="*80)
        print("✓ Step 1 complete: Both linear and exponential visualizations created")
        print("="*80)
        print("  → Open batch_visualizations_linear/index.html")
        print("  → Open batch_visualizations_exponential/index.html")
        print("  → Compare both methods to see which gives better baseline correction")
        
    except Exception as e:
        print(f"\n❌ Step 1 failed: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing with next steps...")
else:
    print("⏭  Skipping visualization step (--skip-viz flag set)")
    print()

# ============================================================================
# STEP 2: APPLY BASELINE CORRECTION
# ============================================================================

print("\n╔════════════════════════════════════════════════════════════════════════════╗")
print("║ STEP 2: APPLYING BASELINE CORRECTION                                      ║")
print("╚════════════════════════════════════════════════════════════════════════════╝")
print()

try:
    from apply_baseline_correction import batch_apply_baseline_correction
    
    summary = batch_apply_baseline_correction(
        root_dir=Path(args.input),
        output_dir=Path("baseline_corrected_data"),
        quality_threshold=args.quality,
        fit_method=args.method,
        save_quality_reports=True
    )
    
    print("\n✓ Step 2 complete: Corrected data saved to baseline_corrected_data/")
    
except Exception as e:
    print(f"\n❌ Step 2 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 3: GENERATE FINAL SUMMARY
# ============================================================================

print("\n╔════════════════════════════════════════════════════════════════════════════╗")
print("║ STEP 3: GENERATING FINAL SUMMARY                                          ║")
print("╚════════════════════════════════════════════════════════════════════════════╝")
print()

try:
    import pandas as pd
    
    # Load the summary files (use linear by default for merging)
    viz_summary = Path("batch_visualizations_linear/SUMMARY_statistics.csv")
    corr_summary = Path("baseline_corrected_data/CORRECTION_SUMMARY.csv")
    
    if viz_summary.exists() and corr_summary.exists():
        df_viz = pd.read_csv(viz_summary)
        df_corr = pd.read_csv(corr_summary)
        
        # Merge on sample name
        df_viz['Sample_Clean'] = df_viz['Sample'].str.replace('_CLEANED', '')
        df_corr['Sample_Clean'] = df_corr['Sample']
        
        merged = pd.merge(df_viz, df_corr, on='Sample_Clean', suffixes=('_Viz', '_Corr'))
        
        # Create final report
        final_report = pd.DataFrame({
            'Sample': merged['Sample_Clean'],
            'Box_Type': merged['Box_Type_Viz'],
            'Total_Channels': merged['Total_Channels_Viz'],
            'Excellent': merged['Excellent'],
            'Good': merged['Good'],
            'Marginal': merged['Marginal'],
            'Poor': merged['Poor'],
            'Dead': merged['Dead'],
            'Included_After_Correction': merged['Included_Channels'],
            'Excluded_After_Correction': merged['Excluded_Channels'],
            'Final_Usable_Percent': merged['Inclusion_Rate_%']
        })
        
        final_file = Path("FINAL_PIPELINE_SUMMARY.csv")
        final_report.to_csv(final_file, index=False)
        
        print(f"✓ Final summary saved: {final_file}")
        
        # Print key statistics
        print("\n" + "="*80)
        print("FINAL PIPELINE STATISTICS")
        print("="*80)
        
        print(f"\nTotal samples processed: {len(final_report)}")
        print(f"Total channels analyzed: {final_report['Total_Channels'].sum()}")
        print(f"Channels included in corrected data: {final_report['Included_After_Correction'].sum()}")
        print(f"Overall inclusion rate: {final_report['Included_After_Correction'].sum() / final_report['Total_Channels'].sum() * 100:.1f}%")
        
        print("\n" + "="*80)
        
    else:
        print("⚠ Could not find summary files to merge")
        
except Exception as e:
    print(f"⚠ Could not generate final summary: {e}")

# ============================================================================
# COMPLETION
# ============================================================================

print("\n╔════════════════════════════════════════════════════════════════════════════╗")
print("║                          PIPELINE COMPLETE!                                ║")
print("╚════════════════════════════════════════════════════════════════════════════╝")

print("""
📂 Output Structure:
   
   batch_visualizations_linear/          ← NEW! Linear method
   ├── index.html                         ← Open this to browse linear results
   ├── all_channels_*.png                 ← Grid plots for each sample
   ├── detailed_plots/                    ← ALL channels, 3-panel plots
   │   └── detailed_*_*.png
   ├── MASTER_quality_report.csv          ← All channels, all samples
   └── SUMMARY_statistics.csv             ← Per-sample summary
   
   batch_visualizations_exponential/      ← NEW! Exponential method
   ├── index.html                         ← Open this to browse exponential results
   ├── all_channels_*.png                 ← Grid plots for each sample
   ├── detailed_plots/                    ← ALL channels, 3-panel plots
   │   └── detailed_*_*.png
   ├── MASTER_quality_report.csv          ← All channels, all samples
   └── SUMMARY_statistics.csv             ← Per-sample summary
   
   baseline_corrected_data/
   ├── *_CORRECTED.xlsx                   ← Corrected data files (NaN = excluded)
   ├── quality_reports/                   ← Per-sample quality details
   │   └── *_quality.csv
   └── CORRECTION_SUMMARY.csv             ← Inclusion/exclusion summary
   
   FINAL_PIPELINE_SUMMARY.csv             ← Complete overview

📊 Next Steps:
   
   1. Compare visualization methods:
      → Open batch_visualizations_linear/index.html in browser
      → Open batch_visualizations_exponential/index.html in browser
      → Compare the same channels in detailed_plots/ folders
      → Pick which method gives better baseline correction
   
   2. Review detailed plots:
      → Look at Panel 1: Does red line fit blue data well?
      → Look at Panel 3: Is baseline (green region) flat near zero?
      → Linear better? → Baseline drifts steadily
      → Exponential better? → Baseline curves/decays
   
   3. Check quality classifications:
      → Review batch_visualizations_*/MASTER_quality_report.csv
   
   4. Verify corrected data:
      → Check baseline_corrected_data/*_CORRECTED.xlsx
      → Excluded channels are set to NaN
   
   5. Proceed to kinetics analysis:
      → Use corrected data for peak detection
      → Fit double exponential binding/dissociation curves
      → Calculate ΔR/R for each channel

""")

print("\n✅ All done!")