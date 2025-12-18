#!/usr/bin/env python3
"""
Automatic integration script - replaces batch_visualize_all.py with improved version.

Usage:
    python integrate_new_baseline.py
    
Or with options:
    python integrate_new_baseline.py --backup-dir my_backups
"""

import shutil
from pathlib import Path
import sys
from datetime import datetime

def integrate_new_baseline(backup_dir: str = "backups", dry_run: bool = False):
    """
    Automatically integrate the new baseline correction script.
    
    Steps:
    1. Check if files exist
    2. Create backup of old file
    3. Copy new file to batch_visualize_all.py
    4. Verify integration
    """
    
    print("\n" + "="*80)
    print("BASELINE CORRECTION INTEGRATION")
    print("="*80 + "\n")
    
    # Files
    current_file = Path("batch_visualize_all.py")
    new_file = Path("outputs/batch_visualize_all_IMPROVED.py")
    backup_folder = Path(backup_dir)
    
    # Step 1: Check if files exist
    print("📋 Step 1: Checking files...")
    
    if not current_file.exists():
        print(f"⚠️  WARNING: {current_file} not found in current directory")
        print(f"   Current directory: {Path.cwd()}")
        print(f"   Please run this script from your project root directory")
        return False
    
    if not new_file.exists():
        print(f"❌ ERROR: {new_file} not found")
        print(f"   Make sure outputs/ folder is in current directory")
        return False
    
    print(f"✅ Found current file: {current_file}")
    print(f"✅ Found new file: {new_file}")
    
    # Step 2: Create backup
    print(f"\n📋 Step 2: Creating backup...")
    
    backup_folder.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_folder / f"batch_visualize_all_BACKUP_{timestamp}.py"
    
    if dry_run:
        print(f"   [DRY RUN] Would backup to: {backup_file}")
    else:
        shutil.copy2(current_file, backup_file)
        print(f"✅ Backed up to: {backup_file}")
    
    # Step 3: Replace with new file
    print(f"\n📋 Step 3: Replacing with new version...")
    
    if dry_run:
        print(f"   [DRY RUN] Would copy {new_file} to {current_file}")
    else:
        shutil.copy2(new_file, current_file)
        print(f"✅ Copied {new_file} → {current_file}")
    
    # Step 4: Verify
    print(f"\n📋 Step 4: Verifying integration...")
    
    if dry_run:
        print(f"   [DRY RUN] Would verify file contents")
    else:
        # Check file size as basic verification
        new_size = current_file.stat().st_size
        expected_size = new_file.stat().st_size
        
        if new_size == expected_size:
            print(f"✅ File size matches: {new_size:,} bytes")
        else:
            print(f"⚠️  File size mismatch: {new_size:,} vs {expected_size:,} bytes")
            return False
        
        # Check for key function
        content = current_file.read_text()
        if "fit_baseline_curve" in content:
            print(f"✅ New function 'fit_baseline_curve' found in file")
        else:
            print(f"❌ ERROR: New function not found - integration may have failed")
            return False
        
        if "create_detailed_channel_plot" in content:
            print(f"✅ New function 'create_detailed_channel_plot' found in file")
        else:
            print(f"❌ ERROR: Detailed plot function not found")
            return False
    
    # Success!
    print("\n" + "="*80)
    print("✅ INTEGRATION SUCCESSFUL!")
    print("="*80)
    
    if not dry_run:
        print(f"""
📂 Files:
   • Current: {current_file} (updated with new version)
   • Backup:  {backup_file}
   • Source:  {new_file}

🚀 Next Steps:
   1. Run your pipeline:
      python run_full_pipeline.py
   
   2. Check outputs in:
      batch_visualizations/all_channels_*_linear.png
      batch_visualizations/detailed_plots/
   
   3. If something goes wrong, restore backup:
      cp {backup_file} {current_file}

💡 Key Changes:
   • Using LINEAR baseline fitting (not median)
   • Creating detailed 3-panel plots
   • File names include method (e.g., _linear.png)
   • Backward compatible with existing pipeline

📖 Documentation:
   • Quick start:  outputs/QUICKSTART.md
   • Integration:  outputs/INTEGRATION_GUIDE.md
   • What changed: outputs/WHATS_CHANGED.md
""")
    else:
        print("\n[DRY RUN] No files were modified")
        print("Run without --dry-run to actually integrate")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Integrate improved baseline correction script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python integrate_new_baseline.py              # Standard integration
  python integrate_new_baseline.py --dry-run    # Preview without changes
  python integrate_new_baseline.py --backup-dir old_versions
        """
    )
    
    parser.add_argument('--backup-dir', type=str, default='backups',
                       help='Directory for backup files (default: backups)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview changes without modifying files')
    
    args = parser.parse_args()
    
    # Run integration
    success = integrate_new_baseline(
        backup_dir=args.backup_dir,
        dry_run=args.dry_run
    )
    
    if success:
        sys.exit(0)
    else:
        print("\n❌ Integration failed - see errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()