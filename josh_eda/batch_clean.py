from pathlib import Path
import sys
import pandas as pd
from data_cleaning import clean_file  

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python batch_clean.py "path/to/folder"')
        sys.exit(1)

    root = Path(sys.argv[1]).resolve()

    if not root.is_dir():
        print(f"Folder not found: {root}")
        sys.exit(1)

    # Only look for RAW files
    files = [
        f for f in root.rglob("*.xlsx")
        if "cleaned" not in f.parts and not f.name.endswith("_CLEANED.xlsx")
    ]

    if not files:
        print("No raw .xlsx files found (already cleaned or none present).")
        sys.exit(0)

    print(f"Found {len(files)} raw file(s) to clean...\n")

    # 🔥 NEW: List to collect all structured faults from all files
    all_faults = []

    for f in files:
        try:
            # 🔥 NEW: Collect both the path and the list of faults
            cleaned_path, sample_faults = clean_file(f) 
            all_faults.extend(sample_faults) 
            print(f"Cleaned → {cleaned_path.name}")
        except Exception as e:
            print(f"FAILED {f.name}: {e}")

    # 🔥 NEW: Save the aggregated fault report
    if all_faults:
        faults_df = pd.DataFrame(all_faults)
        
        # Save the master report in the root directory where the script was run
        master_fault_path = root / "MASTER_FAULT_REPORT.csv"
        
        faults_df.to_csv(master_fault_path, index=False)
        print(f"\n--- Batch Cleaning Complete ---")
        print(f"Master fault report created: {master_fault_path.name} ({len(all_faults)} total entries)")
    else:
        print("\n--- Batch Cleaning Complete ---")
        print("No channel-level faults detected in batch.")