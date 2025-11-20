from pathlib import Path
import sys
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

    for f in files:
        try:
            cleaned_path = clean_file(f)
            print(f"Cleaned → {cleaned_path.name}")
        except Exception as e:
            print(f"FAILED {f.name}: {e}")

    print("\nBatch cleaning complete! Ready for visuals.")