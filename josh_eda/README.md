# Taurus Variable Data Analysis – Josh’s EDA Pipeline

One-click cleaning + visualisation pipeline for all Taurus breath-sensor .xlsx files  
(Box tests and real pregnancy-chip data)


## One-time setup (do this once)

```bash
pip install -r requirements.txt
python josh_eda/batch_clean.py "historical_reference_data"
```

## Folder Structure

Taurus-Variable-Data-Analysis/
├── josh_eda/
│   ├── main.py                      # Entry point: file picker + clean + visualize
│   ├── batch_clean.py               # Bulk-clean all raw files in one command
│   ├── data_cleaning.py             # Core cleaning + fault detection engine
│   ├── visualisation_methods.py     # 6 standard plots per sample
│   └── __init__.py                  # Package marker
├── analysis.py                      # Diagnostic report generator (Box vs Cow Breath comparison)
├── historical_reference_data/       # ← Raw files (never edit)
│   ├── Box_A_B_Test/
│   ├── 7-10-2025_Pregnancy_Chip_10/
│   └── …
└── cleaned/                         # ← Auto-created, mirrored structure
    ├── Box_A_B_Test/
    │   └── …_CLEANED.xlsx
    │   └── …_FAULTS.csv
    └── 7-10-2025_Pregnancy_Chip_10/
        └── …_CLEANED.xlsx
        └── …_FAULTS.csv

## How to Use

| What you want                               | Command (from project root)                                   | Notes                                           |
|---------------------------------------------|---------------------------------------------------------------|-------------------------------------------------|
| Clean **all** raw files at once            | `python josh_eda/batch_clean.py "historical_reference_data"` | Run only when you add new raw folders. Idempotent. |
| Analyse one sample (clean + plots)         | `python josh_eda/main.py` → file picker                       | Works on raw or already-cleaned files           |
| Re-open an old cleaned sample              | `python josh_eda/main.py` → pick any `_CLEANED.xlsx`          | Skips cleaning, goes straight to visualisation  |

## Key Features
- Robust header detection (finds the real "Seq" row automatically)
- Mirrored folder structure in `cleaned/` – provenance is crystal clear
- Clean column names: `Time_ms`, `Temperature_C`, `T1` … `T28`<|eos|>