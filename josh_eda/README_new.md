
# Taurus Variable Data Analysis – Josh’s EDA Pipeline

One-click cleaning + **interactive visualization** pipeline for all Taurus and Variable breath-sensor .xlsx files.

## One-time setup (do this once)

Run this command **from the project root directory** (`Taurus-Variable-Data-Analysis`):

~code~bash
pip install -r requirements.txt
python josh_eda/batch_clean.py "historical_reference_data"
~code~

## Folder Structure

Taurus-Variable-Data-Analysis/
├── josh_eda/
│   ├── dashboard.py                 # **NEW: Interactive Dash web-dashboard**
│   ├── batch_clean.py               # Bulk-clean all raw files in one command
│   ├── data_cleaning.py             # Core cleaning + fault detection engine
│   └── __init__.py                  # Package marker
├── analysis.py                      # Diagnostic report generator (Box vs Cow Breath comparison)
├── historical_reference_data/       # ← Raw files (never edit)
│   └── …
└── cleaned/                         # ← Auto-created, mirrored structure
    └── …_CLEANED.xlsx
└── …_FAULTS.csv

## How to Use

| What you want                                   | Command (MUST be run from **project root** == `Taurus-Variable-Data-Analysis`) | Notes                                                                 |
|-------------------------------------------------|------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| 1. Clean **all** raw files at once              | ~code~python josh_eda/batch_clean.py "historical_reference_data"~code~       | Creates `*_CLEANED.xlsx` and `*_FAULTS.csv` files. Run only when you add new raw data. |
| 2. **Run the Interactive Dashboard**            | ~code~python josh_eda/dashboard.py~code~                                     | Opens the dashboard at `http://localhost:8050`. **Requires cleaned files to exist.** |
| Generate diagnostic report (unused for now)     | ~code~python josh_eda/analysis.py~code~                                               | Console summary + bar chart PNG. Compares calibration vs. cow breath data. Shows dead channels & per-sample statistics. |

## Key Features

* **Interactive Visualization:** The Dash dashboard allows you to select test groups and individual samples to view time-series plots, heatmaps, and summary statistics.
* **Robust Data Cleaning:** Finds the real "Seq" row automatically and handles both Taurus and Variable Box formats.
* **Standardized Output:** Creates a consistent set of standardized column names and a mirrored folder structure for clear data provenance.
* **Automatic Fault Detection:** Detects hardware faults and data quality issues (e.g., channels stuck at 0 $\Omega$, unstable temperature) and logs them to `*_FAULTS.csv`.
* **Centralized Fault Audit:** A `MASTER_FAULT_REPORT.csv` is generated in the `historical_reference_data` directory during batch cleaning, aggregating all channel-level faults.
