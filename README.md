# Taurus/Variable Box Data Analysis Tool

A PyQt5-based desktop application designed for automated statistical analysis and visualization of test data from Taurus and Variable box systems. This tool processes multiple CSV files containing sensor channel data, performs statistical analysis across test runs, and generates comprehensive reports with visualizations.

## Overview

This application is specifically designed for analyzing experimental test data collected from:
- **Taurus Box**: Taurus Box data (28 channels: T1-T28)
- **Variable Box**: Variable Box data (64 channels: D1-D64)

The tool automates the process of:
- Loading and parsing multiple CSV test files from a directory
- Calculating statistical metrics (mean, standard deviation) across test runs
- Generating comparative visualizations for sensor channels
- Consolidating statistics into a single report file

## Features

- **Graphical User Interface**: Simple, intuitive PyQt5 interface for easy operation
- **Dual Data Mode Support**: Toggle between Taurus and Variable data processing
- **Batch Processing**: Automatically processes all CSV files in a selected directory
- **Statistical Analysis**: Computes mean values per test and overall statistics across all tests
- **Visualization**: Generates side-by-side plots comparing channels across multiple test runs
- **Report Generation**: Exports statistics to CSV format and combines all results into a single file
- **Automatic Organization**: Creates output directory structure and manages file organization

## Project Structure

```
Taurus-Variable-Data-Analysis/
├── src/
│   ├── main.py              # Main application entry point
│   ├── build/                # Build artifacts
│   ├── dist/
│   │   └── main.exe          # Windows executable (if built)
│   ├── main.spec             # PyInstaller specification file
│   └── ui/
│       └── file_selector.py  # UI components
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Requirements

- Python 3.6 or higher
- PyQt5
- pandas
- matplotlib
- numpy

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone or download this repository
2. Navigate to the project directory:
   ```bash
   cd Taurus-Variable-Data-Analysis
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Option 1: Run from Source
```bash
python src/main.py
```

### Option 2: Run Executable (Windows)
Double-click `src/dist/main.exe` or run from command line:
```bash
src/dist/main.exe
```

## Usage Guide

### Step-by-Step Instructions

1. **Launch the Application**: Run the application using one of the methods above

2. **Select Data Directory**: 
   - Click the "Select File" button
   - Navigate to and select the directory containing your CSV test files
   - The selected path will be displayed in the interface

3. **Choose Data Type**:
   - Check the "Taurus Data" checkbox if analyzing Taurus Box data (T1-T28)
   - Leave unchecked for Variable Box data (D1-D64)

4. **Run Analysis**:
   - Click the "Submit" button to begin processing
   - The application will:
     - Load all CSV files from the selected directory
     - Process each file (skipping first 9 rows, extracting relevant columns)
     - Generate statistical analysis for each channel group
     - Create visualization plots
     - Save all outputs to a `Stats` subdirectory

5. **View Results**:
   - Navigate to the `Stats` folder in your selected directory
   - Find:
     - `figure_columns_X.png`: Visualization plots for each channel group
     - `statistics_columns_X.csv`: Statistical data for each group
     - `combined_data.csv`: Consolidated statistics from all groups

### Data Format Requirements

**CSV File Structure:**
- Files must be CSV format (`.csv` or `.CSV` extension)
- First 9 rows are skipped (header/metadata)
- Column structure depends on data type:
  - **Taurus Data**: Columns 8-35 (28 sensor channels)
  - **Variable Data**: Columns 9-73 (64 sensor channels)

**Channel Grouping:**
- **Taurus**: Channels grouped into 7 groups of 4 (T1-T4, T5-T8, ..., T25-T28)
- **Variable**: Channels grouped into 16 groups of 4 (D1-D4, D5-D8, ..., D61-D64)

### Output Files

The application generates the following outputs in the `Stats` subdirectory:

1. **Visualization Files** (`figure_columns_X.png`):
   - Grid plots showing each channel's data across all test runs
   - One figure per channel group
   - X-axis: Data point index
   - Y-axis: Sensor value

2. **Statistics Files** (`statistics_columns_X.csv`):
   - Mean value for each channel per test run
   - Overall mean across all tests
   - Standard deviation across tests
   - Transposed format for easy reading

3. **Combined Report** (`combined_data.csv`):
   - Consolidated statistics from all channel groups
   - Single file containing complete analysis results

## How It Works

### Data Processing Pipeline

1. **File Loading**: Scans selected directory for CSV files
2. **Data Extraction**: Reads each CSV, skipping metadata rows and extracting relevant sensor columns
3. **Test Organization**: Assigns each CSV file as a separate test run (test_1, test_2, etc.)
4. **Statistical Calculation**: 
   - Computes mean for each channel in each test
   - Calculates overall mean and standard deviation across tests
5. **Visualization**: Creates subplot grids comparing channels across tests
6. **Report Generation**: Exports statistics to CSV and combines all results

### Statistical Metrics

- **Per-Test Mean**: Average value of each channel for each individual test run
- **Mean Across Tests**: Overall average when combining all test runs
- **Standard Deviation**: Variability measure across test runs

## Troubleshooting

**Error: "Encountered Error During Analysis"**
- Verify CSV files are in the correct format
- Ensure files have the required number of columns
- Check that the directory path is accessible

**Missing Columns Warning**
- Some channels may not exist in all test files
- The application will skip missing channels and continue processing

**Output Directory Issues**
- The application automatically creates the `Stats` folder if it doesn't exist
- Ensure you have write permissions in the selected directory

## Technical Details

- **GUI Framework**: PyQt5
- **Data Processing**: pandas
- **Visualization**: matplotlib
- **File Format**: CSV input/output
- **Platform**: Cross-platform (Windows executable available)


