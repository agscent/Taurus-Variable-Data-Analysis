from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QCheckBox, QLabel, QMessageBox
from PyQt5.QtCore import Qt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import traceback

class FileSelectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Taurus/Variable Data Analysis")
        self.setGeometry(100, 100, 500, 400)
        self.taurus_bool = False
        self.col_list = None
        self.selected_file_path = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.label = QLabel("Selected Directory: None")
        self.label.setWordWrap(True)
        layout.addWidget(self.label)

        self.file_button = QPushButton("Select Directory")
        self.file_button.clicked.connect(self.select_file)
        layout.addWidget(self.file_button)

        self.taurus_checkbox = QCheckBox("Taurus Data (uncheck for Variable Box)")
        self.taurus_checkbox.stateChanged.connect(self.update_taurus_bool)
        layout.addWidget(self.taurus_checkbox)

        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.submit)
        layout.addWidget(self.submit_button)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def update_taurus_bool(self):
        self.taurus_bool = self.taurus_checkbox.isChecked()

    def select_file(self):
        # Use home directory as default starting point
        default_path = os.path.expanduser("~")
        file_path = QFileDialog.getExistingDirectory(
            self, 
            "Select a Directory Containing CSV Files", 
            default_path, 
            QFileDialog.ShowDirsOnly
        )
        
        if file_path:
            self.label.setText(f"Selected Directory: {file_path}")
            self.selected_file_path = file_path

    def detect_file_format(self, file_path):
        """
        Auto-detect if file is Taurus or Variable format based on column names.
        Returns: 'taurus', 'variable', or None if unknown
        """
        try:
            # Read just the header row (after skipping metadata)
            df_header = pd.read_csv(file_path, skiprows=9, nrows=0)
            columns = df_header.columns.tolist()
            
            # Check for Taurus columns (T1, T2, etc.)
            taurus_cols = [c for c in columns if str(c).startswith('T') and str(c)[1:].isdigit()]
            if len(taurus_cols) >= 20:
                return 'taurus'
            
            # Check for Variable columns (D1, D2, etc.)
            variable_cols = [c for c in columns if str(c).startswith('D') and str(c)[1:].isdigit()]
            if len(variable_cols) >= 50:
                return 'variable'
            
            return None
        except Exception as e:
            print(f"Error detecting format: {e}")
            return None

    def load_and_parse_xl(self):
        """
        Parse CSV files and return a list of DataFrames.
        Handles empty rows and validates data.
        """
        # Taurus: columns 8-35 (T1-T28, 28 channels)
        T_Cols = list(range(8, 36))  # indices 8-35 inclusive
        
        # Variable: columns 9-72 (D1-D64, 64 channels)
        # Fixed: was 9-73 which included Humidity column
        V_Cols = list(range(9, 73))  # indices 9-72 inclusive
        
        df_list = []
        files_processed = []
        errors = []
        
        # Check if directory exists and has CSV files
        if not os.path.isdir(self.selected_file_path):
            raise ValueError(f"Directory not found: {self.selected_file_path}")
        
        csv_files = [f for f in os.listdir(self.selected_file_path) 
                     if f.lower().endswith('.csv')]
        
        if not csv_files:
            raise ValueError(f"No CSV files found in: {self.selected_file_path}")
        
        for file in csv_files:
            file_path = os.path.join(self.selected_file_path, file)
            try:
                # Auto-detect format if needed
                detected_format = self.detect_file_format(file_path)
                if detected_format:
                    print(f"File {file}: Detected as {detected_format} format")
                
                # Read the file
                if self.taurus_bool:
                    df = pd.read_csv(file_path, skiprows=9, usecols=T_Cols)
                else:
                    df = pd.read_csv(file_path, skiprows=9, usecols=V_Cols)
                
                # Validate we got the expected columns
                if df.empty:
                    errors.append(f"{file}: No data rows found")
                    continue
                
                # Drop rows where ALL sensor values are NaN (empty data rows like "Initialize system")
                df_cleaned = df.dropna(how='all')
                
                # Also drop rows where most values are NaN (> 50% missing)
                threshold = len(df.columns) * 0.5
                df_cleaned = df_cleaned.dropna(thresh=int(threshold))
                
                if df_cleaned.empty:
                    errors.append(f"{file}: All rows have empty sensor data")
                    continue
                
                # Convert to numeric, coercing errors to NaN
                df_cleaned = df_cleaned.apply(pd.to_numeric, errors='coerce')
                
                df_list.append(df_cleaned)
                files_processed.append(file)
                print(f"Successfully loaded: {file} ({len(df_cleaned)} rows)")
                
            except Exception as e:
                errors.append(f"{file}: {str(e)}")
                print(f"Error loading {file}: {e}")
        
        if errors:
            print("\n--- Errors encountered ---")
            for err in errors:
                print(f"  - {err}")
        
        if not df_list:
            error_msg = "No valid CSV files could be loaded.\n\nErrors:\n" + "\n".join(errors)
            raise ValueError(error_msg)
        
        print(f"\nSuccessfully loaded {len(df_list)} file(s): {files_processed}")
        return df_list

    def plot_channels_and_save_statistics(self, test_dataframes, columns, output_csv, output_figure):
        """
        Plot channels side by side, save the figure, and save statistics to a transposed CSV file.
        """
        num_tests = len(test_dataframes)
        num_columns = len(columns)
        
        if num_tests == 0:
            raise ValueError("No test data to plot")
        
        stats = {test_name: [] for test_name in test_dataframes.keys()}
        stats['Mean_across_tests'] = []
        stats['Std_dev_across_tests'] = []

        # Handle single test case
        if num_tests == 1:
            fig, axs = plt.subplots(num_columns, 1, figsize=(6, 3 * num_columns))
            if num_columns == 1:
                axs = [axs]
        else:
            fig, axs = plt.subplots(num_columns, num_tests, figsize=(4 * num_tests, 3 * num_columns))
        
        missing_columns = []
        
        for col_idx, col in enumerate(columns):
            column_data = []
            for test_name, df in test_dataframes.items():
                if col in df.columns:
                    # Drop NaN values for this column
                    valid_data = df[col].dropna()
                    if len(valid_data) > 0:
                        column_data.append(valid_data.values)
                        stats[test_name].append(valid_data.mean())
                    else:
                        stats[test_name].append(np.nan)
                else:
                    missing_columns.append(f"{col} in {test_name}")
                    stats[test_name].append(np.nan)
            
            if column_data:
                # Pad arrays to same length for combining
                max_len = max(len(arr) for arr in column_data)
                padded_data = [np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan) 
                               for arr in column_data]
                combined_data = pd.DataFrame(padded_data).T
                mean_across_tests = combined_data.mean(axis=1).mean()
                std_dev_across_tests = combined_data.mean(axis=1).std()
            else:
                mean_across_tests = np.nan
                std_dev_across_tests = np.nan
            
            stats['Mean_across_tests'].append(mean_across_tests)
            stats['Std_dev_across_tests'].append(std_dev_across_tests)
            
            for test_idx, (test_name, df) in enumerate(test_dataframes.items()):
                if num_tests == 1:
                    ax = axs[col_idx] if num_columns > 1 else axs[0]
                else:
                    ax = axs[col_idx, test_idx] if num_columns > 1 else axs[test_idx]
                
                if col in df.columns:
                    valid_data = df[col].dropna()
                    if len(valid_data) > 0:
                        ax.plot(valid_data.values, label=f'{test_name} - {col}', color='blue')
                        ax.set_title(f'{test_name} - {col}')
                        ax.set_xlabel('Index')
                        ax.set_ylabel('Value')
                        ax.grid(True, alpha=0.3)
                        ax.legend(fontsize=8, loc='upper right')
                    else:
                        ax.set_visible(False)
                else:
                    ax.set_visible(False)
        
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns[:5]}...")  # Show first 5
        
        plt.tight_layout()
        
        # Create Stats directory if needed
        stats_dir = os.path.join(self.selected_file_path, "Stats")
        os.makedirs(stats_dir, exist_ok=True)
        
        # Save the figure
        fig_path = os.path.join(stats_dir, output_figure)
        plt.savefig(fig_path)
        plt.close(fig)  # Close figure to free memory
        print(f"Figure saved to {fig_path}")
        
        # Save statistics
        stats_df = pd.DataFrame(stats, index=columns)
        stats_df = stats_df.transpose()
        
        csv_path = os.path.join(stats_dir, output_csv)
        stats_df.to_csv(csv_path, index=True)
        print(f"Statistics saved to {csv_path}")

    def combine_and_delete_csv(self):
        """
        Combine all CSV files in Stats directory into a single CSV file and delete the originals.
        """
        stats_dir = os.path.join(self.selected_file_path, "Stats")
        combined_df = pd.DataFrame()
        files_to_delete = []
        
        for file in os.listdir(stats_dir):
            if file.lower().endswith('.csv') and file != 'combined_data.csv':
                file_path = os.path.join(stats_dir, file)
                try:
                    if combined_df.empty:
                        combined_df = pd.read_csv(file_path)
                    else:
                        df = pd.read_csv(file_path)
                        # Get all columns except the first (index) column
                        cols_to_add = df.columns[1:].tolist()
                        combined_df = pd.concat([combined_df, df[cols_to_add]], axis=1)
                    files_to_delete.append(file_path)
                except Exception as e:
                    print(f"Error combining {file}: {e}")
        
        if not combined_df.empty:
            output_path = os.path.join(stats_dir, 'combined_data.csv')
            combined_df.to_csv(output_path, index=False)
            print(f"Combined data saved to {output_path}")
            
            # Delete original files
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Could not delete {file_path}: {e}")

    def perform_analysis(self):
        """
        Main analysis function that coordinates loading, plotting, and statistics.
        """
        if self.selected_file_path is None:
            raise ValueError("Please select a directory first")
        
        # Define column groups based on data type
        if self.taurus_bool:
            # Taurus: T1-T28 (28 channels)
            columns_1 = ['T1', 'T2', 'T3', 'T4']
            columns_2 = ['T5', 'T6', 'T7', 'T8']
            columns_3 = ['T9', 'T10', 'T11', 'T12']
            columns_4 = ['T13', 'T14', 'T15', 'T16']
            columns_5 = ['T17', 'T18', 'T19', 'T20']
            columns_6 = ['T21', 'T22', 'T23', 'T24']
            columns_7 = ['T25', 'T26', 'T27', 'T28']

            self.col_list = [columns_1, columns_2, columns_3, columns_4,
                        columns_5, columns_6, columns_7]
        else:
            # Variable Box: D1-D64 (64 channels)
            columns_D1_D4 = ['D1', 'D2', 'D3', 'D4']
            columns_D5_D8 = ['D5', 'D6', 'D7', 'D8']
            columns_D9_D12 = ['D9', 'D10', 'D11', 'D12']
            columns_D13_D16 = ['D13', 'D14', 'D15', 'D16']
            columns_D17_D20 = ['D17', 'D18', 'D19', 'D20']
            columns_D21_D24 = ['D21', 'D22', 'D23', 'D24']
            columns_D25_D28 = ['D25', 'D26', 'D27', 'D28']
            columns_D29_D32 = ['D29', 'D30', 'D31', 'D32']
            columns_D33_D36 = ['D33', 'D34', 'D35', 'D36']
            columns_D37_D40 = ['D37', 'D38', 'D39', 'D40']
            columns_D41_D44 = ['D41', 'D42', 'D43', 'D44']
            columns_D45_D48 = ['D45', 'D46', 'D47', 'D48']
            columns_D49_D52 = ['D49', 'D50', 'D51', 'D52']
            columns_D53_D56 = ['D53', 'D54', 'D55', 'D56']
            columns_D57_D60 = ['D57', 'D58', 'D59', 'D60']
            columns_D61_D64 = ['D61', 'D62', 'D63', 'D64']

            self.col_list = [columns_D1_D4, columns_D5_D8, columns_D9_D12, columns_D13_D16,
                        columns_D17_D20, columns_D21_D24, columns_D25_D28, columns_D29_D32,
                        columns_D33_D36, columns_D37_D40, columns_D41_D44, columns_D45_D48,
                        columns_D49_D52, columns_D53_D56, columns_D57_D60, columns_D61_D64]
        
        # Load and parse data
        print(f"\n{'='*50}")
        print(f"Starting analysis...")
        print(f"Mode: {'Taurus' if self.taurus_bool else 'Variable Box'}")
        print(f"Directory: {self.selected_file_path}")
        print(f"{'='*50}\n")
        
        dfs = self.load_and_parse_xl()
        
        if not dfs:
            raise ValueError("No data loaded from CSV files")
        
        test_dataframes = {}
        for i, df in enumerate(dfs):
            test_dataframes[f'test_{i+1}'] = df

        # Process each column group
        for i, col in enumerate(self.col_list):
            print(f"\nProcessing column group {i+1}/{len(self.col_list)}: {col}")
            self.plot_channels_and_save_statistics(
                test_dataframes, 
                col, 
                f'statistics_columns_{i+1}.csv', 
                f'figure_columns_{i+1}.png'
            )

        self.combine_and_delete_csv()
        
        print(f"\n{'='*50}")
        print("Analysis completed successfully!")
        print(f"{'='*50}\n")

    def show_error_dialog(self, title, message, details=None):
        """Show a detailed error dialog"""
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        if details:
            msg_box.setDetailedText(details)
        msg_box.exec_()

    def submit(self):
        """Handle submit button click with proper error handling"""
        try:
            self.status_label.setText("Processing...")
            self.status_label.setStyleSheet("font-weight: bold; font-size: 16px; color: blue;")
            QApplication.processEvents()  # Update UI
            
            self.update_taurus_bool()
            self.perform_analysis()
            
            self.status_label.setText("Analysis Performed Successfully!")
            self.status_label.setStyleSheet("font-weight: bold; font-size: 20px; color: green;")
            self.status_label.setAlignment(Qt.AlignCenter)
            
        except ValueError as e:
            # User-friendly errors (missing files, wrong format, etc.)
            error_msg = str(e)
            self.status_label.setText(f"Error: {error_msg[:50]}...")
            self.status_label.setStyleSheet("font-weight: bold; font-size: 14px; color: red;")
            self.show_error_dialog("Analysis Error", error_msg)
            
        except Exception as e:
            # Unexpected errors - show full traceback
            error_msg = str(e)
            full_traceback = traceback.format_exc()
            print(f"Error: {error_msg}")
            print(f"Traceback:\n{full_traceback}")
            
            self.status_label.setText("Error During Analysis (see details)")
            self.status_label.setStyleSheet("font-weight: bold; font-size: 14px; color: red;")
            self.show_error_dialog(
                "Analysis Error", 
                f"An error occurred: {error_msg}",
                details=full_traceback
            )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FileSelectorApp()
    window.show()
    sys.exit(app.exec_())
