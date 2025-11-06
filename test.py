import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, 
                             QFileDialog, QCheckBox, QLabel, QGridLayout, QMessageBox, 
                             QProgressBar, QComboBox)
from PyQt5.QtCore import Qt

class FileSelectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Taurus/Variable Data Analysis App")
        self.setGeometry(100, 100, 500, 500)
        self.taurus_bool = False
        self.col_list = None
        self.breath_indices = []
        self.baseline_indices = []
        self.increasing = True
        self.initUI()

    def initUI(self):
        # Create main widget and layout
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)

        # File selection
        self.label = QLabel("Selected File Path: None")
        layout.addWidget(self.label)

        self.file_button = QPushButton("Select File")
        self.file_button.clicked.connect(self.select_file)
        layout.addWidget(self.file_button)

        # Data type selection
        self.taurus_checkbox = QCheckBox("Taurus Data")
        self.taurus_checkbox.stateChanged.connect(self.update_taurus_bool)
        layout.addWidget(self.taurus_checkbox)

        # SNR threshold
        snr_layout = QGridLayout()
        snr_layout.addWidget(QLabel("Signal-to-Noise Ratio Threshold:"), 0, 0)
        self.snr_threshold = QComboBox()
        self.snr_threshold.addItems([str(i) for i in range(0, 11)])
        self.snr_threshold.setCurrentText("3")
        snr_layout.addWidget(self.snr_threshold, 0, 1)
        layout.addLayout(snr_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Submit button
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.submit)
        layout.addWidget(self.submit_button)

        # Status label
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        self.setCentralWidget(main_widget)

    def update_taurus_bool(self):
        self.taurus_bool = self.taurus_checkbox.isChecked()

    def select_file(self):
        options = QFileDialog.Options()
        file_path = QFileDialog.getExistingDirectory(self, "Select a Directory", "C:/Users/yuthm/Desktop/Agscent/Analysis Files", QFileDialog.ShowDirsOnly)
        
        if file_path:
            self.label.setText(f"Selected File Path: {file_path}")
            self.selected_file_path = file_path

    def load_and_parse_xl(self):
        """
        Parse the Excel file and return a DataFrame with properly named columns.
        """
        T_Cols = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
                22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
        
        V_Cols = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
                24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 
                38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 
                52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 
                66, 67, 68, 69, 70, 71, 72, 73]
        
        df_list = []
        for file in os.listdir(self.selected_file_path):
            if file.endswith('.csv') or file.endswith('.CSV'):
                try:
                    if self.taurus_bool:
                        df = pd.read_csv(self.selected_file_path+"/"+file, skiprows=9, usecols=T_Cols)

                        # Rename columns to T1, T2, etc.
                        new_columns = [f'T{i+1}' for i in range(len(df.columns))]
                        df.columns = new_columns
                        df_list.append(df)
                    else:
                        df = pd.read_csv(self.selected_file_path+"/"+file, skiprows=9, usecols=V_Cols)
                        
                        # Rename columns to D1, D2, etc.
                        new_columns = [f'D{i+1}' for i in range(len(df.columns))]
                        df.columns = new_columns
                        df_list.append(df)
                    
                    # if self.breath_indices == []:
                    df = pd.read_csv(self.selected_file_path+"/"+file, skiprows=8)
                    sequence = np.array((df.iloc[:, 0]))
                    self.baseline_indices = np.where(sequence == '1')[0]
                    self.breath_indices = np.where(sequence == '2')[0]

                    if len(self.breath_indices) == 0:
                        self.breath_indices = np.where(sequence == 2)[0]
                    if len(self.baseline_indices) == 0:
                        self.baseline_indices = np.where(sequence == 1)[0]

                    ## Extend breath region by 2 to ensure we capture maximum
                    self.breath_indices = np.append(self.breath_indices, self.breath_indices[-1]+1)
                    self.breath_indices = np.append(self.breath_indices, self.breath_indices[-1]+1)
                    

                except Exception as e:
                    print(True)
                    print(f"Error loading file {file}: {str(e)}")
            
        return df_list
    
    #############################
    ###### R_t - R_0 / R_0 ######
    #############################

    def find_peak(self, spectrum):
        # Convert to numpy array and handle NaN/Inf values
        spectrum = np.array(spectrum)

        ## If response decreases resistance...
        if np.mean(spectrum[self.breath_indices]) < np.mean(spectrum[self.baseline_indices]):
            peak_position = self.breath_indices[0] + np.argmin(spectrum[self.breath_indices])
            peak_height = spectrum[peak_position]
            self.increasing = False
        else:
            peak_position = self.breath_indices[0] + np.argmax(spectrum[self.breath_indices])
            peak_height = spectrum[peak_position]
            self.increasing = True

        return {
            'peak_position': int(peak_position),
            'peak_height': (peak_height),
            'peak_resistance': np.mean(spectrum[self.breath_indices])
        }
    
    def calculate_snr(self, peak_height, baseline_mean, baseline_std):
        if baseline_std == 0:
            return np.inf

        else:
            if self.increasing == True:
                return (peak_height - baseline_mean) / baseline_std
            
            else:
                return (baseline_mean - peak_height) / baseline_std


    def save_all_statistics(self, test_dataframes, columns, output_csv):
        """
        Save peak statistics for all channels to a single CSV file.
        Uses peak values (peak ±1 point) for statistics.
        Uses 0 (instead of NaN) for channels with SNR < threshold.
        Handles both increasing and decreasing curves.
        
        Parameters:
        - test_dataframes: dict, a dictionary where keys are test names (e.g., 'test_1') and values are DataFrames.
        - columns: list, the column names to include (e.g., ['D1', 'D2', ..., 'D64']).
        - output_csv: str, the path to save the CSV file with statistics.
        """
        # Get SNR threshold from UI
        snr_threshold = float(self.snr_threshold.currentText())
        
        # Create a list of column names for the output DataFrame
        column_names = []
        for test_name in test_dataframes.keys():
            column_names.extend([f'{test_name}_mean', f'{test_name}_std', f'{test_name}_absolute_response'])
            
        # Create an empty DataFrame with rows for each feature column and columns for each test statistic
        output_df = pd.DataFrame(index=columns, columns=column_names)

        # Calculate statistics for each channel
        for col in columns:
            for test_name, df in test_dataframes.items():
                if col in df.columns:
                    # Get data for this channel
                    data = df[col].values
                    
                    # Calculate baseline (last 5 points)
                    baseline = data[self.baseline_indices[::-1][:5]]
                    baseline_mean = np.mean(baseline)
                    baseline_std = np.std(baseline)
                    
                    # Find peak using function
                    peak_info = self.find_peak(data)
                    
                    # Default to 0 for all values (instead of NaN)
                    mean_val = 0
                    std_val = 0
                    absolute_response = 0
                    
                    if peak_info:
                        # Calculate SNR as peak_height - avg_baseline
                        peak_height = peak_info['peak_height']
                        snr = self.calculate_snr(peak_height, baseline_mean, baseline_std)

                        if ((type(snr) == float) or (type(snr) == np.float64)) and (snr > 30):
                            snr = 30
                        
                        # Calculate absolute response - use absolute difference for all curves
                        absolute_response = (peak_height - baseline_mean) / baseline_mean
                        
                        # Only calculate other statistics if SNR >= threshold
                        if snr >= snr_threshold:
                            # Use peak position and surrounding points (peak region) for statistics
                            peak_position = peak_info['peak_position']
                            
                            # Define peak region as peak ± 1 point (3 points total)
                            start_idx = max(0, peak_position - 1)
                            end_idx = min(len(data), peak_position + 2)  # +2 because end index is exclusive
                            peak_region = data[start_idx:end_idx]
                            
                            # Calculate statistics using only the peak region
                            mean_val = np.mean(peak_region)
                            std_val = np.std(peak_region)
                        else:
                            # If SNR < threshold, set absolute_response to 0
                            absolute_response = 0
                    
                    # Store results in the output DataFrame (will be 0 if SNR < threshold or no peak found)
                    output_df.loc[col, f'{test_name}_mean'] = mean_val
                    output_df.loc[col, f'{test_name}_std'] = std_val
                    output_df.loc[col, f'{test_name}_absolute_response'] = absolute_response
        
        # Calculate mean across tests (including 0 values)
        mean_cols = [col for col in output_df.columns if 'mean' in col]
        std_cols = [col for col in output_df.columns if 'std' in col]
        abs_resp_cols = [col for col in output_df.columns if 'absolute_response' in col]
        
        # Don't use skipna=True since we want to include the 0 values in the average
        output_df['mean_across_tests'] = output_df[mean_cols].mean(axis=1)
        output_df['std_across_tests'] = output_df[std_cols].mean(axis=1)
        output_df['absolute_response_across_tests'] = output_df[abs_resp_cols].mean(axis=1)
        
        # Transpose the DataFrame - this will make statistics the rows and channels the columns
        output_df_transposed = output_df.transpose()
        
        # Add a 'Statistic' column to the transposed DataFrame
        output_df_transposed['Statistic'] = output_df_transposed.index
        
        # Move the Statistic column to the front
        cols = output_df_transposed.columns.tolist()
        cols = ['Statistic'] + [col for col in cols if col != 'Statistic']
        output_df_transposed = output_df_transposed[cols]
        
        # Save the statistics - add index=False to remove the index column
        try:
            output_df_transposed.to_csv(self.selected_file_path+"/Stats/"+output_csv, index=False)
        except:
            os.makedirs(self.selected_file_path+"/Stats/", exist_ok=True)
            output_df_transposed.to_csv(self.selected_file_path+"/Stats/"+output_csv, index=False)
            
        # Update print statement for clarity
        print(f"Peak statistics saved to {output_csv}")

    def plot_group_only(self, test_dataframes, columns, output_figure):
        """
        Plot channels side by side and save the figure only (no CSV).
        Highlights peak region used for statistics calculation if SNR >= threshold.
        Uses X marker for peaks with SNR < threshold.
        Handles both increasing and decreasing curves.
        
        Parameters:
        - test_dataframes: dict, a dictionary where keys are test names (e.g., 'test_1') and values are DataFrames.
        - columns: list, the column names to plot (e.g., ['D1', 'D2', ..., 'D28']).
        - output_figure: str, the path to save the figure.
        """
        num_tests = len(test_dataframes)
        num_columns = len(columns)
        
        # Get SNR threshold from UI
        snr_threshold = float(self.snr_threshold.currentText())
        
        # Create a figure with subplots arranged in a grid
        fig, axs = plt.subplots(num_columns, num_tests, figsize=(4 * num_tests, 3 * num_columns))
        
        # If there's only one subplot, axs will be a scalar, not an array
        if num_columns == 1 and num_tests == 1:
            axs = np.array([[axs]])
        elif num_columns == 1:
            axs = np.array([axs]).reshape(1, -1)
        elif num_tests == 1:
            axs = axs.reshape(-1, 1)

        # Loop through each column (feature) and plot it for each test
        for i, col in enumerate(columns):
            for j, (test_name, df) in enumerate(test_dataframes.items()):
                if col in df.columns:
                    # Get data for this channel
                    data = df[col].values
                    
                    # Calculate baseline statistics (Last 5 points)
                    baseline = data[self.baseline_indices[::-1][:5]]
                    baseline_mean = np.mean(baseline)
                    baseline_std = np.std(baseline)

                    # Find peak using autocorrelation
                    peak_info = self.find_peak(data)
                    
                    # Plot the data
                    axs[i, j].plot(data)
                    
                    # Highlight baseline region (Last 5 points)
                    axs[i, j].axvspan(self.baseline_indices[-1], self.baseline_indices[-6], color='lightblue', alpha=0.3, label='Baseline')
                    
                    # Mark detected peak if found
                    if peak_info:
                        peak_position = peak_info['peak_position']
                        peak_height = peak_info['peak_height']
                        peak_resistance = peak_info['peak_resistance']
                        
                        # Calculate SNR
                        snr = self.calculate_snr(peak_height, baseline_mean, baseline_std)
                        if ((type(snr) == float) or (type(snr) == np.float64)) and (snr > 30):
                            snr = 30
                        
                        # If SNR >= threshold, highlight peak region and mark with a dot
                        if snr >= snr_threshold:
                            # Choose marker color based on curve direction
                            marker_color = 'bo'
                            
                            # Mark the peak point with SNR value in the label
                            axs[i, j].plot(peak_position, peak_height, marker_color, markersize=6, 
                                        label=f'Peak (SNR: {snr:.1f})')
                                        
                            # Define peak region as peak ± 1 point (3 points total)
                            start_idx = max(0, peak_position - 1)
                            end_idx = min(len(data), peak_position + 2)  # +2 because end index is exclusive
                            
                            # Highlight the peak region
                            region_color = 'orange'
                            axs[i, j].axvspan(start_idx, end_idx-1, color=region_color, alpha=0.3, 
                                            label='Peak Region (For Stats)')
                        else:
                            # Mark with an X for low SNR peaks
                            marker_color = 'bx'
                            axs[i, j].plot(peak_position, peak_height, marker_color, markersize=8, 
                                        label=f'Peak (SNR: {snr:.1f} < {snr_threshold})')
                                    
                    axs[i, j].set_title(f'{test_name} - {col}')
                    axs[i, j].set_xlabel('Index')
                    axs[i, j].set_ylabel('Value')
                    axs[i, j].grid(True, alpha=0.3)
                    axs[i, j].legend(fontsize=8, loc='upper right')
                else:
                    axs[i, j].set_visible(False)
                    print(f"Warning: Column {col} is missing in {test_name}.")
        
        plt.tight_layout()
        
        # Save the figure
        try:
            plt.savefig(self.selected_file_path+"/Stats/"+output_figure)
        except:
            os.makedirs(self.selected_file_path+"/Stats/", exist_ok=True)
            plt.savefig(self.selected_file_path+"/Stats/"+output_figure)
        print(f"Figure saved to {output_figure}")
        plt.close(fig)

    def save_baseline_statistics(self, test_dataframes, columns, output_csv):
        """
        Save baseline statistics (first 5 points) for all channels to a separate CSV file.
        
        Parameters:
        - test_dataframes: dict, a dictionary where keys are test names (e.g., 'test_1') and values are DataFrames.
        - columns: list, the column names to include (e.g., ['D1', 'D2', ..., 'D64']).
        - output_csv: str, the path to save the CSV file with statistics.
        """
        # Create a list of column names for the output DataFrame
        column_names = []
        for test_name in test_dataframes.keys():
            column_names.extend([f'{test_name}_mean', f'{test_name}_std'])
            
        # Create an empty DataFrame with rows for each feature column and columns for each test statistic
        output_df = pd.DataFrame(index=columns, columns=column_names)

        # Calculate statistics for each channel using baseline (first 5 points)
        for col in columns:
            for test_name, df in test_dataframes.items():
                if col in df.columns:
                    # Get data for this channel
                    data = df[col].values
                    
                    # Use first 5 points as baseline
                    baseline = data[self.baseline_indices[::-1][:5]]
                    
                    if len(baseline) > 0:
                        # Calculate statistics using baseline region
                        mean_val = np.mean(baseline)
                        std_val = np.std(baseline)
                    else:
                        # No baseline data available
                        mean_val = np.nan
                        std_val = np.nan
                    
                    # Store results in the output DataFrame
                    output_df.loc[col, f'{test_name}_mean'] = mean_val
                    output_df.loc[col, f'{test_name}_std'] = std_val
        
        # Calculate mean across tests (ignoring NaN values)
        mean_cols = [col for col in output_df.columns if 'mean' in col]
        std_cols = [col for col in output_df.columns if 'std' in col]
        
        output_df['mean_across_tests'] = output_df[mean_cols].mean(axis=1, skipna=True)
        output_df['std_across_tests'] = output_df[std_cols].mean(axis=1, skipna=True)
        
        # Transpose the DataFrame - this will make statistics the rows and channels the columns
        output_df_transposed = output_df.transpose()
        
        # Add a 'Statistic' column to the transposed DataFrame
        output_df_transposed['Statistic'] = output_df_transposed.index
        
        # Move the Statistic column to the front
        cols = output_df_transposed.columns.tolist()
        cols = ['Statistic'] + [col for col in cols if col != 'Statistic']
        output_df_transposed = output_df_transposed[cols]
        
        # Save the statistics - add index=False to remove the index column
        try:
            output_df_transposed.to_csv(self.selected_file_path+"/Stats/"+output_csv, index=False)
        except:
            os.makedirs(self.selected_file_path+"/Stats/", exist_ok=True)
            output_df_transposed.to_csv(self.selected_file_path+"/Stats/"+output_csv, index=False)
        
        print(f"Baseline statistics saved to {output_csv}")

    def perform_analysis(self):
        """
        Perform data analysis with separate plots for each group,
        and save both peak and baseline statistics in separate CSV files.
        Uses progress bar to track group processing.
        """
        test_dataframes = {}
        dfs = self.load_and_parse_xl()

        for i in range(len(dfs)):
            test_dataframes['test_' + str(i+1)] = dfs[i]
        
        # Define channel groups
        if self.taurus_bool:
            # Taurus data - T1-T28 in groups of 4
            self.col_list = [
                ['T1', 'T2', 'T3', 'T4'],
                ['T5', 'T6', 'T7', 'T8'],
                ['T9', 'T10', 'T11', 'T12'],
                ['T13', 'T14', 'T15', 'T16'],
                ['T17', 'T18', 'T19', 'T20'],
                ['T21', 'T22', 'T23', 'T24'],
                ['T25', 'T26', 'T27', 'T28']
            ]
            # Full list of all channels
            all_columns = [f'T{i+1}' for i in range(28)]
        else:
            # Variable data - D1-D64 in groups of 4
            self.col_list = [
                ['D1', 'D2', 'D3', 'D4'],
                ['D5', 'D6', 'D7', 'D8'],
                ['D9', 'D10', 'D11', 'D12'],
                ['D13', 'D14', 'D15', 'D16'],
                ['D17', 'D18', 'D19', 'D20'],
                ['D21', 'D22', 'D23', 'D24'],
                ['D25', 'D26', 'D27', 'D28'],
                ['D29', 'D30', 'D31', 'D32'],
                ['D33', 'D34', 'D35', 'D36'],
                ['D37', 'D38', 'D39', 'D40'],
                ['D41', 'D42', 'D43', 'D44'],
                ['D45', 'D46', 'D47', 'D48'],
                ['D49', 'D50', 'D51', 'D52'],
                ['D53', 'D54', 'D55', 'D56'],
                ['D57', 'D58', 'D59', 'D60'],
                ['D61', 'D62', 'D63', 'D64']
            ]
            # Full list of all channels
            all_columns = [f'D{i+1}' for i in range(64)]

        # Set up progress bar
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(len(self.col_list))
        self.progress_bar.setValue(0)

        # First create all the plots in groups with progress bar updates
        for i, group_cols in enumerate(self.col_list):
            # Update progress bar
            self.progress_bar.setValue(i + 1)
            QApplication.processEvents()  # Process events to update UI
            
            # Plot only - don't save individual CSVs
            self.plot_group_only(test_dataframes, group_cols, f'figure_group_{i+1}.png')
        
        # Then generate CSVs with all channels
        self.save_all_statistics(test_dataframes, all_columns, 'peak_statistics.csv')
        self.save_baseline_statistics(test_dataframes, all_columns, 'baseline_statistics.csv')
        
        self.status_label.setText("Analysis Performed Successfully!")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 20px;")
        self.status_label.setAlignment(Qt.AlignCenter)

    def submit(self):
        try:
            self.update_taurus_bool()
            self.progress_bar.setValue(0)  # Reset progress bar
            self.status_label.setText("Analysis in progress...")
            QApplication.processEvents()  # Process events to update UI
            self.perform_analysis()
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            self.status_label.setText("Encountered Error During Analysis")
            self.status_label.setStyleSheet("font-weight: bold; font-size: 20px;")
            self.status_label.setAlignment(Qt.AlignCenter)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FileSelectorApp()
    window.show()
    sys.exit(app.exec_())