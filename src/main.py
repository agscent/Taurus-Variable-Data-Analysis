from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QCheckBox, QLabel
from PyQt5.QtCore import Qt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

class FileSelectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("File Selector App")
        self.setGeometry(100, 100, 400, 400)
        self.taurus_bool = False
        self.col_list = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.label = QLabel("Selected File Path: None")
        layout.addWidget(self.label)

        self.file_button = QPushButton("Select File")
        self.file_button.clicked.connect(self.select_file)
        layout.addWidget(self.file_button)

        self.taurus_checkbox = QCheckBox("Taurus Data")
        self.taurus_checkbox.stateChanged.connect(self.update_taurus_bool)
        layout.addWidget(self.taurus_checkbox)

        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.submit)
        layout.addWidget(self.submit_button)

        self.status_label = QLabel("")  # Add a status label at the bottom
        layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

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
        Parse the Excel file and return a DataFrame.
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
                if (self.taurus_bool == True):
                    df_list.append(pd.read_csv(self.selected_file_path+"/"+file, skiprows=9, usecols=T_Cols))
                else:
                    df_list.append(pd.read_csv(self.selected_file_path+"/"+file, skiprows=9, usecols=V_Cols))
        return df_list

    def plot_channels_and_save_statistics(self, test_dataframes, columns, output_csv, output_figure):
        """
        Plot channels side by side, save the figure, and save statistics to a transposed CSV file.

        Parameters:
        - test_dataframes: dict, a dictionary where keys are test names (e.g., 'test_1') and values are DataFrames.
        - columns: list, the column names to plot (e.g., ['D1', 'D2', ..., 'D28']).
        - output_csv: str, the path to save the CSV file with statistics.
        - output_figure: str, the path to save the figure.
        """
        num_tests = len(test_dataframes)
        num_columns = len(columns)
        
        stats = {test_name: [] for test_name in test_dataframes.keys()}
        stats['Mean_across_tests'] = []
        stats['Std_dev_across_tests'] = []

        fig, axs = plt.subplots(num_columns, num_tests, figsize=(4 * num_tests, 3 * num_columns))
        
        for col_idx, col in enumerate(columns):
            column_data = []
            test_averages = {}
            for test_name, df in test_dataframes.items():
                if col in df.columns:
                    column_data.append(df[col].values)
                    test_averages[test_name] = df[col].mean()  
                    stats[test_name].append(df[col].mean())  
            
            if column_data:
                combined_data = pd.DataFrame(column_data).T  
                mean_across_tests = combined_data.mean(axis=1).mean()  
                std_dev_across_tests = combined_data.mean(axis=1).std()  
            else:
                mean_across_tests = None
                std_dev_across_tests = None
            
            stats['Mean_across_tests'].append(mean_across_tests)
            stats['Std_dev_across_tests'].append(std_dev_across_tests)
            
            for test_idx, (test_name, df) in enumerate(test_dataframes.items()):
                ax = axs[col_idx, test_idx] if num_columns > 1 else axs[test_idx]
                
                if col in df.columns:
                    ax.plot(df[col], label=f'{test_name} - {col}', color='blue')
                    ax.set_title(f'{test_name} - {col}')
                    ax.set_xlabel('Index')
                    ax.set_ylabel('Value')
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=8, loc='upper right')
                else:
                    ax.set_visible(False)  
                    print(f"Warning: Column {col} is missing in {test_name}.")
        
        plt.tight_layout()
        
        # Save the figure
        try:
            plt.savefig(self.selected_file_path+"/Stats/"+output_figure)
        except:
            os.makedirs(self.selected_file_path+"/Stats/")
            plt.savefig(self.selected_file_path+"/Stats/"+output_figure)
        print(f"Figure saved to {output_figure}")
        
        # Optionally show the figure
        # plt.show()
        
        stats_df = pd.DataFrame(stats, index=columns)  
        stats_df = stats_df.transpose()  
        print(stats_df)

        try:
            stats_df.to_csv(self.selected_file_path+"/Stats/"+output_csv, index=True)
        except:
            os.makedirs(self.selected_file_path+"/Stats")
            stats_df.to_csv(self.selected_file_path+"/Stats/"+output_csv, index=True)

        print(f"Statistics saved to {output_csv}")

    def combine_and_delete_csv(self):
        """
        Combine all CSV files in the current directory into a single CSV file and delete the original files.
        """
        combined_df = pd.DataFrame()
        for file in os.listdir(self.selected_file_path+"/Stats"):
            if file.endswith('.csv') or file.endswith('.CSV'):
                if combined_df.empty:
                    combined_df = pd.read_csv(self.selected_file_path+"/Stats"+"/"+file)
                else:
                    df = pd.read_csv(self.selected_file_path+"/Stats"+"/"+file, usecols=[1,2,3,4])
                    combined_df = pd.concat([combined_df, df], ignore_index=True, axis=1)
                os.remove(self.selected_file_path+"/Stats"+"/"+file)  # Delete the original file
        combined_df.to_csv(self.selected_file_path+"/Stats"+"/"+'combined_data.csv', index=False)
        print("Combined data saved to combined_data.csv and original files deleted.")

    def perform_analysis(self):
        
        if (self.taurus_bool == True):
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
        
        test_dataframes = {}
        dfs = self.load_and_parse_xl()

        for i in range(len(dfs)):
            test_dataframes['test_' + str(i+1)] = dfs[i]

        for i, col in enumerate(self.col_list):
            self.plot_channels_and_save_statistics(test_dataframes, col, f'statistics_columns_{i+1}.csv', f'figure_columns_{i+1}.png')

        self.combine_and_delete_csv()
        self.status_label.setText("Analysis Performed Successfully!")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 20px;")
        self.status_label.setAlignment(Qt.AlignCenter)

    def submit(self):
        try:
            self.update_taurus_bool()
            self.perform_analysis()
        except:
            self.status_label.setText("Encountered Error During Analysis")
            self.status_label.setStyleSheet("font-weight: bold; font-size: 20px;")
            self.status_label.setAlignment(Qt.AlignCenter)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FileSelectorApp()
    window.show()
    sys.exit(app.exec_())