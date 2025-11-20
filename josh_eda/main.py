'''
Data Cleaning: Automatically cleans .xlsx files (each file represents a separate sample).
Data Visualization: Generatea the visualizations discussed previously for each cleaned sample.
'''
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from data_cleaning import clean_file
from visualisation_methods import create_visuals   

def select_file():
    tk.Tk().withdraw()
    path = filedialog.askopenfilename(
        title="Select any Taurus .xlsx (raw or cleaned)",
        filetypes=[("Excel files", "*.xlsx")]
    )
    return Path(path) if path else None

def main():
    print("Select a Taurus sample...")
    path = select_file()
    if not path:
        return

    print(f"Selected → {path.name}")

    if path.name.endswith("_CLEANED.xlsx"):
        cleaned_path = path
        print("Already cleaned → skipping cleaning")
    else:
        print("Cleaning file...")
        cleaned_path = clean_file(path)
        print(f"Cleaned → {cleaned_path.name}")

    print("Generating plots...")
    create_visuals(cleaned_path)
    print("Done! Check the plots folder or screen.")

if __name__ == "__main__":
    main()