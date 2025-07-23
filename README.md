# PyQt5 Taurus/Variable Box Data Analysis App

This project is a simple PyQt5 application that allows users to select a directory to perform a fast and easy to interepret analysis for the Taurus and Variable box data.

## Project Structure

```
pyqt5-file-selector-app
├── src
│   ├── main.py          # Entry point of the application
|   └── build
|   └── dist
|       └── main.exe     # Executable containing analysis app
│   └── ui
│       └── file_selector.py  # User interface for file selection
|   
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

## Requirements

To run this application, you need to have Python installed along with the following dependencies:

- PyQt5
- pandas
- matplotlib
- numpy

You can install the required dependencies using pip:

```
pip install -r requirements.txt
```

## Running the Application

1. Clone the repository or download the project files.
2. Navigate to the project directory.
3. Run the application using the following command:
```
python src/main.py
```
4. Alternatively you can run directly through the executable

## Usage

- Click the "Select File" button to choose a file path.
- Use the toggle to set the `Taurus Data` variable to True or False.
- The selected file path and the value of `Taurus_bool` will be used in the application logic.

## License

This project is open-source and available under the MIT License.
