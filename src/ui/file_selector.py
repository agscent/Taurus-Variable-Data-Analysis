from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QCheckBox, QFileDialog

class FileSelectorUI(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('File Selector')
        
        layout = QVBoxLayout()

        self.label = QLabel('Selected File Path: None')
        layout.addWidget(self.label)

        self.select_button = QPushButton('Select File')
        self.select_button.clicked.connect(self.select_file)
        layout.addWidget(self.select_button)

        self.taurus_checkbox = QCheckBox('Set Taurus_bool to True')
        layout.addWidget(self.taurus_checkbox)

        self.setLayout(layout)

    def select_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a File", "", "All Files (*);;CSV Files (*.csv)", options=options)
        if file_path:
            self.label.setText(f'Selected File Path: {file_path}')