import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel
import pandas as pd


class ECGApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Set up the layout and widgets
        layout = QVBoxLayout()

        # Label to display the selected file path
        self.file_label = QLabel("No file selected", self)
        layout.addWidget(self.file_label)

        # Browse button to select the file
        browse_button = QPushButton("Browse ECG Excel File", self)
        browse_button.clicked.connect(self.browse_file)
        layout.addWidget(browse_button)

        # Set layout to the window
        self.setLayout(layout)
        self.setWindowTitle("ECG Signal Equalizer")
        self.setGeometry(300, 300, 400, 200)

    def browse_file(self):
        # Open file dialog to select an Excel file
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ECG File",
            "",
            "Excel and CSV Files (*.xlsx *.xls *.xlsm *.csv);;All Files (*)",
            options=options
        )

        if file_path:
            # Display the selected file path in the label
            self.file_label.setText(f"Selected file: {file_path}")

            # Load the ECG data
            self.load_ecg_data(file_path)

    def load_ecg_data(self, file_path):
        # Read the Excel file
        try:
            self.ecg_data = pd.read_csv(file_path)
            print("ECG data loaded successfully!")
            print(self.ecg_data.head())  # Print first few rows to confirm data load
        except Exception as e:
            print(f"Error loading ECG data: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ECGApp()
    ex.show()
    sys.exit(app.exec_())
