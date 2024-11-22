from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel
from PyQt5.QtCore import Qt, pyqtSignal


class ECGAbnormalitiesModeTab(QWidget):
    slider_signal_array=pyqtSignal(int,int)
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()

        self.ecg_slider_values = {}

        self.sliders = []  # List to store references to sliders

        self.create_sliders()

        self.setLayout(self.layout)

    def create_sliders(self):
        num_sliders = 3
        for i in range(num_sliders):
            slider_layout = QHBoxLayout()
            slider_label = QLabel(f"ECG Abnormality Slider {i + 1}")
            slider_label.setStyleSheet("color: white; font-size: 12px;")

            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(50)
            slider_value_label = QLabel("50")
            slider_value_label.setStyleSheet("color: #4287f5; font-size: 12px; font-weight: bold;")

            self.ecg_slider_values[i] = 50

            slider.valueChanged.connect(
                lambda value, index=i, lbl=slider_value_label: self.update_slider_value(index, value, lbl))
            self.sliders.append(slider)

            slider.setStyleSheet("""
                QSlider::groove:horizontal {
                    height: 8px;
                    background: #555;
                    border-radius: 4px;
                }
                QSlider::handle:horizontal {
                    background: #6ba4ff;
                    border: 1px solid #2E8B57;
                    width: 14px;
                    height: 14px;
                    margin: -3px 0;
                    border-radius: 7px;
                }
                QSlider::handle:horizontal:hover {
                    background: #4287f5;
                }
            """)

            slider_layout.addWidget(slider_label)
            slider_layout.addWidget(slider)
            slider_layout.addWidget(slider_value_label)
            self.layout.addLayout(slider_layout)

    def update_slider_value(self, index, value, label):
        """Update the slider value in the dictionary and the label"""
        self.ecg_slider_values[index] = value
        label.setText(str(value))
        # print(f"ECG Slider {index + 1} Value: {value}")
        self.slider_signal_array.emit(index,value)
