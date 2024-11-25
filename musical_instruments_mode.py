from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel
from PyQt5.QtCore import Qt, pyqtSignal


class MusicalInstrumentsModeTab(QWidget):
    sliderValueChanged2 = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()
        self.instrument_slider_values = {}

        self.create_sliders()

        self.setLayout(self.layout)

    def create_sliders(self):
        instrument_names = ["xylophone", "trombone", "triangle", "bass"]  # Names for the sliders
        num_sliders = 4



        for i in range(num_sliders):
            slider_layout = QHBoxLayout()
            slider_label = QLabel(f"{instrument_names[i]} Slider")  # Use the instruments_name here
            slider_label.setStyleSheet("color: white; font-size: 12px;")

            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(50)
            slider_value_label = QLabel("50")
            slider_value_label.setStyleSheet("color: #2E8B57; font-size: 12px; font-weight: bold;")

            self.instrument_slider_values[i] = 50

            slider.valueChanged.connect(
                lambda value, index=i, lbl=slider_value_label: self.update_slider_value2(index, value, lbl))

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

    def update_slider_value2(self, index, value, label):
        """Update the slider value in the dictionary and the label"""
        self.instrument_slider_values[index] = value
        label.setText(str(value))
        print(f"Instrument Slider {index + 1} Value: {value}")
        self.sliderValueChanged2.emit(index, value)
