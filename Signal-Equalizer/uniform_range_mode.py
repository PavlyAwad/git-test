
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QPushButton,QFileDialog
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
import numpy as np
from pyqtgraph import ColorMap

from scipy.fft import fft, ifft
import sounddevice as sd
import pyqtgraph as pg
from scipy.io import wavfile
from scipy.signal import spectrogram
import numpy as np


class UniformRangeModeTab(QWidget):

    frequency_data_updated = pyqtSignal(np.ndarray)
    toggle_frequency = pyqtSignal(bool)
    slider_value_received = pyqtSignal(int)
    def __init__(self, input_viewer, output_viewer, frequency_viewer,input_spectrogram, output_spectrogram):
        super().__init__()

        # Viewer references
        self.inputSignalViewer = input_viewer
        self.outputSignalViewer = output_viewer
        self.frequencyViewer = frequency_viewer
        self.input_spectrogram = input_spectrogram
        self.output_spectrogram = output_spectrogram

        self.layout = QVBoxLayout()
        self.uniform_slider_values = {}

        # Initialize attributes for audio data

        self.original_freq_data = None
        self.adjusted_data = None
        self.sampling_rate = None
        self.data = None



        # Add sliders for frequency control
        self.create_sliders()

        # Timer for animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate_plot)
        # Animation step tracker
        self.animation_step = 0
        self.speed = 100
        self.slider_value_received.connect(self.receive_slider_value)



        self.setLayout(self.layout)

    def create_sliders(self):
        num_sliders = 10
        self.sliders = []  # Store slider references to access values
        for i in range(num_sliders):
            slider_layout = QHBoxLayout()
            slider_label = QLabel(f"Uniform Range Slider {i + 1}")
            slider_label.setStyleSheet("color: white; font-size: 12px;")

            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 150)
            slider.setValue(100)
            slider_value_label = QLabel("100")
            slider_value_label.setStyleSheet("color: #4287f5; font-size: 12px; font-weight: bold;")

            self.uniform_slider_values[i] = 100
            slider.valueChanged.connect(lambda value, index=i, lbl=slider_value_label: self.update_slider_value(index, value, lbl))

            # Style the sliders
            slider.setStyleSheet("""
                QSlider::groove:horizontal { height: 8px; background: #555; border-radius: 4px; }
                QSlider::handle:horizontal { background: #6ba4ff; border: 1px solid #2E8B57; width: 14px; height: 14px; margin: -3px 0; border-radius: 7px; }
                QSlider::handle:horizontal:hover { background: #4287f5; }
            """)

            slider_layout.addWidget(slider_label)
            slider_layout.addWidget(slider)
            slider_layout.addWidget(slider_value_label)
            self.layout.addLayout(slider_layout)
            self.sliders.append(slider)

    def update_slider_value(self, index, value, label):
        self.uniform_slider_values[index] = value
        label.setText(str(value))
        self.update_output()  # Update output signal when slider changes

    def receive_slider_value(self, value):
        # You can choose what to do with the value
        # For example, update the slider's value and reflect it in the GUI:
        self.speed= value
        self.timer.setInterval(value)



    def set_data(self, sampling_rate, data):
        self.sampling_rate = sampling_rate
        self.data = data
        self.original_freq_data = fft(data)

        # Display the signal in the input viewer
        self.inputSignalViewer.clear()
        self.inputSignalViewer.plot(data, pen='g')
        self.inputSignalViewer.enableAutoRange()
        self.plot_spectrogram(self.data, self.input_spectrogram)
        self.update_output()  # Update the output based on the new data

    def play_original_audio(self):
        if self.data is not None:
            sd.play(self.data, self.sampling_rate)

    def play_adjusted_audio(self):
        if self.adjusted_data is not None:
            sd.play(self.adjusted_data, self.sampling_rate)

    def update_output(self):
        if self.original_freq_data is not None:
            slider_values = [slider.value() / 100 for slider in self.sliders]
            adjusted_freq_data = self.adjust_frequencies(self.original_freq_data, slider_values)
            adjusted_data = ifft(adjusted_freq_data).real
            self.adjusted_data = np.int16(np.clip(adjusted_data, -32768, 32767))
            # Reset animation step
            self.animation_step = 0

            # Start the timer for animation
            self.timer.start(self.speed)  # Adjust interval (ms) for animation speed
            print(f'{self.speed} updated')

            #self.plot_full_frequency_range(adjusted_freq_data)
            self.frequency_data_updated.emit(self.adjusted_data)  # Emit adjusted data
            self.toggle_frequency.emit(True)
            self.plot_spectrogram(self.adjusted_data, self.output_spectrogram)

    def adjust_frequencies(self, freq_data, slider_values):
        adjusted_freq_data = freq_data.copy()
        n = len(freq_data) // 2
        segment_size = n // len(slider_values)
        for i, value in enumerate(slider_values):
            start = i * segment_size
            end = start + segment_size
            adjusted_freq_data[start:end] *= value
            adjusted_freq_data[start:end] *= value
            if start == 0:
                adjusted_freq_data[-end:-1] *= value
            else:
                adjusted_freq_data[-end:-start] *= value

        return adjusted_freq_data

    def plot_full_frequency_range(self, freq_data):
        freqs = np.fft.fftfreq(len(freq_data), 1 / self.sampling_rate)
        self.frequencyViewer.clear()
        self.frequencyViewer.plot(freqs, np.abs(freq_data), pen='b')
        self.frequencyViewer.enableAutoRange()

    def plot_spectrogram(self, audio, viewer):
        # Compute spectrogram
        frequencies, times, Sxx = spectrogram(audio, fs=self.sampling_rate, nperseg=256, noverlap=128, nfft=1024)

        # Convert to dB scale
        Sxx_log = 10 * np.log10(Sxx + 1e-10)

        # Check if all values are zero
        if Sxx_log.max() == Sxx_log.min():
            Sxx_log_normalized = np.zeros_like(Sxx_log)  # Set normalized array to zero if all magnitudes are zero
        else:
            # Normalize for color mapping
            Sxx_log_normalized = (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min())

        # Define custom colormap
        colormap = ColorMap(
            [0, 0.25, 0.5, 0.75, 1],
            [
                (0, 0, 128, 255),  # Dark Blue
                (0, 255, 255, 255),  # Cyan
                (255, 255, 0, 255),  # Yellow
                (255, 128, 0, 255),  # Orange
                (255, 0, 0, 255)  # Red
            ]
        )

        # Normalize frequency axis to max frequency (Nyquist frequency)
        normalized_frequencies = frequencies / (self.sampling_rate / 2)

        # Create ImageItem with normalized spectrogram
        img = pg.ImageItem()
        img.setImage(Sxx_log_normalized.T, levels=(0, 1))  # Transpose for correct orientation
        img.setColorMap(colormap)

        # Set rectangle for time (x-axis) and normalized frequency (y-axis from 0 to 1)
        img.setRect(0, 0, times[-1], 1)  # Max y-axis is normalized frequency 1

        # Clear previous plot and add the spectrogram
        viewer.clear()
        viewer.addItem(img)

        # Set x-axis (time) and y-axis (normalized frequency) ranges
        viewer.setXRange(0, times[-1])
        viewer.setYRange(0, 1)  # y-axis from 0 to 1 for normalized frequency

        # Set labels (optional, if viewer supports it)
        viewer.getAxis('bottom').setLabel('Time (s)')
        viewer.getAxis('left').setLabel('Normalized Frequency')

    def animate_plot(self):
        """Animate the plot to show the signal advancing from left to right."""
        if self.data is None:
            return

        # Define the chunk size (e.g., 10% of total data)
        chunk_size = len(self.data) // 100  # Adjust based on desired resolution
        max_steps = len(self.data) // chunk_size

        # Calculate the current window
        start = self.animation_step * chunk_size
        end = start + chunk_size

        # Stop the animation if we reach the end
        if start >= len(self.data):
            self.timer.stop()
            return

        # Plot the signal up to the current chunk
        self.inputSignalViewer.clear()
        self.inputSignalViewer.plot(self.data[:end], pen='g')  # Plot from 0 to 'end'

        # If adjusted data exists, plot it
        if self.adjusted_data is not None:
            self.outputSignalViewer.clear()
            self.outputSignalViewer.plot(self.adjusted_data[:end], pen='b')

        # Update the x-axis to show the advancing window
        self.inputSignalViewer.setXRange(0, end)
        self.outputSignalViewer.setXRange(0, end)
        self.inputSignalViewer.enableAutoRange()
        self.outputSignalViewer.enableAutoRange()

        # Increment the animation step
        self.animation_step += 1
        self.set_graph_limits(end)




    def set_graph_limits(self, end):
        """
        Set limits for panning and zooming for the input and output graphs.
        """
        # Example limits for panning and zooming (adjust as per your data range)
        min_x, max_x = 0, end + 100  # Adjust based on your data

        # Sanitize data to prevent overflows
        sanitized_data = np.clip(self.data, -1e6, 1e6)  # Limit values to a reasonable range
        min_y, max_y = min(sanitized_data) - 50, max(sanitized_data) + 50

        min_zoom, max_zoom = 100, 210000

        # Ensure zoom constraints are valid relative to data ranges
        x_range = max_x - min_x
        y_range = max_y - min_y
        min_zoom = max(min_zoom, 1)  # Prevent zoom constraints from being zero or negative
        max_zoom = max(max_zoom, max(x_range, y_range))
        if min_y == max_y:
            min_y -= 50
            max_y += 50  # Add some padding to avoid zero range

        # Set limits for input graph
        self.inputSignalViewer.getViewBox().setLimits(
            xMin=min_x, xMax=max_x,
            yMin=min_y, yMax=max_y,
            minXRange=min_zoom, maxXRange=max_zoom,
            minYRange=min_zoom, maxYRange=max_zoom
        )

        # Set limits for output graph
        self.outputSignalViewer.getViewBox().setLimits(
            xMin=min_x, xMax=max_x,
            yMin=min_y, yMax=max_y,
            minXRange=min_zoom, maxXRange=max_zoom,
            minYRange=min_zoom, maxYRange=max_zoom
        )




