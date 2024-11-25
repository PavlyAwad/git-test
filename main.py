import os
import wfdb
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QWidget, QGroupBox, \
    QFileDialog, QComboBox, QSlider
import pyqtgraph as pg
from pyqtgraph import ColorMap
from scipy.io import wavfile
import numpy as np
import sounddevice as sd
from scipy.signal import spectrogram
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from animal_sounds_mode import AnimalSoundsModeTab
from ecg_abnormalities_mode import ECGAbnormalitiesModeTab
from musical_instruments_mode import MusicalInstrumentsModeTab
from uniform_range_mode import UniformRangeModeTab
from pydub import AudioSegment
from pydub.playback import play
from scipy.fft import fft, ifft


class AudioProcessingThread(QThread):
    audio_processed = pyqtSignal(np.ndarray)

    def __init__(self, audio_data, sample_rate, instruments_gain, instruments_masks, parent=None):
        super().__init__(parent)
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.instruments_gain = instruments_gain
        self.instruments_masks = instruments_masks
        self.modified_spectrum = []

    def run(self):
        spectrum = fft(self.audio_data)
        self.modified_spectrum = np.copy(spectrum)

        for instrument, gain in self.instruments_gain.items():
            if instrument in self.instruments_masks:
                self.modified_spectrum[self.instruments_masks[instrument]] *= gain

        equalized_signal = np.real(ifft(self.modified_spectrum))
        self.audio_processed.emit(equalized_signal)


class SignalEqualizerUI(QMainWindow):
    slider_value_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.thread = None
        self.setWindowTitle("Signal Equalizer")
        self.setGeometry(100, 100, 1200, 700)

        self.dark_theme()
        self.sampling_rate, self.data = None, None

        self.play_input = False
        self.play_output = False
        self.show_spectrograms = True
        self.use_audiogram_scale = False  # Track current scale
        self.adjusted_data = None

        self.chunk_size = 1024  # Number of samples to plot at a time
        self.current_sample = 0
        self.audio_samples = None
        self.sample_rate = None
        self.combined_signal = []
        self.modified_freq_audio = []
        self.play_first_time = True

        self.instrument_frequency_ranges = {
            'trombone': [(0, 345)],
            'xylophone': [(345, 1000)],
            'bass': [(860, 4000)],
            'triangle': [(4000, 22000)],



        }
        self.animal_frequency_ranges = {
            'whale': [(0, 800)],
            'frog': [(800, 2000)],
            'monkey': [(2000, 4000)],
            'bat': [(4000, 22000)]
        }
        self.instruments_gain = {
            'xylophone': 1.0,
            'trombone': 1.0,
            'triangle': 1.0,
            'bass': 1.0
        }
        self.animal_gain = {
            'whale': 1.0,
            'frog': 1.0,
            'monkey': 1.0,
            'bat': 1.0
        }

        # تهيئة instruments_masks
        self.instruments_masks = {}
        self.animal_masks = {}

        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()

        # Create an instance of the ECGAbnormalitiesModeTab
        self.ecg_abnormalities_tab = ECGAbnormalitiesModeTab()
        self.ecg_slider1_value = 50
        self.ecg_slider2_value = 50
        self.ecg_slider3_value = 50
        self.afib_range_positive = 0
        self.afib_range_negative = 0
        self.mrd2_range_positive = 0
        self.mrd2_range_negative = 0
        self.mrd3_range_positive = 0
        self.mrd3_range_negative = 0
        self.freqs = None
        self.buttons_in = {}
        self.buttons_out = {}

        input_output_layout = QHBoxLayout()

        input_layout = QVBoxLayout()
        self.input_graph = pg.PlotWidget(title="Input Signal")
        self.input_graph.setBackground('k')
        self.input_graph.setLabel('left', "Amplitude")  # Adding y-axis label
        self.input_graph.setLabel('bottom', "Time", units="s")     # Adding x-axis label
        self.input_spectrogram = pg.PlotWidget(title="Input Spectrogram")
        self.input_spectrogram.setBackground('k')
        self.input_spectrogram.setLabel('left', "Frequency")
        self.input_spectrogram.setLabel('bottom', "Time", units="s")
        self.playOriginalButton = QtWidgets.QPushButton("Play Original")
        self.playOriginalButton.setStyleSheet("""
            QPushButton {
                background-color: #4287f5;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 12px;
            }
            QPushButton:hover {
                background-color: #6ba4ff;
            }
        """)
        self.playOriginalButton.clicked.connect(self.play_original_audio)
        self.input_controls = self.create_controls_group("Input Controls")
        input_layout.addWidget(self.input_graph)
        input_layout.addWidget(self.playOriginalButton)
        input_layout.addWidget(self.input_controls)
        input_layout.addWidget(self.input_spectrogram)

        output_layout = QVBoxLayout()
        self.output_graph = pg.PlotWidget(title="Output Signal")
        self.output_graph.setBackground('k')
        self.output_graph.setLabel('left', "Amplitude")
        self.output_graph.setLabel('bottom', "Time", units="s")
        self.output_spectrogram = pg.PlotWidget(title="Output Spectrogram")
        self.output_spectrogram.setBackground('k')
        self.output_spectrogram.setLabel('left', "Frequency")
        self.output_spectrogram.setLabel('bottom', "Time", units="s")
        self.playAdjustedButton = QtWidgets.QPushButton("Play Adjusted")
        self.playAdjustedButton.setStyleSheet("""
            QPushButton {
                background-color: #4287f5;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 12px;
            }
            QPushButton:hover {
                background-color: #6ba4ff;
            }
        """)
        self.playAdjustedButton.clicked.connect(self.play_adjusted_audio)
        self.output_controls = self.create_controls_group("Output Controls")
        output_layout.addWidget(self.output_graph)
        output_layout.addWidget(self.playAdjustedButton)
        output_layout.addWidget(self.output_controls)
        output_layout.addWidget(self.output_spectrogram)

        self.audio_data = None
        self.sample_rate = None
        self.animal_sounds_tab = AnimalSoundsModeTab()
        self.instrument_sound_tab = MusicalInstrumentsModeTab()
        self.animal_sounds_tab.sliderValueChanged.connect(self.update_slider)
        self.instrument_sound_tab.sliderValueChanged2.connect(self.update_slider)

        input_output_layout.addLayout(input_layout)
        input_output_layout.addLayout(output_layout)

        # Frequency Domain Graph with Dropdown for Scale Selection
        self.frequency_domain = pg.PlotWidget(title="Frequency Domain")
        self.frequency_domain.setBackground('k')
        self.frequency_domain.setLabel('left', "Amplitude")
        self.frequency_domain.setLabel('bottom', "Frequency", units="Hz")
        self.scale_dropdown = QComboBox()
        self.scale_dropdown.addItems(["Linear", "Audiogram"])
        self.scale_dropdown.setStyleSheet("""
            QComboBox {
                background-color: #4287f5;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 8px;
            }
            QComboBox::drop-down {
                background-color: #6ba4ff;
            }
        """)
        self.scale_dropdown.currentIndexChanged.connect(self.change_frequency_scale)

        left_layout.addLayout(input_output_layout)
        left_layout.addWidget(self.scale_dropdown, alignment=QtCore.Qt.AlignCenter)
        left_layout.addWidget(self.frequency_domain)

        right_layout = QVBoxLayout()
        self.mode_tabs = QTabWidget()
        self.mode_tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #333; }
            QTabBar::tab { background: #4287f5; padding: 10px; margin: 2px; border-radius: 8px; color: white; font-size: 14px; }
            QTabBar::tab:selected { background: #4287f5; }
        """)
        self.uniform_range_tab = UniformRangeModeTab(self.input_graph, self.output_graph, self.frequency_domain,
                                                     self.input_spectrogram, self.output_spectrogram)
        self.mode_tabs.addTab(self.uniform_range_tab, "Uniform Range")
        self.mode_tabs.addTab(self.instrument_sound_tab, "Musical Instruments")
        self.mode_tabs.addTab(self.animal_sounds_tab, "Animal Sounds")
        self.mode_tabs.addTab(self.ecg_abnormalities_tab, "ECG Abnormalities")

        self.current_index = self.mode_tabs.currentIndex()
        self.ecg_abnormalities_index = self.mode_tabs.indexOf(self.ecg_abnormalities_tab)

        right_layout.addWidget(self.mode_tabs)
        self.mode_tabs.currentChanged.connect(self.load_ecg)
        # Spectrogram Toggle Button
        self.spectrogram_toggle_btn = QPushButton("Toggle Spectrogram Visibility")
        self.spectrogram_toggle_btn.clicked.connect(self.toggle_spectrogram_visibility)
        self.spectrogram_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #F0B429;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 12px;
            }
            QPushButton:hover {
                background-color: #F6D55C;
            }
        """)
        right_layout.addWidget(self.spectrogram_toggle_btn)
        file_upload_btn = QPushButton("Upload Signal File")
        file_upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #F0B429;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 12px;
            }
            QPushButton:hover {
                background-color: #F6D55C;
            }
        """)
        file_upload_btn.clicked.connect(self.load_file)
        right_layout.addWidget(file_upload_btn, alignment=QtCore.Qt.AlignBottom)

        main_layout.addLayout(left_layout, 4)
        main_layout.addLayout(right_layout, 1)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.ecg_abnormalities_tab.slider_signal_array.connect(self.slider_of_ecg)
        self.uniform_range_tab.frequency_data_updated.connect(self.handle_frequency_update)
        self.uniform_range_tab.toggle_frequency.connect(self.handle_frequency_toggle)

        self.uniform_range_tab.slider_value_received.connect(self.uniform_range_tab.receive_slider_value)
        slider_value = 100
        self.uniform_range_tab.slider_value_received.emit(slider_value)

        self.ecg_signal = None
        self.fft_ecg_signal = None
        self.adjusted_ecg_signal = None
        self.audio_playing = False
        self.edited_audio_playing = False
        self.current_tab = self.mode_tabs.currentWidget()
        # input_viewbox = self.input_graph.getViewBox()
        # output_viewbox = self.output_graph.getViewBox()
        #
        # # Connect view synchronization signals
        # input_viewbox.sigRangeChanged.connect(self.sync_output_view)
        # output_viewbox.sigRangeChanged.connect(self.sync_input_view)

        self.timer_interval = 20  # Time in milliseconds (for smooth animation)
        self.plot_step = 500  # Number of data points to plot per timer tick (adjust as needed)
        self.plot_index = 0
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_signals)
        self.time_input = []
        self.time_output = []

    def dark_theme(self):
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(40, 40, 40))
        dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Base, QColor(30, 30, 30))
        dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Button, QColor(50, 50, 50))
        dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        self.setPalette(dark_palette)
        self.setStyleSheet("background-color: #282828; color: white; font-family: Arial; font-size: 14px;")

    def create_controls_group(self, title):
        group_box = QGroupBox(title)
        layout = QHBoxLayout()

        buttons = {
            "Play": self.play_pause,
            "Replay": self.replay,
            "Zoomin": self.zoom_in,
            "Zoomout": self.zoom_out,
            "Pan": self.pan,
            "Reset": self.reset
        }
        for btn_name, function in buttons.items():
            button = QPushButton(btn_name)
            button.setStyleSheet("""
                QPushButton {
                    background-color: #4287f5;
                    color: white;
                    font-weight: bold;
                    padding: 8px;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background-color: #6ba4ff;
                }
            """)
            button.clicked.connect(function)
            layout.addWidget(button)
            print(title)
            # Save reference to each button so it can be accessed later
            if "Input Controls" == str(title):
                self.buttons_in[btn_name] = button
            else:
                self.buttons_out[btn_name] = button
            print(self.buttons_in)
            print(self.buttons_out)
        # Create Speed slider to replace the Speed button
        self.speed_slider = QSlider(Qt.Horizontal)  # Horizontal slider
        self.speed_slider.setRange(1, 200)  # Set the range (e.g., from 1 to 10)
        self.speed_slider.setValue(100)  # Set default value
        self.speed_slider.setTickInterval(20)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)  # Display ticks below slider
        self.speed_slider.valueChanged.connect(self.adjust_speed)
        # Add the slider to the layout
        layout.addWidget(self.speed_slider)

        group_box.setLayout(layout)
        return group_box

    def sync_input_view(self, view_box, _):
        """
        Sync the input viewer's view range with the output viewer.
        This is called when the output viewer changes its view.
        """
        # Get the range from the output graph and apply it to the input graph
        input_range = self.output_graph.getViewBox().viewRange()
        self.input_graph.getViewBox().setRange(xRange=input_range[0], yRange=input_range[1], padding=0)

    def sync_output_view(self, view_box, _):
        """
        Sync the output viewer's view range with the input viewer.
        This is called when the input viewer changes its view.
        """
        # Get the range from the input graph and apply it to the output graph
        output_range = self.input_graph.getViewBox().viewRange()
        self.output_graph.getViewBox().setRange(xRange=output_range[0], yRange=output_range[1], padding=0)

    def upload_file(self):
        # Open file dialog to load a WAV file
        file_path, _ = QFileDialog.getOpenFileName(self, "Open WAV File", "", "WAV Files (*.wav)")
        if file_path:
            # Load audio data and start playback
            self.load_audio(file_path)
            # Start timer based on sample rate

    def load_audio(self, file_path):
        abs_path = os.path.abspath(file_path)
        print(f"Attempting to load file from: {abs_path}")

        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"File not found at: {abs_path}")

        file_extension = os.path.splitext(abs_path)[1].lower()
        print(f"File extension: {file_extension}")

        try:
            print("Converting to WAV...")
            audio = AudioSegment.from_file(abs_path)

            # Create temp directory if it doesn't exist
            temp_dir = "temp_audio"
            os.makedirs(temp_dir, exist_ok=True)

            temp_wav = os.path.join(temp_dir, "temp_audio.wav")
            print(f"Saving temporary WAV file to: {temp_wav}")

            audio.export(temp_wav, format="wav")
            self.sample_rate, self.audio_data = wavfile.read(temp_wav)

            # Clean up
            os.remove(temp_wav)
            os.rmdir(temp_dir)

            print(f"Successfully loaded audio file. Sample rate: {self.sample_rate}Hz")

        except Exception as e:
            raise Exception(f"Error processing audio file: {str(e)}")

        # Convert to mono if stereo
        if len(self.audio_data.shape) > 1:
            self.audio_data = np.mean(self.audio_data, axis=1)

        # Normalize audio data
        self.audio_data = self.audio_data.astype(float)
        if self.audio_data.max() > 1.0 or self.audio_data.min() < -1.0:
            self.audio_data = self.audio_data / 32768.0
        self.plot_spectrogram(self.audio_data, self.input_spectrogram)
        # self.timer.start(int(1000 * self.chunk_size / self.sample_rate))
        self.adjusted_data = self.audio_data
        time = np.linspace(0, len(self.audio_data) / self.sample_rate, num=len(self.audio_data))

        # self.input_graph.plot(self.audio_data)

        self.animation_timer.start(self.timer_interval)
        self.modified_freq_audio = fft(self.adjusted_data)
        self.plot_spectrogram(self.adjusted_data, self.output_spectrogram)
        # self.plot_full_frequency_range(np.abs(self.modified_freq_audio))
        self.update_frequency_plot()
        self._create_masks()



    def _create_masks(self):
        if isinstance(self.current_tab, MusicalInstrumentsModeTab):
            self.make_masks(self.instrument_frequency_ranges, self.instruments_masks)
        elif isinstance(self.current_tab, AnimalSoundsModeTab):
            self.make_masks(self.animal_frequency_ranges, self.animal_masks)

    def make_masks(self, frequency_ranges, masks):
        n_samples = len(self.audio_data)
        frequencies = np.fft.fftfreq(n_samples, 1 / self.sample_rate)

        for sound, ranges in frequency_ranges.items():
            mask = np.zeros(n_samples, dtype=bool)
            for freq_range in ranges:
                mask |= (np.abs(frequencies) >= freq_range[0]) & (np.abs(frequencies) <= freq_range[1])
            masks[sound] = mask

    def plot_audio_real_time(self, input_graph, output_graph, sampling_rate, input_audio, modified_audio):
        """
        Plots input audio and modified audio in real time, stopping when the data ends.

        Parameters:
            input_graph (pyqtgraph.PlotWidget): The graph widget for input audio.
            output_graph (pyqtgraph.PlotWidget): The graph widget for modified audio.
            sampling_rate (int): Sampling rate of the audio (in Hz).
            input_audio (numpy.ndarray): Original audio data as a 1D array.
            modified_audio (numpy.ndarray): Modified audio data as a 1D array.
        """
        # Internal state variables
        current_index = [0]  # To allow modification in the nested function
        chunk_size = 1024  # Number of samples per update

        # Create plots on the provided graphs
        input_plot = input_graph.plot(pen="y")  # Yellow line for input
        output_plot = output_graph.plot(pen="c")  # Cyan line for modified

        def update_plots():
            """Update the plots with the next chunk of audio data."""
            if current_index[0] >= len(input_audio):
                # Stop the timer when all data has been visualized
                timer.stop()
                return

            # Get the next chunk of data
            input_chunk = input_audio[current_index[0]: current_index[0] + chunk_size]
            modified_chunk = modified_audio[current_index[0]: current_index[0] + chunk_size]

            # Update the plots
            input_plot.setData(input_chunk)
            output_plot.setData(modified_chunk)

            current_index[0] += chunk_size

            # Timer for real-time updates
        timer = QTimer()
        timer.timeout.connect(update_plots)
        timer.start(500 * chunk_size / sampling_rate)  # Update interval in ms

        # Start the first update immediately
        update_plots()

        return timer

    def apply_equalization(self):
        if isinstance(self.current_tab, MusicalInstrumentsModeTab):
            self.access_audio_processing(self.instruments_gain, self.instruments_masks)
        else:
            self.access_audio_processing(self.animal_gain, self.animal_masks)

    def access_audio_processing(self, gain, masks):
        self.thread = AudioProcessingThread(self.audio_data, self.sample_rate, gain,
                                            masks)
        self.thread.audio_processed.connect(self.play_audio)
        self.thread.start()

    def update_slider(self, index, value):
        if isinstance(self.current_tab, MusicalInstrumentsModeTab):
            tab = 1
            self.plot_all_view(self.instruments_gain, index, value, tab)
        elif isinstance(self.current_tab, AnimalSoundsModeTab):
            tab = 2
            self.plot_all_view(self.animal_gain, index, value, tab)

    def plot_all_view(self, gain, index, value, tab):
        # Define a dictionary mapping each (index, tab) pair to an instrument.
        instrument_mapping = {
            (0, 1): 'xylophone',
            (0, 2): 'whale',
            (1, 1): 'trombone',
            (1, 2): 'frog',
            (2, 1): 'triangle',
            (2, 2): 'monkey',
            (3, 1): 'bass',
            (3, 2): 'bat'
        }

        # Get the instrument based on index and tab.
        instrument = instrument_mapping.get((index, tab), "Unknown Instrument")

        gain[instrument] = value / 50.0  # Convert 0-100 to 0.0-2.0 scale
        print(f"Updated {instrument} gain to {gain[instrument]}")
        # self.first_slider_move = True

        # Delay processing until the user stops moving the slider
        if hasattr(self, '_slider_timer'):
            self._slider_timer.stop()

        self._slider_timer = QTimer()
        self._slider_timer.setSingleShot(True)
        if tab == 1:
            self._slider_timer.timeout.connect(
                lambda: self.access_audio_processing(self.instruments_gain, self.instruments_masks))
        elif tab == 2:
            self._slider_timer.timeout.connect(
                lambda: self.access_audio_processing(self.animal_gain, self.animal_masks))
        self._slider_timer.start(500)
        self.modified_freq_audio = fft(self.adjusted_data)
        # self.plot_full_frequency_range(np.abs(self.modified_freq_audio))
        self.update_frequency_plot()
        self.plot_spectrogram(self.adjusted_data, self.output_spectrogram)
        # self.plotModifiedSignal(self.adjusted_data, self.sample_rate)
        self.plot_index = 0  # Start the animation from the beginning
        self.animation_timer.start(self.timer_interval)

    def animate_signals(self):
        """Update the plot incrementally for animation effect."""
        # Define the current time steps based on plot index
        end_index = min(self.plot_index + self.plot_step, len(self.audio_data))

        # Time axis for both signals
        time_input = np.linspace(0, len(self.audio_data) / self.sample_rate, num=len(self.audio_data))
        self.time_input = time_input
        time_output = np.linspace(0, len(self.adjusted_data) / self.sample_rate, num=len(self.adjusted_data))
        self.time_output = time_output

        # Plot the input signal (original)
        self.input_graph.clear()
        self.input_graph.plot(time_input[:end_index], self.audio_data[:end_index], pen='g')

        # Plot the adjusted signal (output)
        self.output_graph.clear()
        self.output_graph.plot(time_output[:end_index], self.adjusted_data[:end_index],
                               pen='r')  # Red for adjusted signal
        # self.set_graph_limits(self.plot_index )
        self.input_graph.enableAutoRange()
        # self.output_graph.enableAutoRange()
        self.set_graph_limits()

        # Update the plot index
        self.plot_index += self.plot_step

        # Stop the animation when all data is plotted
        if self.plot_index >= len(self.audio_data):
            self.animation_timer.stop()

    def play_audio(self, equalized_signal):
        # تشغيل الصوت المعالج باستخدام sounddevice
        if equalized_signal is None:
            print("No audio to play!")
            return

        # Stop any currently playing audio
        sd.stop()
        self.adjusted_data = equalized_signal

        # Normalize the audio to the range expected by sounddevice
        audio_to_play = np.clip(equalized_signal, -1.0, 1.0)

        # Play the audio
        sd.play(audio_to_play, self.sample_rate)
        print("Audio playing.")

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
        self.normalized_frequencies = frequencies / (self.sampling_rate / 2)

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


    def play(self):
        self.is_playing = True
        self.timer.start(int(1000 * self.chunk_size / self.sample_rate))
        sd.play(self.audio_data[self.current_sample:], samplerate=self.sample_rate)
        print("play")

    def plot_full_frequency_range(self, freq_data):
        freqs = np.fft.fftfreq(len(freq_data), 1 / self.sampling_rate)
        self.frequency_domain.clear()
        self.frequency_domain.plot(freqs, np.abs(freq_data), pen='b')
        self.frequency_domain.enableAutoRange()

    def play_pause(self):
        play_button_in = self.buttons_in.get("Play")
        play_button_out = self.buttons_out.get("Play")
        if play_button_in and play_button_out:
            if isinstance(self.current_tab, UniformRangeModeTab):
                if self.uniform_range_tab.timer.isActive():
                    self.uniform_range_tab.timer.stop()
                    play_button_in.setText("Play")  # Change text of the 'Play' button
                    play_button_out.setText("Play")  # Change text of the 'Play' button
                else:
                    self.uniform_range_tab.timer.start(100)  # Adjust interval as needed
                    play_button_in.setText("Pause")  # Change text of the 'Play' button
                    play_button_out.setText("Pause")  # Change text of the 'Play' button
            else:
                if self.animation_timer.isActive():
                    self.animation_timer.stop()
                    play_button_in.setText("Play")  # Change text of the 'Play' button
                    play_button_out.setText("Play")  # Change text of the 'Play' button
                else:
                    self.animation_timer.start(20)  # Adjust interval as needed
                    play_button_in.setText("Pause")
        else:
            print("Play button not found!")

    def replay(self):
        replay_button_in = self.buttons_in.get("Replay")
        replay_button_out = self.buttons_out.get("Replay")
        play_button_in = self.buttons_in.get("Play")
        play_button_out = self.buttons_out.get("Play")
        if replay_button_in and replay_button_out:
            if isinstance(self.current_tab, UniformRangeModeTab):
                self.uniform_range_tab.animation_step = 0
                self.input_graph.clear()
                self.output_graph.clear()
                self.uniform_range_tab.timer.start(100)
                play_button_in.setText("Pause")
                play_button_out.setText("Pause")
            else:
                self.plot_index = 0  # Reset to the start of the data
                self.input_graph.clear()  # Clear any current plots
                self.output_graph.clear()
                self.animation_timer.start(self.timer_interval)  # Restart the animation
                play_button_in.setText("Pause")
                play_button_out.setText("Pause")
        else:
            print("Replay button not found!")

    def adjust_speed(self, value):
        if isinstance(self.current_tab, UniformRangeModeTab):
            self.uniform_range_tab.slider_value_received.emit(value)
        # This method will be called whenever the slider value changes.
        # You can use the `value` to adjust the speed in your application.
        print(f"Speed changed to: {value}")
        # Implement the logic for adjusting speed here

    def zoom_out(self):

        # Apply zoom out scaling
        self.input_graph.getViewBox().scaleBy((1.1, 1.1))
        self.output_graph.getViewBox().scaleBy((1.1, 1.1))

        print("Zoom out applied.")

    def zoom_in(self):
        print("zoom")
        self.input_graph.getViewBox().scaleBy((0.8, 0.8))
        self.output_graph.getViewBox().scaleBy((0.8, 0.8))

    def pan(self, direction="right"):
        """
        Pan the graphs left or right, with constraints to prevent panning out of bounds.
        """
        pan_amount = 500  # Adjust panning step
        if direction == "right":
            self.input_graph.getViewBox().translateBy(x=pan_amount)
            self.output_graph.getViewBox().translateBy(x=pan_amount)
        elif direction == "left":
            self.input_graph.getViewBox().translateBy(x=-pan_amount)
            self.output_graph.getViewBox().translateBy(x=-pan_amount)

    def reset(self):
        self.input_graph.enableAutoRange()
        self.output_graph.enableAutoRange()

    def clear(self):
        self.input_graph.clear()
        self.output_graph.clear()
        self.input_spectrogram.clear()
        self.output_spectrogram.clear()
        self.frequency_domain.clear()
        self.sampling_rate, self.data = None, None

        self.play_input = False
        self.play_output = False
        self.adjusted_data = None
        self.current_sample = 0
        self.audio_samples = None
        self.sample_rate = None
        self.combined_signal = []
        self.modified_freq_audio = []
        self.play_first_time = True
        self.ecg_signal = None
        self.fft_ecg_signal = None
        self.adjusted_ecg_signal = None
        self.time_input=[]
        self.time_output=[]
        sd.stop()

    def load_file(self):
        self.clear()
        self.current_tab = self.mode_tabs.currentWidget()
        play_button_in = self.buttons_in.get("Play")
        play_button_out = self.buttons_out.get("Play")
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open WAV file", "", "Audio Files (*.wav)")
        if filePath:

            self.sampling_rate, self.data = wavfile.read(filePath)
            if isinstance(self.current_tab, UniformRangeModeTab):
                self.uniform_range_tab.set_data(self.sampling_rate, self.data)



            elif isinstance(self.current_tab, AnimalSoundsModeTab):
                self.load_audio(filePath)
            elif isinstance(self.current_tab, MusicalInstrumentsModeTab):

                self.load_audio(filePath)
            else:
                pass
        if play_button_in and play_button_out:
            play_button_in.setText("Pause")  # Change text of the 'Play' button
            play_button_out.setText("Pause")  # Change text of the 'Play' button
        else:
            print("Play button not found!")

    def load_ecg(self):
        self.clear()
        self.current_tab2 = self.mode_tabs.currentWidget()
        if isinstance(self.current_tab2, ECGAbnormalitiesModeTab):
            self.record1 = wfdb.rdrecord('Task3_Data/00001_hr')
            self.ecg_normal = self.record1.p_signal
            self.record2 = wfdb.rdrecord('Task3_Data/00015_hr')
            self.ecg_arythm = self.record2.p_signal
            self.record3 = wfdb.rdrecord('Task3_Data/00228_hr')
            self.ecg_arythm_left_axis = self.record3.p_signal
            self.record4 = wfdb.rdrecord('Task3_Data/21804_hr')
            self.ecg_tachy = self.record3.p_signal
            self.ecg_signal = self.ecg_normal + self.ecg_arythm + self.ecg_arythm_left_axis + self.ecg_tachy

            print(self.ecg_signal[:10, 0])
            print(self.ecg_signal.shape)
            self.plot_ecg_signal()
            self.plot_ecg_spectrogram()

    def fourier_ecg(self):
        self.fft_ecg_signal = fft(self.ecg_signal[:, 0])

    def plot_ecg_signal(self):
        # Generate time array based on the sampling rate and duration
        fs = 500  # Sampling rate (500 samples per second)
        duration = 10  # Duration in seconds
        time_axis = np.linspace(0, duration, fs * duration)  # Generates time values from 0 to 10 seconds

        # Get the first column of the ECG signal for plotting
        ecg_signal_first_column = self.ecg_signal[:fs * duration, 0]  # First 5000 samples of the first column

        # Clear any previous plots
        self.input_graph.clear()

        # Plot the ECG signal on the graph
        self.input_graph.plot(time_axis, ecg_signal_first_column, pen='b')  # 'b' for blue color

    def normalize_signal(self, ecg_signal_first_col):
        """
        Normalize the first column of the ECG signal to the range [-1, 1].
        """
        ecg_signal_min = np.min(ecg_signal_first_col)
        ecg_signal_max = np.max(ecg_signal_first_col)
        return 2 * (ecg_signal_first_col - ecg_signal_min) / (ecg_signal_max - ecg_signal_min) - 1

    def plot_ecg_spectrogram(self):
        # Define parameters for spectrogram
        fs = 500  # Sampling rate in Hz
        ecg_signal_first_column = self.ecg_signal[:, 0]  # Use only the first column of ECG data
        normalized_ecg_signal = self.normalize_signal(ecg_signal_first_column)

        # Compute the spectrogram
        frequencies, times, Sxx = spectrogram(normalized_ecg_signal, fs=fs, nperseg=256, noverlap=128, nfft=1024)
        Sxx_log = 10 * np.log10(Sxx + 1e-10)  # Log scale to see details
        # Normalize the spectrogram values for better color mapping
        Sxx_log_normalized = (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min())

        # Define a color map for better visualization
        colormap = pg.ColorMap(
            [0, 0.25, 0.5, 0.75, 1],
            [
                (0, 0, 128, 255),  # Dark blue
                (0, 255, 255, 255),  # Cyan
                (255, 255, 0, 255),  # Yellow
                (255, 128, 0, 255),  # Orange
                (255, 0, 0, 255)  # Red
            ]
        )

        # Create an ImageItem to display the spectrogram
        img = pg.ImageItem()
        img.setImage(Sxx_log_normalized.T,
                     levels=(0, 1))  # Transpose to align time with x-axis and frequency with y-axis
        img.setColorMap(colormap)

        # Set the scaling of the spectrogram to match time and frequency
        img_rect = pg.QtCore.QRectF(0, 0, times[-1], frequencies[-1])
        img.setRect(img_rect)

        # Clear any previous plots in the graph
        self.input_spectrogram.clear()

        # Add the spectrogram image to the graph
        self.input_spectrogram.addItem(img)

        # Set the x-axis and y-axis ranges to match the time and frequency scales
        self.input_spectrogram.setXRange(0, times[-1])
        self.input_spectrogram.setYRange(0, fs / 2)  # Nyquist frequency (half the sampling rate)

    def plot_output_ecg_spectrogram(self, output_signal):
        self.output_signal = output_signal
        # Define parameters for spectrogram
        fs = 500  # Sampling rate in Hz

        # Compute the spectrogram
        frequencies, times, Sxx = spectrogram(self.output_signal, fs=fs, nperseg=256, noverlap=128, nfft=1024)
        Sxx_log = 10 * np.log10(Sxx + 1e-10)  # Log scale to see details

        # Normalize the spectrogram values for better color mapping
        Sxx_log_normalized = (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min())

        # Define a custom color map
        colormap = pg.ColorMap(
            [0, 0.25, 0.5, 0.75, 1],
            [
                (0, 0, 128, 255),  # Dark blue
                (0, 255, 255, 255),  # Cyan
                (255, 255, 0, 255),  # Yellow
                (255, 128, 0, 255),  # Orange
                (255, 0, 0, 255)  # Red
            ]
        )

        # Create an ImageItem to display the spectrogram
        img = pg.ImageItem()
        img.setImage(Sxx_log_normalized.T, levels=(0, 1))  # Transpose for correct orientation
        img.setColorMap(colormap)

        # Define the rectangle for time and frequency mapping
        img.setRect(pg.QtCore.QRectF(0, 0, times[-1], frequencies[-1]))

        # Clear any previous spectrograms from the graph
        self.output_spectrogram.clear()

        # Add the spectrogram image to the graph
        self.output_spectrogram.addItem(img)

        # Set the x-axis and y-axis ranges to match the time and frequency scales
        self.output_spectrogram.setXRange(0, times[-1])
        self.output_spectrogram.setYRange(0, fs / 2)  # Nyquist frequency (fs/2)

    def update_first_ecg(self):
        """
        Adjusts only the magnitude of the atrial fibrillation frequency range (3-10 Hz)
        of the ECG signal based on the slider value, while keeping other frequencies unchanged.
        """
        print(f"Slider value: {self.ecg_slider1_value}")  # Debugging
        # Sampling rate
        fs = 500  # Adjust based on your actual sampling rate

        # Perform FFT on the first column of the ECG signal
        self.fft_ecg_signal = (lambda x: fft(x[:, 0]) if x.ndim > 1 else fft(x))(self.ecg_signal)

        # Calculate frequency array corresponding to FFT output
        self.freqs = np.fft.fftfreq(len(self.fft_ecg_signal), 1 / fs)
        print(self.freqs)
        print(len(self.freqs))

        # Calculate the scale factor based on the slider value
        scale_factor = self.ecg_slider1_value / 100.0

        # Create a copy of the FFT to apply changes only to the target frequency range
        self.adjusted_fft = np.copy(self.fft_ecg_signal)

        # Create a mask to find the indices within the target frequency range (both positive and negative frequencies)
        self.afib_range_positive = (self.freqs > 0) & (self.freqs <= 50)
        self.afib_range_negative = (self.freqs < -0) & (self.freqs >= -50)

        print(self.adjusted_fft[self.afib_range_negative])
        # Apply the scale factor to these frequencies
        self.adjusted_fft[self.afib_range_positive] *= scale_factor
        self.adjusted_fft[self.afib_range_negative] *= scale_factor
        self.adjusted_fft[self.mrd2_range_positive] *= self.ecg_slider2_value / 100
        self.adjusted_fft[self.mrd2_range_negative] *= self.ecg_slider2_value / 100
        self.adjusted_fft[self.mrd3_range_positive] *= self.ecg_slider3_value / 100
        self.adjusted_fft[self.mrd3_range_negative] *= self.ecg_slider3_value / 100

        print(self.adjusted_fft[self.afib_range_negative])
        self.update_frequency_plot()

        # Reconstruct the time-domain signal with the adjusted frequencies
        self.adjusted_ecg_signal = np.real(ifft(self.adjusted_fft))
        self.output_graph.clear()  # Clear previous plot data
        self.output_graph.plot(self.adjusted_ecg_signal)
        self.plot_output_ecg_spectrogram(self.adjusted_ecg_signal)

    def update_second_ecg(self):
        """
               Adjusts only the magnitude of the atrial fibrillation frequency range (3-10 Hz)
               of the ECG signal based on the slider value, while keeping other frequencies unchanged.
               """
        print(f"Slider value: {self.ecg_slider2_value}")  # Debugging
        # Sampling rate
        fs = 500  # Adjust based on your actual sampling rate

        # Perform FFT on the first column of the ECG signal
        self.fft_ecg_signal = (lambda x: fft(x[:, 0]) if x.ndim > 1 else fft(x))(self.ecg_signal)

        # Calculate frequency array corresponding to FFT output
        freqs = np.fft.fftfreq(len(self.fft_ecg_signal), 1 / fs)
        print(freqs)

        # Calculate the scale factor based on the slider value
        scale_factor = self.ecg_slider2_value / 100.0

        # Create a copy of the FFT to apply changes only to the target frequency range
        self.adjusted_fft = np.copy(self.fft_ecg_signal)

        # Create a mask to find the indices within the target frequency range (both positive and negative frequencies)
        self.mrd2_range_positive = (freqs > 0) & (freqs <= 20)
        self.mrd2_range_negative = (freqs < -0) & (freqs >= -20)

        print(self.adjusted_fft[self.mrd2_range_negative])
        # Apply the scale factor to these frequencies
        self.adjusted_fft[self.mrd2_range_positive] *= scale_factor
        self.adjusted_fft[self.mrd2_range_negative] *= scale_factor
        self.adjusted_fft[self.afib_range_positive] *= self.ecg_slider1_value / 100
        self.adjusted_fft[self.afib_range_negative] *= self.ecg_slider1_value / 100
        self.adjusted_fft[self.mrd3_range_positive] *= self.ecg_slider3_value / 100
        self.adjusted_fft[self.mrd3_range_negative] *= self.ecg_slider3_value / 100
        self.update_frequency_plot()

        # Reconstruct the time-domain signal with the adjusted frequencies
        self.adjusted_ecg_signal = np.real(ifft(self.adjusted_fft))
        self.output_graph.clear()  # Clear previous plot data
        self.output_graph.plot(self.adjusted_ecg_signal)
        self.plot_output_ecg_spectrogram(self.adjusted_ecg_signal)

    def update_third_ecg(self):
        """
               Adjusts only the magnitude of the atrial fibrillation frequency range (3-10 Hz)
               of the ECG signal based on the slider value, while keeping other frequencies unchanged.
               """
        print(f"Slider value: {self.ecg_slider3_value}")  # Debugging
        # Sampling rate
        fs = 500  # Adjust based on your actual sampling rate

        # Perform FFT on the first column of the ECG signal
        self.fft_ecg_signal = (lambda x: fft(x[:, 0]) if x.ndim > 1 else fft(x))(self.ecg_signal)

        # Calculate frequency array corresponding to FFT output
        freqs = np.fft.fftfreq(len(self.fft_ecg_signal), 1 / fs)
        print(freqs)

        # Calculate the scale factor based on the slider value
        scale_factor = self.ecg_slider3_value / 100.0

        # Create a copy of the FFT to apply changes only to the target frequency range
        self.adjusted_fft = np.copy(self.fft_ecg_signal)

        # Create a mask to find the indices within the target frequency range (both positive and negative frequencies)
        self.mrd3_range_positive = (freqs > 0) & (freqs <= 40)
        self.mrd3_range_negative = (freqs < -0) & (freqs >= -40)

        print(self.adjusted_fft[self.mrd3_range_negative])
        # Apply the scale factor to these frequencies
        self.adjusted_fft[self.mrd3_range_positive] *= scale_factor
        self.adjusted_fft[self.mrd3_range_negative] *= scale_factor
        self.adjusted_fft[self.afib_range_positive] *= self.ecg_slider1_value / 100
        self.adjusted_fft[self.afib_range_negative] *= self.ecg_slider1_value / 100
        self.adjusted_fft[self.mrd2_range_positive] *= self.ecg_slider2_value / 100
        self.adjusted_fft[self.mrd2_range_negative] *= self.ecg_slider2_value / 100

        print(self.adjusted_fft[self.mrd3_range_negative])
        self.update_frequency_plot()

        # Reconstruct the time-domain signal with the adjusted frequencies
        self.adjusted_ecg_signal = np.real(ifft(self.adjusted_fft))
        self.output_graph.clear()  # Clear previous plot data
        self.output_graph.plot(self.adjusted_ecg_signal)
        self.plot_output_ecg_spectrogram(self.adjusted_ecg_signal)

    def slider_of_ecg(self, index, value):
        if index == 0:
            self.ecg_slider1_value = value
            self.update_first_ecg()
        elif index == 1:
            self.ecg_slider2_value = value
            self.update_second_ecg()
        else:
            self.ecg_slider3_value = value
            self.update_third_ecg()
        print(self.ecg_slider3_value)

    def toggle_spectrogram_visibility(self):
        self.show_spectrograms = not self.show_spectrograms
        self.input_spectrogram.setVisible(self.show_spectrograms)
        self.input_spectrogram.setMinimumHeight(300)
        self.output_spectrogram.setVisible(self.show_spectrograms)
        self.output_spectrogram.setMinimumHeight(300)

    def change_frequency_scale(self):
        if self.scale_dropdown.currentText() == "Audiogram":
            self.use_audiogram_scale = True
        else:
            self.use_audiogram_scale = False
        self.handle_units()
        self.update_frequency_plot()

    def update_frequency_plot(self):
        # Identify the current tab
        self.current_tab = self.mode_tabs.currentWidget()

        if self.current_tab == self.ecg_abnormalities_tab:
            # Audiogram or Linear Scale for ECG abnormalities tab
            self.frequency_domain.clear()  # Clear plot before updating
            if self.use_audiogram_scale:
                # Convert magnitude to dB scale for Audiogram
                db_freq = 20 * np.log10(np.abs(self.adjusted_fft) + 1e-10)  # Avoid log(0)
                self.frequency_domain.plot(self.freqs, db_freq, pen='b')
                self.frequency_domain.setTitle("Frequency Domain (Audiogram Scale)")
            else:
                # Linear scale magnitude
                self.frequency_domain.plot(self.freqs, np.abs(self.adjusted_fft), pen='b')
                self.frequency_domain.setTitle("Frequency Domain (Linear Scale)")
                self.frequency_domain.setYRange(0, 300)  # Adjust Y-range as needed

            self.frequency_domain.enableAutoRange()

        elif self.current_tab == self.uniform_range_tab:
            # Perform FFT on adjusted data for Uniform Range Tab
            if self.adjusted_data is None:
                return

            # Compute FFT and frequencies
            freqs = np.fft.fftfreq(len(self.adjusted_data), 1 / self.sampling_rate)
            freq_magnitudes = np.abs(np.fft.fft(self.adjusted_data))

            # # Filter negative frequencies for plotting
            # pos_mask = freqs >= 0
            # freqs = freqs[pos_mask]
            # freq_magnitudes = freq_magnitudes[pos_mask]

            self.frequency_domain.clear()  # Clear plot before updating

            # Plot based on scale
            if self.use_audiogram_scale:
                # Convert magnitude to dB scale for Audiogram
                db_magnitudes = 20 * np.log10(freq_magnitudes + 1e-10)
                self.frequency_domain.plot(freqs, db_magnitudes, pen='g')
                self.frequency_domain.setTitle("Frequency Domain (Audiogram Scale)")
            else:
                # Linear scale magnitude
                self.frequency_domain.plot(freqs, freq_magnitudes, pen='g')
                self.frequency_domain.setTitle("Frequency Domain (Linear Scale)")

        elif self.current_tab == self.animal_sounds_tab or self.current_tab== self.instrument_sound_tab:
            # Perform FFT on adjusted data for Uniform Range Tab
            if self.adjusted_data is None:
                return

            # Compute FFT and frequencies
            freqs = np.fft.fftfreq(len(self.adjusted_data), 1 / self.sampling_rate)
            freq_magnitudes = np.abs(np.fft.fft(self.adjusted_data))

            # # Filter negative frequencies for plotting
            # pos_mask = freqs >= 0
            # freqs = freqs[pos_mask]
            # freq_magnitudes = freq_magnitudes[pos_mask]

            self.frequency_domain.clear()  # Clear plot before updating

            # Plot based on scale
            if self.use_audiogram_scale:
                # Convert magnitude to dB scale for Audiogram
                db_magnitudes = 20 * np.log10(freq_magnitudes + 1e-10)
                self.frequency_domain.plot(freqs, db_magnitudes, pen='g')

                self.frequency_domain.setTitle("Frequency Domain (Audiogram Scale)")
            else:
                # Linear scale magnitude
                self.frequency_domain.plot(freqs, freq_magnitudes, pen='g')
                self.frequency_domain.setTitle("Frequency Domain (Linear Scale)")

            self.frequency_domain.enableAutoRange()
    def handle_units(self):
        if self.use_audiogram_scale:
            self.frequency_domain.setLabel('left', "Amplitude",units='dB')
        else:
            self.frequency_domain.setLabel('left', "Amplitude")


    def handle_frequency_update(self, adjusted_data):
        # Process or display the adjusted frequency data in the main file

        self.adjusted_data = adjusted_data
        print("Received updated frequency data:", adjusted_data)

    def handle_frequency_toggle(self, boolen):
        # Process or display the adjusted frequency data in the main file

        if boolen:
            self.change_frequency_scale()

    def play_original_audio(self):
        if self.data is not None and not self.audio_playing:
            sd.play(self.data, self.sampling_rate)
            self.audio_playing = True
            self.playOriginalButton.setText("Pause Original")
        elif self.data is not None and self.audio_playing:
            sd.stop()
            self.audio_playing = False
            self.playOriginalButton.setText("Play Original")

    def play_adjusted_audio(self):
        if self.adjusted_data is not None and not self.edited_audio_playing:
            sd.play(self.adjusted_data, self.sampling_rate)
            self.edited_audio_playing = True
            self.playAdjustedButton.setText("Pause Adjisted")
        elif self.adjusted_data is not None and self.edited_audio_playing:
            sd.stop()
            self.edited_audio_playing = False
            self.playAdjustedButton.setText("Play Adjisted")

    def set_graph_limits(self):
        if isinstance(self.current_tab,AnimalSoundsModeTab) or isinstance(self.current_tab,MusicalInstrumentsModeTab):
            self.time_input = []
            self.time_output = []
            time_input = np.linspace(0, len(self.audio_data) / self.sample_rate, num=len(self.audio_data))
            self.time_input = time_input
            time_output = np.linspace(0, len(self.adjusted_data) / self.sample_rate, num=len(self.adjusted_data))
            self.time_output = time_output

            viewbox = self.output_graph.getViewBox()
            viewbox2 = self.input_graph.getViewBox()

            # Define limits for the output graph
            xmin_output = 0
            xmax_output = max(self.time_output)
            ymin_output = min(self.adjusted_data)
            ymax_output = max(self.adjusted_data)
            xmin_input = 0
            xmax_input = max(self.time_input)
            ymin_input = min(self.audio_data)
            ymax_input = max(self.audio_data)


            # Set limits for the ViewBox
            if isinstance(self.current_tab,MusicalInstrumentsModeTab):
                minYRange=0.001
                minXRange=1
            if isinstance(self.current_tab,AnimalSoundsModeTab):
                minYRange=0.6
                minXRange=0.3
            viewbox.setLimits(
                xMin=xmin_output, xMax=xmax_output,
                yMin=ymin_output, yMax=ymax_output,
                minXRange=minXRange, maxXRange=xmax_output - xmin_output,
                minYRange=minYRange, maxYRange=ymax_output - ymin_output
            )
            viewbox2.setLimits(
                xMin=xmin_input, xMax=xmax_input,
                yMin=ymin_input, yMax=ymax_input,
                minXRange=minXRange, maxXRange=xmax_input - xmin_input,
                minYRange=minYRange, maxYRange=ymax_input - ymin_input
            )

        self.input_graph.setXLink(self.output_graph)  # Link x-axis
        self.output_graph.setXLink(self.input_graph)  # Link x-axis
        self.input_graph.setYLink(self.output_graph)  # Optional: Link y-axis if required
        self.output_graph.setYLink(self.input_graph)  # Optional: Link y-axis if required


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = SignalEqualizerUI()
    window.show()
    app.exec_()
