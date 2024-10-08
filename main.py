import sys
import threading
import queue
import sounddevice as sd
import numpy as np
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    WhisperProcessor,
)
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtCore import QTimer, Qt, QDateTime
import warnings
import logging

# Suppress DeprecationWarning for PyQt5 (Temporary Fix)
warnings.filterwarnings("ignore", category=DeprecationWarning, module='PyQt5')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Suppress specific FutureWarnings temporarily
warnings.filterwarnings("ignore", category=FutureWarning, message="Passing a tuple of past_key_values is deprecated")

# Import the UI layout
from ui_main import Ui_MainWindow

# Import Silero VAD
from silero_vad import get_speech_timestamps, VADIterator, collect_chunks

# Constants
AUDIO_RATE = 16000  # Whisper's expected sample rate
CHANNELS = 1         # Mono audio
FRAME_DURATION = 30  # Duration of a frame in ms (must be 10, 20, or 30)


class RealTimeTranslator(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Real-Time Transcription and Translation")
        self.setStyleSheet(self.load_stylesheet())

        # Initialize buttons
        self.startButton.clicked.connect(self.start_transcription)
        self.stopButton.clicked.connect(self.stop_transcription)
        self.stopButton.setEnabled(False)

        # Initialize text areas
        self.transcriptionArea.setReadOnly(True)
        self.translationArea.setReadOnly(True)

        # Queues for threading
        self.audio_queue = queue.Queue(maxsize=10)  # Limit queue size to prevent memory issues
        self.transcription_queue = queue.Queue()
        self.audio_frames_queue = queue.Queue(maxsize=100)

        # Flags
        self.is_recording = False
        self.should_exit = False  # Flag to signal threads to exit on application close

        # Settings
        self.settings_lock = threading.Lock()
        self.language = "auto"  # default language

        # Connect settings controls
        self.languageComboBox.currentTextChanged.connect(self.update_language)
        self.vadWindowSizeSpinBox.valueChanged.connect(self.update_vad_window_size)
        self.vadHopSizeSpinBox.valueChanged.connect(self.update_vad_hop_size)

        # Initialize Silero VAD
        self.load_vad_model()

        # Buffer to hold audio frames
        self.audio_buffer = np.array([], dtype=np.float32)
        # Initialize vad_window_size and vad_hop_size based on spin box defaults
        self.vad_window_size = int(AUDIO_RATE * self.vadWindowSizeSpinBox.value())  # seconds to samples
        self.vad_hop_size = int(AUDIO_RATE * self.vadHopSizeSpinBox.value())        # seconds to samples

        # Load models
        self.load_models()

        # Start the transcription thread
        self.transcription_thread = threading.Thread(target=self.process_transcription, daemon=True)
        self.transcription_thread.start()

        # Start the audio processing thread
        self.audio_processing_thread = threading.Thread(target=self.process_audio_frames, daemon=True)
        self.audio_processing_thread.start()

        # Set up a timer to periodically update the UI
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(500)  # Update every 500 ms

    def load_stylesheet(self):
        try:
            with open("resources/style.qss", "r") as f:
                return f.read()
        except FileNotFoundError:
            logging.warning("Style sheet not found. Proceeding without styles.")
            return ""

    def load_vad_model(self):
        logging.info("Loading Silero VAD model...")
        self.vad_model, self.vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                        model='silero_vad',
                                                        force_reload=False,
                                                        onnx=False)
        (self.get_speech_ts, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks) = self.vad_utils
        logging.info("Silero VAD model loaded.")

    def load_models(self):
        logging.info("Loading models...")

        # Set device and data types
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        logging.info(f"Using device: {self.device}")
        if self.device == "cuda":
            logging.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
            logging.info(f"Number of GPUs: {torch.cuda.device_count()}")
            logging.info(f"CUDA Version: {torch.version.cuda}")
            logging.info(f"cuDNN Version: {torch.backends.cudnn.version()}")

        # Load Whisper model for ASR with translation capability
        asr_model_id = "openai/whisper-large-v3-turbo"
        try:
            self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                asr_model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                trust_remote_code=True  # Ensure the repository is trusted
            ).to(self.device)
            logging.info(f"Whisper model loaded successfully on {self.device}")
        except Exception as e:
            logging.error(f"Error loading ASR model: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load ASR model: {e}")
            sys.exit(1)

        try:
            self.asr_processor = WhisperProcessor.from_pretrained(
                asr_model_id,
                trust_remote_code=True  # Ensure the repository is trusted
            )
            logging.info("Whisper processor initialized.")
        except Exception as e:
            logging.error(f"Error initializing ASR processor: {e}")
            QMessageBox.critical(self, "Error", f"Failed to initialize ASR processor: {e}")
            sys.exit(1)

        # Remove forced_decoder_ids to prevent conflicts with task
        if hasattr(self.asr_model.generation_config, 'forced_decoder_ids'):
            self.asr_model.generation_config.forced_decoder_ids = None

        # Ensure pad_token_id is set
        if self.asr_processor.tokenizer.pad_token is None:
            self.asr_processor.tokenizer.pad_token = self.asr_processor.tokenizer.eos_token
            self.asr_model.config.pad_token_id = self.asr_processor.tokenizer.pad_token_id

    def update_language(self, new_language):
        language = new_language.strip().lower()
        with self.settings_lock:
            if language == "auto detect":
                self.language = "auto"
            else:
                self.language = language if language else "auto"
            logging.info(f"Source language updated to {self.language}.")

    def update_vad_window_size(self, new_value):
        with self.settings_lock:
            self.vad_window_size = int(AUDIO_RATE * new_value)
            logging.info(f"VAD window size updated to {new_value} seconds ({self.vad_window_size} samples).")

    def update_vad_hop_size(self, new_value):
        with self.settings_lock:
            self.vad_hop_size = int(AUDIO_RATE * new_value)
            logging.info(f"VAD hop size updated to {new_value} seconds ({self.vad_hop_size} samples).")

    def start_transcription(self):
        if self.is_recording:
            return
        self.is_recording = True
        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        self.transcriptionArea.clear()
        self.translationArea.clear()
        self.audio_thread = threading.Thread(target=self.capture_audio, daemon=True)
        self.audio_thread.start()
        logging.info("Started recording.")

    def stop_transcription(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        logging.info("Stopped recording.")

        # Clear the audio queues safely
        self.audio_queue = queue.Queue(maxsize=10)
        self.audio_frames_queue = queue.Queue(maxsize=100)
        self.audio_buffer = np.array([], dtype=np.float32)

    def capture_audio(self):
        try:
            with sd.RawInputStream(samplerate=AUDIO_RATE, blocksize=int(FRAME_DURATION * AUDIO_RATE / 1000),
                                   dtype='int16', channels=CHANNELS, callback=self.audio_callback):
                while self.is_recording:
                    sd.sleep(100)
        except Exception as e:
            logging.error(f"Audio capture error: {e}")
            self.is_recording = False
            self.startButton.setEnabled(True)
            self.stopButton.setEnabled(False)
            QMessageBox.critical(self, "Error", f"Audio capture error: {e}")

    def audio_callback(self, indata, frames, time_info, status):
        if not self.is_recording:
            return

        if status:
            logging.warning(f"Audio status: {status}")

        try:
            # Convert indata to bytes and enqueue
            self.audio_frames_queue.put_nowait(bytes(indata))
        except queue.Full:
            logging.warning("Audio frames queue is full. Dropping audio frames.")

    def process_audio_frames(self):
        while not self.should_exit:
            try:
                indata_bytes = self.audio_frames_queue.get(timeout=1)
                # Convert bytes to numpy array
                audio_data = np.frombuffer(indata_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                with self.settings_lock:
                    # Append audio data to buffer
                    self.audio_buffer = np.concatenate((self.audio_buffer, audio_data))

                    # If we have enough data, process with VAD
                    while len(self.audio_buffer) >= self.vad_window_size:
                        # Prepare the audio chunk
                        audio_chunk = self.audio_buffer[:self.vad_window_size]
                        # Remove processed data from buffer
                        self.audio_buffer = self.audio_buffer[self.vad_hop_size:]  # Move the window

                        # Convert to torch tensor
                        audio_tensor = torch.from_numpy(audio_chunk).unsqueeze(0)

                        # Apply VAD
                        speech_timestamps = self.get_speech_ts(audio_tensor, self.vad_model, sampling_rate=AUDIO_RATE)

                        if speech_timestamps:
                            # Speech detected, collect speech chunks
                            speech_chunks = self.collect_chunks(speech_timestamps, audio_tensor)
                            for chunk in speech_chunks:
                                # Convert chunk to numpy array
                                speech_np = chunk.squeeze().numpy()
                                try:
                                    self.audio_queue.put_nowait(speech_np)
                                    logging.info("Speech segment added to queue.")
                                except queue.Full:
                                    logging.warning("Audio queue is full. Dropping speech segment.")
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in audio processing: {e}")
                continue

    def process_transcription(self):
        while not self.should_exit:
            try:
                # Get audio to process
                audio_to_process = self.audio_queue.get(timeout=1)

                with self.settings_lock:
                    current_language = self.language

                # Prepare generate_kwargs for transcription
                transcribe_kwargs = {
                    "task": "transcribe",
                    "max_new_tokens": 440,
                    "num_beams": 3,
                    "condition_on_prev_tokens": False,
                    "compression_ratio_threshold": 1.35,
                    "temperature": 0.0,
                    "logprob_threshold": -1.0,   # Adjusted for better filtering
                    "no_speech_threshold": 0.5,  # Adjusted for better VAD handling
                    "return_timestamps": True
                }

                if current_language != "auto":
                    transcribe_kwargs["language"] = current_language

                # Perform transcription using Whisper
                inputs = self.asr_processor(
                    audio_to_process,
                    sampling_rate=AUDIO_RATE,
                    return_tensors="pt",
                    return_attention_mask=True
                ).to(self.device, dtype=self.torch_dtype)

                input_features = inputs.input_features
                attention_mask = inputs.attention_mask

                with torch.no_grad():
                    # Transcribe the audio
                    generated_ids = self.asr_model.generate(
                        input_features,
                        attention_mask=attention_mask,
                        **transcribe_kwargs
                    )
                    transcription = self.asr_processor.batch_decode(
                        generated_ids, skip_special_tokens=True)[0].strip()
                    logging.info(f"Transcription: {transcription}")

                translation = ""

                # Perform translation to English if current_language is not 'en' or 'auto'
                if current_language not in ["en", "auto"]:
                    translate_kwargs = {
                        "task": "translate",
                        "max_new_tokens": 440,
                        "num_beams": 1,
                        "condition_on_prev_tokens": False,
                        "compression_ratio_threshold": 1.35,
                        "temperature": 0.0,
                        "logprob_threshold": -0.5,   # Adjusted
                        "no_speech_threshold": 0.5,  # Adjusted
                        "return_timestamps": True
                    }
                    if current_language != "auto":
                        translate_kwargs["language"] = current_language

                    with torch.no_grad():
                        translation_ids = self.asr_model.generate(
                            input_features,
                            attention_mask=attention_mask,
                            **translate_kwargs
                        )
                        translation = self.asr_processor.batch_decode(
                            translation_ids, skip_special_tokens=True)[0].strip()
                        logging.info(f"Translation: {translation}")

                # Put the transcription and translation into the queue
                self.transcription_queue.put((transcription, translation))

            except queue.Empty:
                continue
            except Exception as e:
                error_message = f"[Error in transcription]: {e}"
                logging.error(f"Transcription processing error: {e}")
                self.transcription_queue.put((error_message, ""))

    def update_ui(self):
        # Update transcription and translation areas
        while not self.transcription_queue.empty():
            transcription, translation = self.transcription_queue.get()
            timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
            if transcription:
                self.transcriptionArea.append(f"[{timestamp}] {transcription}")
            if translation:
                self.translationArea.append(f"[{timestamp}] {translation}")

        # Update the segment count label if it exists
        if hasattr(self, 'chunkCountLabel'):
            segment_count = self.audio_queue.qsize()
            self.chunkCountLabel.setText(f"Segments in Queue: {segment_count}")

    def closeEvent(self, event):
        """Handle the window close event to ensure clean shutdown."""
        logging.info("Application closing. Initiating shutdown sequence.")
        self.is_recording = False
        self.should_exit = True

        # Wait for the audio thread to finish
        if hasattr(self, 'audio_thread') and self.audio_thread.is_alive():
            self.audio_thread.join()

        # Wait for the audio processing thread to finish
        if hasattr(self, 'audio_processing_thread') and self.audio_processing_thread.is_alive():
            self.audio_processing_thread.join()

        # Wait for the transcription thread to finish
        if self.transcription_thread.is_alive():
            self.transcription_thread.join()

        event.accept()
        logging.info("Shutdown sequence completed.")


def main():
    app = QApplication(sys.argv)
    window = RealTimeTranslator()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
