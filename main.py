
import sys
import threading
import queue
from collections import deque
from itertools import islice
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
import argparse
import time

# ------------------------ Logging Configuration ------------------------

def configure_logging(verbose=False):
    """
    Configure logging settings.
    
    Args:
        verbose (bool): If True, set logging level to DEBUG. Otherwise, INFO.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# ------------------------ Import Dependencies ------------------------

# Suppress DeprecationWarning for PyQt5 (Temporary Fix)
warnings.filterwarnings("ignore", category=DeprecationWarning, module='PyQt5')

# Suppress specific FutureWarnings temporarily
warnings.filterwarnings("ignore", category=FutureWarning, message="Passing a tuple of past_key_values is deprecated")

# Import the UI layout
from ui_main import Ui_MainWindow

# Import Silero VAD
from silero_vad import get_speech_timestamps, VADIterator, collect_chunks

# ------------------------ Constants ------------------------

AUDIO_RATE = 16000  # Whisper's expected sample rate
CHANNELS = 1         # Mono audio
FRAME_DURATION = 30  # Duration of a frame in ms (must be 10, 20, or 30)
MAX_BUFFER_SIZE = AUDIO_RATE * 10  # Maximum buffer size (e.g., 10 seconds of audio)

# VAD Sensitivity Parameters
VAD_THRESHOLD = 0.5   # Adjusted threshold as desired

# Post-Processing Parameters (in seconds)
POST_MIN_DURATION = 0.5      # Minimum gap to merge segments
POST_SMOOTH_PADDING = 0.1    # Padding to add to segment boundaries 
POST_MERGE_MIN_GAP = 0.2     # Minimum duration for speech segments



# ------------------------ Main Class ------------------------

class RealTimeTranslator(QMainWindow, Ui_MainWindow):
    def __init__(self, verbose=False):
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
        self.audio_queue = queue.Queue(maxsize=10)      # Limit queue size to prevent memory issues
        self.transcription_queue = queue.Queue()
        self.audio_frames_queue = queue.Queue(maxsize=100)

        # Flags
        self.is_recording = False
        self.should_exit = False  # Flag to signal threads to exit on application close

        # Settings
        self.settings_lock = threading.Lock()
        self.language = "auto"          # default language
        self.no_speech_threshold = 0.1  # default value, can be updated from UI

        # Initialize a separate lock for buffer management
        self.buffer_lock = threading.Lock()

        # Connect settings controls
        self.languageComboBox.currentTextChanged.connect(self.update_language)
        self.vadWindowSizeSpinBox.valueChanged.connect(self.update_vad_window_size)
        self.vadHopSizeSpinBox.valueChanged.connect(self.update_vad_hop_size)
        self.noSpeechThresholdSpinBox.valueChanged.connect(self.update_no_speech_threshold)

        # Initialize Silero VAD
        self.load_vad_model()

        # Buffer to hold audio frames using deque
        self.audio_buffer = deque(maxlen=MAX_BUFFER_SIZE)

        # Initialize vad_window_size and vad_hop_size based on spin box defaults
        self.vad_window_size_sec = self.vadWindowSizeSpinBox.value()  # in seconds
        self.vad_hop_size_sec = self.vadHopSizeSpinBox.value()        # in seconds

        self.vad_window_size = int(AUDIO_RATE * self.vad_window_size_sec)
        self.vad_hop_size = int(AUDIO_RATE * self.vad_hop_size_sec)

        # Define overlap (50% of window size)
        self.overlap = int(self.vad_window_size * 0.5)
        self.shift = self.vad_window_size - self.overlap

        logging.info(f"Initial VAD window size: {self.vad_window_size_sec} sec ({self.vad_window_size} samples)")
        logging.info(f"Initial VAD hop size: {self.vad_hop_size_sec} sec ({self.vad_hop_size} samples)")
        logging.debug(f"Initial overlap: {self.overlap} samples")
        logging.debug(f"Initial shift: {self.shift} samples")

        # Load models
        self.load_models()

        
        # Post-Processing Parameters (in seconds)
        self.post_min_duration = POST_MIN_DURATION
        self.post_smooth_padding = POST_SMOOTH_PADDING
        self.post_merge_min_gap = POST_MERGE_MIN_GAP

        # Initialize threads
        self.audio_thread = None
        self.audio_processing_thread = None
        self.transcription_thread = None

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
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
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

        # NOTE: User requested not to change this model ID
        asr_model_id = "openai/whisper-large-v3-turbo" 
        try:
            self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                asr_model_id,
                torch_dtype=self.torch_dtype,
                use_safetensors=True,
                trust_remote_code=True
            ).to(self.device)
            logging.info(f"Whisper model loaded successfully on {self.device}")
        except Exception as e:
            logging.error(f"Error loading ASR model: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load ASR model: {e}")
            sys.exit(1)

        try:
            self.asr_processor = WhisperProcessor.from_pretrained(
                asr_model_id,
                trust_remote_code=True
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

    def update_vad_window_size(self, new_value_sec):
        with self.settings_lock:
            if new_value_sec < self.vadHopSizeSpinBox.value():
                # Adjust hop size to be less than or equal to window size
                self.vadHopSizeSpinBox.setValue(new_value_sec)
            self.vad_window_size_sec = new_value_sec
            self.vad_window_size = int(AUDIO_RATE * self.vad_window_size_sec)
            # Update overlap and shift based on new window size
            self.overlap = int(self.vad_window_size * 0.5)
            self.shift = self.vad_window_size - self.overlap
            logging.info(f"VAD window size updated to {new_value_sec} sec ({self.vad_window_size} samples).")
            logging.debug(f"Updated overlap: {self.overlap} samples")
            logging.debug(f"Updated shift: {self.shift} samples")

    def update_vad_hop_size(self, new_value_sec):
        with self.settings_lock:
            if new_value_sec > self.vadWindowSizeSpinBox.value():
                # Adjust window size to be greater than or equal to hop size
                self.vadWindowSizeSpinBox.setValue(new_value_sec)
            self.vad_hop_size_sec = new_value_sec
            self.vad_hop_size = int(AUDIO_RATE * self.vad_hop_size_sec)
            logging.info(f"VAD hop size updated to {new_value_sec} sec ({self.vad_hop_size} samples).")

    def update_no_speech_threshold(self, new_value):
        with self.settings_lock:
            self.no_speech_threshold = new_value
            logging.info(f"No Speech Threshold updated to {new_value} seconds.")

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

        # Clear the audio queues and buffer safely instead of reassigning
        with self.buffer_lock:
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()
            while not self.audio_frames_queue.empty():
                self.audio_frames_queue.get_nowait()
            self.audio_buffer.clear()

        logging.debug("Cleared audio_queue, audio_frames_queue, and reset audio_buffer.")

    def capture_audio(self):
        """
        Capture audio from the default input device using SoundDevice's InputStream.
        This method runs in a separate thread to prevent blocking the main GUI thread.
        """
        try:
            block_duration_ms = FRAME_DURATION
            blocksize = int(AUDIO_RATE * block_duration_ms / 1000)
            logging.debug(f"Audio capture blocksize set to {blocksize} samples (block_duration_ms={block_duration_ms}ms).")

            stream = sd.InputStream(
                samplerate=AUDIO_RATE,
                blocksize=blocksize,
                dtype='int16',
                channels=CHANNELS,
                callback=self.audio_callback,
                latency='low'
            )

            with stream:
                logging.info("Audio stream opened successfully.")
                while self.is_recording and not self.should_exit:
                    sd.sleep(100)
        except Exception as e:
            logging.error(f"Audio capture error: {e}")
            self.is_recording = False
            self.startButton.setEnabled(True)
            self.stopButton.setEnabled(False)
            QMessageBox.critical(self, "Audio Capture Error", f"An error occurred while capturing audio:\n{e}")

    def audio_callback(self, indata, frames, time_info, status):
        """
        Callback function called by sounddevice for each audio block.
        """
        if not self.is_recording:
            return

        if status:
            logging.warning(f"Audio Callback Status: {status}")

        try:
            audio_data = np.frombuffer(indata, dtype=np.int16).astype(np.float32) / 32768.0

            with self.buffer_lock:
                self.audio_buffer.extend(audio_data)
                logging.debug(f"Appended {len(audio_data)} samples to audio_buffer. Current size: {len(self.audio_buffer)}")

            try:
                self.audio_frames_queue.put_nowait(audio_data)
            except queue.Full:
                logging.warning("Audio frames queue is full. Dropping audio frames.")
        except Exception as e:
            logging.error(f"Unexpected error in audio_callback: {e}")

    # -------------------- Post-Processing Methods --------------------
    def merge_close_segments(self, speech_timestamps, min_gap=0.3):
        if not speech_timestamps:
            return []

        merged_segments = [speech_timestamps[0]]
        min_gap_samples = int(min_gap * AUDIO_RATE)

        for current in speech_timestamps[1:]:
            previous = merged_segments[-1]
            gap = current['start'] - previous['end']
            if gap <= min_gap_samples:
                merged_segments[-1]['end'] = current['end']
            else:
                merged_segments.append(current)

        return merged_segments

    def smooth_boundaries(self, speech_timestamps, padding=0.05):
        smoothed_segments = []
        padding_samples = int(padding * AUDIO_RATE)

        for ts in speech_timestamps:
            start = max(0, ts['start'] - padding_samples)
            end = ts['end'] + padding_samples
            smoothed_segments.append({'start': start, 'end': end})

        return smoothed_segments

    def remove_short_segments(self, speech_timestamps, min_duration=0.3):
        filtered_segments = []
        min_duration_samples = int(min_duration * AUDIO_RATE)

        for ts in speech_timestamps:
            duration = ts['end'] - ts['start']
            if duration >= min_duration_samples:
                filtered_segments.append(ts)

        return filtered_segments
    # -------------------- End of Post-Processing Methods --------------------

    def process_audio_frames(self):
        while not self.should_exit:
            if not self.is_recording:
                # If not recording, sleep to reduce CPU usage
                time.sleep(0.1)
                continue

            try:
                self.audio_frames_queue.get(timeout=1)  # Just a signal to proceed
                with self.buffer_lock:
                    if len(self.audio_buffer) >= self.vad_window_size:
                        audio_chunk = list(islice(self.audio_buffer, self.vad_window_size))
                        audio_chunk = np.array(audio_chunk, dtype=np.float32)
                        audio_tensor = torch.from_numpy(audio_chunk).unsqueeze(0).to('cpu')

                        speech_timestamps = self.get_speech_ts(
                            audio_tensor,
                            self.vad_model,
                            sampling_rate=AUDIO_RATE,
                            threshold=VAD_THRESHOLD
                        )

                        if speech_timestamps:
                            merged_timestamps = self.merge_close_segments(
                                speech_timestamps,
                                min_gap=self.post_merge_min_gap
                            )
                            smoothed_timestamps = self.smooth_boundaries(
                                merged_timestamps,
                                padding=self.post_smooth_padding
                            )
                            
                            final_timestamps = self.remove_short_segments(
                                smoothed_timestamps,
                                min_duration=self.post_min_duration
                            )

                            if final_timestamps:  # Check if non-empty
                                speech_chunks = self.collect_chunks(final_timestamps, audio_tensor)
                                for chunk in speech_chunks:
                                    speech_np = chunk.squeeze().cpu().numpy()
                                    try:
                                        self.audio_queue.put_nowait(speech_np)
                                        logging.info("Speech segment added to queue after post-processing.")
                                    except queue.Full:
                                        logging.warning("Audio queue is full. Dropping speech segment.")
                            else:
                                # No segments detected, skip collecting chunks.
                                logging.debug("No final timestamps detected, skipping chunk collection.")


                        # Remove shift samples
                        if self.shift > 0:
                            remaining_samples = list(islice(self.audio_buffer, self.shift, len(self.audio_buffer)))
                            self.audio_buffer = deque(remaining_samples, maxlen=MAX_BUFFER_SIZE)

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in audio processing: {e}")
                continue

    def process_transcription(self):
        while not self.should_exit:
            if not self.is_recording:
                time.sleep(0.1)
                continue

            try:
                audio_to_process = self.audio_queue.get(timeout=1)
                logging.debug(f"Retrieved speech segment of {len(audio_to_process)} samples from audio_queue.")

                with self.settings_lock:
                    current_language = self.language
                    # current_no_speech_threshold = self.no_speech_threshold 
                    # Not used directly with generate()

                # Prepare generate_kwargs for transcription
                transcribe_kwargs = {
                    "task": "transcribe",
                    "max_new_tokens": 440,
                    "num_beams": 4,
                    "condition_on_prev_tokens": True,
                    "compression_ratio_threshold": 1.35,
                    "temperature": 0.0,
                    "logprob_threshold": -0.5,
                    "return_timestamps": True
                }

                if current_language != "auto":
                    transcribe_kwargs["language"] = current_language

                inputs = self.asr_processor(
                    audio_to_process,
                    sampling_rate=AUDIO_RATE,
                    return_tensors="pt",
                    return_attention_mask=True
                ).to(self.device, dtype=self.torch_dtype)

                input_features = inputs.input_features
                attention_mask = inputs.attention_mask

                with torch.no_grad():
                    generated_ids = self.asr_model.generate(
                        input_features,
                        attention_mask=attention_mask,
                        **transcribe_kwargs
                    )
                    transcription = self.asr_processor.batch_decode(
                        generated_ids, skip_special_tokens=True)[0].strip()
                    logging.info(f"Transcription: {transcription}")

                translation = ""
                if current_language not in ["en", "auto"]:
                    translate_kwargs = {
                        "task": "translate",
                        "max_new_tokens": 440,
                        "num_beams": 3,
                        "condition_on_prev_tokens": False,
                        "compression_ratio_threshold": 1.35,
                        "temperature": 0.0,
                        "logprob_threshold": -1.0,
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

                self.transcription_queue.put((transcription, translation))
                logging.debug("Placed transcription and translation into transcription_queue.")

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
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1)
            logging.debug("Audio thread has been terminated.")

        # Wait for the audio processing thread to finish
        if self.audio_processing_thread and self.audio_processing_thread.is_alive():
            self.audio_processing_thread.join(timeout=1)
            logging.debug("Audio processing thread has been terminated.")

        # Wait for the transcription thread to finish
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.transcription_thread.join(timeout=1)
            logging.debug("Transcription thread has been terminated.")

        # Explicitly release GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        event.accept()
        logging.info("Shutdown sequence completed.")

# ------------------------ Entry Point ------------------------

def main():
    parser = argparse.ArgumentParser(description="Real-Time Transcription and Translation Application")
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose (DEBUG) logging.'
    )
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)

    app = QApplication(sys.argv)
    window = RealTimeTranslator(verbose=args.verbose)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()