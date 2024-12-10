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
from caption_window import CaptionWindow

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
MAX_BUFFER_SIZE = AUDIO_RATE * 30  # Maximum buffer size (e.g., 30 seconds of audio)

# VAD Sensitivity Parameters
VAD_CHUNK_SIZE = 512  # Silero VAD expects 512 samples for 16kHz audio
VAD_THRESHOLD = 0.3  # Lower threshold for more sensitive speech detection
MIN_SILENCE_DURATION = 0.2  # Minimum silence duration to consider a sentence break (in seconds)
MIN_SPEECH_DURATION = 0.3  # Minimum speech duration to consider valid speech (in seconds)
MAX_SENTENCE_DURATION = 10.0  # Maximum duration for a single sentence (in seconds)

# Post-Processing Parameters (in seconds)
POST_MIN_DURATION = 0.5
POST_SMOOTH_PADDING = 0.1
POST_MERGE_MIN_GAP = 0.2

# Add these constants near the other constants
MAX_CONTINUOUS_SPEECH = 3  # Maximum continuous speech duration in seconds
MAX_CONTINUOUS_SAMPLES = AUDIO_RATE * MAX_CONTINUOUS_SPEECH  # Maximum samples for continuous speech

# Add this class to track processed segments
class ProcessedSegment:
    def __init__(self, start_time, end_time, audio_data):
        self.start_time = start_time
        self.end_time = end_time
        self.audio_data = audio_data
        self.hash = hash(audio_data.tobytes())

    def overlaps_with(self, other):
        return not (self.end_time < other.start_time or self.start_time > other.end_time)

def samples_to_seconds(samples):
    return samples / AUDIO_RATE

def seconds_to_samples(seconds):
    return int(seconds * AUDIO_RATE)

def chunk_audio(audio_data, chunk_size=VAD_CHUNK_SIZE):
    """Split audio data into chunks of specified size."""
    return [audio_data[i:i + chunk_size] for i in range(0, len(audio_data), chunk_size)]

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

        # Define overlap (reduced to 25% of window size to minimize overlap)
        self.overlap = int(self.vad_window_size * 0.25)
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

        # Variable to track the last transcription for deduplication
        self.last_transcription = ""

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

        # Add processed segments tracking
        self.processed_segments = []
        self.processed_segments_lock = threading.Lock()

        # Initialize caption window
        self.caption_window = CaptionWindow()
        self.caption_window_visible = False
        self.captionButton.clicked.connect(self.toggle_caption_window)

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

        asr_model_id = "openai/whisper-large-v3-turbo"  # turbo model is the best model so far
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
            self.overlap = int(self.vad_window_size * 0.25)  # Reduced overlap to 25%
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


    def process_audio_frames(self):
        current_speech_buffer = []
        silence_buffer = []
        speech_start_time = None
        in_speech = False
        last_cleanup_time = time.time()
        cleanup_interval = 5  # Cleanup every 5 seconds
        
        while not self.should_exit:
            if not self.is_recording:
                time.sleep(0.1)
                continue

            try:
                audio_frame = self.audio_frames_queue.get(timeout=1)
                current_time = time.time()

                # Periodic cleanup of old data
                if current_time - last_cleanup_time > cleanup_interval:
                    with self.buffer_lock:
                        # Clear old data from audio buffer
                        if len(self.audio_buffer) > MAX_BUFFER_SIZE:
                            excess = len(self.audio_buffer) - MAX_BUFFER_SIZE
                            for _ in range(excess):
                                self.audio_buffer.popleft()
                            logging.debug(f"Cleaned up {excess} samples from audio buffer")

                    # Clear old processed segments
                    with self.processed_segments_lock:
                        cleanup_time = current_time - 30  # Keep last 30 seconds
                        original_count = len(self.processed_segments)
                        self.processed_segments = [
                            ps for ps in self.processed_segments 
                            if ps.end_time > cleanup_time
                        ]
                        removed_count = original_count - len(self.processed_segments)
                        if removed_count > 0:
                            logging.debug(f"Cleaned up {removed_count} old processed segments")

                    # Clear old items from queues if they're getting too full
                    while self.audio_queue.qsize() > 10:  # Keep last 10 items
                        try:
                            self.audio_queue.get_nowait()
                        except queue.Empty:
                            break

                    while self.audio_frames_queue.qsize() > 50:  # Keep last 50 frames
                        try:
                            self.audio_frames_queue.get_nowait()
                        except queue.Empty:
                            break

                    last_cleanup_time = current_time
                    logging.debug("Completed periodic cleanup of audio data")

                with self.buffer_lock:
                    if len(self.audio_buffer) >= VAD_CHUNK_SIZE:
                        # Process audio in VAD-compatible chunks
                        audio_data = list(islice(self.audio_buffer, 0, self.vad_window_size))
                        audio_chunks = chunk_audio(audio_data)
                        
                        # Track speech probability for the window
                        speech_probs = []
                        for chunk in audio_chunks:
                            if len(chunk) == VAD_CHUNK_SIZE:  # Only process complete chunks
                                chunk = np.array(chunk, dtype=np.float32)
                                audio_tensor = torch.from_numpy(chunk).unsqueeze(0).to('cpu')
                                speech_prob = self.vad_model(audio_tensor, AUDIO_RATE).item()
                                speech_probs.append(speech_prob)
                        
                        # Consider it speech if the average probability exceeds threshold
                        if speech_probs:
                            avg_speech_prob = sum(speech_probs) / len(speech_probs)
                            is_speech = avg_speech_prob > VAD_THRESHOLD
                            
                            if is_speech and not in_speech:
                                # Speech start detected
                                if len(silence_buffer) > 0:
                                    silence_buffer = []
                                if speech_start_time is None:
                                    speech_start_time = current_time
                                in_speech = True
                                current_speech_buffer.extend(audio_data)
                            
                            elif is_speech and in_speech:
                                # Continuing speech
                                current_speech_buffer.extend(audio_data)
                                
                                # Check if we've exceeded max sentence duration
                                if len(current_speech_buffer) >= seconds_to_samples(MAX_SENTENCE_DURATION):
                                    speech_segment = np.array(current_speech_buffer, dtype=np.float32)
                                    try:
                                        self.audio_queue.put_nowait((
                                            speech_segment,
                                            True,  # Mark as complete
                                            speech_start_time,
                                            current_time
                                        ))
                                    except queue.Full:
                                        logging.warning("Audio queue is full. Dropping speech segment.")
                                    
                                    # Reset buffers
                                    current_speech_buffer = []
                                    speech_start_time = current_time
                            
                            elif not is_speech and in_speech:
                                # Potential speech end - accumulate silence
                                silence_buffer.extend(audio_data)
                                
                                if samples_to_seconds(len(silence_buffer)) >= MIN_SILENCE_DURATION:
                                    # Confirmed sentence break
                                    if samples_to_seconds(len(current_speech_buffer)) >= MIN_SPEECH_DURATION:
                                        speech_segment = np.array(current_speech_buffer, dtype=np.float32)
                                        try:
                                            self.audio_queue.put_nowait((
                                                speech_segment,
                                                True,  # Mark as complete
                                                speech_start_time,
                                                current_time - samples_to_seconds(len(silence_buffer))
                                            ))
                                        except queue.Full:
                                            logging.warning("Audio queue is full. Dropping speech segment.")
                                    
                                    # Reset all buffers
                                    current_speech_buffer = []
                                    silence_buffer = []
                                    speech_start_time = None
                                    in_speech = False
                            
                            # Remove processed audio from buffer
                            if self.shift > 0:
                                remaining_samples = list(islice(self.audio_buffer, self.shift, len(self.audio_buffer)))
                                self.audio_buffer = deque(remaining_samples, maxlen=MAX_BUFFER_SIZE)

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in audio processing: {e}")
                continue

    def process_transcription(self):
        def find_largest_overlap(partial: str, final: str) -> int:
            """
            Find the length of the largest case-insensitive suffix of 'partial'
            that is a prefix of 'final'.
            """
            partial_lower = partial.lower()
            final_lower = final.lower()
            max_overlap = 0
            for i in range(min(len(partial), len(final)), 0, -1):
                if final_lower.startswith(partial_lower[-i:]):
                    max_overlap = i
                    break
            return max_overlap

        previous_partial = ""
        while not self.should_exit:
            if not self.is_recording:
                time.sleep(0.1)
                continue

            try:
                audio_to_process, is_complete, start_time, end_time = self.audio_queue.get(timeout=1)
                logging.debug("Retrieved speech segment from audio_queue.")

                with self.settings_lock:
                    current_language = self.language

                # Prepare ASR generation kwargs
                transcribe_kwargs = {
                    "task": "transcribe",
                    "max_new_tokens": 440,
                    "num_beams": 4,
                    "condition_on_prev_tokens": True,
                    "compression_ratio_threshold": 2.0,
                    "temperature": 0.0,
                    "logprob_threshold": -1.0,
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
                        generated_ids, skip_special_tokens=True
                    )[0].strip()

                    if not is_complete:
                        # For partial transcripts, store the raw transcription
                        # without ellipses for future overlap checking
                        previous_partial = transcription
                        display_transcription = transcription + " ..."
                    else:
                        # For complete transcripts, try to remove overlapping text
                        if previous_partial:
                            overlap_len = find_largest_overlap(previous_partial, transcription)
                            # Remove the overlapping portion from the final transcription
                            transcription = transcription[overlap_len:].strip()
                        display_transcription = transcription
                        previous_partial = ""

                    logging.info(f"Transcription ({'complete' if is_complete else 'partial'}): {display_transcription}")

                translation = ""
                if current_language not in ["en", "auto"] and is_complete:
                    # Only translate complete segments
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
                            translation_ids, skip_special_tokens=True
                        )[0].strip()
                        logging.info(f"Translation: {translation}")

                self.transcription_queue.put((display_transcription, translation))
                logging.debug("Placed transcription and translation into transcription_queue.")

            except queue.Empty:
                continue
            except Exception as e:
                error_message = f"[Error in transcription]: {e}"
                logging.error(f"Transcription processing error: {e}")
                self.transcription_queue.put((error_message, ""))


    def update_ui(self):
        while not self.transcription_queue.empty():
            transcription, translation = self.transcription_queue.get()
            timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
            if transcription:
                new_transcription = transcription.strip()
                if new_transcription and new_transcription != self.last_transcription:
                    # Update main window
                    self.transcriptionArea.append(f"[{timestamp}] {new_transcription}")
                    # Update caption window
                    self.caption_window.update_text(new_transcription)
                    self.last_transcription = new_transcription
                else:
                    logging.debug("Duplicate transcription detected. Skipping display.")

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

        # Close caption window
        self.caption_window.close()

        event.accept()
        logging.info("Shutdown sequence completed.")

    def toggle_caption_window(self):
        """Toggle the visibility of the caption window"""
        if self.caption_window_visible:
            self.caption_window.hide()
            self.captionButton.setText("Show Captions")
        else:
            self.caption_window.show()
            self.captionButton.setText("Hide Captions")
        self.caption_window_visible = not self.caption_window_visible



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
