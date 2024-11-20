# Real-Time Speech Transcription and Translation Application

## Overview

This project is a **real-time speech transcription and translation application** developed using **Python**. It leverages advanced machine learning models and provides a user-friendly interface for capturing audio, transcribing speech, and translating it into English in real-time.

**Note**: This is currently a conceptual project and contains known bugs. It serves as a demonstration and requires additional development for production use.


## Key Features

- **Real-Time Audio Capture**: Utilizes the microphone to capture audio streams with minimal latency.
- **Advanced Speech Recognition**: Implements OpenAI's **Whisper Large-V3 Turbo** model for high-accuracy transcription.
- **Language Support**: Capable of transcribing over **90 languages**, with automatic language detection.
- **Real-Time Translation**: Translates transcribed text into English on-the-fly using the same model.
- **Voice Activity Detection (VAD)**: Integrates **Silero VAD** for efficient and accurate speech segment detection.
- **Customizable Settings**: Offers adjustable parameters for VAD and transcription to optimize performance.
- **Graphical User Interface**: Built with **PyQt5**, providing an intuitive and responsive user experience.
- **Multithreaded Processing**: Employs threading for concurrent audio capture, processing, and UI updates without blocking.

## Technologies and Libraries Used

- **Python 3.7+**
- **PyQt5**: For the graphical user interface.
- **PyTorch**: As the deep learning framework.
- **Hugging Face Transformers**: To access and run the Whisper model.
- **SoundDevice**: For real-time audio capture.
- **NumPy**: For efficient numerical computations.
- **Silero VAD**: For voice activity detection.
- **Multithreading**: To handle real-time processing efficiently.

## Architecture and Design

- **Modular Codebase**: The project is divided into clear modules for audio capture, processing, transcription, and UI.
- **Thread-Safe Operations**: Utilizes locks and thread-safe queues to manage data between threads safely.
- **Asynchronous Processing**: Ensures that audio capture and transcription do not block the main thread, providing a smooth user experience.
- **Optimized for Performance**:
  - **GPU Acceleration**: Automatically detects and utilizes CUDA-compatible GPUs for running the Whisper model.
  - **Efficient Memory Management**: Implements deque and queues with maximum sizes to prevent memory leaks.

## Installation and Setup

### Prerequisites

- **Python 3.7** or higher
- **Pip** package manager
- **Virtual Environment** (recommended)
- **CUDA-compatible GPU** (optional for performance enhancement)
- **Microphone** connected to your computer

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/real-time-transcription.git
   cd real-time-transcription
   ```

2. **Set Up Virtual Environment**

   ```bash
   python -m venv venv
   # Activate the virtual environment:
   # On Windows:
   venv\Scripts\activate
   # On Unix or MacOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the Application**

   ```bash
   python main.py
   ```

## Usage

### User Interface Overview

- **Start Button**: Begins capturing audio and starts the transcription and translation process.
- **Stop Button**: Stops the audio capture and processing.
- **Language Selection Dropdown**: Allows the user to select the source language or opt for automatic detection.
- **VAD Settings**:
  - **VAD Window Size**: Adjusts the duration of audio analyzed for speech detection.
  - **VAD Hop Size**: Sets the step size between consecutive VAD windows.
  - **No Speech Threshold**: Configures the sensitivity for detecting the end of speech.
- **Transcription Area**: Displays the real-time transcribed text.
- **Translation Area**: Shows the translated text in English.

### Keyboard Shortcuts

- **Ctrl+S**: Start transcription and translation.
- **Ctrl+Q**: Stop the process.

## Challenges Overcome

- **Real-Time Processing**: Achieved low-latency audio capture and processing by optimizing buffer management and using efficient data structures like `deque`.
- **Thread Synchronization**: Ensured thread-safe operations with locks and thread-safe queues to prevent race conditions.
- **Model Integration**: Integrated the large Whisper model efficiently, handling device management (CPU/GPU) and data types for optimal performance.
- **VAD Accuracy**: Fine-tuned the VAD parameters and implemented post-processing steps to improve speech segment detection and reduce false positives.

## Potential Improvements

- **Dockerization**: Containerizing the application for easier deployment.
- **Cross-Platform Support**: Ensuring compatibility across different operating systems.
- **Enhanced UI/UX**: Adding features like dark mode, custom themes, and more interactive controls.
- **Additional Language Translation**: Extending translation capabilities beyond English.

## Skills Demonstrated

- **Machine Learning**: Practical experience with state-of-the-art NLP models.
- **Software Development**: Proficient in Python programming and software architecture design.
- **Multithreading and Concurrency**: Managed complex threading scenarios to maintain application responsiveness.
- **User Interface Design**: Developed a functional and user-friendly GUI using PyQt5.
- **Performance Optimization**: Optimized resource usage for real-time processing requirements.
- **Problem-Solving**: Addressed and resolved challenges related to audio processing and model integration.


*Published as part of a portfolio to showcase expertise in machine learning, real-time systems, and software development.*