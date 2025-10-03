"""
Audio Recorder Module.

This module provides real-time audio recording functionality with
support for streaming callbacks and WAV file export.
"""

# PyAudio is optional (not needed for RISC-V with whisper.cpp)
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    pyaudio = None
    PYAUDIO_AVAILABLE = False

import wave
import numpy as np
import threading
import time
from typing import Optional, Callable, Any
from pathlib import Path

from src.utils.logger import get_logger
from src.exceptions import (
    AudioRecordingError,
    AudioDeviceError,
    AudioSaveError
)

# Initialize logger
logger = get_logger(__name__)


class AudioRecorder:
    """Audio recording with real-time streaming support.

    This class provides audio recording functionality using PyAudio,
    with support for real-time processing callbacks and WAV file export.

    Attributes:
        config: Configuration dictionary for audio settings
        sample_rate: Audio sample rate in Hz
        channels: Number of audio channels (1=mono, 2=stereo)
        chunk_size: Audio buffer chunk size
        format: PyAudio format (paInt16)
        audio: PyAudio instance
        stream: Active audio stream (when recording)
        is_recording: Whether recording is currently active
        recording_thread: Background thread for recording
        audio_data: Accumulated audio data buffers
        chunk_callback: Optional callback for real-time processing

    Example:
        >>> recorder = AudioRecorder({'sample_rate': 16000, 'channels': 1})
        >>> recorder.initialize()
        >>> recorder.start_recording()
        >>> # ... recording happens ...
        >>> audio_file = recorder.stop_recording()
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the audio recorder with configuration.

        Args:
            config: Configuration dictionary containing audio settings
        """
        logger.debug("Initializing AudioRecorder")

        if not PYAUDIO_AVAILABLE:
            logger.warning("PyAudio not available - audio recording disabled")
            logger.info("For RISC-V: Use whisper.cpp which handles audio directly")

        self.config = config
        self.sample_rate = config.get('sample_rate', 16000)
        self.channels = config.get('channels', 1)
        self.chunk_size = config.get('chunk_size', 1024)
        self.format = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None

        self.audio = None
        self.stream = None
        self.is_recording = False
        self.recording_thread = None
        self.audio_data = []

        # Callback for real-time processing
        self.chunk_callback: Optional[Callable[[np.ndarray], None]] = None

    def initialize(self) -> bool:
        """Initialize PyAudio.

        Returns:
            True if initialization successful, False otherwise

        Raises:
            AudioDeviceError: If PyAudio initialization fails
        """
        if not PYAUDIO_AVAILABLE:
            logger.warning("PyAudio not available - skipping audio recorder initialization")
            return False

        try:
            self.audio = pyaudio.PyAudio()
            logger.info("Audio recorder initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize audio recorder: {e}", exc_info=True)
            return False

    def list_input_devices(self) -> list[dict[str, Any]]:
        """List available audio input devices.

        Returns:
            List of dictionaries containing device information:
            - index (int): Device index
            - name (str): Device name
            - sample_rate (int): Default sample rate

        Example:
            >>> devices = recorder.list_input_devices()
            >>> for device in devices:
            ...     print(f"{device['index']}: {device['name']}")
        """
        if not self.audio:
            return []

        devices = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': info['name'],
                    'sample_rate': int(info['defaultSampleRate'])
                })
        return devices

    def start_recording(self, output_file: Optional[str] = None) -> bool:
        """Start recording audio.

        Args:
            output_file: Optional output file path (unused, kept for compatibility)

        Returns:
            True if recording started successfully, False otherwise

        Raises:
            AudioRecordingError: If recording fails to start
        """
        if self.is_recording:
            logger.warning("Recording already in progress")
            return False

        logger.info("Starting audio recording")

        try:
            # Find default input device
            input_device = self.config.get('input_device')
            if input_device is None:
                input_device = self.audio.get_default_input_device_info()['index']

            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=input_device,
                frames_per_buffer=self.chunk_size
            )

            self.is_recording = True
            self.audio_data = []

            # Start recording thread
            self.recording_thread = threading.Thread(
                target=self._recording_loop,
                args=(output_file,)
            )
            self.recording_thread.start()

            logger.info("Audio recording started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start recording: {e}", exc_info=True)
            raise AudioRecordingError(
                f"Failed to start recording: {str(e)}",
                details={'sample_rate': self.sample_rate, 'channels': self.channels}
            ) from e

    def stop_recording(self) -> Optional[str]:
        """Stop recording and save audio file.

        Returns:
            Path to saved audio file, or None if no recording active

        Raises:
            AudioSaveError: If saving audio file fails
        """
        if not self.is_recording:
            logger.warning("Cannot stop recording - no recording active")
            return None

        logger.info("Stopping audio recording")

        self.is_recording = False

        # Wait for recording thread to finish
        if self.recording_thread:
            self.recording_thread.join()

        # Close stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        logger.info("Audio recording stopped")
        return self._save_audio_data()

    def _recording_loop(self, output_file: Optional[str]) -> None:
        """Main recording loop (runs in background thread).

        Args:
            output_file: Optional output file path (unused)
        """
        while self.is_recording:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16)

                # Store audio data
                self.audio_data.append(data)

                # Call chunk callback for real-time processing
                if self.chunk_callback:
                    # Convert to float32 for processing
                    float_chunk = audio_chunk.astype(np.float32) / 32768.0
                    self.chunk_callback(float_chunk)

            except Exception as e:
                logger.error(f"Error in recording loop: {e}", exc_info=True)
                break

    def _save_audio_data(self) -> Optional[str]:
        """Save recorded audio data to WAV file.

        Returns:
            Path to saved file, or None if no data to save

        Raises:
            AudioSaveError: If file save operation fails
        """
        if not self.audio_data:
            return None

        try:
            timestamp = int(time.time())
            output_file = f"recording_{timestamp}.wav"
            output_path = Path("data/recordings") / output_file

            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save as WAV file
            with wave.open(str(output_path), 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.audio_data))

            logger.info(f"Audio saved to: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to save audio: {e}", exc_info=True)
            raise AudioSaveError(
                f"Failed to save audio: {str(e)}",
                details={'output_path': str(output_path)}
            ) from e

    def set_chunk_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Set callback for real-time audio processing.

        Args:
            callback: Function to call with each audio chunk
        """
        self.chunk_callback = callback

    def get_audio_array(self) -> Optional[np.ndarray]:
        """Get recorded audio as numpy array.

        Returns:
            Normalized float32 audio array, or None if no data
        """
        if not self.audio_data:
            return None

        # Combine all chunks
        audio_bytes = b''.join(self.audio_data)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

        # Convert to float32 and normalize
        return audio_array.astype(np.float32) / 32768.0

    def cleanup(self) -> None:
        """Clean up audio resources.

        Stops any active recording and releases PyAudio resources.
        """
        logger.debug("Cleaning up AudioRecorder resources")
        if self.is_recording:
            self.stop_recording()

        if self.audio:
            self.audio.terminate()
            self.audio = None

    def __del__(self) -> None:
        """Destructor to ensure cleanup on object deletion."""
        try:
            self.cleanup()
        except Exception:
            # Suppress errors during cleanup in destructor
            pass