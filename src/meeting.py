"""
Main Meeting Assistant Module.

This module provides the core MeetingAssistant class that orchestrates
all components including audio recording, speech-to-text transcription,
and summarization.
"""

import time
import json
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

from src.config import config
from src.stt import STTManager
from src.summarization import SummarizationManager
from src.audio import AudioRecorder
from src.utils.logger import get_logger
from src.exceptions import (
    MeetingAlreadyActiveError,
    MeetingNotActiveError,
    MeetingSaveError,
    AudioRecordingError,
    TranscriptionError,
    SummarizationError
)

# Initialize logger
logger = get_logger(__name__)


class MeetingAssistant:
    """Main meeting assistant orchestrator.

    This class coordinates all meeting-related operations including:
    - Audio recording and real-time transcription
    - Meeting lifecycle management (start/stop)
    - Transcript generation and summarization
    - Engine management (STT and summarization)
    - Meeting data persistence

    Attributes:
        stt_manager: Speech-to-text manager instance
        summarization_manager: Summarization manager instance
        audio_recorder: Audio recorder instance
        current_meeting: Currently active meeting data (None if no meeting active)
        real_time_transcript: Accumulated real-time transcript text

    Example:
        >>> assistant = MeetingAssistant()
        >>> assistant.initialize()
        >>> result = assistant.start_meeting(title="Team Standup")
        >>> # ... meeting happens ...
        >>> result = assistant.stop_meeting()
        >>> print(result['summary'])
    """

    def __init__(self) -> None:
        """Initialize the Meeting Assistant with all required components."""
        logger.info("Initializing Meeting Assistant")

        self.stt_manager = STTManager(config.stt.to_dict())
        self.summarization_manager = SummarizationManager(
            config.summarization.to_dict()
        )
        self.audio_recorder = AudioRecorder(config.audio.to_dict())

        self.current_meeting: Optional[dict[str, Any]] = None
        self.real_time_transcript = ""

        logger.debug("Meeting Assistant components created")

    def initialize(self) -> bool:
        """Initialize all components and check system readiness.

        Returns:
            True if initialization successful (always returns True to allow
            web server to start even if audio is unavailable)

        Raises:
            None - catches all exceptions to allow partial initialization
        """
        logger.info("Starting component initialization")
        success = True
        audio_success = False

        # Initialize audio recorder
        try:
            audio_success = self.audio_recorder.initialize()
            if audio_success:
                # Set up real-time transcription callback only if audio works
                self.audio_recorder.set_chunk_callback(self._process_audio_chunk)
                logger.info("Audio recorder initialized successfully")
            else:
                logger.warning(
                    "Audio recorder failed to initialize - "
                    "recording features will be disabled"
                )
        except Exception as e:
            logger.error(
                f"Audio initialization error: {e} - "
                f"recording features will be disabled",
                exc_info=True
            )

        logger.info(
            f"Meeting Assistant initialized successfully (audio: {audio_success})"
        )
        return True  # Always return True for web server to start

    def get_available_stt_engines(self) -> list[str]:
        """Get list of available STT engine names.

        Returns:
            List of registered STT engine identifiers

        Example:
            >>> assistant.get_available_stt_engines()
            ['whisper-medium', 'vosk-en-us-0.22']
        """
        engines = self.stt_manager.get_available_engines()
        logger.debug(f"Available STT engines: {engines}")
        return engines

    def get_available_summarization_engines(self) -> list[str]:
        """Get list of available summarization engine names.

        Returns:
            List of registered summarization engine identifiers

        Example:
            >>> assistant.get_available_summarization_engines()
            ['qwen-Qwen2.5-3B-Instruct', 'ollama-qwen2.5:1.5b']
        """
        engines = self.summarization_manager.get_available_engines()
        logger.debug(f"Available summarization engines: {engines}")
        return engines

    def switch_stt_engine(self, engine_name: str) -> bool:
        """Switch to a different STT engine.

        Args:
            engine_name: Name of the engine to switch to

        Returns:
            True if switch was successful, False otherwise

        Example:
            >>> assistant.switch_stt_engine('whisper-base')
            True
        """
        logger.info(f"Switching STT engine to: {engine_name}")
        result = self.stt_manager.switch_engine(engine_name)

        if result:
            logger.info(f"Successfully switched to STT engine: {engine_name}")
        else:
            logger.warning(f"Failed to switch to STT engine: {engine_name}")

        return result

    def switch_summarization_engine(self, engine_name: str) -> bool:
        """Switch to a different summarization engine.

        Args:
            engine_name: Name of the engine to switch to

        Returns:
            True if switch was successful, False otherwise

        Example:
            >>> assistant.switch_summarization_engine('ollama-qwen2.5:1.5b')
            True
        """
        logger.info(f"Switching summarization engine to: {engine_name}")
        result = self.summarization_manager.switch_engine(engine_name)

        if result:
            logger.info(
                f"Successfully switched to summarization engine: {engine_name}"
            )
        else:
            logger.warning(
                f"Failed to switch to summarization engine: {engine_name}"
            )

        return result

    def get_engine_status(self) -> dict[str, Any]:
        """Get current status of all engines and audio devices.

        Returns:
            Dictionary containing status information for:
            - STT engine
            - Summarization engine
            - Available audio devices

        Example:
            >>> status = assistant.get_engine_status()
            >>> print(status['stt']['name'])
            'whisper-medium'
        """
        status = {
            'stt': self.stt_manager.get_current_engine_info(),
            'summarization': self.summarization_manager.get_current_engine_info(),
            'audio_devices': self.audio_recorder.list_input_devices()
        }
        logger.debug(f"Engine status: {status}")
        return status

    def start_meeting(
        self,
        title: Optional[str] = None,
        participants: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """Start a new meeting with audio recording.

        Args:
            title: Optional meeting title. If not provided, generates one
                   based on current timestamp
            participants: Optional list of participant names

        Returns:
            Dictionary with:
            - success (bool): Whether meeting started successfully
            - meeting_id (str): Unique meeting identifier
            - title (str): Meeting title
            - error (str): Error message if success is False

        Raises:
            MeetingAlreadyActiveError: If a meeting is already in progress

        Example:
            >>> result = assistant.start_meeting(
            ...     title="Team Standup",
            ...     participants=["Alice", "Bob"]
            ... )
            >>> if result['success']:
            ...     print(f"Meeting started: {result['meeting_id']}")
        """
        if self.current_meeting:
            error_msg = "Meeting already in progress"
            logger.error(error_msg)
            raise MeetingAlreadyActiveError(error_msg)

        timestamp = datetime.now()
        meeting_id = f"meeting_{int(timestamp.timestamp())}"
        meeting_title = title or f"Meeting {timestamp.strftime('%Y-%m-%d %H:%M')}"

        logger.info(
            f"Starting meeting: {meeting_title} (ID: {meeting_id})"
        )

        self.current_meeting = {
            'id': meeting_id,
            'title': meeting_title,
            'participants': participants or [],
            'start_time': timestamp.isoformat(),
            'transcript_segments': [],
            'real_time_transcript': ""
        }

        # Start audio recording
        try:
            recording_started = self.audio_recorder.start_recording()

            if recording_started:
                logger.info(f"Meeting '{meeting_title}' started successfully")
                return {
                    'success': True,
                    'meeting_id': meeting_id,
                    'title': meeting_title
                }
            else:
                error_msg = "Failed to start audio recording"
                logger.error(error_msg)
                self.current_meeting = None
                raise AudioRecordingError(
                    error_msg,
                    details={'meeting_id': meeting_id}
                )

        except Exception as e:
            logger.error(f"Error starting meeting: {e}", exc_info=True)
            self.current_meeting = None
            return {'success': False, 'error': str(e)}

    def stop_meeting(self) -> dict[str, Any]:
        """Stop the current meeting and generate summary.

        Returns:
            Dictionary with:
            - success (bool): Whether meeting stopped successfully
            - meeting_id (str): Meeting identifier
            - title (str): Meeting title
            - transcript (str): Full meeting transcript
            - audio_file (str): Path to audio recording
            - meeting_file (str): Path to saved meeting data
            - summary (dict): Meeting summary (if auto_summarize enabled)
            - error (str): Error message if success is False

        Raises:
            MeetingNotActiveError: If no meeting is currently active

        Example:
            >>> result = assistant.stop_meeting()
            >>> if result['success']:
            ...     print(f"Transcript: {result['transcript']}")
            ...     print(f"Summary: {result['summary']['summary']}")
        """
        if not self.current_meeting:
            error_msg = "No meeting in progress"
            logger.error(error_msg)
            raise MeetingNotActiveError(error_msg)

        meeting_id = self.current_meeting['id']
        logger.info(f"Stopping meeting: {meeting_id}")

        # Stop recording
        try:
            audio_file = self.audio_recorder.stop_recording()
        except Exception as e:
            logger.error(f"Error stopping audio recording: {e}", exc_info=True)
            audio_file = None

        # Finalize meeting data
        self.current_meeting['end_time'] = datetime.now().isoformat()
        self.current_meeting['audio_file'] = audio_file

        # Get full transcript if not using real-time
        if not config.processing.real_time_stt and audio_file:
            logger.info("Transcribing full audio file")
            try:
                transcript_result = self.stt_manager.transcribe(audio_file)
                full_transcript = transcript_result.get('text', '')
            except Exception as e:
                logger.error(f"Transcription failed: {e}", exc_info=True)
                raise TranscriptionError(
                    f"Failed to transcribe audio: {str(e)}",
                    details={'audio_file': audio_file}
                ) from e
        else:
            full_transcript = self.current_meeting['real_time_transcript']

        self.current_meeting['full_transcript'] = full_transcript
        logger.info(
            f"Transcript generated: {len(full_transcript)} characters"
        )

        # Generate summary if auto-summarize is enabled
        summary_result = None
        if config.processing.auto_summarize and full_transcript:
            logger.info("Generating meeting summary")
            try:
                summary_result = self.summarization_manager.generate_meeting_summary(
                    full_transcript,
                    self.current_meeting['participants']
                )
                logger.info("Summary generated successfully")
            except Exception as e:
                logger.error(f"Summarization failed: {e}", exc_info=True)
                # Don't raise - summarization is optional

        # Save meeting data
        try:
            meeting_file = self._save_meeting(self.current_meeting, summary_result)
        except Exception as e:
            logger.error(f"Failed to save meeting data: {e}", exc_info=True)
            raise MeetingSaveError(
                f"Failed to save meeting data: {str(e)}",
                details={'meeting_id': meeting_id}
            ) from e

        result = {
            'success': True,
            'meeting_id': self.current_meeting['id'],
            'title': self.current_meeting['title'],
            'transcript': full_transcript,
            'audio_file': audio_file,
            'meeting_file': meeting_file
        }

        if summary_result:
            result['summary'] = summary_result

        # Clear current meeting
        logger.info(f"Meeting stopped successfully: {meeting_id}")
        self.current_meeting = None
        self.real_time_transcript = ""

        return result

    def get_current_meeting_status(self) -> dict[str, Any]:
        """Get status information about the current meeting.

        Returns:
            Dictionary with:
            - active (bool): Whether a meeting is active
            - meeting_id (str): Meeting identifier (if active)
            - title (str): Meeting title (if active)
            - duration (int): Meeting duration in seconds (if active)
            - transcript_length (int): Current transcript length (if active)
            - participants (list[str]): Participant names (if active)

        Example:
            >>> status = assistant.get_current_meeting_status()
            >>> if status['active']:
            ...     print(f"Duration: {status['duration']}s")
        """
        if not self.current_meeting:
            return {'active': False}

        start_time = datetime.fromisoformat(self.current_meeting['start_time'])
        duration = time.time() - start_time.timestamp()

        status = {
            'active': True,
            'meeting_id': self.current_meeting['id'],
            'title': self.current_meeting['title'],
            'duration': int(duration),
            'transcript_length': len(self.current_meeting['real_time_transcript']),
            'participants': self.current_meeting['participants']
        }

        logger.debug(f"Meeting status: {status}")
        return status

    def transcribe_audio_file(self, audio_file: str) -> dict[str, Any]:
        """Transcribe a standalone audio file.

        Args:
            audio_file: Path to the audio file to transcribe

        Returns:
            Dictionary with transcription results including:
            - text (str): Transcribed text
            - confidence (float): Confidence score
            - engine (str): Engine used
            - error (str): Error message if failed

        Example:
            >>> result = assistant.transcribe_audio_file('/path/to/audio.wav')
            >>> print(result['text'])
        """
        logger.info(f"Transcribing audio file: {audio_file}")

        try:
            result = self.stt_manager.transcribe(audio_file)
            logger.info(
                f"Transcription completed: {len(result.get('text', ''))} characters"
            )
            return result
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return {
                'text': '',
                'error': str(e),
                'success': False
            }

    def summarize_text(self, text: str) -> dict[str, Any]:
        """Summarize provided text.

        Args:
            text: Text to summarize

        Returns:
            Dictionary with summarization results including:
            - summary (str): Generated summary
            - key_points (list[str]): Key points extracted
            - action_items (list[str]): Action items identified
            - error (str): Error message if failed

        Example:
            >>> result = assistant.summarize_text("Long meeting transcript...")
            >>> print(result['summary'])
        """
        logger.info(f"Summarizing text: {len(text)} characters")

        try:
            result = self.summarization_manager.generate_meeting_summary(text)
            logger.info("Summarization completed successfully")
            return result
        except Exception as e:
            logger.error(f"Summarization error: {e}", exc_info=True)
            return {
                'summary': '',
                'error': str(e),
                'success': False
            }

    def _process_audio_chunk(self, audio_chunk) -> None:
        """Process real-time audio chunk for streaming transcription.

        This is called as a callback from the audio recorder for each
        audio chunk when real-time STT is enabled.

        Args:
            audio_chunk: NumPy array containing audio samples

        Note:
            Errors are logged but not raised to avoid breaking the
            recording stream.
        """
        if not self.current_meeting or not config.processing.real_time_stt:
            return

        try:
            # Transcribe audio chunk
            partial_text = self.stt_manager.transcribe_stream(audio_chunk)

            if partial_text:
                # Update real-time transcript
                self.current_meeting['real_time_transcript'] += partial_text + " "

                # Store segment with timestamp
                segment = {
                    'timestamp': time.time(),
                    'text': partial_text
                }
                self.current_meeting['transcript_segments'].append(segment)

                logger.debug(f"Transcribed chunk: {partial_text[:50]}...")

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}", exc_info=True)

    def _save_meeting(
        self,
        meeting_data: dict[str, Any],
        summary_data: Optional[dict[str, Any]] = None
    ) -> str:
        """Save meeting data to file system.

        Args:
            meeting_data: Complete meeting data dictionary
            summary_data: Optional summary data to include

        Returns:
            Path to the saved meeting data file

        Raises:
            MeetingSaveError: If saving fails
        """
        try:
            # Create meetings directory
            meetings_dir = Path(config.storage.meetings_dir)
            meetings_dir.mkdir(parents=True, exist_ok=True)

            # Create meeting-specific directory
            meeting_dir = meetings_dir / meeting_data['id']
            meeting_dir.mkdir(exist_ok=True)

            # Prepare meeting data for saving
            save_data = meeting_data.copy()

            if summary_data:
                save_data['summary'] = summary_data

            # Save as JSON
            meeting_file = meeting_dir / "meeting_data.json"
            with open(meeting_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

            # Save transcript as text file
            transcript_file = meeting_dir / "transcript.txt"
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(save_data.get('full_transcript', ''))

            logger.info(f"Meeting data saved to: {meeting_dir}")
            return str(meeting_file)

        except Exception as e:
            logger.error(f"Failed to save meeting data: {e}", exc_info=True)
            raise MeetingSaveError(
                f"Failed to save meeting data: {str(e)}",
                details={'meeting_id': meeting_data.get('id')}
            ) from e

    def cleanup(self) -> None:
        """Clean up all resources and shut down components.

        This should be called before application shutdown to ensure
        all resources are properly released.

        Example:
            >>> assistant.cleanup()
        """
        logger.info("Cleaning up Meeting Assistant resources")

        try:
            self.stt_manager.cleanup()
            self.summarization_manager.cleanup()
            self.audio_recorder.cleanup()
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)

    def __del__(self) -> None:
        """Destructor to ensure cleanup on object deletion."""
        try:
            self.cleanup()
        except Exception:
            # Suppress errors during cleanup in destructor
            pass
