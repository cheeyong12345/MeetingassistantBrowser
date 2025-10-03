"""
Custom Exception Classes for Meeting Assistant.

This module defines application-specific exceptions to provide
better error handling and debugging capabilities. Each exception
includes context and can be chained from underlying errors.
"""

from typing import Optional, Any


class MeetingAssistantError(Exception):
    """Base exception class for all Meeting Assistant errors.

    All custom exceptions in the application should inherit from this class.
    This allows for catching all application-specific errors with a single
    except clause.

    Attributes:
        message: Human-readable error message
        details: Optional dictionary with additional error context
    """

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None
    ) -> None:
        """Initialize the exception.

        Args:
            message: Error message describing what went wrong
            details: Optional dictionary with additional context
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


# Audio-related exceptions

class AudioError(MeetingAssistantError):
    """Base exception for audio-related errors."""
    pass


class AudioRecordingError(AudioError):
    """Raised when audio recording fails.

    This can occur due to:
    - Missing or inaccessible audio input device
    - Permission issues accessing the microphone
    - Audio stream interruption
    - Buffer overflow/underflow
    """
    pass


class AudioDeviceError(AudioError):
    """Raised when there are issues with audio devices.

    This can occur due to:
    - No audio input devices available
    - Invalid device index specified
    - Device disconnected during operation
    """
    pass


class AudioSaveError(AudioError):
    """Raised when saving audio data fails.

    This can occur due to:
    - Insufficient disk space
    - Permission issues writing to file
    - Invalid audio format
    - File system errors
    """
    pass


# Model and Engine-related exceptions

class ModelError(MeetingAssistantError):
    """Base exception for model-related errors."""
    pass


class ModelLoadingError(ModelError):
    """Raised when loading an AI model fails.

    This can occur due to:
    - Model files not found
    - Insufficient memory
    - Incompatible model format
    - Corrupted model files
    - Missing dependencies
    """
    pass


class ModelNotFoundError(ModelError):
    """Raised when a requested model cannot be found.

    This can occur due to:
    - Model not downloaded
    - Incorrect model path specified
    - Model directory missing
    """
    pass


class EngineError(MeetingAssistantError):
    """Base exception for engine-related errors."""
    pass


class EngineInitializationError(EngineError):
    """Raised when an engine fails to initialize.

    This can occur due to:
    - Missing required libraries
    - Invalid configuration
    - Resource allocation failure
    - Hardware compatibility issues
    """
    pass


class EngineNotAvailableError(EngineError):
    """Raised when requested engine is not available.

    This can occur due to:
    - Engine not registered
    - Engine disabled in configuration
    - Missing dependencies for the engine
    """
    pass


# Speech-to-Text exceptions

class STTError(MeetingAssistantError):
    """Base exception for speech-to-text errors."""
    pass


class TranscriptionError(STTError):
    """Raised when audio transcription fails.

    This can occur due to:
    - Invalid audio format
    - Audio quality too poor
    - Language detection failure
    - Model inference error
    - Insufficient audio data
    """
    pass


class StreamTranscriptionError(STTError):
    """Raised when real-time stream transcription fails.

    This can occur due to:
    - Audio chunk too small
    - Buffer issues
    - Model not supporting streaming
    - Stream processing timeout
    """
    pass


# Summarization exceptions

class SummarizationError(MeetingAssistantError):
    """Base exception for summarization errors."""
    pass


class SummaryGenerationError(SummarizationError):
    """Raised when summary generation fails.

    This can occur due to:
    - Text too long or too short
    - Model inference error
    - Invalid prompt
    - Context length exceeded
    - Generation timeout
    """
    pass


class ActionItemExtractionError(SummarizationError):
    """Raised when extracting action items fails.

    This can occur due to:
    - No actionable items in text
    - Parsing error
    - Model output format invalid
    """
    pass


# Configuration exceptions

class ConfigurationError(MeetingAssistantError):
    """Base exception for configuration errors."""
    pass


class ConfigValidationError(ConfigurationError):
    """Raised when configuration validation fails.

    This can occur due to:
    - Missing required fields
    - Invalid field values
    - Type mismatches
    - Constraint violations
    """
    pass


class ConfigFileError(ConfigurationError):
    """Raised when there are issues with the config file.

    This can occur due to:
    - Config file not found
    - Invalid YAML syntax
    - File permission issues
    - Malformed configuration structure
    """
    pass


# Meeting-related exceptions

class MeetingError(MeetingAssistantError):
    """Base exception for meeting-related errors."""
    pass


class MeetingAlreadyActiveError(MeetingError):
    """Raised when trying to start a meeting while one is already active."""
    pass


class MeetingNotActiveError(MeetingError):
    """Raised when trying to stop a meeting when none is active."""
    pass


class MeetingSaveError(MeetingError):
    """Raised when saving meeting data fails.

    This can occur due to:
    - Disk space issues
    - Permission problems
    - Serialization errors
    - File system errors
    """
    pass


# Storage exceptions

class StorageError(MeetingAssistantError):
    """Base exception for storage-related errors."""
    pass


class FileNotFoundError(StorageError):
    """Raised when a required file cannot be found."""
    pass


class DirectoryCreationError(StorageError):
    """Raised when directory creation fails.

    This can occur due to:
    - Permission issues
    - Invalid path
    - File system full
    """
    pass


# Network/API exceptions

class NetworkError(MeetingAssistantError):
    """Base exception for network-related errors."""
    pass


class APIError(NetworkError):
    """Raised when external API calls fail.

    This can occur due to:
    - Network connectivity issues
    - API rate limiting
    - Invalid API credentials
    - API service unavailable
    """
    pass


class APIAuthenticationError(APIError):
    """Raised when API authentication fails."""
    pass


class APIRateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    pass


# Helper functions for exception handling

def create_error_details(**kwargs: Any) -> dict[str, Any]:
    """Create a dictionary of error details.

    Args:
        **kwargs: Key-value pairs of error context

    Returns:
        Dictionary with error details

    Example:
        >>> details = create_error_details(
        ...     engine='whisper',
        ...     file_path='/path/to/audio.wav',
        ...     error_code='LOAD_FAILED'
        ... )
    """
    return {k: v for k, v in kwargs.items() if v is not None}
