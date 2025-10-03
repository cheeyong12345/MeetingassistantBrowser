"""
Configuration Validation using Pydantic Models.

This module provides Pydantic models for validating the application
configuration loaded from config.yaml. It ensures all settings are
valid before the application starts, providing clear error messages
for any misconfigurations.
"""

from typing import Optional, Literal
from pathlib import Path
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict
)

from src.exceptions import ConfigValidationError


class AppConfig(BaseModel):
    """Application configuration settings.

    Attributes:
        name: Application name
        version: Application version string
        debug: Enable debug mode
    """
    model_config = ConfigDict(extra='allow')

    name: str = Field(
        default="Meeting Assistant",
        description="Application name"
    )
    version: str = Field(
        default="1.0.0",
        pattern=r'^\d+\.\d+\.\d+$',
        description="Application version in semver format"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )


class ServerConfig(BaseModel):
    """Server configuration settings.

    Attributes:
        host: Server host address
        port: Server port number
        reload: Enable auto-reload in development
    """
    model_config = ConfigDict(extra='allow')

    host: str = Field(
        default="localhost",
        description="Server host address"
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Server port number (1-65535)"
    )
    reload: bool = Field(
        default=False,
        description="Enable auto-reload for development"
    )


class AudioConfig(BaseModel):
    """Audio recording configuration settings.

    Attributes:
        sample_rate: Audio sample rate in Hz
        channels: Number of audio channels (1=mono, 2=stereo)
        chunk_size: Audio buffer chunk size
        format: Audio file format
        input_device: Input device index (None for default)
    """
    model_config = ConfigDict(extra='allow')

    sample_rate: int = Field(
        default=16000,
        ge=8000,
        le=48000,
        description="Audio sample rate in Hz (8000-48000)"
    )
    channels: int = Field(
        default=1,
        ge=1,
        le=2,
        description="Number of audio channels (1=mono, 2=stereo)"
    )
    chunk_size: int = Field(
        default=1024,
        ge=256,
        le=8192,
        description="Audio buffer chunk size (256-8192)"
    )
    format: Literal['wav', 'mp3', 'flac', 'ogg'] = Field(
        default='wav',
        description="Audio file format"
    )
    input_device: Optional[int] = Field(
        default=None,
        description="Input device index (None for auto-detect)"
    )


class WhisperEngineConfig(BaseModel):
    """Whisper STT engine configuration.

    Attributes:
        model_size: Whisper model size
        language: Language code or 'auto' for detection
        device: Device to run model on
    """
    model_config = ConfigDict(extra='allow')

    model_size: Literal['tiny', 'base', 'small', 'medium', 'large'] = Field(
        default='medium',
        description="Whisper model size"
    )
    language: str = Field(
        default='auto',
        description="Language code or 'auto' for detection"
    )
    device: Literal['auto', 'cpu', 'cuda', 'mps'] = Field(
        default='auto',
        description="Device to run model on"
    )


class VoskEngineConfig(BaseModel):
    """Vosk STT engine configuration.

    Attributes:
        model_path: Path to Vosk model directory
        language: Language code
    """
    model_config = ConfigDict(extra='allow')

    model_path: str = Field(
        description="Path to Vosk model directory"
    )
    language: str = Field(
        default='en-us',
        description="Language code"
    )


class GoogleEngineConfig(BaseModel):
    """Google STT engine configuration.

    Attributes:
        api_key: Google Cloud API key
        language: Language code
    """
    model_config = ConfigDict(extra='allow')

    api_key: Optional[str] = Field(
        default=None,
        description="Google Cloud API key"
    )
    language: str = Field(
        default='en-US',
        description="Language code"
    )


class STTEnginesConfig(BaseModel):
    """Configuration for all STT engines.

    Attributes:
        whisper: Whisper engine configuration
        vosk: Vosk engine configuration
        google: Google engine configuration
    """
    model_config = ConfigDict(extra='allow')

    whisper: Optional[WhisperEngineConfig] = None
    vosk: Optional[VoskEngineConfig] = None
    google: Optional[GoogleEngineConfig] = None

    @model_validator(mode='after')
    def validate_at_least_one_engine(self) -> 'STTEnginesConfig':
        """Ensure at least one STT engine is configured."""
        if not any([self.whisper, self.vosk, self.google]):
            raise ConfigValidationError(
                "At least one STT engine must be configured",
                details={'section': 'stt.engines'}
            )
        return self


class STTConfig(BaseModel):
    """Speech-to-Text configuration settings.

    Attributes:
        default_engine: Default STT engine to use
        engines: Configuration for each STT engine
    """
    model_config = ConfigDict(extra='allow')

    default_engine: str = Field(
        default='whisper',
        description="Default STT engine"
    )
    engines: STTEnginesConfig = Field(
        description="STT engine configurations"
    )


class Qwen3EngineConfig(BaseModel):
    """Qwen3 summarization engine configuration.

    Attributes:
        model_name: Hugging Face model name
        device: Device to run model on
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    """
    model_config = ConfigDict(extra='allow')

    model_name: str = Field(
        default='Qwen/Qwen2.5-3B-Instruct',
        description="Hugging Face model name"
    )
    device: Literal['auto', 'cpu', 'cuda', 'mps'] = Field(
        default='auto',
        description="Device to run model on"
    )
    max_tokens: int = Field(
        default=1000,
        ge=100,
        le=4096,
        description="Maximum tokens to generate (100-4096)"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0-2.0)"
    )


class OllamaEngineConfig(BaseModel):
    """Ollama summarization engine configuration.

    Attributes:
        base_url: Ollama API base URL
        model_name: Ollama model name
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    """
    model_config = ConfigDict(extra='allow')

    base_url: str = Field(
        default='http://localhost:11434',
        description="Ollama API base URL"
    )
    model_name: str = Field(
        default='qwen2.5:1.5b',
        description="Ollama model name"
    )
    max_tokens: int = Field(
        default=1000,
        ge=100,
        le=4096,
        description="Maximum tokens to generate (100-4096)"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0-2.0)"
    )


class OpenAIEngineConfig(BaseModel):
    """OpenAI summarization engine configuration.

    Attributes:
        api_key: OpenAI API key
        model: OpenAI model name
        max_tokens: Maximum tokens to generate
    """
    model_config = ConfigDict(extra='allow')

    api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    model: str = Field(
        default='gpt-3.5-turbo',
        description="OpenAI model name"
    )
    max_tokens: int = Field(
        default=1000,
        ge=100,
        le=4096,
        description="Maximum tokens to generate (100-4096)"
    )


class SummarizationEnginesConfig(BaseModel):
    """Configuration for all summarization engines.

    Attributes:
        qwen3: Qwen3 engine configuration
        ollama: Ollama engine configuration
        openai: OpenAI engine configuration
    """
    model_config = ConfigDict(extra='allow')

    qwen3: Optional[Qwen3EngineConfig] = None
    ollama: Optional[OllamaEngineConfig] = None
    openai: Optional[OpenAIEngineConfig] = None

    @model_validator(mode='after')
    def validate_at_least_one_engine(self) -> 'SummarizationEnginesConfig':
        """Ensure at least one summarization engine is configured."""
        if not any([self.qwen3, self.ollama, self.openai]):
            raise ConfigValidationError(
                "At least one summarization engine must be configured",
                details={'section': 'summarization.engines'}
            )
        return self


class SummarizationConfig(BaseModel):
    """Summarization configuration settings.

    Attributes:
        default_engine: Default summarization engine to use
        engines: Configuration for each summarization engine
    """
    model_config = ConfigDict(extra='allow')

    default_engine: str = Field(
        default='qwen3',
        description="Default summarization engine"
    )
    engines: SummarizationEnginesConfig = Field(
        description="Summarization engine configurations"
    )


class StorageConfig(BaseModel):
    """Storage configuration settings.

    Attributes:
        data_dir: Root data directory
        meetings_dir: Meetings storage directory
        models_dir: Models storage directory
        database_url: Database connection URL
    """
    model_config = ConfigDict(extra='allow')

    data_dir: str = Field(
        default='./data',
        description="Root data directory"
    )
    meetings_dir: str = Field(
        default='./data/meetings',
        description="Meetings storage directory"
    )
    models_dir: str = Field(
        default='./models',
        description="Models storage directory"
    )
    database_url: str = Field(
        default='sqlite:///./data/meetings.db',
        description="Database connection URL"
    )

    @field_validator('data_dir', 'meetings_dir', 'models_dir')
    @classmethod
    def validate_directory_path(cls, v: str) -> str:
        """Validate that directory paths are reasonable."""
        if not v or v.strip() == '':
            raise ValueError("Directory path cannot be empty")
        return v


class ProcessingConfig(BaseModel):
    """Processing configuration settings.

    Attributes:
        real_time_stt: Enable real-time speech-to-text
        auto_summarize: Automatically summarize meetings
        speaker_detection: Enable speaker detection
        chunk_duration: Audio chunk duration in seconds
        max_meeting_duration: Maximum meeting duration in seconds
    """
    model_config = ConfigDict(extra='allow')

    real_time_stt: bool = Field(
        default=True,
        description="Enable real-time speech-to-text"
    )
    auto_summarize: bool = Field(
        default=True,
        description="Automatically summarize meetings"
    )
    speaker_detection: bool = Field(
        default=False,
        description="Enable speaker detection"
    )
    chunk_duration: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Audio chunk duration in seconds (5-300)"
    )
    max_meeting_duration: int = Field(
        default=14400,
        ge=60,
        le=86400,
        description="Maximum meeting duration in seconds (1min-24hrs)"
    )


class MeetingAssistantConfig(BaseModel):
    """Complete Meeting Assistant configuration.

    This is the root configuration model that validates the entire
    config.yaml file.

    Attributes:
        app: Application settings
        server: Server settings
        audio: Audio settings
        stt: Speech-to-text settings
        summarization: Summarization settings
        storage: Storage settings
        processing: Processing settings
    """
    model_config = ConfigDict(extra='allow')

    app: AppConfig = Field(
        default_factory=AppConfig,
        description="Application configuration"
    )
    server: ServerConfig = Field(
        default_factory=ServerConfig,
        description="Server configuration"
    )
    audio: AudioConfig = Field(
        default_factory=AudioConfig,
        description="Audio configuration"
    )
    stt: STTConfig = Field(
        description="Speech-to-text configuration"
    )
    summarization: SummarizationConfig = Field(
        description="Summarization configuration"
    )
    storage: StorageConfig = Field(
        default_factory=StorageConfig,
        description="Storage configuration"
    )
    processing: ProcessingConfig = Field(
        default_factory=ProcessingConfig,
        description="Processing configuration"
    )


def validate_config(config_dict: dict) -> MeetingAssistantConfig:
    """Validate configuration dictionary against Pydantic models.

    Args:
        config_dict: Configuration dictionary loaded from YAML

    Returns:
        Validated configuration model

    Raises:
        ConfigValidationError: If validation fails with details

    Example:
        >>> import yaml
        >>> with open('config.yaml') as f:
        ...     config_data = yaml.safe_load(f)
        >>> validated_config = validate_config(config_data)
    """
    try:
        return MeetingAssistantConfig(**config_dict)
    except Exception as e:
        raise ConfigValidationError(
            f"Configuration validation failed: {str(e)}",
            details={'error': str(e)}
        ) from e
