"""
Speech-to-Text Manager Module.

This module provides the STTManager class that manages multiple STT engines,
allowing dynamic switching between different transcription engines and
providing a unified interface for speech-to-text operations.
"""

from typing import Any, Optional, Union
import numpy as np

from src.stt.base import STTEngine
from src.utils.logger import get_logger
from src.exceptions import (
    EngineNotAvailableError,
    EngineInitializationError,
    TranscriptionError,
    StreamTranscriptionError
)

# Whisper is optional (requires PyTorch which may not be available on RISC-V)
try:
    from src.stt.whisper_engine import WhisperEngine
    WHISPER_AVAILABLE = True
except ImportError:
    WhisperEngine = None
    WHISPER_AVAILABLE = False

# Whisper.cpp is optional (C++ implementation, no PyTorch - recommended for RISC-V)
try:
    from src.stt.whispercpp_engine import WhisperCppEngine
    WHISPERCPP_AVAILABLE = True
except ImportError:
    WhisperCppEngine = None
    WHISPERCPP_AVAILABLE = False

# Vosk is optional (may not be installed)
try:
    from src.stt.vosk_engine import VoskEngine
    VOSK_AVAILABLE = True
except ImportError:
    VoskEngine = None
    VOSK_AVAILABLE = False

# Initialize logger
logger = get_logger(__name__)


class STTManager:
    """Manager for multiple STT (Speech-to-Text) engines.

    This class handles registration, initialization, and switching between
    different STT engines. It provides a unified interface for transcription
    operations regardless of the underlying engine.

    Attributes:
        config: Configuration dictionary for STT settings
        engines: Dictionary mapping engine names to engine instances
        current_engine: Currently active STT engine
        current_engine_name: Name of the currently active engine

    Example:
        >>> config = {'default_engine': 'whisper', 'engines': {...}}
        >>> manager = STTManager(config)
        >>> result = manager.transcribe('audio.wav')
        >>> print(result['text'])
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the STT Manager.

        Args:
            config: Configuration dictionary containing engine settings
        """
        logger.info("Initializing STT Manager")

        self.config = config
        self.engines: dict[str, STTEngine] = {}
        self.current_engine: Optional[STTEngine] = None
        self.current_engine_name: str = ""

        self._register_engines()

    def _register_engines(self) -> None:
        """Register all available STT engines from configuration.

        This method instantiates and registers each configured STT engine.
        Engines that fail to register are logged but don't prevent other
        engines from being registered.

        Raises:
            EngineInitializationError: If no engines could be registered
        """
        logger.info("Registering STT engines")
        engines_config = self.config.get('engines', {})

        # Register Whisper engine (if available)
        if 'whisper' in engines_config and WHISPER_AVAILABLE:
            try:
                whisper_config = engines_config['whisper']
                model_size = whisper_config.get('model_size', 'medium')
                engine_name = f"whisper-{model_size}"
                self.engines[engine_name] = WhisperEngine(whisper_config)
                logger.info(f"Registered STT engine: {engine_name}")
            except Exception as e:
                logger.error(
                    f"Failed to register Whisper engine: {e}",
                    exc_info=True
                )
        elif 'whisper' in engines_config and not WHISPER_AVAILABLE:
            logger.warning("Whisper engine requested but not available (PyTorch not installed)")

        # Register Whisper.cpp engine (if available) - recommended for RISC-V
        if 'whispercpp' in engines_config and WHISPERCPP_AVAILABLE:
            try:
                whispercpp_config = engines_config['whispercpp']
                model_size = whispercpp_config.get('model_size', 'base')
                engine_name = f"whispercpp-{model_size}"
                self.engines[engine_name] = WhisperCppEngine(whispercpp_config)
                logger.info(f"Registered STT engine: {engine_name} (no PyTorch needed!)")
            except Exception as e:
                logger.error(
                    f"Failed to register Whisper.cpp engine: {e}",
                    exc_info=True
                )
        elif 'whispercpp' in engines_config and not WHISPERCPP_AVAILABLE:
            logger.warning("Whisper.cpp engine requested but not available")

        # Register Vosk engine (if available)
        if 'vosk' in engines_config and VOSK_AVAILABLE:
            try:
                vosk_config = engines_config['vosk']
                model_path = vosk_config.get('model_path', 'vosk-model')
                # Extract model name from path
                model_name = (
                    model_path.split('/')[-1]
                    if '/' in model_path
                    else model_path
                )
                engine_name = f"vosk-{model_name}"
                self.engines[engine_name] = VoskEngine(vosk_config)
                logger.info(f"Registered STT engine: {engine_name}")
            except Exception as e:
                logger.error(
                    f"Failed to register Vosk engine: {e}",
                    exc_info=True
                )
        elif 'vosk' in engines_config and not VOSK_AVAILABLE:
            logger.warning("Vosk engine requested but not available (vosk not installed)")

        # Verify at least one engine was registered
        if not self.engines:
            warning_msg = "No STT engines could be registered - STT features will be disabled"
            logger.warning(warning_msg)
            logger.info("App can still work for text summarization without STT")
            # Don't raise error - allow app to work without STT
            return

        # Set default engine (need to find the actual registered name)
        default_base = self.config.get('default_engine', 'whisper')
        default_engine = None
        for engine_name in self.engines.keys():
            if engine_name.startswith(default_base):
                default_engine = engine_name
                break

        if default_engine:
            logger.info(f"Setting default STT engine: {default_engine}")
            self.switch_engine(default_engine)
        else:
            # Use first available engine as default
            if self.engines:
                first_engine = list(self.engines.keys())[0]
                logger.info(f"Default engine '{default_base}' not found, using: {first_engine}")
                self.switch_engine(first_engine)
            else:
                logger.warning(
                    f"Default engine '{default_base}' not found, "
                    f"no engine activated"
                )

    def get_available_engines(self) -> list[str]:
        """Get list of available engine names.

        Returns:
            List of registered engine identifiers

        Example:
            >>> manager.get_available_engines()
            ['whisper-medium', 'vosk-en-us-0.22']
        """
        engines = list(self.engines.keys())
        logger.debug(f"Available STT engines: {engines}")
        return engines

    def switch_engine(self, engine_name: str) -> bool:
        """Switch to a different STT engine.

        Args:
            engine_name: Name of the engine to switch to

        Returns:
            True if switch was successful, False otherwise

        Raises:
            EngineNotAvailableError: If requested engine is not available
            EngineInitializationError: If engine initialization fails

        Example:
            >>> manager.switch_engine('whisper-base')
            True
        """
        logger.info(f"Switching to STT engine: {engine_name}")

        if engine_name not in self.engines:
            error_msg = f"Engine '{engine_name}' not available"
            logger.error(
                f"{error_msg}. Available engines: {list(self.engines.keys())}"
            )
            raise EngineNotAvailableError(
                error_msg,
                details={
                    'requested_engine': engine_name,
                    'available_engines': list(self.engines.keys())
                }
            )

        try:
            # Cleanup current engine
            if self.current_engine:
                logger.debug(
                    f"Cleaning up current engine: {self.current_engine_name}"
                )
                self.current_engine.cleanup()

            # Initialize new engine
            engine = self.engines[engine_name]
            if not engine.is_initialized:
                logger.info(f"Initializing engine: {engine_name}")
                if not engine.initialize():
                    error_msg = f"Failed to initialize engine '{engine_name}'"
                    logger.error(error_msg)
                    raise EngineInitializationError(
                        error_msg,
                        details={'engine': engine_name}
                    )

            self.current_engine = engine
            self.current_engine_name = engine_name
            logger.info(f"Successfully switched to STT engine: {engine_name}")
            return True

        except EngineInitializationError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            error_msg = f"Error switching to engine '{engine_name}'"
            logger.error(f"{error_msg}: {e}", exc_info=True)
            raise EngineInitializationError(
                error_msg,
                details={'engine': engine_name, 'error': str(e)}
            ) from e

    def transcribe(
        self,
        audio_data: Union[str, np.ndarray]
    ) -> dict[str, Any]:
        """Transcribe audio using current engine.

        Args:
            audio_data: Either a file path (str) or audio array (np.ndarray)

        Returns:
            Dictionary containing:
            - text (str): Transcribed text
            - confidence (float): Confidence score (if available)
            - engine (str): Name of engine used
            - error (str): Error message if transcription failed

        Raises:
            EngineNotAvailableError: If no engine is active
            TranscriptionError: If transcription fails

        Example:
            >>> result = manager.transcribe('audio.wav')
            >>> print(result['text'])
            'Hello world'
        """
        if not self.current_engine:
            error_msg = "No STT engine available"
            logger.error(error_msg)
            raise EngineNotAvailableError(error_msg)

        audio_source = (
            audio_data if isinstance(audio_data, str)
            else f"audio array ({len(audio_data)} samples)"
        )
        logger.info(
            f"Transcribing audio with {self.current_engine_name}: {audio_source}"
        )

        try:
            result = self.current_engine.transcribe(audio_data)
            result['engine'] = self.current_engine_name

            text_length = len(result.get('text', ''))
            logger.info(
                f"Transcription completed: {text_length} characters, "
                f"confidence: {result.get('confidence', 'N/A')}"
            )

            return result

        except Exception as e:
            error_msg = "Transcription failed"
            logger.error(f"{error_msg}: {e}", exc_info=True)
            raise TranscriptionError(
                f"{error_msg}: {str(e)}",
                details={
                    'engine': self.current_engine_name,
                    'audio_source': str(audio_source)[:100]
                }
            ) from e

    def transcribe_stream(self, audio_chunk: np.ndarray) -> Optional[str]:
        """Transcribe streaming audio chunk using current engine.

        Args:
            audio_chunk: NumPy array containing audio samples

        Returns:
            Transcribed text from the chunk, or None if no text detected

        Raises:
            EngineNotAvailableError: If no engine is active
            StreamTranscriptionError: If streaming transcription fails

        Example:
            >>> text = manager.transcribe_stream(audio_chunk)
            >>> if text:
            ...     print(f"Partial: {text}")
        """
        if not self.current_engine:
            logger.warning("No STT engine available for stream transcription")
            return None

        try:
            result = self.current_engine.transcribe_stream(audio_chunk)

            # Handle different return types (string or dict)
            if isinstance(result, dict):
                # Engine returned a dictionary (like whispercpp_engine)
                text = result.get('text', '')
                if text:
                    logger.debug(
                        f"Stream transcription result: {text[:50]}..."
                    )
                return text if text else None
            elif isinstance(result, str):
                # Engine returned a string directly
                if result:
                    logger.debug(
                        f"Stream transcription result: {result[:50]}..."
                    )
                return result
            else:
                # No result or None
                return None

        except Exception as e:
            logger.error(
                f"Streaming transcription error: {e}",
                exc_info=True
            )
            raise StreamTranscriptionError(
                f"Streaming transcription failed: {str(e)}",
                details={'engine': self.current_engine_name}
            ) from e

    def get_current_engine_info(self) -> dict[str, Any]:
        """Get information about current engine.

        Returns:
            Dictionary containing:
            - name (str): Engine name
            - initialized (bool): Whether engine is initialized
            - Additional engine-specific information

        Example:
            >>> info = manager.get_current_engine_info()
            >>> print(info['name'])
            'whisper-medium'
        """
        if not self.current_engine:
            logger.debug("No current engine set")
            return {'name': 'None', 'initialized': False}

        info = self.current_engine.get_info()
        info['name'] = self.current_engine_name

        logger.debug(f"Current engine info: {info}")
        return info

    def get_supported_languages(self) -> list[str]:
        """Get supported languages for current engine.

        Returns:
            List of supported language codes

        Example:
            >>> langs = manager.get_supported_languages()
            >>> print('en' in langs)
            True
        """
        if not self.current_engine:
            logger.warning("No engine available to get supported languages")
            return []

        languages = self.current_engine.get_supported_languages()
        logger.debug(
            f"Supported languages for {self.current_engine_name}: "
            f"{len(languages)} languages"
        )
        return languages

    def set_language(self, language: str) -> bool:
        """Set language for current engine.

        Args:
            language: Language code (e.g., 'en', 'es', 'fr')

        Returns:
            True if language was set successfully, False otherwise

        Example:
            >>> manager.set_language('es')
            True
        """
        if not self.current_engine:
            logger.error("No engine available to set language")
            return False

        logger.info(
            f"Setting language to '{language}' for {self.current_engine_name}"
        )

        result = self.current_engine.set_language(language)

        if result:
            logger.info(
                f"Language set to '{language}' for {self.current_engine_name}"
            )
        else:
            logger.warning(
                f"Failed to set language '{language}' "
                f"for {self.current_engine_name}"
            )

        return result

    def cleanup(self) -> None:
        """Clean up all engine resources.

        This method should be called before application shutdown to ensure
        all engines are properly cleaned up and resources are released.

        Example:
            >>> manager.cleanup()
        """
        logger.info("Cleaning up STT Manager resources")

        for engine_name, engine in self.engines.items():
            try:
                logger.debug(f"Cleaning up engine: {engine_name}")
                engine.cleanup()
            except Exception as e:
                logger.error(
                    f"Error cleaning up engine {engine_name}: {e}",
                    exc_info=True
                )

        self.current_engine = None
        self.current_engine_name = ""

        logger.info("STT Manager cleanup completed")

    def __del__(self) -> None:
        """Destructor to ensure cleanup on object deletion."""
        try:
            self.cleanup()
        except Exception:
            # Suppress errors during cleanup in destructor
            pass
