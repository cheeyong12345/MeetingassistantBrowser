"""
Summarization Manager Module.

This module provides the SummarizationManager class that manages multiple
summarization engines, allowing dynamic switching and providing a unified
interface for text summarization operations.
"""

from typing import Any, Optional

from src.summarization.base import SummarizationEngine
from src.utils.logger import get_logger
from src.exceptions import (
    EngineNotAvailableError,
    EngineInitializationError,
    SummaryGenerationError,
    ActionItemExtractionError
)

# Initialize logger
logger = get_logger(__name__)

# Don't import engines at module level - import when needed to avoid RISC-V errors


class SummarizationManager:
    """Manager for multiple summarization engines.

    This class handles registration, initialization, and switching between
    different summarization engines. It provides a unified interface for
    text summarization operations regardless of the underlying engine.

    Attributes:
        config: Configuration dictionary for summarization settings
        engines: Dictionary mapping engine names to engine instances
        current_engine: Currently active summarization engine
        current_engine_name: Name of the currently active engine

    Example:
        >>> config = {'default_engine': 'qwen3', 'engines': {...}}
        >>> manager = SummarizationManager(config)
        >>> result = manager.summarize('Long text to summarize...')
        >>> print(result['summary'])
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the Summarization Manager.

        Args:
            config: Configuration dictionary containing engine settings
        """
        logger.info("Initializing Summarization Manager")

        self.config = config
        self.engines: dict[str, SummarizationEngine] = {}
        self.current_engine: Optional[SummarizationEngine] = None
        self.current_engine_name: str = ""

        self._register_engines()

    def _register_engines(self) -> None:
        """Register all available summarization engines from configuration.

        This method instantiates and registers each configured summarization
        engine. Engines that fail to register are logged but don't prevent
        other engines from being registered.

        Raises:
            EngineInitializationError: If no engines could be registered
        """
        logger.info("Registering summarization engines")
        engines_config = self.config.get('engines', {})

        # Register Qwen engine (import only when needed to avoid RISC-V errors)
        if 'qwen3' in engines_config:
            try:
                from src.summarization.qwen_engine import QwenEngine
                qwen_config = engines_config['qwen3']
                model_name = qwen_config.get('model_name', 'Qwen/Qwen2.5-3B-Instruct')
                # Extract model version from the full name
                if '/' in model_name:
                    model_short = model_name.split('/')[-1]
                else:
                    model_short = model_name
                engine_name = f"qwen-{model_short}"
                self.engines[engine_name] = QwenEngine(qwen_config)
                logger.info(f"Registered summarization engine: {engine_name}")
            except ImportError as e:
                logger.warning(f"Qwen engine not available (transformers import failed): {e}")
            except Exception as e:
                logger.error(
                    f"Failed to register Qwen engine: {e}",
                    exc_info=True
                )

        # Register Ollama engine (import only when needed)
        if 'ollama' in engines_config:
            try:
                from src.summarization.ollama_engine import OllamaEngine
                ollama_config = engines_config['ollama']
                model_name = ollama_config.get('model_name', 'qwen2.5:1.5b')
                engine_name = f"ollama-{model_name}"
                self.engines[engine_name] = OllamaEngine(ollama_config)
                logger.info(f"Registered summarization engine: {engine_name}")
            except ImportError as e:
                logger.warning(f"Ollama engine not available: {e}")
            except Exception as e:
                logger.error(
                    f"Failed to register Ollama engine: {e}",
                    exc_info=True
                )

        # Verify at least one engine was registered
        if not self.engines:
            warning_msg = "No summarization engines could be registered - summarization features will be disabled"
            logger.warning(warning_msg)
            logger.info("App can still work for transcription without summarization")
            # Don't raise error - allow app to work without summarization
            return

        # Set default engine (need to find the actual registered name)
        default_base = self.config.get('default_engine', 'qwen3')
        default_engine = None
        for engine_name in self.engines.keys():
            if engine_name.startswith(default_base) or engine_name.startswith('qwen'):
                default_engine = engine_name
                break

        if default_engine:
            logger.info(f"Setting default summarization engine: {default_engine}")
            self.switch_engine(default_engine)
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
            ['qwen-Qwen2.5-3B-Instruct', 'ollama-qwen2.5:1.5b']
        """
        engines = list(self.engines.keys())
        logger.debug(f"Available summarization engines: {engines}")
        return engines

    def switch_engine(self, engine_name: str) -> bool:
        """Switch to a different summarization engine.

        Args:
            engine_name: Name of the engine to switch to

        Returns:
            True if switch was successful, False otherwise

        Raises:
            EngineNotAvailableError: If requested engine is not available
            EngineInitializationError: If engine initialization fails

        Example:
            >>> manager.switch_engine('ollama-qwen2.5:1.5b')
            True
        """
        logger.info(f"Switching to summarization engine: {engine_name}")

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
            logger.info(
                f"Successfully switched to summarization engine: {engine_name}"
            )
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

    def summarize(
        self,
        text: str,
        max_tokens: Optional[int] = None
    ) -> dict[str, Any]:
        """Summarize text using current engine.

        Args:
            text: Text to summarize
            max_tokens: Maximum tokens for summary (None uses engine default)

        Returns:
            Dictionary containing:
            - summary (str): Generated summary
            - success (bool): Whether summarization succeeded
            - engine (str): Name of engine used
            - error (str): Error message if failed

        Raises:
            EngineNotAvailableError: If no engine is active
            SummaryGenerationError: If summarization fails

        Example:
            >>> result = manager.summarize('Long meeting transcript...')
            >>> print(result['summary'])
        """
        if not self.current_engine:
            error_msg = "No summarization engine available"
            logger.error(error_msg)
            raise EngineNotAvailableError(error_msg)

        logger.info(
            f"Summarizing text with {self.current_engine_name}: "
            f"{len(text)} characters"
        )

        try:
            result = self.current_engine.summarize(text, max_tokens)
            result['engine'] = self.current_engine_name

            summary_length = len(result.get('summary', ''))
            logger.info(
                f"Summarization completed: {summary_length} characters"
            )

            return result

        except Exception as e:
            error_msg = "Summarization failed"
            logger.error(f"{error_msg}: {e}", exc_info=True)
            raise SummaryGenerationError(
                f"{error_msg}: {str(e)}",
                details={
                    'engine': self.current_engine_name,
                    'text_length': len(text)
                }
            ) from e

    def extract_action_items(self, text: str) -> list[str]:
        """Extract action items from text using current engine.

        Args:
            text: Text to extract action items from

        Returns:
            List of action items extracted from the text

        Raises:
            EngineNotAvailableError: If no engine is active
            ActionItemExtractionError: If extraction fails

        Example:
            >>> items = manager.extract_action_items(transcript)
            >>> for item in items:
            ...     print(f"- {item}")
        """
        if not self.current_engine:
            logger.warning("No engine available for action item extraction")
            return []

        logger.info(
            f"Extracting action items with {self.current_engine_name}"
        )

        try:
            action_items = self.current_engine.extract_action_items(text)
            logger.info(
                f"Extracted {len(action_items)} action items"
            )
            return action_items

        except Exception as e:
            logger.error(
                f"Action item extraction error: {e}",
                exc_info=True
            )
            raise ActionItemExtractionError(
                f"Action item extraction failed: {str(e)}",
                details={'engine': self.current_engine_name}
            ) from e

    def extract_key_points(self, text: str) -> list[str]:
        """Extract key points from text using current engine.

        Args:
            text: Text to extract key points from

        Returns:
            List of key points extracted from the text

        Example:
            >>> points = manager.extract_key_points(transcript)
            >>> for point in points:
            ...     print(f"- {point}")
        """
        if not self.current_engine:
            logger.warning("No engine available for key point extraction")
            return []

        logger.info(
            f"Extracting key points with {self.current_engine_name}"
        )

        try:
            key_points = self.current_engine.extract_key_points(text)
            logger.info(
                f"Extracted {len(key_points)} key points"
            )
            return key_points

        except Exception as e:
            logger.error(
                f"Key point extraction error: {e}",
                exc_info=True
            )
            return []

    def generate_meeting_summary(
        self,
        transcript: str,
        participants: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """Generate comprehensive meeting summary with all components.

        Args:
            transcript: Meeting transcript text
            participants: Optional list of participant names

        Returns:
            Dictionary containing:
            - summary (str): Overall meeting summary
            - key_points (list[str]): Key discussion points
            - action_items (list[str]): Action items identified
            - participants (list[str]): Participant list
            - engine (str): Engine used
            - success (bool): Whether generation succeeded
            - error (str): Error message if failed

        Example:
            >>> result = manager.generate_meeting_summary(
            ...     transcript,
            ...     participants=["Alice", "Bob"]
            ... )
            >>> print(result['summary'])
            >>> for item in result['action_items']:
            ...     print(f"TODO: {item}")
        """
        if not self.current_engine:
            error_msg = "No summarization engine available"
            logger.error(error_msg)
            return {
                'summary': '',
                'key_points': [],
                'action_items': [],
                'participants': participants or [],
                'engine': 'None',
                'success': False,
                'error': error_msg
            }

        logger.info(
            f"Generating meeting summary with {self.current_engine_name}: "
            f"{len(transcript)} characters, "
            f"{len(participants or [])} participants"
        )

        try:
            result = self.current_engine.generate_meeting_summary(
                transcript,
                participants
            )
            logger.info("Meeting summary generated successfully")
            return result

        except Exception as e:
            logger.error(
                f"Meeting summary generation error: {e}",
                exc_info=True
            )
            return {
                'summary': '',
                'key_points': [],
                'action_items': [],
                'participants': participants or [],
                'engine': self.current_engine_name,
                'success': False,
                'error': str(e)
            }

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
            'qwen-Qwen2.5-3B-Instruct'
        """
        if not self.current_engine:
            logger.debug("No current engine set")
            return {'name': 'None', 'initialized': False}

        info = self.current_engine.get_info()
        info['name'] = self.current_engine_name

        logger.debug(f"Current engine info: {info}")
        return info

    def cleanup(self) -> None:
        """Clean up all engine resources.

        This method should be called before application shutdown to ensure
        all engines are properly cleaned up and resources are released.

        Example:
            >>> manager.cleanup()
        """
        logger.info("Cleaning up Summarization Manager resources")

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

        logger.info("Summarization Manager cleanup completed")

    def __del__(self) -> None:
        """Destructor to ensure cleanup on object deletion."""
        try:
            self.cleanup()
        except Exception:
            # Suppress errors during cleanup in destructor
            pass
