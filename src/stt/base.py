from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import numpy as np

class STTEngine(ABC):
    """Base class for Speech-to-Text engines"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the STT engine"""
        pass

    @abstractmethod
    def transcribe(self, audio_data: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Transcribe audio data to text

        Args:
            audio_data: Either file path (str) or audio array (np.ndarray)

        Returns:
            Dict containing 'text', 'confidence', and optional 'segments'
        """
        pass

    @abstractmethod
    def transcribe_stream(self, audio_chunk: np.ndarray) -> Optional[str]:
        """
        Transcribe streaming audio chunk

        Args:
            audio_chunk: Audio data chunk

        Returns:
            Partial or complete transcription text, or None
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> list[str]:
        """Get list of supported languages"""
        pass

    @abstractmethod
    def set_language(self, language: str) -> bool:
        """Set the language for transcription"""
        pass

    def cleanup(self):
        """Clean up resources"""
        pass

    def get_info(self) -> Dict[str, Any]:
        """Get engine information"""
        return {
            'name': self.__class__.__name__,
            'initialized': self.is_initialized,
            'config': self.config
        }