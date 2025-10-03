from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class SummarizationEngine(ABC):
    """Base class for summarization engines"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the summarization engine"""
        pass

    @abstractmethod
    def summarize(self, text: str, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Summarize text

        Args:
            text: Input text to summarize
            max_tokens: Maximum tokens for summary

        Returns:
            Dict containing 'summary', 'key_points', and 'action_items'
        """
        pass

    @abstractmethod
    def extract_action_items(self, text: str) -> List[str]:
        """Extract action items from meeting text"""
        pass

    @abstractmethod
    def extract_key_points(self, text: str) -> List[str]:
        """Extract key points from meeting text"""
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

    def generate_meeting_summary(self, transcript: str, participants: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive meeting summary"""
        try:
            # Basic summarization
            summary_result = self.summarize(transcript)

            # Extract additional insights
            key_points = self.extract_key_points(transcript)
            action_items = self.extract_action_items(transcript)

            return {
                'summary': summary_result.get('summary', ''),
                'key_points': key_points,
                'action_items': action_items,
                'participants': participants or [],
                'engine': self.__class__.__name__,
                'success': True
            }
        except Exception as e:
            return {
                'summary': '',
                'key_points': [],
                'action_items': [],
                'participants': participants or [],
                'engine': self.__class__.__name__,
                'error': str(e),
                'success': False
            }