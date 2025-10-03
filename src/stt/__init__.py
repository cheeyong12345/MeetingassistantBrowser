from src.stt.manager import STTManager
from src.stt.base import STTEngine

# Whisper is optional (requires PyTorch which may not be available on RISC-V)
try:
    from src.stt.whisper_engine import WhisperEngine
    WHISPER_AVAILABLE = True
except ImportError:
    WhisperEngine = None
    WHISPER_AVAILABLE = False

# Whisper.cpp is optional (C++ implementation, no PyTorch needed - recommended for RISC-V)
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

__all__ = [
    'STTManager',
    'STTEngine',
    'WhisperEngine',
    'WhisperCppEngine',
    'VoskEngine',
    'WHISPER_AVAILABLE',
    'WHISPERCPP_AVAILABLE',
    'VOSK_AVAILABLE'
]