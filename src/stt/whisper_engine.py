import whisper
import numpy as np
from typing import Dict, Any, Optional, Union
import warnings
import logging
warnings.filterwarnings("ignore")

from src.stt.base import STTEngine
from src.utils.hardware import get_hardware_detector
from src.utils.npu_acceleration import get_npu_accelerator, is_npu_model_available

logger = logging.getLogger(__name__)

class WhisperEngine(STTEngine):
    """OpenAI Whisper STT Engine with NPU acceleration support"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.model_size = config.get('model_size', 'base')
        self.language = config.get('language', 'auto')
        self.device = config.get('device', 'auto')
        self.use_npu = config.get('use_npu', True)
        self.hardware = get_hardware_detector()
        self.npu_accelerator = None
        self.using_npu = False

    def initialize(self) -> bool:
        """Initialize Whisper model with NPU acceleration if available"""
        try:
            # Auto-detect device
            if self.device == 'auto':
                self.device = self.hardware.get_optimal_device()

            # Try to initialize NPU acceleration first
            if self.use_npu and self.hardware.supports_npu_acceleration():
                npu_info = self.hardware.get_npu_info()
                logger.info(f"NPU detected: {npu_info['description']}")

                # Check if NPU-optimized model is available
                model_path = f"./models/{npu_info['type']}/whisper_{self.model_size}"
                if is_npu_model_available(model_path, npu_info['type']):
                    logger.info("Loading NPU-optimized Whisper model...")
                    self.npu_accelerator = get_npu_accelerator(npu_info['type'])

                    if self.npu_accelerator:
                        # Load NPU model
                        npu_model_ext = ".rknn" if npu_info['type'] == "rk3588" else ".onnx"
                        npu_model_path = f"{model_path}{npu_model_ext}"

                        if self.npu_accelerator.load_model(npu_model_path):
                            self.using_npu = True
                            logger.info("Whisper model loaded on NPU successfully")
                            print(f"✅ Whisper running on {npu_info['description']}")
                            self.is_initialized = True
                            return True
                        else:
                            logger.warning("Failed to load NPU model, falling back to CPU/GPU")
                else:
                    logger.info(f"NPU model not found at {model_path}")
                    logger.info(f"Convert model using: python scripts/convert_models_npu.py --model whisper --size {self.model_size}")

            # Fallback to standard PyTorch model
            try:
                import torch
                logger.info(f"Loading Whisper model '{self.model_size}' on device '{self.device}'...")
                self.model = whisper.load_model(self.model_size, device=self.device)
                self.is_initialized = True

                # Display hardware info
                system_info = self.hardware.get_system_info()
                print(f"Whisper model loaded successfully on {system_info['architecture']} ({system_info['soc_type']})")

                return True

            except ImportError:
                logger.error("PyTorch not available and no NPU model found")
                logger.info("Either install PyTorch or convert model for NPU")
                return False

        except Exception as e:
            logger.error(f"Failed to initialize Whisper: {e}")
            return False

    def transcribe(self, audio_data: Union[str, np.ndarray]) -> Dict[str, Any]:
        """Transcribe audio file or numpy array"""
        if not self.is_initialized:
            raise RuntimeError("WhisperEngine not initialized")

        try:
            # Handle different input types
            if isinstance(audio_data, str):
                # File path
                result = self.model.transcribe(
                    audio_data,
                    language=None if self.language == 'auto' else self.language,
                    verbose=False
                )
            else:
                # Numpy array - ensure it's float32 and normalized
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)

                # Normalize audio if needed
                if audio_data.max() > 1.0:
                    audio_data = audio_data / np.max(np.abs(audio_data))

                result = self.model.transcribe(
                    audio_data,
                    language=None if self.language == 'auto' else self.language,
                    verbose=False
                )

            # Format response
            segments = []
            if 'segments' in result:
                for segment in result['segments']:
                    segments.append({
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': segment['text'].strip(),
                        'confidence': segment.get('avg_logprob', 0.0)
                    })

            return {
                'text': result['text'].strip(),
                'language': result.get('language', 'unknown'),
                'segments': segments,
                'confidence': sum([s.get('avg_logprob', 0.0) for s in result.get('segments', [])]) / max(len(result.get('segments', [])), 1)
            }

        except Exception as e:
            return {
                'text': '',
                'error': str(e),
                'confidence': 0.0,
                'segments': []
            }

    def transcribe_stream(self, audio_chunk: np.ndarray) -> Optional[str]:
        """Transcribe streaming audio chunk"""
        if not self.is_initialized:
            return None

        try:
            # For streaming, we'll use a simplified approach
            # In production, you might want to use a more sophisticated streaming solution
            if len(audio_chunk) < 16000:  # Less than 1 second of audio
                return None

            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)

            if audio_chunk.max() > 1.0:
                audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))

            result = self.model.transcribe(audio_chunk, language=None if self.language == 'auto' else self.language, verbose=False)
            return result['text'].strip() if result['text'].strip() else None

        except Exception as e:
            print(f"Streaming transcription error: {e}")
            return None

    def get_supported_languages(self) -> list[str]:
        """Get supported languages"""
        return [
            'auto', 'af', 'am', 'ar', 'as', 'az', 'ba', 'be', 'bg', 'bn', 'bo', 'br', 'bs', 'ca',
            'cs', 'cy', 'da', 'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'gl',
            'gu', 'ha', 'haw', 'he', 'hi', 'hr', 'ht', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'jw',
            'ka', 'kk', 'km', 'kn', 'ko', 'la', 'lb', 'ln', 'lo', 'lt', 'lv', 'mg', 'mi', 'mk',
            'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'nn', 'no', 'oc', 'pa', 'pl', 'ps',
            'pt', 'ro', 'ru', 'sa', 'sd', 'si', 'sk', 'sl', 'sn', 'so', 'sq', 'sr', 'su', 'sv',
            'sw', 'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'uk', 'ur', 'uz', 'vi', 'yi',
            'yo', 'zh'
        ]

    def set_language(self, language: str) -> bool:
        """Set language for transcription"""
        if language in self.get_supported_languages():
            self.language = language
            return True
        return False

    def cleanup(self):
        """Clean up model resources"""
        if self.npu_accelerator is not None:
            self.npu_accelerator.cleanup()
            self.npu_accelerator = None

        if self.model is not None:
            del self.model
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        self.model = None
        self.using_npu = False
        self.is_initialized = False