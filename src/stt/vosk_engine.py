import json
import os
import numpy as np
from typing import Dict, Any, Optional, Union
import vosk

from src.stt.base import STTEngine

class VoskEngine(STTEngine):
    """Vosk STT Engine for offline recognition"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.recognizer = None
        self.model_path = config.get('model_path', 'models/vosk-model-en-us-0.22')
        self.language = config.get('language', 'en-us')
        self.sample_rate = config.get('sample_rate', 16000)

    def initialize(self) -> bool:
        """Initialize Vosk model"""
        try:
            if not os.path.exists(self.model_path):
                print(f"Vosk model not found at {self.model_path}")
                print("Please download a Vosk model from https://alphacephei.com/vosk/models")
                return False

            vosk.SetLogLevel(-1)  # Suppress Vosk logs
            print(f"Loading Vosk model from {self.model_path}...")

            self.model = vosk.Model(self.model_path)
            self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
            self.recognizer.SetWords(True)  # Enable word-level timestamps

            self.is_initialized = True
            print("Vosk model loaded successfully")
            return True

        except Exception as e:
            print(f"Failed to initialize Vosk: {e}")
            return False

    def transcribe(self, audio_data: Union[str, np.ndarray]) -> Dict[str, Any]:
        """Transcribe audio file or numpy array"""
        if not self.is_initialized:
            raise RuntimeError("VoskEngine not initialized")

        try:
            if isinstance(audio_data, str):
                # For file paths, we need to load and process the audio
                import wave
                with wave.open(audio_data, 'rb') as wf:
                    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != self.sample_rate:
                        raise ValueError(f"Audio file must be mono 16kHz 16-bit PCM")

                    audio_bytes = wf.readframes(wf.getnframes())
            else:
                # Convert numpy array to bytes
                if audio_data.dtype != np.int16:
                    # Convert from float to int16
                    if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                        audio_data = (audio_data * 32767).astype(np.int16)
                    else:
                        audio_data = audio_data.astype(np.int16)

                audio_bytes = audio_data.tobytes()

            # Reset recognizer for new transcription
            self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
            self.recognizer.SetWords(True)

            # Process audio in chunks
            chunk_size = 4096
            full_text = ""
            segments = []

            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i + chunk_size]
                if self.recognizer.AcceptWaveform(chunk):
                    result = json.loads(self.recognizer.Result())
                    if result.get('text'):
                        full_text += result['text'] + " "

                        # Add segments if available
                        if 'result' in result:
                            for word in result['result']:
                                segments.append({
                                    'start': word['start'],
                                    'end': word['end'],
                                    'text': word['word'],
                                    'confidence': word.get('conf', 0.5)
                                })

            # Get final result
            final_result = json.loads(self.recognizer.FinalResult())
            if final_result.get('text'):
                full_text += final_result['text']

            return {
                'text': full_text.strip(),
                'language': self.language,
                'segments': segments,
                'confidence': sum([s['confidence'] for s in segments]) / max(len(segments), 1) if segments else 0.5
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
            # Convert to appropriate format
            if audio_chunk.dtype != np.int16:
                if audio_chunk.dtype == np.float32 or audio_chunk.dtype == np.float64:
                    audio_chunk = (audio_chunk * 32767).astype(np.int16)
                else:
                    audio_chunk = audio_chunk.astype(np.int16)

            audio_bytes = audio_chunk.tobytes()

            if self.recognizer.AcceptWaveform(audio_bytes):
                result = json.loads(self.recognizer.Result())
                return result.get('text', '').strip() if result.get('text') else None
            else:
                # Get partial result
                partial = json.loads(self.recognizer.PartialResult())
                return partial.get('partial', '').strip() if partial.get('partial') else None

        except Exception as e:
            print(f"Vosk streaming error: {e}")
            return None

    def get_supported_languages(self) -> list[str]:
        """Get supported languages (depends on model)"""
        # This is model-dependent, common Vosk models include:
        return ['en-us', 'en-in', 'ru', 'fr', 'de', 'es', 'pt', 'zh', 'ja', 'ko']

    def set_language(self, language: str) -> bool:
        """Set language (requires appropriate model)"""
        if language in self.get_supported_languages():
            self.language = language
            return True
        return False

    def cleanup(self):
        """Clean up resources"""
        self.recognizer = None
        self.model = None
        self.is_initialized = False