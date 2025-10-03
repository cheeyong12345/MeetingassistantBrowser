from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Dict, Any, List, Optional
import re
import logging

from src.summarization.base import SummarizationEngine
from src.utils.hardware import get_hardware_detector
from src.utils.npu_acceleration import get_npu_accelerator, is_npu_model_available
from src.utils.eswin_npu import ESWINNPUInterface

logger = logging.getLogger(__name__)

class QwenEngine(SummarizationEngine):
    """Qwen3 local summarization engine with NPU acceleration support"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_name = config.get('model_name', 'Qwen/Qwen2.5-3B-Instruct')
        self.device = config.get('device', 'auto')
        self.max_tokens = config.get('max_tokens', 1000)
        self.temperature = config.get('temperature', 0.7)
        self.use_npu = config.get('use_npu', True)
        self.hardware = get_hardware_detector()
        self.npu_accelerator = None
        self.eswin_npu = None  # ESWIN NPU binary interface
        self.using_npu = False
        self.using_eswin_npu = False  # Flag for ESWIN NPU binary

    def initialize(self) -> bool:
        """Initialize Qwen model with NPU acceleration if available"""
        try:
            # Priority 1: Try ESWIN NPU binary (best performance for Qwen2 7B)
            if self.use_npu and ESWINNPUInterface.is_available():
                logger.info("ESWIN NPU binary detected - using hardware-optimized Qwen2 7B")
                try:
                    self.eswin_npu = ESWINNPUInterface()
                    if self.eswin_npu.start():
                        self.using_eswin_npu = True
                        self.is_initialized = True
                        print("✅ Qwen2 7B running on ESWIN NPU (hardware binary)")
                        print("   Model: INT8 quantized, 1024 token context")
                        print("   Performance: ~20-50 tokens/second")
                        return True
                    else:
                        logger.warning("ESWIN NPU binary failed to start, trying alternatives...")
                        self.eswin_npu = None
                except Exception as e:
                    logger.warning(f"ESWIN NPU initialization failed: {e}, trying alternatives...")
                    self.eswin_npu = None

            # Priority 2: Auto-detect device for standard inference
            if self.device == 'auto':
                self.device = self.hardware.get_optimal_device()

            # Priority 3: Try ONNX Runtime with NPU acceleration
            if self.use_npu and self.hardware.supports_npu_acceleration():
                npu_info = self.hardware.get_npu_info()
                logger.info(f"NPU detected: {npu_info['description']}")

                # For large language models, ONNX Runtime with ENNP EP is preferred
                logger.info("Note: LLM NPU acceleration uses ONNX Runtime with ENNP Execution Provider")

                # Check if NPU-optimized model is available
                model_short_name = self.model_name.split('/')[-1].lower().replace('-', '_')
                model_path = f"./models/{npu_info['type']}/{model_short_name}"

                if is_npu_model_available(model_path, npu_info['type']):
                    logger.info("Loading NPU-optimized Qwen model...")
                    self.npu_accelerator = get_npu_accelerator(npu_info['type'])

                    if self.npu_accelerator:
                        npu_model_path = f"{model_path}.onnx"  # ONNX format for LLMs

                        if self.npu_accelerator.load_model(npu_model_path, use_ennp=True):
                            self.using_npu = True
                            logger.info("Qwen model loaded on NPU successfully")
                            print(f"✅ Qwen running on {npu_info['description']}")

                            # Still need tokenizer
                            self.tokenizer = AutoTokenizer.from_pretrained(
                                self.model_name,
                                trust_remote_code=True
                            )
                            self.is_initialized = True
                            return True
                        else:
                            logger.warning("Failed to load NPU model, falling back to PyTorch")
                else:
                    logger.info(f"NPU model not found at {model_path}")
                    logger.info(f"Note: Convert model using: python scripts/convert_models_npu.py --model qwen --size {self.model_name}")

            # Fallback to standard PyTorch model
            try:
                import torch

                logger.info(f"Loading Qwen model '{self.model_name}' on device '{self.device}'...")

                # Load tokenizer and model
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device != 'cpu' else torch.float32,
                    device_map=self.device if self.device != 'cpu' else None,
                    trust_remote_code=True
                )

                # Create text generation pipeline
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    torch_dtype=torch.float16 if self.device != 'cpu' else torch.float32,
                    device_map=self.device if self.device != 'cpu' else None
                )

                self.is_initialized = True

                # Display hardware info
                system_info = self.hardware.get_system_info()
                print(f"Qwen model loaded successfully on {system_info['architecture']} ({system_info['soc_type']})")

                return True

            except ImportError:
                logger.error("PyTorch not available and no NPU model found")
                logger.info("Either install PyTorch or use ONNX Runtime with converted model")
                return False

        except Exception as e:
            logger.error(f"Failed to initialize Qwen: {e}")
            return False

    def _generate_response(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate response using Qwen model"""
        if not self.is_initialized:
            raise RuntimeError("QwenEngine not initialized")

        try:
            messages = [{"role": "user", "content": prompt}]
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            outputs = self.pipeline(
                prompt_text,
                max_new_tokens=max_tokens or self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Extract the generated text
            response = outputs[0]['generated_text']
            # Remove the prompt from the response
            response = response[len(prompt_text):].strip()

            return response

        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

    def summarize(self, text: str, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Summarize meeting transcript with improved prompt"""
        prompt = f"""You are an expert meeting summarizer. Analyze the transcript and create a professional summary.

TRANSCRIPT:
{text}

INSTRUCTIONS:
1. Write a brief overview (2-3 sentences) summarizing the main purpose
2. Extract 3-5 key discussion points
3. List any action items or decisions mentioned
4. Keep the summary concise - aim for 30-40% of original length
5. Use clear, professional language
6. DO NOT copy sentences verbatim - paraphrase and condense

SUMMARY FORMAT:
**Overview:**
[Brief 2-3 sentence summary of the meeting]

**Key Points:**
• [Main point 1]
• [Main point 2]
• [Main point 3]

**Action Items:**
• [Action item 1 if any]

**Decisions:**
• [Decision 1 if any]

Now provide the summary:"""

        try:
            summary = self._generate_response(prompt, max_tokens or 800)

            # Validate summary is not just copying
            if summary and len(summary) < len(text) * 0.9:
                return {
                    'summary': summary,
                    'success': True
                }
            else:
                # Fallback to extractive summary
                return {
                    'summary': self._extractive_summary(text),
                    'success': True,
                    'fallback': True
                }
        except Exception as e:
            return {
                'summary': self._extractive_summary(text),
                'success': True,
                'fallback': True,
                'error': str(e)
            }

    def _extractive_summary(self, text: str) -> str:
        """Simple extractive summarization fallback"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        # Take first sentence, middle sentences, and last sentence
        if len(sentences) <= 3:
            return '. '.join(sentences) + '.'

        summary_sentences = [
            sentences[0],  # First sentence
            sentences[len(sentences)//2],  # Middle
            sentences[-1]  # Last sentence
        ]

        return "**Summary (Key Points):**\n\n" + '. '.join(summary_sentences) + '.'

    def extract_action_items(self, text: str) -> List[str]:
        """Extract action items from meeting text"""
        prompt = f"""Please extract all action items from the following meeting transcript.
List each action item as a separate bullet point. Include who is responsible if mentioned.

Meeting Transcript:
{text}

Action Items:"""

        try:
            response = self._generate_response(prompt, 500)

            # Parse action items from response
            action_items = []
            lines = response.split('\n')

            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or
                           re.match(r'^\d+\.', line)):
                    # Clean up the action item
                    action_item = re.sub(r'^[-•*\d\.]\s*', '', line)
                    if action_item:
                        action_items.append(action_item)

            return action_items[:10]  # Limit to 10 action items

        except Exception as e:
            print(f"Error extracting action items: {e}")
            return []

    def extract_key_points(self, text: str) -> List[str]:
        """Extract key points from meeting text"""
        prompt = f"""Please extract the key points and main topics discussed in the following meeting transcript.
List each key point as a separate bullet point.

Meeting Transcript:
{text}

Key Points:"""

        try:
            response = self._generate_response(prompt, 500)

            # Parse key points from response
            key_points = []
            lines = response.split('\n')

            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or
                           re.match(r'^\d+\.', line)):
                    # Clean up the key point
                    key_point = re.sub(r'^[-•*\d\.]\s*', '', line)
                    if key_point:
                        key_points.append(key_point)

            return key_points[:8]  # Limit to 8 key points

        except Exception as e:
            print(f"Error extracting key points: {e}")
            return []

    def cleanup(self):
        """Clean up model resources"""
        if self.npu_accelerator is not None:
            self.npu_accelerator.cleanup()
            self.npu_accelerator = None

        if self.pipeline is not None:
            del self.pipeline
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        self.pipeline = None
        self.model = None
        self.tokenizer = None
        self.using_npu = False
        self.is_initialized = False