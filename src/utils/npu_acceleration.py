"""
NPU Acceleration Module
Provides NPU acceleration support for RK3588 and EIC7700 platforms
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union
import numpy as np

logger = logging.getLogger(__name__)


class NPUAccelerator:
    """Base class for NPU acceleration"""

    def __init__(self, npu_type: str):
        self.npu_type = npu_type
        self.is_available = False
        self.runtime = None

    def initialize(self) -> bool:
        """Initialize NPU runtime"""
        raise NotImplementedError

    def load_model(self, model_path: str) -> bool:
        """Load model for NPU inference"""
        raise NotImplementedError

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on NPU"""
        raise NotImplementedError

    def cleanup(self):
        """Clean up NPU resources"""
        pass


class RK3588NPU(NPUAccelerator):
    """RK3588 NPU acceleration using RKNN toolkit"""

    def __init__(self):
        super().__init__("rk3588")
        self.rknn_model = None

    def initialize(self) -> bool:
        """Initialize RKNN runtime"""
        try:
            # Try to import RKNN toolkit
            from rknnlite.api import RKNNLite
            self.runtime = RKNNLite()
            self.is_available = True
            logger.info("RK3588 RKNN runtime initialized successfully")
            return True
        except ImportError:
            logger.warning("RKNN toolkit not available. Install with: pip install rknn-toolkit-lite2")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize RKNN runtime: {e}")
            return False

    def load_model(self, model_path: str) -> bool:
        """Load RKNN model"""
        if not self.is_available:
            logger.warning("RKNN runtime not available")
            return False

        try:
            # Load RKNN model
            ret = self.runtime.load_rknn(model_path)
            if ret != 0:
                logger.error(f"Failed to load RKNN model: {model_path}")
                return False

            # Initialize runtime
            ret = self.runtime.init_runtime()
            if ret != 0:
                logger.error("Failed to initialize RKNN runtime")
                return False

            self.rknn_model = model_path
            logger.info(f"Loaded RKNN model: {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading RKNN model: {e}")
            return False

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on RK3588 NPU"""
        if not self.is_available or self.rknn_model is None:
            raise RuntimeError("RKNN model not loaded")

        try:
            outputs = self.runtime.inference(inputs=[input_data])
            return outputs[0] if outputs else None
        except Exception as e:
            logger.error(f"RKNN inference error: {e}")
            raise

    def cleanup(self):
        """Clean up RKNN resources"""
        if self.runtime is not None:
            try:
                self.runtime.release()
            except:
                pass
        self.runtime = None
        self.rknn_model = None


class EIC7700NPU(NPUAccelerator):
    """EIC7700 NPU acceleration using ENNP SDK"""

    def __init__(self):
        super().__init__("eic7700")
        self.ennp_model = None
        self.session = None

    def initialize(self) -> bool:
        """Initialize ENNP runtime"""
        try:
            # Try to import ENNP SDK
            # Note: This is a placeholder for actual ENNP Python bindings
            # The actual import will depend on Eswin's SDK structure
            try:
                import ennp
                self.runtime = ennp
                self.is_available = True
                logger.info("EIC7700 ENNP runtime initialized successfully")
                return True
            except ImportError:
                # Fallback: Try to use ONNX Runtime with ENNP EP
                logger.info("ENNP Python bindings not found, trying ONNX Runtime...")
                return self._initialize_onnxruntime()

        except Exception as e:
            logger.error(f"Failed to initialize ENNP runtime: {e}")
            return False

    def _initialize_onnxruntime(self) -> bool:
        """Initialize using ONNX Runtime with ENNP Execution Provider"""
        try:
            import onnxruntime as ort

            # Check for ENNP execution provider
            available_providers = ort.get_available_providers()
            logger.info(f"Available ONNX Runtime providers: {available_providers}")

            if "ENNPExecutionProvider" in available_providers:
                self.runtime = ort
                self.is_available = True
                logger.info("ONNX Runtime with ENNP EP initialized")
                return True
            else:
                logger.warning("ENNP Execution Provider not available in ONNX Runtime")
                # Still mark as available - we can use CPU as fallback
                self.runtime = ort
                self.is_available = True
                logger.info("Using ONNX Runtime (CPU fallback)")
                return True

        except ImportError:
            logger.warning("ONNX Runtime not installed. Install with: pip install onnxruntime")
            return False
        except Exception as e:
            logger.error(f"Error initializing ONNX Runtime: {e}")
            return False

    def load_model(self, model_path: str, use_ennp: bool = True) -> bool:
        """Load model for EIC7700 NPU

        Args:
            model_path: Path to model file (.onnx or .ennp format)
            use_ennp: Whether to try ENNP acceleration (fallback to CPU if unavailable)
        """
        if not self.is_available:
            logger.warning("ENNP runtime not available")
            return False

        try:
            # If using native ENNP SDK
            if hasattr(self.runtime, 'Model'):
                self.ennp_model = self.runtime.Model(model_path)
                logger.info(f"Loaded ENNP model: {model_path}")
                return True

            # If using ONNX Runtime
            else:
                import onnxruntime as ort

                # Configure session options
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                # Set up execution providers
                providers = []
                if use_ennp and "ENNPExecutionProvider" in ort.get_available_providers():
                    providers.append("ENNPExecutionProvider")
                    logger.info("Using ENNP Execution Provider")
                providers.append("CPUExecutionProvider")

                # Create inference session
                self.session = ort.InferenceSession(
                    model_path,
                    sess_options=sess_options,
                    providers=providers
                )

                logger.info(f"Loaded ONNX model with providers: {self.session.get_providers()}")
                self.ennp_model = model_path
                return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def infer(self, input_data: Union[np.ndarray, Dict[str, np.ndarray]]) -> np.ndarray:
        """Run inference on EIC7700 NPU"""
        if not self.is_available or self.ennp_model is None:
            raise RuntimeError("ENNP model not loaded")

        try:
            # Native ENNP inference
            if hasattr(self.runtime, 'Model') and hasattr(self.ennp_model, 'run'):
                outputs = self.ennp_model.run(input_data)
                return outputs

            # ONNX Runtime inference
            elif self.session is not None:
                # Get input name
                input_name = self.session.get_inputs()[0].name

                # Prepare input dict
                if isinstance(input_data, dict):
                    ort_inputs = input_data
                else:
                    ort_inputs = {input_name: input_data}

                # Run inference
                outputs = self.session.run(None, ort_inputs)
                return outputs[0] if outputs else None

            else:
                raise RuntimeError("No valid inference method available")

        except Exception as e:
            logger.error(f"ENNP inference error: {e}")
            raise

    def cleanup(self):
        """Clean up ENNP resources"""
        if self.session is not None:
            try:
                del self.session
            except:
                pass
        self.session = None
        self.ennp_model = None
        self.runtime = None


class NPUModelConverter:
    """Convert PyTorch/TensorFlow models to NPU format"""

    @staticmethod
    def convert_to_rknn(
        model_path: str,
        output_path: str,
        input_shape: tuple,
        mean_values: list = None,
        std_values: list = None
    ) -> bool:
        """Convert model to RKNN format for RK3588"""
        try:
            from rknn.api import RKNN

            # Create RKNN object
            rknn = RKNN(verbose=True)

            # Configure model
            logger.info(f"Loading model: {model_path}")

            # Detect model type
            if model_path.endswith('.onnx'):
                ret = rknn.load_onnx(model=model_path)
            elif model_path.endswith('.tflite'):
                ret = rknn.load_tflite(model=model_path)
            elif model_path.endswith('.pb'):
                ret = rknn.load_tensorflow(
                    tf_pb=model_path,
                    inputs=['input'],
                    outputs=['output'],
                    input_size_list=[input_shape]
                )
            else:
                logger.error(f"Unsupported model format: {model_path}")
                return False

            if ret != 0:
                logger.error("Failed to load model")
                return False

            # Build model
            logger.info("Building RKNN model...")
            ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
            if ret != 0:
                logger.error("Failed to build RKNN model")
                return False

            # Export RKNN model
            logger.info(f"Exporting RKNN model to: {output_path}")
            ret = rknn.export_rknn(output_path)
            if ret != 0:
                logger.error("Failed to export RKNN model")
                return False

            rknn.release()
            logger.info("Successfully converted to RKNN format")
            return True

        except ImportError:
            logger.error("RKNN toolkit not installed")
            return False
        except Exception as e:
            logger.error(f"Error converting to RKNN: {e}")
            return False

    @staticmethod
    def convert_to_ennp(
        model_path: str,
        output_path: str,
        input_shape: tuple,
        quantize: bool = True
    ) -> bool:
        """Convert model to ENNP format for EIC7700

        This uses the ENNP offline toolkit:
        1. EsQuant - for model quantization
        2. EsAAC - for model compilation
        """
        try:
            # First, ensure model is in ONNX format
            if not model_path.endswith('.onnx'):
                logger.info("Converting model to ONNX format first...")
                onnx_path = model_path.replace(Path(model_path).suffix, '.onnx')

                # Convert PyTorch to ONNX
                if model_path.endswith('.pt') or model_path.endswith('.pth'):
                    import torch
                    model = torch.load(model_path)
                    dummy_input = torch.randn(input_shape)
                    torch.onnx.export(
                        model,
                        dummy_input,
                        onnx_path,
                        export_params=True,
                        opset_version=11,
                        do_constant_folding=True
                    )
                    model_path = onnx_path
                else:
                    logger.error(f"Unsupported model format: {model_path}")
                    return False

            # Use ENNP offline tools (command-line based)
            # Note: This requires ENNP SDK to be installed
            logger.info("Converting to ENNP format using offline toolkit...")

            # Step 1: Quantize (if requested)
            if quantize:
                quantize_cmd = [
                    "esquant",
                    "--model", model_path,
                    "--output", output_path.replace('.ennp', '_quant.onnx'),
                    "--quantize_type", "int8"
                ]

                import subprocess
                result = subprocess.run(quantize_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.warning(f"Quantization failed: {result.stderr}")
                    logger.info("Proceeding without quantization...")
                    quantized_model = model_path
                else:
                    quantized_model = output_path.replace('.ennp', '_quant.onnx')
                    logger.info("Model quantized successfully")
            else:
                quantized_model = model_path

            # Step 2: Compile for ENNP
            compile_cmd = [
                "esaac",
                "--model", quantized_model,
                "--output", output_path,
                "--target", "eic7700"
            ]

            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"ENNP compilation failed: {result.stderr}")
                # Fallback: Just copy ONNX file
                logger.info("Using ONNX format as fallback")
                import shutil
                shutil.copy(model_path, output_path.replace('.ennp', '.onnx'))
                return True

            logger.info("Successfully converted to ENNP format")
            return True

        except Exception as e:
            logger.error(f"Error converting to ENNP: {e}")
            logger.info("Model will use CPU/GPU fallback")
            return False


def get_npu_accelerator(npu_type: str) -> Optional[NPUAccelerator]:
    """Factory function to get appropriate NPU accelerator"""
    if npu_type == "rk3588":
        accelerator = RK3588NPU()
        if accelerator.initialize():
            return accelerator
    elif npu_type == "eic7700":
        accelerator = EIC7700NPU()
        if accelerator.initialize():
            return accelerator

    logger.warning(f"NPU accelerator not available for type: {npu_type}")
    return None


def is_npu_model_available(model_path: str, npu_type: str) -> bool:
    """Check if NPU-optimized model exists"""
    if npu_type == "rk3588":
        rknn_path = str(model_path).replace('.onnx', '.rknn')
        return Path(rknn_path).exists()
    elif npu_type == "eic7700":
        ennp_path = str(model_path).replace('.onnx', '.ennp')
        onnx_path = str(model_path) if model_path.endswith('.onnx') else model_path + '.onnx'
        return Path(ennp_path).exists() or Path(onnx_path).exists()
    return False
