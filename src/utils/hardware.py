"""
Hardware Detection Module
Detects CPU architecture, NPU capabilities, and optimizes for different platforms
"""

import platform
import subprocess
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class HardwareDetector:
    """Detect and manage hardware capabilities"""

    def __init__(self):
        self.architecture = None
        self.soc_type = None
        self.has_npu = False
        self.npu_type = None
        self.npu_tops = 0
        self.cpu_info = {}
        self._detect_hardware()

    def _detect_hardware(self):
        """Detect hardware capabilities"""
        self.architecture = platform.machine().lower()
        self._detect_soc_type()
        self._detect_npu()
        self._get_cpu_info()

    def _detect_soc_type(self) -> str:
        """Detect specific SBC/SoC type"""
        device_tree_model = Path("/proc/device-tree/model")

        if device_tree_model.exists():
            try:
                with open(device_tree_model, 'r') as f:
                    model = f.read().strip('\x00').lower()

                # Check for RK3588
                if "rk3588" in model:
                    self.soc_type = "rk3588"
                    logger.info(f"Detected RK3588 SBC: {model}")
                    return "rk3588"

                # Check for EIC7700 (RISC-V)
                elif "eic7700" in model or "eswin" in model:
                    self.soc_type = "eic7700"
                    logger.info(f"Detected EIC7700 RISC-V SoC: {model}")
                    return "eic7700"

                # Check for Raspberry Pi
                elif "raspberry pi" in model:
                    self.soc_type = "rpi"
                    logger.info(f"Detected Raspberry Pi: {model}")
                    return "rpi"

                else:
                    self.soc_type = "generic"
                    logger.info(f"Generic SBC detected: {model}")
                    return "generic"

            except Exception as e:
                logger.warning(f"Could not read device tree model: {e}")

        # Fallback: Check architecture
        if self.architecture in ["riscv64", "riscv"]:
            # RISC-V detected, check if it's EIC7700
            if self._check_eic7700_specific():
                self.soc_type = "eic7700"
                return "eic7700"
            self.soc_type = "riscv_generic"
            return "riscv_generic"

        elif self.architecture in ["aarch64", "arm64"]:
            self.soc_type = "arm64_generic"
            return "arm64_generic"

        else:
            self.soc_type = "x86_64"
            return "x86_64"

    def _check_eic7700_specific(self) -> bool:
        """Check for EIC7700-specific indicators"""
        # Check for ENNP SDK presence
        ennp_paths = [
            "/usr/lib/libennp.so",
            "/usr/local/lib/libennp.so",
            "/opt/eswin/ennp/lib/libennp.so"
        ]

        for path in ennp_paths:
            if Path(path).exists():
                logger.info(f"Found ENNP SDK at {path}")
                return True

        # Check for EIC7700 in CPU info
        try:
            with open("/proc/cpuinfo", 'r') as f:
                cpuinfo = f.read().lower()
                if "eic7700" in cpuinfo or "eswin" in cpuinfo:
                    return True
        except:
            pass

        return False

    def _detect_npu(self):
        """Detect NPU hardware"""
        if self.soc_type == "rk3588":
            self.has_npu = self._detect_rk3588_npu()
            if self.has_npu:
                self.npu_type = "rk3588"
                self.npu_tops = 6.0  # RK3588 has 6 TOPS NPU

        elif self.soc_type == "eic7700":
            self.has_npu = self._detect_eic7700_npu()
            if self.has_npu:
                self.npu_type = "eic7700"
                # EIC7700 has 13.3 TOPS, EIC7700X has 19.95 TOPS
                # Try to detect which variant
                self.npu_tops = self._get_eic7700_tops()

        else:
            self.has_npu = False
            self.npu_type = None

    def _detect_rk3588_npu(self) -> bool:
        """Detect RK3588 NPU"""
        # Check for NPU device nodes
        npu_devices = [
            "/dev/rknpu",
            "/sys/class/misc/rknpu"
        ]

        for device in npu_devices:
            if Path(device).exists():
                logger.info(f"RK3588 NPU detected at {device}")
                return True

        # Check for RKNN toolkit
        try:
            import rknn
            logger.info("RKNN toolkit available")
            return True
        except ImportError:
            pass

        return False

    def _detect_eic7700_npu(self) -> bool:
        """Detect EIC7700 NPU (ENNP)"""
        # Check for ENNP device nodes
        ennp_devices = [
            "/dev/ennp",
            "/dev/eswin-npu",
            "/sys/class/misc/ennp"
        ]

        for device in ennp_devices:
            if Path(device).exists():
                logger.info(f"EIC7700 NPU detected at {device}")
                return True

        # Check for ENNP SDK
        try:
            # Try to import ENNP Python bindings if available
            import ennp
            logger.info("ENNP SDK available")
            return True
        except ImportError:
            pass

        # Check for ENNP library files
        ennp_libs = [
            "/usr/lib/libennp.so",
            "/usr/local/lib/libennp.so",
            "/opt/eswin/ennp/lib/libennp.so"
        ]

        for lib in ennp_libs:
            if Path(lib).exists():
                logger.info(f"ENNP library found at {lib}")
                return True

        return False

    def _get_eic7700_tops(self) -> float:
        """Get EIC7700 NPU TOPS rating"""
        # Try to read from device tree or system info
        try:
            # Check for EIC7700X indicators (higher performance variant)
            with open("/proc/cpuinfo", 'r') as f:
                cpuinfo = f.read().lower()
                if "eic7700x" in cpuinfo:
                    logger.info("Detected EIC7700X variant (19.95 TOPS)")
                    return 19.95
        except:
            pass

        # Default to standard EIC7700 (13.3 TOPS)
        logger.info("Detected EIC7700 standard variant (13.3 TOPS)")
        return 13.3

    def _get_cpu_info(self):
        """Get CPU information"""
        self.cpu_info = {
            'architecture': self.architecture,
            'processor': platform.processor(),
            'cpu_count': os.cpu_count() or 1,
            'system': platform.system(),
            'machine': platform.machine()
        }

        # Get CPU frequency if available
        try:
            with open("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", 'r') as f:
                max_freq_khz = int(f.read().strip())
                self.cpu_info['max_freq_mhz'] = max_freq_khz / 1000
        except:
            self.cpu_info['max_freq_mhz'] = None

    def get_optimal_device(self) -> str:
        """Get optimal device for PyTorch/AI workloads"""
        try:
            import torch

            # First priority: NPU if available
            if self.has_npu and self.npu_type:
                logger.info(f"NPU available: {self.npu_type} ({self.npu_tops} TOPS)")
                # Note: Actual NPU usage requires model conversion
                # For now, we'll use CPU with NPU as fallback for compatible models

            # Second priority: CUDA
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                return "cuda"

            # Third priority: MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("Apple MPS available")
                return "mps"

            # Default: CPU
            logger.info(f"Using CPU: {self.cpu_info.get('processor', 'Unknown')}")
            return "cpu"

        except ImportError:
            # PyTorch not available (e.g., on RISC-V without build)
            logger.warning("PyTorch not available, defaulting to CPU for ONNX Runtime")
            return "cpu"

    def get_npu_info(self) -> Dict[str, Any]:
        """Get NPU information"""
        return {
            'available': self.has_npu,
            'type': self.npu_type,
            'tops': self.npu_tops,
            'description': self._get_npu_description()
        }

    def _get_npu_description(self) -> str:
        """Get human-readable NPU description"""
        if not self.has_npu:
            return "No NPU detected"

        if self.npu_type == "rk3588":
            return f"Rockchip RK3588 NPU (6.0 TOPS)"
        elif self.npu_type == "eic7700":
            if self.npu_tops >= 19:
                return f"ESWIN EIC7700X NPU (19.95 TOPS INT8)"
            else:
                return f"ESWIN EIC7700 NPU (13.3 TOPS INT8)"
        else:
            return f"Unknown NPU ({self.npu_tops} TOPS)"

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            'architecture': self.architecture,
            'soc_type': self.soc_type,
            'cpu_info': self.cpu_info,
            'npu_info': self.get_npu_info(),
            'optimal_device': self.get_optimal_device()
        }

    def supports_npu_acceleration(self) -> bool:
        """Check if NPU acceleration is supported"""
        return self.has_npu and self.npu_type in ["rk3588", "eic7700"]

    def get_recommended_models(self) -> Dict[str, str]:
        """Get recommended model sizes based on hardware"""
        recommendations = {
            'whisper': 'base',
            'qwen': 'Qwen/Qwen2.5-3B-Instruct'
        }

        # Adjust based on hardware capabilities
        if self.soc_type == "eic7700":
            # EIC7700 has powerful NPU, can handle larger models
            if self.npu_tops >= 19:
                recommendations['whisper'] = 'medium'
                recommendations['qwen'] = 'Qwen/Qwen2.5-7B-Instruct'
            else:
                recommendations['whisper'] = 'small'
                recommendations['qwen'] = 'Qwen/Qwen2.5-3B-Instruct'

        elif self.soc_type == "rk3588":
            # RK3588 has moderate NPU
            recommendations['whisper'] = 'small'
            recommendations['qwen'] = 'Qwen/Qwen2.5-3B-Instruct'

        elif self.soc_type == "rpi":
            # Raspberry Pi - use smaller models
            recommendations['whisper'] = 'tiny'
            recommendations['qwen'] = 'Qwen/Qwen2.5-1.5B-Instruct'

        elif self.soc_type == "x86_64":
            # x86 - can typically handle larger models
            recommendations['whisper'] = 'medium'
            recommendations['qwen'] = 'Qwen/Qwen2.5-7B-Instruct'

        return recommendations


# Global hardware detector instance
_hardware_detector = None

def get_hardware_detector() -> HardwareDetector:
    """Get global hardware detector instance"""
    global _hardware_detector
    if _hardware_detector is None:
        _hardware_detector = HardwareDetector()
    return _hardware_detector


def get_system_info() -> Dict[str, Any]:
    """Convenience function to get system info"""
    return get_hardware_detector().get_system_info()


def get_optimal_device() -> str:
    """Convenience function to get optimal device"""
    return get_hardware_detector().get_optimal_device()


def supports_npu() -> bool:
    """Convenience function to check NPU support"""
    return get_hardware_detector().supports_npu_acceleration()


def get_npu_info() -> Dict[str, Any]:
    """Convenience function to get NPU info"""
    return get_hardware_detector().get_npu_info()
