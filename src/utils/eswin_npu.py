"""
ESWIN NPU Interface for Qwen2 7B
Wrapper for the ESWIN NPU binary: /opt/eswin/sample-code/npu_sample/qwen_sample/bin/es_qwen2
"""

import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class ESWINNPUInterface:
    """
    Interface to ESWIN NPU binary for Qwen2 7B inference.

    Based on HARDWARE_STACK.md integration guide.
    Uses stdin/stdout protocol with character-by-character streaming.
    """

    # Default paths (can be overridden)
    DEFAULT_BINARY = "/opt/eswin/sample-code/npu_sample/qwen_sample/bin/es_qwen2"
    DEFAULT_CONFIG = "/opt/eswin/sample-code/npu_sample/qwen_sample/config.json"
    DEFAULT_MODEL_DIR = "/opt/eswin/sample-code/npu_sample/qwen_sample/models/qwen2_7b_1k_int8"

    def __init__(
        self,
        binary_path: Optional[str] = None,
        config_path: Optional[str] = None,
        model_dir: Optional[str] = None,
        startup_timeout: int = 120,
        response_timeout: int = 60
    ):
        """
        Initialize ESWIN NPU interface.

        Args:
            binary_path: Path to es_qwen2 binary (default: /opt/eswin/.../es_qwen2)
            config_path: Path to config.json (default: /opt/eswin/.../config.json)
            model_dir: Path to model directory (default: /opt/eswin/.../qwen2_7b_1k_int8)
            startup_timeout: Timeout for model loading (seconds)
            response_timeout: Timeout for response generation (seconds)
        """
        self.binary_path = binary_path or self.DEFAULT_BINARY
        self.config_path = config_path or self.DEFAULT_CONFIG
        self.model_dir = model_dir or self.DEFAULT_MODEL_DIR
        self.startup_timeout = startup_timeout
        self.response_timeout = response_timeout

        self.process: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()
        self.is_ready = False

        logger.info(f"ESWIN NPU Interface initialized")
        logger.info(f"  Binary: {self.binary_path}")
        logger.info(f"  Config: {self.config_path}")
        logger.info(f"  Models: {self.model_dir}")

    @classmethod
    def is_available(cls, binary_path: Optional[str] = None) -> bool:
        """
        Check if ESWIN NPU is available on this system.

        Args:
            binary_path: Optional custom binary path

        Returns:
            True if NPU binary and models exist
        """
        binary = binary_path or cls.DEFAULT_BINARY

        # Check binary
        if not os.path.exists(binary):
            logger.debug(f"ESWIN NPU binary not found: {binary}")
            return False

        if not os.access(binary, os.X_OK):
            logger.debug(f"ESWIN NPU binary not executable: {binary}")
            return False

        # Check model directory
        model_dir = Path(cls.DEFAULT_MODEL_DIR)
        if not model_dir.exists():
            logger.debug(f"ESWIN NPU model directory not found: {model_dir}")
            return False

        # Check for essential model files
        essential_files = [
            "modified_block_0_npu_b1.model",
            "lm_npu_b1.model",
            "embedding.bin"
        ]

        for file in essential_files:
            if not (model_dir / file).exists():
                logger.debug(f"ESWIN NPU model file missing: {file}")
                return False

        logger.info("ESWIN NPU is available")
        return True

    def start(self) -> bool:
        """
        Start the NPU process and wait for initialization.

        Returns:
            True if started successfully
        """
        if self.process is not None:
            logger.warning("NPU process already running")
            return True

        try:
            logger.info("Starting ESWIN NPU process...")

            # Change to config directory for relative paths
            config_dir = os.path.dirname(self.config_path)
            config_file = os.path.basename(self.config_path)

            self.process = subprocess.Popen(
                [self.binary_path, config_file],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                cwd=config_dir if config_dir else None
            )

            logger.info("Waiting for model to load (looking for 100.00%)...")

            if not self._wait_ready():
                logger.error("NPU failed to initialize")
                self.stop()
                return False

            self.is_ready = True
            logger.info("✅ ESWIN NPU ready for inference")
            return True

        except Exception as e:
            logger.error(f"Failed to start NPU: {e}")
            self.stop()
            return False

    def _wait_ready(self) -> bool:
        """
        Wait for NPU to finish loading (detect "100.00%" in stdout).

        Returns:
            True if ready within timeout
        """
        buffer = ""
        start_time = time.time()

        while time.time() - start_time < self.startup_timeout:
            # Check if process died
            if self.process.poll() is not None:
                stderr = self.process.stderr.read() if self.process.stderr else ""
                logger.error(f"NPU process died during startup: {stderr}")
                return False

            try:
                char = self.process.stdout.read(1)
                if not char:
                    time.sleep(0.1)
                    continue

                buffer += char

                # Keep buffer size manageable
                if len(buffer) > 10000:
                    buffer = buffer[-1000:]

                # Check for ready signal
                if "100.00%" in buffer:
                    logger.info("NPU model loaded successfully")
                    return True

            except Exception as e:
                logger.error(f"Error reading NPU stdout: {e}")
                return False

        logger.error(f"NPU initialization timeout ({self.startup_timeout}s)")
        return False

    def generate(
        self,
        prompt: str,
        streaming_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Generate text using the NPU.

        Args:
            prompt: Input prompt text
            streaming_callback: Optional callback(char) for real-time streaming

        Returns:
            Generated text response
        """
        if not self.is_ready:
            raise RuntimeError("NPU not ready. Call start() first.")

        with self.lock:
            try:
                # Send mode selector (3 = custom prompt)
                self.process.stdin.write("3\n")
                self.process.stdin.flush()

                # Send prompt
                self.process.stdin.write(f"{prompt}\n")
                self.process.stdin.flush()

                logger.debug(f"Sent prompt to NPU ({len(prompt)} chars)")

                # Read response character-by-character
                output = []
                start_time = time.time()
                skip_first_chars = 7  # Skip control characters

                while time.time() - start_time < self.response_timeout:
                    char = self.process.stdout.read(1)

                    if not char:
                        # Check if process died
                        if self.process.poll() is not None:
                            raise RuntimeError("NPU process died during generation")
                        time.sleep(0.01)
                        continue

                    # Skip first few control characters
                    if len(output) < skip_first_chars:
                        output.append(char)
                        continue

                    output.append(char)

                    # Call streaming callback if provided
                    if streaming_callback and len(output) >= skip_first_chars:
                        streaming_callback(char)

                    # Check for termination marker
                    recent = ''.join(output[-10:])
                    if "-------" in recent:
                        logger.debug("NPU generation complete (termination marker found)")
                        break

                # Remove control characters and termination marker
                response = ''.join(output[skip_first_chars:])
                response = response.replace("-------", "").strip()

                logger.debug(f"NPU response: {len(response)} chars")
                return response

            except Exception as e:
                logger.error(f"NPU generation error: {e}")
                raise

    def stop(self):
        """Stop the NPU process."""
        if self.process is None:
            return

        try:
            logger.info("Stopping ESWIN NPU process...")
            self.process.terminate()

            # Wait for graceful shutdown
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("NPU process didn't terminate, killing...")
                self.process.kill()
                self.process.wait()

            logger.info("ESWIN NPU stopped")

        except Exception as e:
            logger.error(f"Error stopping NPU: {e}")

        finally:
            self.process = None
            self.is_ready = False

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Check if NPU is available
    if not ESWINNPUInterface.is_available():
        print("❌ ESWIN NPU not available on this system")
        exit(1)

    # Use context manager
    with ESWINNPUInterface() as npu:
        # Simple generation
        print("\n=== Simple Generation ===")
        response = npu.generate("What is artificial intelligence?")
        print(f"Response: {response}")

        # Streaming generation
        print("\n=== Streaming Generation ===")
        print("Response: ", end="", flush=True)
        response = npu.generate(
            "Explain quantum computing in simple terms.",
            streaming_callback=lambda char: print(char, end="", flush=True)
        )
        print("\n")

    print("✅ ESWIN NPU demo complete")
