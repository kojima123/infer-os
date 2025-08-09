"""
ONNX Runtime session management with IOBinding optimization.

This module provides optimized session creation and zero-copy I/O operations
to minimize CPU↔GPU/NPU transfer overhead.
"""

import onnxruntime as ort
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
import logging
import time
from .device import pick_providers, device_string_from_provider, get_device_info

logger = logging.getLogger(__name__)

class OptimizedORTSession:
    """
    Optimized ONNX Runtime session with IOBinding and memory reuse.
    """
    
    def __init__(self, model_path: str, for_stage: str = "decode"):
        """
        Initialize optimized ORT session.
        
        Args:
            model_path: Path to ONNX model
            for_stage: "decode" or "prefill" for provider selection
        """
        self.model_path = model_path
        self.for_stage = for_stage
        self.session = None
        self.provider = None
        self.device_info = None
        self.input_buffers = {}
        self.output_buffers = {}
        self.buffer_pool = {}
        
        self._create_session()
    
    def _create_session(self):
        """Create optimized ONNX Runtime session."""
        # Session options for performance
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.enable_mem_pattern = True
        session_options.enable_cpu_mem_arena = True
        session_options.enable_mem_reuse = True
        
        # Thread configuration for latency optimization
        session_options.intra_op_num_threads = max(1, torch.get_num_threads() // 2)
        session_options.inter_op_num_threads = 1  # Minimize context switching
        
        # Select providers
        providers = pick_providers(self.for_stage)
        
        try:
            self.session = ort.InferenceSession(
                self.model_path, 
                sess_options=session_options, 
                providers=providers
            )
            self.provider = providers[0]  # First available provider
            self.device_info = get_device_info(self.provider)
            
            logger.info(f"Created session with provider: {self.provider}")
            logger.info(f"Device info: {self.device_info}")
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            # Fallback to CPU
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=["CPUExecutionProvider"]
            )
            self.provider = "CPUExecutionProvider"
            self.device_info = get_device_info(self.provider)
    
    def _get_buffer(self, name: str, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """
        Get or create reusable buffer for tensor.
        
        Args:
            name: Buffer name
            shape: Tensor shape
            dtype: Tensor dtype
            
        Returns:
            Reusable tensor buffer
        """
        key = (name, shape, dtype)
        
        if key not in self.buffer_pool:
            device = device_string_from_provider(self.provider)
            try:
                # Try to create on target device
                if device == "cuda" and torch.cuda.is_available():
                    buffer = torch.empty(shape, dtype=dtype, device="cuda")
                elif device == "rocm":
                    # ROCm uses cuda device string in PyTorch
                    buffer = torch.empty(shape, dtype=dtype, device="cuda")
                else:
                    buffer = torch.empty(shape, dtype=dtype, device="cpu")
                
                self.buffer_pool[key] = buffer
                logger.debug(f"Created buffer {name} with shape {shape} on {device}")
                
            except Exception as e:
                logger.warning(f"Failed to create buffer on {device}: {e}, falling back to CPU")
                buffer = torch.empty(shape, dtype=dtype, device="cpu")
                self.buffer_pool[key] = buffer
        
        return self.buffer_pool[key]
    
    def _bind_tensor_input(self, io_binding: ort.IOBinding, name: str, tensor: torch.Tensor):
        """
        Bind tensor as input with zero-copy when possible.
        
        Args:
            io_binding: ORT IOBinding object
            name: Input name
            tensor: Input tensor
        """
        try:
            # Try zero-copy binding for GPU tensors
            if tensor.is_cuda and self.device_info["supports_iobinding"]:
                # Use DLPack for zero-copy GPU→ORT
                ort_value = ort.OrtValue.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))
                io_binding.bind_ortvalue_input(name, ort_value)
                logger.debug(f"Zero-copy binding for input {name}")
                return
        except Exception as e:
            logger.debug(f"Zero-copy binding failed for {name}: {e}")
        
        # Fallback: CPU binding
        np_array = tensor.detach().to("cpu", non_blocking=True).contiguous().numpy()
        io_binding.bind_cpu_input(name, np_array)
        logger.debug(f"CPU binding for input {name}")
    
    def _bind_output(self, io_binding: ort.IOBinding, name: str, shape: Tuple[int, ...], dtype: torch.dtype):
        """
        Bind output tensor with pre-allocated buffer.
        
        Args:
            io_binding: ORT IOBinding object
            name: Output name
            shape: Expected output shape
            dtype: Expected output dtype
        """
        try:
            if self.device_info["supports_iobinding"]:
                # Try device-side output binding
                io_binding.bind_output(name)
                logger.debug(f"Device binding for output {name}")
                return
        except Exception as e:
            logger.debug(f"Device output binding failed for {name}: {e}")
        
        # Fallback: CPU output binding
        io_binding.bind_output(name)
    
    def run_with_iobinding(self, feeds: Dict[str, torch.Tensor], 
                          output_names: List[str],
                          output_shapes: Optional[Dict[str, Tuple[int, ...]]] = None) -> Dict[str, Any]:
        """
        Run inference with IOBinding optimization.
        
        Args:
            feeds: Input tensors
            output_names: Names of outputs to retrieve
            output_shapes: Expected output shapes (for buffer pre-allocation)
            
        Returns:
            Dictionary of output tensors/values
        """
        if not self.device_info["supports_iobinding"]:
            # Fallback to regular run
            return self._run_regular(feeds, output_names)
        
        start_time = time.perf_counter()
        
        try:
            io_binding = self.session.io_binding()
            
            # Bind inputs
            for name, tensor in feeds.items():
                self._bind_tensor_input(io_binding, name, tensor)
            
            # Bind outputs
            for name in output_names:
                if output_shapes and name in output_shapes:
                    shape = output_shapes[name]
                    dtype = torch.float32  # Default, should be configurable
                    self._bind_output(io_binding, name, shape, dtype)
                else:
                    io_binding.bind_output(name)
            
            # Run inference
            self.session.run_with_iobinding(io_binding)
            
            # Retrieve outputs
            outputs = {}
            ort_outputs = io_binding.get_outputs()
            
            for i, name in enumerate(output_names):
                if i < len(ort_outputs):
                    ort_value = ort_outputs[i]
                    # Convert OrtValue to tensor/numpy as needed
                    try:
                        # Try to get as tensor
                        outputs[name] = torch.from_dlpack(ort_value.to_dlpack())
                    except:
                        # Fallback to numpy
                        outputs[name] = ort_value.numpy()
                else:
                    logger.warning(f"Output {name} not found in results")
            
            elapsed = time.perf_counter() - start_time
            logger.debug(f"IOBinding inference took {elapsed*1000:.2f}ms")
            
            return outputs
            
        except Exception as e:
            logger.warning(f"IOBinding failed: {e}, falling back to regular run")
            return self._run_regular(feeds, output_names)
    
    def _run_regular(self, feeds: Dict[str, torch.Tensor], output_names: List[str]) -> Dict[str, Any]:
        """
        Fallback to regular ONNX Runtime inference.
        
        Args:
            feeds: Input tensors
            output_names: Names of outputs to retrieve
            
        Returns:
            Dictionary of output arrays
        """
        start_time = time.perf_counter()
        
        # Convert tensors to numpy
        np_feeds = {}
        for name, tensor in feeds.items():
            np_feeds[name] = tensor.detach().to("cpu").numpy()
        
        # Run inference
        outputs = self.session.run(output_names, np_feeds)
        
        # Convert to dictionary
        result = {name: outputs[i] for i, name in enumerate(output_names)}
        
        elapsed = time.perf_counter() - start_time
        logger.debug(f"Regular inference took {elapsed*1000:.2f}ms")
        
        return result
    
    def get_input_metadata(self) -> Dict[str, Any]:
        """Get input metadata for the model."""
        inputs = {}
        for inp in self.session.get_inputs():
            inputs[inp.name] = {
                "shape": inp.shape,
                "type": inp.type
            }
        return inputs
    
    def get_output_metadata(self) -> Dict[str, Any]:
        """Get output metadata for the model."""
        outputs = {}
        for out in self.session.get_outputs():
            outputs[out.name] = {
                "shape": out.shape,
                "type": out.type
            }
        return outputs

def create_optimized_session(model_path: str, for_stage: str = "decode") -> OptimizedORTSession:
    """
    Factory function to create optimized ORT session.
    
    Args:
        model_path: Path to ONNX model
        for_stage: "decode" or "prefill"
        
    Returns:
        Optimized ORT session
    """
    return OptimizedORTSession(model_path, for_stage)

