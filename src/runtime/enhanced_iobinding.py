"""
Enhanced IOBinding implementation with advanced memory management.

This module provides optimized IOBinding with:
- Intelligent memory pool management
- Zero-copy operations where possible
- Adaptive buffer sizing
- Memory usage monitoring
"""

import onnxruntime as ort
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import time
import threading
import gc
from collections import defaultdict
from .device import pick_providers, device_string_from_provider, get_device_info

logger = logging.getLogger(__name__)

class MemoryPool:
    """
    Advanced memory pool for tensor buffer reuse.
    """
    
    def __init__(self, device: str, max_pool_size: int = 100):
        """
        Initialize memory pool.
        
        Args:
            device: Target device ("cpu", "cuda", etc.)
            max_pool_size: Maximum number of buffers to keep
        """
        self.device = device
        self.max_pool_size = max_pool_size
        self.buffers = defaultdict(list)  # (shape, dtype) -> [buffers]
        self.usage_count = defaultdict(int)
        self.lock = threading.Lock()
        
    def get_buffer(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """
        Get or create buffer with specified shape and dtype.
        
        Args:
            shape: Tensor shape
            dtype: Tensor dtype
            
        Returns:
            Reusable tensor buffer
        """
        key = (shape, dtype)
        
        with self.lock:
            if key in self.buffers and self.buffers[key]:
                # Reuse existing buffer
                buffer = self.buffers[key].pop()
                self.usage_count[key] += 1
                logger.debug(f"Reused buffer {shape} {dtype} (usage: {self.usage_count[key]})")
                return buffer
        
        # Create new buffer
        try:
            if self.device == "cuda" and torch.cuda.is_available():
                buffer = torch.empty(shape, dtype=dtype, device="cuda", pin_memory=False)
            elif self.device == "rocm":
                buffer = torch.empty(shape, dtype=dtype, device="cuda")  # ROCm uses cuda in PyTorch
            else:
                buffer = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
            
            with self.lock:
                self.usage_count[key] += 1
            
            logger.debug(f"Created new buffer {shape} {dtype} on {self.device}")
            return buffer
            
        except Exception as e:
            logger.warning(f"Failed to create buffer on {self.device}: {e}")
            # Fallback to CPU
            buffer = torch.empty(shape, dtype=dtype, device="cpu")
            with self.lock:
                self.usage_count[key] += 1
            return buffer
    
    def return_buffer(self, buffer: torch.Tensor):
        """
        Return buffer to pool for reuse.
        
        Args:
            buffer: Buffer to return
        """
        if buffer.device.type != self.device.split(":")[0]:
            return  # Wrong device
        
        key = (tuple(buffer.shape), buffer.dtype)
        
        with self.lock:
            if len(self.buffers[key]) < self.max_pool_size:
                # Clear buffer content for security
                buffer.zero_()
                self.buffers[key].append(buffer)
                logger.debug(f"Returned buffer {key} to pool")
            else:
                logger.debug(f"Pool full for {key}, discarding buffer")
    
    def clear_pool(self):
        """Clear all buffers from pool."""
        with self.lock:
            self.buffers.clear()
            self.usage_count.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        logger.info("Memory pool cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self.lock:
            total_buffers = sum(len(buffers) for buffers in self.buffers.values())
            return {
                "total_buffer_types": len(self.buffers),
                "total_buffers": total_buffers,
                "usage_counts": dict(self.usage_count),
                "device": self.device
            }

class EnhancedORTSession:
    """
    Enhanced ONNX Runtime session with advanced IOBinding and memory management.
    """
    
    def __init__(self, model_path: str, for_stage: str = "decode", 
                 enable_memory_pool: bool = True, pool_size: int = 50):
        """
        Initialize enhanced ORT session.
        
        Args:
            model_path: Path to ONNX model
            for_stage: "decode" or "prefill" for provider selection
            enable_memory_pool: Whether to use memory pooling
            pool_size: Maximum buffers per pool
        """
        self.model_path = model_path
        self.for_stage = for_stage
        self.enable_memory_pool = enable_memory_pool
        self.session = None
        self.provider = None
        self.device_info = None
        self.memory_pools = {}
        self.io_binding_cache = {}
        self.performance_stats = {
            "total_runs": 0,
            "iobinding_runs": 0,
            "fallback_runs": 0,
            "avg_latency_ms": 0.0,
            "memory_reuse_rate": 0.0
        }
        
        self._create_session()
        
        if self.enable_memory_pool:
            device = device_string_from_provider(self.provider)
            self.memory_pools[device] = MemoryPool(device, pool_size)
    
    def _create_session(self):
        """Create optimized ONNX Runtime session with enhanced settings."""
        session_options = ort.SessionOptions()
        
        # Enhanced optimization settings
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.enable_mem_pattern = True
        session_options.enable_cpu_mem_arena = True
        session_options.enable_mem_reuse = True
        
        # Memory optimization
        session_options.add_session_config_entry("session.use_env_allocators", "1")
        session_options.add_session_config_entry("session.disable_prepacking", "0")
        
        # Thread configuration for latency
        session_options.intra_op_num_threads = max(1, torch.get_num_threads() // 2)
        session_options.inter_op_num_threads = 1
        
        # Select providers
        providers = pick_providers(self.for_stage)
        
        try:
            self.session = ort.InferenceSession(
                self.model_path, 
                sess_options=session_options, 
                providers=providers
            )
            self.provider = providers[0]
            self.device_info = get_device_info(self.provider)
            
            logger.info(f"Enhanced session created with provider: {self.provider}")
            
        except Exception as e:
            logger.error(f"Failed to create enhanced session: {e}")
            # Fallback to CPU
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=["CPUExecutionProvider"]
            )
            self.provider = "CPUExecutionProvider"
            self.device_info = get_device_info(self.provider)
    
    def _get_memory_pool(self, device: str) -> MemoryPool:
        """Get memory pool for device."""
        if not self.enable_memory_pool:
            return None
        
        if device not in self.memory_pools:
            self.memory_pools[device] = MemoryPool(device)
        
        return self.memory_pools[device]
    
    def _prepare_tensor_input(self, tensor: torch.Tensor, target_device: str) -> torch.Tensor:
        """
        Prepare tensor for optimal transfer to target device.
        
        Args:
            tensor: Input tensor
            target_device: Target device string
            
        Returns:
            Optimized tensor
        """
        # Ensure tensor is contiguous
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        # Move to target device if needed
        current_device = tensor.device.type
        if current_device != target_device and target_device != "cpu":
            try:
                tensor = tensor.to(target_device, non_blocking=True)
            except Exception as e:
                logger.debug(f"Failed to move tensor to {target_device}: {e}")
        
        return tensor
    
    def _bind_input_optimized(self, io_binding: ort.IOBinding, name: str, 
                            tensor: torch.Tensor) -> bool:
        """
        Bind input tensor with optimized zero-copy when possible.
        
        Args:
            io_binding: ORT IOBinding object
            name: Input name
            tensor: Input tensor
            
        Returns:
            True if zero-copy binding succeeded
        """
        try:
            # Prepare tensor for optimal binding
            device = device_string_from_provider(self.provider)
            tensor = self._prepare_tensor_input(tensor, device)
            
            # Try zero-copy binding for compatible tensors
            if (tensor.is_cuda and self.device_info["supports_iobinding"] and 
                tensor.is_contiguous()):
                
                # Use DLPack for zero-copy GPU→ORT
                ort_value = ort.OrtValue.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))
                io_binding.bind_ortvalue_input(name, ort_value)
                logger.debug(f"Zero-copy binding for input {name}")
                return True
                
        except Exception as e:
            logger.debug(f"Zero-copy binding failed for {name}: {e}")
        
        # Fallback: CPU binding with optimized copy
        try:
            if tensor.is_cuda:
                # Async copy to CPU
                cpu_tensor = tensor.to("cpu", non_blocking=True)
                torch.cuda.synchronize()  # Ensure copy is complete
                np_array = cpu_tensor.detach().numpy()
            else:
                np_array = tensor.detach().numpy()
            
            io_binding.bind_cpu_input(name, np_array)
            logger.debug(f"CPU binding for input {name}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to bind input {name}: {e}")
            raise
    
    def run_optimized(self, feeds: Dict[str, torch.Tensor], 
                     output_names: List[str],
                     output_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
                     return_buffers: bool = False) -> Dict[str, Any]:
        """
        Run inference with enhanced IOBinding and memory management.
        
        Args:
            feeds: Input tensors
            output_names: Names of outputs to retrieve
            output_shapes: Expected output shapes for pre-allocation
            return_buffers: Whether to return buffers to pool after use
            
        Returns:
            Dictionary of output tensors/values
        """
        start_time = time.perf_counter()
        self.performance_stats["total_runs"] += 1
        
        if not self.device_info["supports_iobinding"]:
            result = self._run_fallback(feeds, output_names)
            self.performance_stats["fallback_runs"] += 1
        else:
            try:
                result = self._run_with_iobinding(feeds, output_names, output_shapes)
                self.performance_stats["iobinding_runs"] += 1
            except Exception as e:
                logger.warning(f"IOBinding failed: {e}, falling back")
                result = self._run_fallback(feeds, output_names)
                self.performance_stats["fallback_runs"] += 1
        
        # Update performance statistics
        elapsed = time.perf_counter() - start_time
        self.performance_stats["avg_latency_ms"] = (
            (self.performance_stats["avg_latency_ms"] * (self.performance_stats["total_runs"] - 1) + 
             elapsed * 1000) / self.performance_stats["total_runs"]
        )
        
        # Calculate memory reuse rate
        if self.enable_memory_pool:
            total_buffers = sum(pool.get_stats()["total_buffers"] 
                              for pool in self.memory_pools.values())
            total_usage = sum(sum(pool.usage_count.values()) 
                            for pool in self.memory_pools.values())
            if total_usage > 0:
                self.performance_stats["memory_reuse_rate"] = total_buffers / total_usage
        
        return result
    
    def _run_with_iobinding(self, feeds: Dict[str, torch.Tensor], 
                           output_names: List[str],
                           output_shapes: Optional[Dict[str, Tuple[int, ...]]]) -> Dict[str, Any]:
        """Run inference with IOBinding."""
        io_binding = self.session.io_binding()
        zero_copy_count = 0
        
        # Bind inputs
        for name, tensor in feeds.items():
            if self._bind_input_optimized(io_binding, name, tensor):
                zero_copy_count += 1
        
        # Bind outputs
        for name in output_names:
            if output_shapes and name in output_shapes:
                # Pre-allocate output buffer if shape is known
                shape = output_shapes[name]
                dtype = torch.float32  # Should be configurable
                
                if self.enable_memory_pool:
                    device = device_string_from_provider(self.provider)
                    pool = self._get_memory_pool(device)
                    if pool:
                        buffer = pool.get_buffer(shape, dtype)
                        try:
                            ort_value = ort.OrtValue.from_dlpack(torch.utils.dlpack.to_dlpack(buffer))
                            io_binding.bind_ortvalue_output(name, ort_value)
                            continue
                        except Exception as e:
                            logger.debug(f"Failed to bind pre-allocated buffer: {e}")
                            pool.return_buffer(buffer)
            
            # Default output binding
            io_binding.bind_output(name)
        
        # Run inference
        self.session.run_with_iobinding(io_binding)
        
        # Retrieve outputs
        outputs = {}
        ort_outputs = io_binding.get_outputs()
        
        for i, name in enumerate(output_names):
            if i < len(ort_outputs):
                ort_value = ort_outputs[i]
                try:
                    # Try to get as tensor
                    outputs[name] = torch.from_dlpack(ort_value.to_dlpack())
                except:
                    # Fallback to numpy
                    outputs[name] = ort_value.numpy()
        
        logger.debug(f"IOBinding run: {zero_copy_count}/{len(feeds)} zero-copy inputs")
        return outputs
    
    def _run_fallback(self, feeds: Dict[str, torch.Tensor], 
                     output_names: List[str]) -> Dict[str, Any]:
        """Fallback to regular ONNX Runtime inference."""
        # Convert tensors to numpy
        np_feeds = {}
        for name, tensor in feeds.items():
            np_feeds[name] = tensor.detach().to("cpu").numpy()
        
        # Run inference
        outputs = self.session.run(output_names, np_feeds)
        
        # Convert to dictionary
        return {name: outputs[i] for i, name in enumerate(output_names)}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        
        if self.enable_memory_pool:
            stats["memory_pools"] = {
                device: pool.get_stats() 
                for device, pool in self.memory_pools.items()
            }
        
        return stats
    
    def clear_memory_pools(self):
        """Clear all memory pools."""
        if self.enable_memory_pool:
            for pool in self.memory_pools.values():
                pool.clear_pool()
    
    def __del__(self):
        """Cleanup resources."""
        self.clear_memory_pools()

def create_enhanced_session(model_path: str, for_stage: str = "decode",
                          enable_memory_pool: bool = True, 
                          pool_size: int = 50) -> EnhancedORTSession:
    """
    Factory function to create enhanced ORT session.
    
    Args:
        model_path: Path to ONNX model
        for_stage: "decode" or "prefill"
        enable_memory_pool: Whether to enable memory pooling
        pool_size: Maximum buffers per pool
        
    Returns:
        Enhanced ORT session
    """
    return EnhancedORTSession(model_path, for_stage, enable_memory_pool, pool_size)

