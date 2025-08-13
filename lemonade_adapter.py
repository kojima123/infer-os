#!/usr/bin/env python3
"""
Lemonade Adapter for Infer-OS Integration
Middleware/Plugin for GAIA Lemonade Server integration
"""

import json
import logging
import time
import threading
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
import requests
import numpy as np
from contextlib import contextmanager

# Configuration and data models
@dataclass
class InferenceContext:
    """Context information for inference run"""
    model_name: str
    seq_len: int
    batch_size: int
    target_ftl_ms: int = 300
    quality_budget: float = 0.5
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

@dataclass
class InferenceStats:
    """Statistics from inference execution"""
    start_time: float
    end_time: float
    tokens_generated: int
    first_token_latency: float
    total_latency: float
    memory_peak: float
    quality_score: float = 1.0
    
    @property
    def tps(self) -> float:
        """Tokens per second"""
        if self.total_latency > 0:
            return self.tokens_generated / (self.total_latency / 1000.0)
        return 0.0

class KVQuantizer:
    """KV-cache quantization manager"""
    
    def __init__(self):
        self.quantization_levels = {
            "L0": {"precision": "fp16", "scale": 1.0},
            "L1": {"precision": "int8", "scale": 127.0},
            "L2": {"precision": "int4", "scale": 15.0},
            "L3": {"precision": "evict", "scale": 0.0}
        }
        self.logger = logging.getLogger("KVQuantizer")
    
    def quantize_kv(self, kv_tensor: np.ndarray, level: str = "L1") -> Dict[str, Any]:
        """Quantize KV tensor to specified level"""
        try:
            if level not in self.quantization_levels:
                level = "L1"  # Default fallback
            
            config = self.quantization_levels[level]
            
            if config["precision"] == "fp16":
                # No quantization needed
                return {
                    "data": kv_tensor.astype(np.float16),
                    "meta": {"level": level, "scale": 1.0, "zero_point": 0}
                }
            
            elif config["precision"] == "int8":
                # INT8 quantization
                tensor_max = np.abs(kv_tensor).max()
                scale = tensor_max / config["scale"] if tensor_max > 0 else 1.0
                
                quantized = np.round(kv_tensor / scale).clip(-128, 127).astype(np.int8)
                
                return {
                    "data": quantized,
                    "meta": {
                        "level": level,
                        "scale": scale,
                        "zero_point": 0,
                        "shape": kv_tensor.shape,
                        "dtype": str(kv_tensor.dtype)
                    }
                }
            
            elif config["precision"] == "int4":
                # INT4 quantization (stored as int8)
                tensor_max = np.abs(kv_tensor).max()
                scale = tensor_max / config["scale"] if tensor_max > 0 else 1.0
                
                quantized = np.round(kv_tensor / scale).clip(-8, 7).astype(np.int8)
                
                return {
                    "data": quantized,
                    "meta": {
                        "level": level,
                        "scale": scale,
                        "zero_point": 0,
                        "shape": kv_tensor.shape,
                        "dtype": str(kv_tensor.dtype)
                    }
                }
            
            elif config["precision"] == "evict":
                # Evict from cache (compress or drop)
                return {
                    "data": None,
                    "meta": {
                        "level": level,
                        "evicted": True,
                        "shape": kv_tensor.shape,
                        "dtype": str(kv_tensor.dtype)
                    }
                }
            
        except Exception as e:
            self.logger.error(f"KV quantization error: {e}")
            # Fallback to FP16
            return {
                "data": kv_tensor.astype(np.float16),
                "meta": {"level": "L0", "scale": 1.0, "zero_point": 0, "error": str(e)}
            }
    
    def dequantize_kv(self, quantized_data: Dict[str, Any]) -> np.ndarray:
        """Dequantize KV tensor from quantized format"""
        try:
            data = quantized_data["data"]
            meta = quantized_data["meta"]
            level = meta["level"]
            
            if level == "L0" or data is None:
                # FP16 or evicted data
                if data is None:
                    # Return zeros for evicted data
                    shape = meta.get("shape", (1, 1, 1))
                    return np.zeros(shape, dtype=np.float16)
                return data.astype(np.float16)
            
            elif level in ["L1", "L2"]:
                # Dequantize INT8/INT4
                scale = meta["scale"]
                dequantized = data.astype(np.float32) * scale
                return dequantized.astype(np.float16)
            
            else:
                self.logger.warning(f"Unknown quantization level: {level}")
                return data.astype(np.float16)
                
        except Exception as e:
            self.logger.error(f"KV dequantization error: {e}")
            # Return zeros as fallback
            return np.zeros((1, 1, 1), dtype=np.float16)

class LemonadeAdapter:
    """
    Lemonade Adapter for Infer-OS Integration
    
    Provides hooks for:
    - PreRun: Context setup and policy application
    - KV Read/Write: Quantization management
    - PostRun: Metrics collection and feedback
    """
    
    def __init__(self, agent_url: str = "http://127.0.0.1:7031", enable_kv_quantization: bool = True):
        self.agent_url = agent_url
        self.enable_kv_quantization = enable_kv_quantization
        self.kv_quantizer = KVQuantizer() if enable_kv_quantization else None
        self.logger = self._setup_logging()
        
        # State tracking
        self.current_context: Optional[InferenceContext] = None
        self.current_policy: Dict[str, Any] = {}
        self.kv_cache: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.inference_stats: List[InferenceStats] = []
        self.max_stats_history = 100
        
        # Thread safety
        self.lock = threading.RLock()
        
        self.logger.info("Lemonade Adapter initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("LemonadeAdapter")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _call_agent_api(self, endpoint: str, method: str = "GET", data: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Call Infer-OS Control Agent API"""
        try:
            url = f"{self.agent_url}{endpoint}"
            
            if method == "GET":
                response = requests.get(url, timeout=5.0)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=5.0)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            if response.status_code in [200, 204]:
                if response.content:
                    return response.json()
                return {}
            else:
                self.logger.warning(f"Agent API call failed: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Agent API unavailable: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Agent API error: {e}")
            return None
    
    def prerun_hook(self, context: InferenceContext) -> Dict[str, Any]:
        """
        PreRun hook - called before inference execution
        
        Args:
            context: Inference context information
            
        Returns:
            Policy configuration to apply
        """
        with self.lock:
            self.current_context = context
            
            try:
                # Notify agent of run context
                context_data = {
                    "seq_len": context.seq_len,
                    "batch": context.batch_size,
                    "target_ftl_ms": context.target_ftl_ms,
                    "quality_budget": context.quality_budget
                }
                
                self._call_agent_api("/v1/run-context", "POST", context_data)
                
                # Get suggested policy (placeholder - would be implemented in agent)
                policy_response = self._call_agent_api("/v1/policy/suggest")
                
                if policy_response:
                    self.current_policy = policy_response
                else:
                    # Default policy
                    self.current_policy = {
                        "kv": {"mode": "dynamic", "recent_window": 64},
                        "io": {"enable_iobinding": True},
                        "scheduler": {"mode": "hybrid"}
                    }
                
                self.logger.info(f"PreRun: Context={context.model_name}, Policy={self.current_policy}")
                
                return self.current_policy
                
            except Exception as e:
                self.logger.error(f"PreRun hook error: {e}")
                return {"error": str(e)}
    
    def kv_write_hook(self, kv_id: str, kv_tensor: np.ndarray, meta: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        KV Write hook - called when writing KV cache
        
        Args:
            kv_id: Unique identifier for KV tensor
            kv_tensor: KV tensor data
            meta: Additional metadata
            
        Returns:
            Quantized KV data and metadata
        """
        if not self.enable_kv_quantization or self.kv_quantizer is None:
            # Pass through without quantization
            return {"data": kv_tensor, "meta": meta or {}}
        
        with self.lock:
            try:
                # Determine quantization level based on policy and conditions
                level = self._determine_quantization_level(kv_id, meta)
                
                # Quantize KV tensor
                quantized = self.kv_quantizer.quantize_kv(kv_tensor, level)
                
                # Store in cache
                self.kv_cache[kv_id] = {
                    "quantized": quantized,
                    "timestamp": time.time(),
                    "access_count": 1,
                    "original_shape": kv_tensor.shape
                }
                
                self.logger.debug(f"KV Write: {kv_id} -> {level}")
                
                return quantized
                
            except Exception as e:
                self.logger.error(f"KV write hook error: {e}")
                # Fallback to original data
                return {"data": kv_tensor, "meta": meta or {}, "error": str(e)}
    
    def kv_read_hook(self, kv_id: str) -> Optional[np.ndarray]:
        """
        KV Read hook - called when reading KV cache
        
        Args:
            kv_id: Unique identifier for KV tensor
            
        Returns:
            Dequantized KV tensor data
        """
        if not self.enable_kv_quantization or self.kv_quantizer is None:
            # Return from simple cache
            return self.kv_cache.get(kv_id, {}).get("data")
        
        with self.lock:
            try:
                if kv_id not in self.kv_cache:
                    self.logger.warning(f"KV Read: {kv_id} not found in cache")
                    return None
                
                cache_entry = self.kv_cache[kv_id]
                quantized_data = cache_entry["quantized"]
                
                # Update access statistics
                cache_entry["access_count"] += 1
                cache_entry["last_access"] = time.time()
                
                # Dequantize
                dequantized = self.kv_quantizer.dequantize_kv(quantized_data)
                
                self.logger.debug(f"KV Read: {kv_id} dequantized")
                
                return dequantized
                
            except Exception as e:
                self.logger.error(f"KV read hook error: {e}")
                return None
    
    def _determine_quantization_level(self, kv_id: str, meta: Dict[str, Any] = None) -> str:
        """Determine appropriate quantization level for KV tensor"""
        try:
            # Get current policy
            kv_policy = self.current_policy.get("kv", {})
            recent_window = kv_policy.get("recent_window", 64)
            
            # Parse KV ID to extract position information
            # Format: "layer{layer}.head{head}.pos{start}-{end}"
            if "pos" in kv_id:
                pos_part = kv_id.split("pos")[-1]
                if "-" in pos_part:
                    start_pos = int(pos_part.split("-")[0])
                    # Recent positions stay in FP16
                    if start_pos >= (self.current_context.seq_len - recent_window):
                        return "L0"
            
            # Get memory pressure from agent
            metrics = self._call_agent_api("/v1/metrics")
            if metrics:
                host_gb = metrics.get("host_gb", 0)
                # Simple heuristic based on memory usage
                if host_gb > 20:  # High memory usage
                    return "L2"  # INT4
                elif host_gb > 15:  # Medium memory usage
                    return "L1"  # INT8
                else:
                    return "L0"  # FP16
            
            # Default to INT8 if no metrics available
            return "L1"
            
        except Exception as e:
            self.logger.error(f"Error determining quantization level: {e}")
            return "L0"  # Safe fallback
    
    def postrun_hook(self, stats: InferenceStats):
        """
        PostRun hook - called after inference execution
        
        Args:
            stats: Inference execution statistics
        """
        with self.lock:
            try:
                # Add to history
                self.inference_stats.append(stats)
                if len(self.inference_stats) > self.max_stats_history:
                    self.inference_stats.pop(0)
                
                # Send metrics to agent
                metrics_data = {
                    "tps": stats.tps,
                    "ftl_ms": stats.first_token_latency,
                    "total_latency_ms": stats.total_latency,
                    "tokens": stats.tokens_generated,
                    "memory_peak_gb": stats.memory_peak,
                    "quality_score": stats.quality_score,
                    "timestamp": stats.end_time
                }
                
                self._call_agent_api("/v1/metrics/push", "POST", metrics_data)
                
                self.logger.info(f"PostRun: TPS={stats.tps:.1f}, FTL={stats.first_token_latency:.1f}ms")
                
            except Exception as e:
                self.logger.error(f"PostRun hook error: {e}")
    
    def apply_ort_policy(self, session_options, run_options):
        """
        Apply current policy to ORT session and run options
        
        Args:
            session_options: ORT SessionOptions object
            run_options: ORT RunOptions object
        """
        try:
            policy = self.current_policy
            
            # Apply IO policy
            io_policy = policy.get("io", {})
            if io_policy.get("enable_iobinding", False):
                # Enable IOBinding optimizations
                # In real implementation, would configure ORT IOBinding
                self.logger.debug("IOBinding enabled")
            
            # Apply scheduler policy
            scheduler_policy = policy.get("scheduler", {})
            mode = scheduler_policy.get("mode", "hybrid")
            
            if mode == "hybrid":
                # Configure hybrid NPU+iGPU execution
                # In real implementation, would set execution providers
                self.logger.debug("Hybrid execution mode enabled")
            elif mode == "gpu_only":
                # GPU-only execution
                self.logger.debug("GPU-only execution mode")
            elif mode == "npu_only":
                # NPU-only execution
                self.logger.debug("NPU-only execution mode")
            
            # Apply threading and optimization settings
            # session_options.intra_op_num_threads = policy.get("threads", 4)
            # session_options.graph_optimization_level = policy.get("opt_level", "all")
            
        except Exception as e:
            self.logger.error(f"Error applying ORT policy: {e}")
    
    @contextmanager
    def inference_session(self, context: InferenceContext):
        """
        Context manager for inference session with hooks
        
        Usage:
            with adapter.inference_session(context) as session:
                # Run inference
                result = session.run(...)
        """
        start_time = time.time()
        
        try:
            # PreRun hook
            policy = self.prerun_hook(context)
            
            # Yield control to inference code
            yield self
            
        finally:
            # PostRun hook (placeholder stats)
            end_time = time.time()
            stats = InferenceStats(
                start_time=start_time,
                end_time=end_time,
                tokens_generated=50,  # Would be actual count
                first_token_latency=(end_time - start_time) * 200,  # Estimate
                total_latency=(end_time - start_time) * 1000,
                memory_peak=16.0,  # Would be actual measurement
                quality_score=0.95
            )
            
            self.postrun_hook(stats)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from recent inference runs"""
        with self.lock:
            if not self.inference_stats:
                return {"message": "No inference statistics available"}
            
            recent_stats = self.inference_stats[-10:]  # Last 10 runs
            
            avg_tps = sum(s.tps for s in recent_stats) / len(recent_stats)
            avg_ftl = sum(s.first_token_latency for s in recent_stats) / len(recent_stats)
            avg_quality = sum(s.quality_score for s in recent_stats) / len(recent_stats)
            
            return {
                "runs": len(recent_stats),
                "avg_tps": avg_tps,
                "avg_ftl_ms": avg_ftl,
                "avg_quality": avg_quality,
                "kv_cache_entries": len(self.kv_cache),
                "quantization_enabled": self.enable_kv_quantization
            }
    
    def clear_kv_cache(self):
        """Clear KV cache"""
        with self.lock:
            self.kv_cache.clear()
            self.logger.info("KV cache cleared")

# Example usage and testing
def example_usage():
    """Example of how to use the Lemonade Adapter"""
    
    # Initialize adapter
    adapter = LemonadeAdapter(
        agent_url="http://127.0.0.1:7031",
        enable_kv_quantization=True
    )
    
    # Create inference context
    context = InferenceContext(
        model_name="llama-7b",
        seq_len=1024,
        batch_size=1,
        target_ftl_ms=250,
        quality_budget=0.8
    )
    
    # Use context manager for inference
    with adapter.inference_session(context) as session:
        print("🚀 Running inference with Infer-OS optimization")
        
        # Simulate KV cache operations
        kv_tensor = np.random.randn(16, 128, 64).astype(np.float32)
        
        # Write to KV cache (with quantization)
        quantized = session.kv_write_hook("layer12.head3.pos0-127", kv_tensor)
        print(f"📝 KV Write: {quantized['meta']['level']} quantization applied")
        
        # Read from KV cache (with dequantization)
        dequantized = session.kv_read_hook("layer12.head3.pos0-127")
        print(f"📖 KV Read: Shape {dequantized.shape}, MSE: {np.mean((kv_tensor - dequantized)**2):.6f}")
        
        # Simulate inference delay
        time.sleep(0.1)
    
    # Get performance summary
    summary = adapter.get_performance_summary()
    print(f"📊 Performance: {summary}")

if __name__ == "__main__":
    # Run example
    example_usage()

