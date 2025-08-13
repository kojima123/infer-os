#!/usr/bin/env python3
"""
KV Quantization Engine for Infer-OS GAIA Integration
Advanced KV-cache quantization with reversible multi-level compression
"""

import numpy as np
import logging
import time
import threading
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
import zlib
from collections import defaultdict, OrderedDict

class QuantizationLevel(Enum):
    """Quantization levels for KV cache"""
    L0_FP16 = "L0"      # Full precision FP16
    L1_INT8 = "L1"      # INT8 quantization
    L2_INT4 = "L2"      # INT4 quantization  
    L3_EVICT = "L3"     # Evicted/compressed

@dataclass
class KVChunkMeta:
    """Metadata for KV cache chunk"""
    chunk_id: str
    layer_id: int
    head_id: int
    seq_start: int
    seq_end: int
    level: QuantizationLevel
    shape: Tuple[int, ...]
    dtype: str
    scale: float = 1.0
    zero_point: int = 0
    timestamp: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    importance_score: float = 0.0
    pinned: bool = False
    compressed_size: int = 0
    original_size: int = 0
    
    def update_access(self):
        """Update access statistics"""
        self.last_access = time.time()
        self.access_count += 1

@dataclass
class QuantizationStats:
    """Statistics for quantization operations"""
    total_chunks: int = 0
    level_counts: Dict[str, int] = field(default_factory=lambda: {
        "L0": 0, "L1": 0, "L2": 0, "L3": 0
    })
    total_memory_saved: int = 0
    total_operations: int = 0
    quantization_time: float = 0.0
    dequantization_time: float = 0.0
    compression_ratio: float = 0.0
    quality_loss: float = 0.0

class ImportanceCalculator:
    """Calculate importance scores for KV chunks"""
    
    def __init__(self, alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.2, eta: float = 0.1):
        """
        Initialize importance calculator
        
        Args:
            alpha: Weight for attention score
            beta: Weight for access frequency
            gamma: Weight for semantic relevance
            eta: Weight for age penalty
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.logger = logging.getLogger("ImportanceCalculator")
    
    def calculate_importance(self, 
                           chunk_meta: KVChunkMeta,
                           attention_weights: Optional[np.ndarray] = None,
                           semantic_score: float = 0.5,
                           current_time: float = None) -> float:
        """
        Calculate importance score for KV chunk
        
        I = α·AttnWeight + β·AccessFreq + γ·SemanticRel – η·Age
        
        Args:
            chunk_meta: Chunk metadata
            attention_weights: Attention weights for this chunk
            semantic_score: Semantic relevance score
            current_time: Current timestamp
            
        Returns:
            Importance score (0.0 to 1.0)
        """
        if current_time is None:
            current_time = time.time()
        
        try:
            # Attention weight component
            if attention_weights is not None:
                attn_score = float(np.mean(attention_weights))
            else:
                # Estimate based on position (recent tokens more important)
                seq_len = chunk_meta.seq_end - chunk_meta.seq_start
                position_score = 1.0 - (chunk_meta.seq_start / max(chunk_meta.seq_end, 1))
                attn_score = min(position_score, 1.0)
            
            # Access frequency component (normalized)
            max_access = max(chunk_meta.access_count, 1)
            freq_score = min(chunk_meta.access_count / max_access, 1.0)
            
            # Semantic relevance component
            semantic_score = max(0.0, min(semantic_score, 1.0))
            
            # Age penalty component
            age_seconds = current_time - chunk_meta.timestamp
            age_score = np.exp(-age_seconds / 3600.0)  # Decay over 1 hour
            
            # Combined importance score
            importance = (
                self.alpha * attn_score +
                self.beta * freq_score +
                self.gamma * semantic_score -
                self.eta * (1.0 - age_score)
            )
            
            # Clamp to [0, 1]
            importance = max(0.0, min(importance, 1.0))
            
            return importance
            
        except Exception as e:
            self.logger.error(f"Error calculating importance: {e}")
            return 0.5  # Default middle importance

class KVQuantizationEngine:
    """
    Advanced KV-cache quantization engine with reversible multi-level compression
    
    Features:
    - 4-level quantization (FP16/INT8/INT4/Evict)
    - Importance-based quantization decisions
    - Reversible quantization with quality preservation
    - Memory pressure adaptation
    - Performance monitoring
    """
    
    def __init__(self, 
                 recent_window: int = 64,
                 max_cache_size: int = 10000,
                 quality_threshold: float = 0.5):
        """
        Initialize KV quantization engine
        
        Args:
            recent_window: Number of recent tokens to keep in FP16
            max_cache_size: Maximum number of chunks in cache
            quality_threshold: Minimum quality threshold for quantization
        """
        self.recent_window = recent_window
        self.max_cache_size = max_cache_size
        self.quality_threshold = quality_threshold
        
        # Core components
        self.importance_calc = ImportanceCalculator()
        self.logger = self._setup_logging()
        
        # Cache storage
        self.kv_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.metadata_cache: Dict[str, KVChunkMeta] = {}
        
        # Statistics and monitoring
        self.stats = QuantizationStats()
        self.performance_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.level_thresholds = {
            "L1_int8": 0.7,
            "L2_int4": 0.5,
            "L3_evict": 0.3
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        self.logger.info("KV Quantization Engine initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("KVQuantizationEngine")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _generate_chunk_id(self, layer_id: int, head_id: int, seq_start: int, seq_end: int) -> str:
        """Generate unique chunk ID"""
        return f"layer{layer_id}.head{head_id}.pos{seq_start}-{seq_end}"
    
    def _quantize_to_int8(self, tensor: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """Quantize tensor to INT8"""
        tensor_flat = tensor.flatten()
        tensor_max = np.abs(tensor_flat).max()
        
        if tensor_max == 0:
            return np.zeros_like(tensor, dtype=np.int8), 1.0, 0
        
        scale = tensor_max / 127.0
        zero_point = 0
        
        quantized = np.round(tensor / scale).clip(-128, 127).astype(np.int8)
        
        return quantized, scale, zero_point
    
    def _quantize_to_int4(self, tensor: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """Quantize tensor to INT4 (stored as INT8)"""
        tensor_flat = tensor.flatten()
        tensor_max = np.abs(tensor_flat).max()
        
        if tensor_max == 0:
            return np.zeros_like(tensor, dtype=np.int8), 1.0, 0
        
        scale = tensor_max / 7.0  # INT4 range: -8 to 7
        zero_point = 0
        
        quantized = np.round(tensor / scale).clip(-8, 7).astype(np.int8)
        
        return quantized, scale, zero_point
    
    def _dequantize_int8(self, quantized: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
        """Dequantize INT8 tensor"""
        return (quantized.astype(np.float32) - zero_point) * scale
    
    def _dequantize_int4(self, quantized: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
        """Dequantize INT4 tensor"""
        return (quantized.astype(np.float32) - zero_point) * scale
    
    def _compress_tensor(self, tensor: np.ndarray) -> bytes:
        """Compress tensor for L3 eviction"""
        try:
            # Serialize and compress
            tensor_bytes = pickle.dumps(tensor)
            compressed = zlib.compress(tensor_bytes, level=6)
            return compressed
        except Exception as e:
            self.logger.error(f"Compression error: {e}")
            return b""
    
    def _decompress_tensor(self, compressed_data: bytes) -> np.ndarray:
        """Decompress tensor from L3 storage"""
        try:
            decompressed = zlib.decompress(compressed_data)
            tensor = pickle.loads(decompressed)
            return tensor
        except Exception as e:
            self.logger.error(f"Decompression error: {e}")
            return np.array([])
    
    def _decide_quantization_level(self, 
                                 chunk_meta: KVChunkMeta,
                                 memory_pressure: float,
                                 quality_budget: float) -> QuantizationLevel:
        """
        Decide quantization level for KV chunk
        
        Args:
            chunk_meta: Chunk metadata
            memory_pressure: Memory pressure (0.0 to 1.0)
            quality_budget: Quality budget (0.0 to 1.0)
            
        Returns:
            Quantization level to apply
        """
        try:
            # Recent window always stays FP16
            if chunk_meta.pinned or chunk_meta.seq_end >= (chunk_meta.seq_start + self.recent_window):
                return QuantizationLevel.L0_FP16
            
            # Calculate importance score
            importance = self.importance_calc.calculate_importance(chunk_meta)
            chunk_meta.importance_score = importance
            
            # Decision logic based on importance, memory pressure, and quality budget
            if memory_pressure > 0.8 and importance < self.level_thresholds["L3_evict"]:
                return QuantizationLevel.L3_EVICT
            elif memory_pressure > 0.6 and importance < self.level_thresholds["L2_int4"]:
                return QuantizationLevel.L2_INT4
            elif importance < self.level_thresholds["L1_int8"] or quality_budget < 0.3:
                return QuantizationLevel.L1_INT8
            else:
                return QuantizationLevel.L0_FP16
                
        except Exception as e:
            self.logger.error(f"Error deciding quantization level: {e}")
            return QuantizationLevel.L0_FP16  # Safe fallback
    
    def quantize_kv_chunk(self,
                         tensor: np.ndarray,
                         layer_id: int,
                         head_id: int,
                         seq_start: int,
                         seq_end: int,
                         memory_pressure: float = 0.5,
                         quality_budget: float = 0.5,
                         attention_weights: Optional[np.ndarray] = None) -> str:
        """
        Quantize KV chunk and store in cache
        
        Args:
            tensor: KV tensor to quantize
            layer_id: Layer identifier
            head_id: Head identifier  
            seq_start: Sequence start position
            seq_end: Sequence end position
            memory_pressure: Current memory pressure
            quality_budget: Quality budget for quantization
            attention_weights: Optional attention weights
            
        Returns:
            Chunk ID for retrieval
        """
        start_time = time.time()
        
        with self.lock:
            try:
                # Generate chunk ID
                chunk_id = self._generate_chunk_id(layer_id, head_id, seq_start, seq_end)
                
                # Create metadata
                chunk_meta = KVChunkMeta(
                    chunk_id=chunk_id,
                    layer_id=layer_id,
                    head_id=head_id,
                    seq_start=seq_start,
                    seq_end=seq_end,
                    level=QuantizationLevel.L0_FP16,
                    shape=tensor.shape,
                    dtype=str(tensor.dtype),
                    original_size=tensor.nbytes
                )
                
                # Check if in recent window
                if seq_end >= (seq_start + self.recent_window):
                    chunk_meta.pinned = True
                
                # Decide quantization level
                target_level = self._decide_quantization_level(chunk_meta, memory_pressure, quality_budget)
                chunk_meta.level = target_level
                
                # Apply quantization
                if target_level == QuantizationLevel.L0_FP16:
                    # No quantization
                    quantized_data = tensor.astype(np.float16)
                    chunk_meta.scale = 1.0
                    chunk_meta.zero_point = 0
                    
                elif target_level == QuantizationLevel.L1_INT8:
                    # INT8 quantization
                    quantized_data, scale, zero_point = self._quantize_to_int8(tensor)
                    chunk_meta.scale = scale
                    chunk_meta.zero_point = zero_point
                    
                elif target_level == QuantizationLevel.L2_INT4:
                    # INT4 quantization
                    quantized_data, scale, zero_point = self._quantize_to_int4(tensor)
                    chunk_meta.scale = scale
                    chunk_meta.zero_point = zero_point
                    
                elif target_level == QuantizationLevel.L3_EVICT:
                    # Compress and evict
                    quantized_data = self._compress_tensor(tensor)
                    chunk_meta.compressed_size = len(quantized_data)
                    chunk_meta.scale = 1.0
                    chunk_meta.zero_point = 0
                
                # Store in cache
                cache_entry = {
                    "data": quantized_data,
                    "meta": chunk_meta,
                    "timestamp": time.time()
                }
                
                self.kv_cache[chunk_id] = cache_entry
                self.metadata_cache[chunk_id] = chunk_meta
                
                # Update statistics
                self.stats.total_chunks += 1
                self.stats.level_counts[target_level.value] += 1
                self.stats.quantization_time += (time.time() - start_time)
                self.stats.total_operations += 1
                
                # Calculate memory savings
                original_size = tensor.nbytes
                if target_level == QuantizationLevel.L3_EVICT:
                    new_size = chunk_meta.compressed_size
                else:
                    new_size = quantized_data.nbytes
                
                memory_saved = original_size - new_size
                self.stats.total_memory_saved += memory_saved
                
                # Manage cache size
                self._manage_cache_size()
                
                self.logger.debug(f"Quantized {chunk_id} to {target_level.value}, saved {memory_saved} bytes")
                
                return chunk_id
                
            except Exception as e:
                self.logger.error(f"Quantization error: {e}")
                # Fallback: store as FP16
                chunk_id = self._generate_chunk_id(layer_id, head_id, seq_start, seq_end)
                fallback_meta = KVChunkMeta(
                    chunk_id=chunk_id,
                    layer_id=layer_id,
                    head_id=head_id,
                    seq_start=seq_start,
                    seq_end=seq_end,
                    level=QuantizationLevel.L0_FP16,
                    shape=tensor.shape,
                    dtype=str(tensor.dtype)
                )
                
                self.kv_cache[chunk_id] = {
                    "data": tensor.astype(np.float16),
                    "meta": fallback_meta,
                    "timestamp": time.time()
                }
                self.metadata_cache[chunk_id] = fallback_meta
                
                return chunk_id
    
    def dequantize_kv_chunk(self, chunk_id: str) -> Optional[np.ndarray]:
        """
        Dequantize and retrieve KV chunk
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Dequantized tensor or None if not found
        """
        start_time = time.time()
        
        with self.lock:
            try:
                if chunk_id not in self.kv_cache:
                    self.logger.warning(f"Chunk {chunk_id} not found in cache")
                    return None
                
                cache_entry = self.kv_cache[chunk_id]
                quantized_data = cache_entry["data"]
                chunk_meta = cache_entry["meta"]
                
                # Update access statistics
                chunk_meta.update_access()
                
                # Dequantize based on level
                if chunk_meta.level == QuantizationLevel.L0_FP16:
                    # No dequantization needed
                    result = quantized_data.astype(np.float32)
                    
                elif chunk_meta.level == QuantizationLevel.L1_INT8:
                    # Dequantize INT8
                    result = self._dequantize_int8(quantized_data, chunk_meta.scale, chunk_meta.zero_point)
                    
                elif chunk_meta.level == QuantizationLevel.L2_INT4:
                    # Dequantize INT4
                    result = self._dequantize_int4(quantized_data, chunk_meta.scale, chunk_meta.zero_point)
                    
                elif chunk_meta.level == QuantizationLevel.L3_EVICT:
                    # Decompress
                    result = self._decompress_tensor(quantized_data)
                    
                else:
                    self.logger.error(f"Unknown quantization level: {chunk_meta.level}")
                    return None
                
                # Update statistics
                self.stats.dequantization_time += (time.time() - start_time)
                
                # Move to end (LRU)
                self.kv_cache.move_to_end(chunk_id)
                
                self.logger.debug(f"Dequantized {chunk_id} from {chunk_meta.level.value}")
                
                return result
                
            except Exception as e:
                self.logger.error(f"Dequantization error for {chunk_id}: {e}")
                return None
    
    def _manage_cache_size(self):
        """Manage cache size by evicting old entries"""
        try:
            while len(self.kv_cache) > self.max_cache_size:
                # Remove oldest entry (LRU)
                oldest_id, _ = self.kv_cache.popitem(last=False)
                if oldest_id in self.metadata_cache:
                    del self.metadata_cache[oldest_id]
                self.logger.debug(f"Evicted {oldest_id} from cache")
                
        except Exception as e:
            self.logger.error(f"Cache management error: {e}")
    
    def update_quantization_policy(self, 
                                 memory_pressure: float,
                                 quality_budget: float,
                                 recent_window: Optional[int] = None):
        """
        Update quantization policy based on current conditions
        
        Args:
            memory_pressure: Current memory pressure (0.0 to 1.0)
            quality_budget: Quality budget (0.0 to 1.0)
            recent_window: New recent window size
        """
        with self.lock:
            try:
                if recent_window is not None:
                    self.recent_window = recent_window
                
                # Adjust thresholds based on conditions
                if memory_pressure > 0.8:
                    # Aggressive quantization
                    self.level_thresholds = {
                        "L1_int8": 0.8,
                        "L2_int4": 0.6,
                        "L3_evict": 0.4
                    }
                elif memory_pressure > 0.6:
                    # Balanced quantization
                    self.level_thresholds = {
                        "L1_int8": 0.7,
                        "L2_int4": 0.5,
                        "L3_evict": 0.3
                    }
                else:
                    # Conservative quantization
                    self.level_thresholds = {
                        "L1_int8": 0.6,
                        "L2_int4": 0.4,
                        "L3_evict": 0.2
                    }
                
                self.logger.info(f"Updated quantization policy: memory_pressure={memory_pressure:.2f}, quality_budget={quality_budget:.2f}")
                
            except Exception as e:
                self.logger.error(f"Policy update error: {e}")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with self.lock:
            try:
                # Calculate compression ratio
                total_original = sum(meta.original_size for meta in self.metadata_cache.values())
                total_compressed = sum(
                    entry["data"].nbytes if hasattr(entry["data"], "nbytes") else len(entry["data"])
                    for entry in self.kv_cache.values()
                )
                
                compression_ratio = total_compressed / total_original if total_original > 0 else 1.0
                self.stats.compression_ratio = compression_ratio
                
                # Level distribution
                level_distribution = defaultdict(int)
                for meta in self.metadata_cache.values():
                    level_distribution[meta.level.value] += 1
                
                return {
                    "total_chunks": len(self.kv_cache),
                    "level_distribution": dict(level_distribution),
                    "compression_ratio": compression_ratio,
                    "total_memory_saved_mb": self.stats.total_memory_saved / (1024 * 1024),
                    "avg_quantization_time_ms": (self.stats.quantization_time / max(self.stats.total_operations, 1)) * 1000,
                    "avg_dequantization_time_ms": (self.stats.dequantization_time / max(self.stats.total_operations, 1)) * 1000,
                    "cache_hit_rate": self._calculate_hit_rate(),
                    "recent_window": self.recent_window,
                    "level_thresholds": self.level_thresholds
                }
                
            except Exception as e:
                self.logger.error(f"Statistics calculation error: {e}")
                return {"error": str(e)}
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        try:
            total_accesses = sum(meta.access_count for meta in self.metadata_cache.values())
            if total_accesses == 0:
                return 0.0
            
            # Simplified hit rate calculation
            return min(total_accesses / len(self.metadata_cache), 1.0) if self.metadata_cache else 0.0
            
        except Exception as e:
            self.logger.error(f"Hit rate calculation error: {e}")
            return 0.0
    
    def clear_cache(self):
        """Clear all cache entries"""
        with self.lock:
            self.kv_cache.clear()
            self.metadata_cache.clear()
            self.stats = QuantizationStats()
            self.logger.info("Cache cleared")
    
    def export_cache_state(self, filepath: str):
        """Export cache state to file"""
        try:
            state = {
                "metadata": {k: {
                    "chunk_id": v.chunk_id,
                    "layer_id": v.layer_id,
                    "head_id": v.head_id,
                    "seq_start": v.seq_start,
                    "seq_end": v.seq_end,
                    "level": v.level.value,
                    "shape": v.shape,
                    "importance_score": v.importance_score,
                    "access_count": v.access_count
                } for k, v in self.metadata_cache.items()},
                "statistics": {
                    "total_chunks": self.stats.total_chunks,
                    "level_counts": self.stats.level_counts,
                    "compression_ratio": self.stats.compression_ratio,
                    "total_memory_saved": self.stats.total_memory_saved
                },
                "config": {
                    "recent_window": self.recent_window,
                    "level_thresholds": self.level_thresholds
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
                
            self.logger.info(f"Cache state exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Export error: {e}")

# Example usage and testing
def test_kv_quantization():
    """Test KV quantization engine"""
    print("🧪 Testing KV Quantization Engine")
    
    # Initialize engine
    engine = KVQuantizationEngine(recent_window=32, max_cache_size=100)
    
    # Test quantization with different scenarios
    test_cases = [
        {"memory_pressure": 0.3, "quality_budget": 0.9, "desc": "Low pressure, high quality"},
        {"memory_pressure": 0.7, "quality_budget": 0.5, "desc": "Medium pressure, medium quality"},
        {"memory_pressure": 0.9, "quality_budget": 0.2, "desc": "High pressure, low quality"}
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n📊 Test Case {i+1}: {case['desc']}")
        
        # Create test tensor
        test_tensor = np.random.randn(16, 64, 128).astype(np.float32)
        
        # Quantize
        chunk_id = engine.quantize_kv_chunk(
            tensor=test_tensor,
            layer_id=12,
            head_id=8,
            seq_start=i * 64,
            seq_end=(i + 1) * 64,
            memory_pressure=case["memory_pressure"],
            quality_budget=case["quality_budget"]
        )
        
        # Dequantize
        dequantized = engine.dequantize_kv_chunk(chunk_id)
        
        if dequantized is not None:
            # Calculate quality metrics
            mse = np.mean((test_tensor - dequantized) ** 2)
            max_error = np.max(np.abs(test_tensor - dequantized))
            
            print(f"  ✅ Chunk ID: {chunk_id}")
            print(f"  📏 MSE: {mse:.6f}")
            print(f"  📐 Max Error: {max_error:.6f}")
        else:
            print(f"  ❌ Dequantization failed")
    
    # Print statistics
    stats = engine.get_cache_statistics()
    print(f"\n📈 Final Statistics:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Level distribution: {stats['level_distribution']}")
    print(f"  Compression ratio: {stats['compression_ratio']:.3f}")
    print(f"  Memory saved: {stats['total_memory_saved_mb']:.2f} MB")

if __name__ == "__main__":
    test_kv_quantization()

