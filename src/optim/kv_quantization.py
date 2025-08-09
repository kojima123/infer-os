"""
KV Cache Gradual Quantization for Infer-OS.

This module implements progressive quantization of KV cache to reduce memory usage
while maintaining quality through reversible quantization schemes.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)

class QuantizationScheme(Enum):
    """Quantization schemes for KV cache."""
    INT8 = "int8"
    INT4 = "int4"
    FP16 = "fp16"
    DYNAMIC = "dynamic"

@dataclass
class QuantizationConfig:
    """Configuration for KV cache quantization."""
    scheme: QuantizationScheme = QuantizationScheme.INT8
    enable_gradual: bool = True  # Enable gradual quantization
    age_threshold: int = 10  # Age threshold for quantization
    importance_threshold: float = 0.1  # Importance threshold
    reversible: bool = True  # Enable reversible quantization
    memory_pressure_threshold: float = 0.8  # Memory pressure threshold
    quality_preservation: bool = True  # Preserve quality for important tokens

class KVCacheEntry:
    """Single entry in KV cache with quantization metadata."""
    
    def __init__(self, key: torch.Tensor, value: torch.Tensor, 
                 layer_idx: int, position: int):
        """
        Initialize KV cache entry.
        
        Args:
            key: Key tensor
            value: Value tensor
            layer_idx: Layer index
            position: Position in sequence
        """
        self.key = key
        self.value = value
        self.layer_idx = layer_idx
        self.position = position
        self.age = 0
        self.access_count = 0
        self.importance_score = 1.0
        self.is_quantized = False
        self.quantization_scheme = None
        self.quantization_params = {}
        self.creation_time = time.time()
        self.last_access_time = time.time()
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_access_time = time.time()
    
    def update_age(self):
        """Update age of the entry."""
        self.age += 1
    
    def calculate_importance(self, attention_weights: Optional[torch.Tensor] = None) -> float:
        """
        Calculate importance score for this cache entry.
        
        Args:
            attention_weights: Attention weights for this position
            
        Returns:
            Importance score
        """
        # Base importance from access patterns
        access_importance = math.log(1 + self.access_count)
        
        # Recency importance (newer entries are more important)
        recency_importance = 1.0 / (1.0 + self.age * 0.1)
        
        # Position importance (earlier positions often more important)
        position_importance = 1.0 / (1.0 + self.position * 0.01)
        
        # Attention-based importance
        attention_importance = 1.0
        if attention_weights is not None:
            attention_importance = float(attention_weights.mean().item())
        
        # Combined importance score
        self.importance_score = (
            0.3 * access_importance +
            0.3 * recency_importance +
            0.2 * position_importance +
            0.2 * attention_importance
        )
        
        return self.importance_score

class KVQuantizer:
    """KV cache quantizer with multiple quantization schemes."""
    
    def __init__(self, config: QuantizationConfig = None):
        """
        Initialize KV quantizer.
        
        Args:
            config: Quantization configuration
        """
        self.config = config or QuantizationConfig()
        self.quantization_stats = {
            "total_entries": 0,
            "quantized_entries": 0,
            "memory_saved": 0,
            "quality_loss": 0.0
        }
    
    def quantize_tensor(self, tensor: torch.Tensor, 
                       scheme: QuantizationScheme) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Quantize tensor using specified scheme.
        
        Args:
            tensor: Input tensor to quantize
            scheme: Quantization scheme
            
        Returns:
            Tuple of (quantized_tensor, quantization_params)
        """
        if scheme == QuantizationScheme.INT8:
            return self._quantize_int8(tensor)
        elif scheme == QuantizationScheme.INT4:
            return self._quantize_int4(tensor)
        elif scheme == QuantizationScheme.FP16:
            return self._quantize_fp16(tensor)
        elif scheme == QuantizationScheme.DYNAMIC:
            return self._quantize_dynamic(tensor)
        else:
            raise ValueError(f"Unsupported quantization scheme: {scheme}")
    
    def _quantize_int8(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize tensor to INT8."""
        # Calculate scale and zero point
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        # Avoid division by zero
        if max_val == min_val:
            scale = 1.0
            zero_point = 0
        else:
            scale = (max_val - min_val) / 255.0
            zero_point = int(-min_val / scale)
        
        # Quantize
        quantized = torch.round(tensor / scale + zero_point).clamp(0, 255).to(torch.uint8)
        
        params = {
            "scale": scale,
            "zero_point": zero_point,
            "min_val": min_val,
            "max_val": max_val
        }
        
        return quantized, params
    
    def _quantize_int4(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize tensor to INT4 (stored as INT8)."""
        # Calculate scale and zero point for 4-bit range
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        if max_val == min_val:
            scale = 1.0
            zero_point = 0
        else:
            scale = (max_val - min_val) / 15.0  # 4-bit range: 0-15
            zero_point = int(-min_val / scale)
        
        # Quantize to 4-bit range but store as uint8
        quantized = torch.round(tensor / scale + zero_point).clamp(0, 15).to(torch.uint8)
        
        params = {
            "scale": scale,
            "zero_point": zero_point,
            "min_val": min_val,
            "max_val": max_val,
            "bits": 4
        }
        
        return quantized, params
    
    def _quantize_fp16(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize tensor to FP16."""
        quantized = tensor.to(torch.float16)
        params = {"original_dtype": tensor.dtype}
        return quantized, params
    
    def _quantize_dynamic(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Dynamic quantization based on tensor characteristics."""
        # Analyze tensor to choose best quantization
        tensor_range = tensor.max() - tensor.min()
        tensor_std = tensor.std()
        
        # Choose quantization scheme based on characteristics
        if tensor_range < 0.1 and tensor_std < 0.05:
            # Small range and low variance -> INT4
            return self._quantize_int4(tensor)
        elif tensor_range < 1.0:
            # Medium range -> INT8
            return self._quantize_int8(tensor)
        else:
            # Large range -> FP16
            return self._quantize_fp16(tensor)
    
    def dequantize_tensor(self, quantized_tensor: torch.Tensor, 
                         params: Dict[str, Any],
                         scheme: QuantizationScheme) -> torch.Tensor:
        """
        Dequantize tensor back to original precision.
        
        Args:
            quantized_tensor: Quantized tensor
            params: Quantization parameters
            scheme: Quantization scheme used
            
        Returns:
            Dequantized tensor
        """
        if scheme == QuantizationScheme.INT8:
            return self._dequantize_int8(quantized_tensor, params)
        elif scheme == QuantizationScheme.INT4:
            return self._dequantize_int4(quantized_tensor, params)
        elif scheme == QuantizationScheme.FP16:
            return self._dequantize_fp16(quantized_tensor, params)
        elif scheme == QuantizationScheme.DYNAMIC:
            # Determine original scheme from params
            if "bits" in params and params["bits"] == 4:
                return self._dequantize_int4(quantized_tensor, params)
            elif "original_dtype" in params:
                return self._dequantize_fp16(quantized_tensor, params)
            else:
                return self._dequantize_int8(quantized_tensor, params)
        else:
            raise ValueError(f"Unsupported quantization scheme: {scheme}")
    
    def _dequantize_int8(self, quantized_tensor: torch.Tensor, 
                        params: Dict[str, Any]) -> torch.Tensor:
        """Dequantize INT8 tensor."""
        scale = params["scale"]
        zero_point = params["zero_point"]
        
        # Dequantize
        dequantized = (quantized_tensor.to(torch.float32) - zero_point) * scale
        return dequantized
    
    def _dequantize_int4(self, quantized_tensor: torch.Tensor, 
                        params: Dict[str, Any]) -> torch.Tensor:
        """Dequantize INT4 tensor."""
        scale = params["scale"]
        zero_point = params["zero_point"]
        
        # Dequantize
        dequantized = (quantized_tensor.to(torch.float32) - zero_point) * scale
        return dequantized
    
    def _dequantize_fp16(self, quantized_tensor: torch.Tensor, 
                        params: Dict[str, Any]) -> torch.Tensor:
        """Dequantize FP16 tensor."""
        original_dtype = params.get("original_dtype", torch.float32)
        return quantized_tensor.to(original_dtype)

class GradualKVCache:
    """
    KV cache with gradual quantization capabilities.
    
    This class manages KV cache entries and applies quantization based on
    age, importance, and memory pressure.
    """
    
    def __init__(self, config: QuantizationConfig = None, max_entries: int = 1000):
        """
        Initialize gradual KV cache.
        
        Args:
            config: Quantization configuration
            max_entries: Maximum number of cache entries
        """
        self.config = config or QuantizationConfig()
        self.max_entries = max_entries
        self.quantizer = KVQuantizer(config)
        
        # Cache storage: layer_idx -> position -> KVCacheEntry
        self.cache: Dict[int, Dict[int, KVCacheEntry]] = {}
        self.memory_usage = 0
        self.total_memory_limit = None
        
        self.stats = {
            "total_entries": 0,
            "quantized_entries": 0,
            "memory_saved_bytes": 0,
            "quantization_operations": 0,
            "dequantization_operations": 0
        }
    
    def add_entry(self, key: torch.Tensor, value: torch.Tensor, 
                  layer_idx: int, position: int,
                  attention_weights: Optional[torch.Tensor] = None) -> KVCacheEntry:
        """
        Add new KV cache entry.
        
        Args:
            key: Key tensor
            value: Value tensor
            layer_idx: Layer index
            position: Position in sequence
            attention_weights: Attention weights for importance calculation
            
        Returns:
            Created cache entry
        """
        # Create new entry
        entry = KVCacheEntry(key, value, layer_idx, position)
        
        # Calculate importance
        entry.calculate_importance(attention_weights)
        
        # Add to cache
        if layer_idx not in self.cache:
            self.cache[layer_idx] = {}
        
        self.cache[layer_idx][position] = entry
        self.stats["total_entries"] += 1
        
        # Update memory usage
        self.memory_usage += self._calculate_entry_memory(entry)
        
        # Check if gradual quantization should be applied
        if self.config.enable_gradual:
            self._apply_gradual_quantization()
        
        return entry
    
    def get_entry(self, layer_idx: int, position: int) -> Optional[KVCacheEntry]:
        """
        Get KV cache entry and update access statistics.
        
        Args:
            layer_idx: Layer index
            position: Position in sequence
            
        Returns:
            Cache entry if found, None otherwise
        """
        if layer_idx in self.cache and position in self.cache[layer_idx]:
            entry = self.cache[layer_idx][position]
            entry.update_access()
            
            # Dequantize if needed
            if entry.is_quantized and self.config.reversible:
                self._dequantize_entry(entry)
            
            return entry
        
        return None
    
    def get_kv_tensors(self, layer_idx: int, position: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get key and value tensors for specified position.
        
        Args:
            layer_idx: Layer index
            position: Position in sequence
            
        Returns:
            Tuple of (key_tensor, value_tensor)
        """
        entry = self.get_entry(layer_idx, position)
        if entry:
            return entry.key, entry.value
        return None, None
    
    def _calculate_entry_memory(self, entry: KVCacheEntry) -> int:
        """Calculate memory usage of cache entry in bytes."""
        key_memory = entry.key.numel() * entry.key.element_size()
        value_memory = entry.value.numel() * entry.value.element_size()
        return key_memory + value_memory
    
    def _apply_gradual_quantization(self):
        """Apply gradual quantization based on age and importance."""
        # Update ages
        for layer_cache in self.cache.values():
            for entry in layer_cache.values():
                entry.update_age()
        
        # Check memory pressure
        memory_pressure = self._calculate_memory_pressure()
        
        # Find candidates for quantization
        candidates = []
        for layer_idx, layer_cache in self.cache.items():
            for position, entry in layer_cache.items():
                if not entry.is_quantized and self._should_quantize(entry, memory_pressure):
                    candidates.append(entry)
        
        # Sort by importance (quantize less important first)
        candidates.sort(key=lambda e: e.importance_score)
        
        # Quantize candidates
        for entry in candidates:
            self._quantize_entry(entry)
    
    def _should_quantize(self, entry: KVCacheEntry, memory_pressure: float) -> bool:
        """
        Determine if entry should be quantized.
        
        Args:
            entry: Cache entry to check
            memory_pressure: Current memory pressure (0-1)
            
        Returns:
            True if entry should be quantized
        """
        # Age-based quantization
        if entry.age >= self.config.age_threshold:
            return True
        
        # Importance-based quantization
        if entry.importance_score < self.config.importance_threshold:
            return True
        
        # Memory pressure-based quantization
        if memory_pressure > self.config.memory_pressure_threshold:
            # More aggressive quantization under memory pressure
            if entry.age >= self.config.age_threshold // 2:
                return True
        
        return False
    
    def _calculate_memory_pressure(self) -> float:
        """Calculate current memory pressure (0-1)."""
        if self.total_memory_limit is None:
            # Estimate based on current usage and max entries
            estimated_limit = self.memory_usage * (self.max_entries / max(1, self.stats["total_entries"]))
            return min(1.0, self.memory_usage / estimated_limit)
        else:
            return min(1.0, self.memory_usage / self.total_memory_limit)
    
    def _quantize_entry(self, entry: KVCacheEntry):
        """Quantize a cache entry."""
        if entry.is_quantized:
            return
        
        # Choose quantization scheme
        scheme = self.config.scheme
        if scheme == QuantizationScheme.DYNAMIC:
            # Dynamic scheme selection based on entry characteristics
            if entry.importance_score > 0.7:
                scheme = QuantizationScheme.FP16  # High importance -> less aggressive
            elif entry.importance_score > 0.3:
                scheme = QuantizationScheme.INT8  # Medium importance -> moderate
            else:
                scheme = QuantizationScheme.INT4  # Low importance -> aggressive
        
        # Store original memory usage
        original_memory = self._calculate_entry_memory(entry)
        
        # Quantize key and value
        quantized_key, key_params = self.quantizer.quantize_tensor(entry.key, scheme)
        quantized_value, value_params = self.quantizer.quantize_tensor(entry.value, scheme)
        
        # Update entry
        entry.key = quantized_key
        entry.value = quantized_value
        entry.is_quantized = True
        entry.quantization_scheme = scheme
        entry.quantization_params = {
            "key_params": key_params,
            "value_params": value_params
        }
        
        # Update statistics
        new_memory = self._calculate_entry_memory(entry)
        memory_saved = original_memory - new_memory
        
        self.stats["quantized_entries"] += 1
        self.stats["memory_saved_bytes"] += memory_saved
        self.stats["quantization_operations"] += 1
        self.memory_usage -= memory_saved
        
        logger.debug(f"Quantized entry at layer {entry.layer_idx}, position {entry.position} "
                    f"with {scheme.value}, saved {memory_saved} bytes")
    
    def _dequantize_entry(self, entry: KVCacheEntry):
        """Dequantize a cache entry."""
        if not entry.is_quantized:
            return
        
        # Dequantize key and value
        key_params = entry.quantization_params["key_params"]
        value_params = entry.quantization_params["value_params"]
        
        dequantized_key = self.quantizer.dequantize_tensor(
            entry.key, key_params, entry.quantization_scheme
        )
        dequantized_value = self.quantizer.dequantize_tensor(
            entry.value, value_params, entry.quantization_scheme
        )
        
        # Calculate memory change
        old_memory = self._calculate_entry_memory(entry)
        
        # Update entry
        entry.key = dequantized_key
        entry.value = dequantized_value
        entry.is_quantized = False
        entry.quantization_scheme = None
        entry.quantization_params = {}
        
        # Update statistics
        new_memory = self._calculate_entry_memory(entry)
        memory_increase = new_memory - old_memory
        
        self.stats["dequantization_operations"] += 1
        self.memory_usage += memory_increase
        
        logger.debug(f"Dequantized entry at layer {entry.layer_idx}, position {entry.position}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = self.stats["total_entries"]
        quantized_entries = self.stats["quantized_entries"]
        
        stats = {
            **self.stats,
            "quantization_rate": quantized_entries / max(1, total_entries),
            "memory_usage_bytes": self.memory_usage,
            "memory_pressure": self._calculate_memory_pressure(),
            "avg_memory_per_entry": self.memory_usage / max(1, total_entries)
        }
        
        return stats
    
    def clear_cache(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.memory_usage = 0
        self.stats = {
            "total_entries": 0,
            "quantized_entries": 0,
            "memory_saved_bytes": 0,
            "quantization_operations": 0,
            "dequantization_operations": 0
        }

def create_gradual_kv_cache(scheme: QuantizationScheme = QuantizationScheme.INT8,
                           age_threshold: int = 10,
                           importance_threshold: float = 0.1,
                           max_entries: int = 1000) -> GradualKVCache:
    """
    Factory function to create gradual KV cache.
    
    Args:
        scheme: Quantization scheme
        age_threshold: Age threshold for quantization
        importance_threshold: Importance threshold
        max_entries: Maximum cache entries
        
    Returns:
        Configured gradual KV cache
    """
    config = QuantizationConfig(
        scheme=scheme,
        age_threshold=age_threshold,
        importance_threshold=importance_threshold
    )
    
    return GradualKVCache(config, max_entries)

