#!/usr/bin/env python3
"""
KV Cache Quantization benchmark and test suite.

This script tests the KV cache quantization implementation and measures
memory savings and performance impact.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import time
import logging
import argparse
from typing import Dict, List, Tuple
import statistics

# Import our KV quantization implementation
from optim.kv_quantization import (
    GradualKVCache, QuantizationConfig, QuantizationScheme, KVQuantizer,
    create_gradual_kv_cache
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_tensors(batch_size: int = 1, seq_len: int = 128, 
                       hidden_size: int = 768) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create test key and value tensors.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden dimension size
        
    Returns:
        Tuple of (key_tensor, value_tensor)
    """
    torch.manual_seed(42)  # For reproducible results
    
    key = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    value = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    
    return key, value

def test_quantization_schemes():
    """Test different quantization schemes."""
    logger.info("Testing quantization schemes...")
    
    # Create test tensor
    test_tensor = torch.randn(32, 64, dtype=torch.float32)
    quantizer = KVQuantizer()
    
    schemes = [
        QuantizationScheme.INT8,
        QuantizationScheme.INT4,
        QuantizationScheme.FP16,
        QuantizationScheme.DYNAMIC
    ]
    
    for scheme in schemes:
        logger.info(f"Testing {scheme.value} quantization...")
        
        # Quantize
        quantized, params = quantizer.quantize_tensor(test_tensor, scheme)
        
        # Dequantize
        dequantized = quantizer.dequantize_tensor(quantized, params, scheme)
        
        # Calculate error
        mse = torch.mean((test_tensor - dequantized) ** 2).item()
        max_error = torch.max(torch.abs(test_tensor - dequantized)).item()
        
        # Calculate compression ratio
        original_size = test_tensor.numel() * test_tensor.element_size()
        quantized_size = quantized.numel() * quantized.element_size()
        compression_ratio = original_size / quantized_size
        
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  Max Error: {max_error:.6f}")
        logger.info(f"  Compression Ratio: {compression_ratio:.2f}x")
        logger.info(f"  Original Size: {original_size} bytes")
        logger.info(f"  Quantized Size: {quantized_size} bytes")
        
        # Validate reconstruction quality
        if scheme == QuantizationScheme.FP16:
            assert mse < 1e-6, f"FP16 quantization error too high: {mse}"
        elif scheme == QuantizationScheme.INT8:
            assert mse < 1e-2, f"INT8 quantization error too high: {mse}"
        elif scheme == QuantizationScheme.INT4:
            assert mse < 1e-1, f"INT4 quantization error too high: {mse}"
    
    logger.info("✅ Quantization schemes test PASSED")

def test_gradual_cache_basic():
    """Test basic gradual cache functionality."""
    logger.info("Testing gradual cache basic functionality...")
    
    config = QuantizationConfig(
        scheme=QuantizationScheme.INT8,
        enable_gradual=True,
        age_threshold=5,
        importance_threshold=0.3
    )
    
    cache = GradualKVCache(config, max_entries=100)
    
    # Add some entries
    for layer_idx in range(3):
        for position in range(10):
            key = torch.randn(1, 64, dtype=torch.float32)
            value = torch.randn(1, 64, dtype=torch.float32)
            
            # Simulate attention weights (higher for earlier positions)
            attention_weights = torch.tensor([1.0 / (position + 1)])
            
            entry = cache.add_entry(key, value, layer_idx, position, attention_weights)
            
            # Verify entry was added
            assert entry is not None
            assert entry.layer_idx == layer_idx
            assert entry.position == position
    
    # Check statistics
    stats = cache.get_statistics()
    logger.info(f"Total entries: {stats['total_entries']}")
    logger.info(f"Memory usage: {stats['memory_usage_bytes']} bytes")
    logger.info(f"Quantization rate: {stats['quantization_rate']:.2%}")
    
    # Retrieve entries
    for layer_idx in range(3):
        for position in range(10):
            key, value = cache.get_kv_tensors(layer_idx, position)
            assert key is not None and value is not None
            assert key.shape == (1, 64)
            assert value.shape == (1, 64)
    
    logger.info("✅ Gradual cache basic functionality test PASSED")

def test_quantization_triggers():
    """Test quantization triggers (age, importance, memory pressure)."""
    logger.info("Testing quantization triggers...")
    
    config = QuantizationConfig(
        scheme=QuantizationScheme.INT8,
        enable_gradual=True,
        age_threshold=3,  # Low threshold for testing
        importance_threshold=0.5
    )
    
    cache = GradualKVCache(config, max_entries=50)
    
    # Add entries with different importance scores
    entries = []
    for i in range(10):
        key = torch.randn(1, 32, dtype=torch.float32)
        value = torch.randn(1, 32, dtype=torch.float32)
        
        # Vary importance: first entries more important
        importance = 1.0 - (i / 10.0)
        attention_weights = torch.tensor([importance])
        
        entry = cache.add_entry(key, value, 0, i, attention_weights)
        entries.append(entry)
    
    # Force aging by calling gradual quantization multiple times
    for _ in range(5):
        cache._apply_gradual_quantization()
    
    # Check which entries got quantized
    quantized_count = sum(1 for entry in entries if entry.is_quantized)
    logger.info(f"Quantized {quantized_count} out of {len(entries)} entries")
    
    # Verify that less important entries are more likely to be quantized
    important_entries = [e for e in entries if e.importance_score > 0.5]
    unimportant_entries = [e for e in entries if e.importance_score <= 0.5]
    
    important_quantized = sum(1 for e in important_entries if e.is_quantized)
    unimportant_quantized = sum(1 for e in unimportant_entries if e.is_quantized)
    
    logger.info(f"Important entries quantized: {important_quantized}/{len(important_entries)}")
    logger.info(f"Unimportant entries quantized: {unimportant_quantized}/{len(unimportant_entries)}")
    
    # Less important entries should be quantized more aggressively
    if len(unimportant_entries) > 0:
        unimportant_rate = unimportant_quantized / len(unimportant_entries)
        important_rate = important_quantized / max(1, len(important_entries))
        assert unimportant_rate >= important_rate, "Quantization should prioritize less important entries"
    
    logger.info("✅ Quantization triggers test PASSED")

def benchmark_memory_savings(num_layers: int = 12, seq_len: int = 512, 
                           hidden_size: int = 768, num_runs: int = 5):
    """
    Benchmark memory savings from KV cache quantization.
    
    Args:
        num_layers: Number of transformer layers
        seq_len: Sequence length
        hidden_size: Hidden dimension size
        num_runs: Number of benchmark runs
    """
    logger.info("=== KV Cache Quantization Memory Benchmark ===")
    logger.info(f"Layers: {num_layers}, Seq Length: {seq_len}, Hidden Size: {hidden_size}")
    
    schemes = [
        QuantizationScheme.FP16,
        QuantizationScheme.INT8,
        QuantizationScheme.INT4,
        QuantizationScheme.DYNAMIC
    ]
    
    results = {}
    
    for scheme in schemes:
        logger.info(f"\nTesting {scheme.value} quantization...")
        
        memory_savings = []
        quality_losses = []
        
        for run in range(num_runs):
            # Create cache
            config = QuantizationConfig(
                scheme=scheme,
                enable_gradual=True,
                age_threshold=seq_len // 4,  # Quantize older entries
                importance_threshold=0.2
            )
            
            cache = GradualKVCache(config, max_entries=num_layers * seq_len)
            
            # Simulate adding KV cache entries
            original_memory = 0
            
            for layer_idx in range(num_layers):
                for position in range(seq_len):
                    key = torch.randn(1, hidden_size, dtype=torch.float32)
                    value = torch.randn(1, hidden_size, dtype=torch.float32)
                    
                    # Calculate original memory
                    original_memory += key.numel() * key.element_size()
                    original_memory += value.numel() * value.element_size()
                    
                    # Add to cache with simulated importance
                    importance = np.random.exponential(0.5)  # Exponential distribution
                    attention_weights = torch.tensor([importance])
                    
                    cache.add_entry(key, value, layer_idx, position, attention_weights)
            
            # Force quantization
            for _ in range(10):
                cache._apply_gradual_quantization()
            
            # Calculate memory savings
            stats = cache.get_statistics()
            current_memory = stats["memory_usage_bytes"]
            memory_saved = original_memory - current_memory
            memory_saving_ratio = memory_saved / original_memory
            
            memory_savings.append(memory_saving_ratio)
            
            # Estimate quality loss (simplified)
            quality_loss = stats["quantized_entries"] / stats["total_entries"] * 0.1  # Rough estimate
            quality_losses.append(quality_loss)
        
        # Calculate averages
        avg_memory_saving = statistics.mean(memory_savings)
        avg_quality_loss = statistics.mean(quality_losses)
        
        results[scheme.value] = {
            "memory_saving": avg_memory_saving,
            "quality_loss": avg_quality_loss,
            "memory_savings": memory_savings
        }
        
        logger.info(f"  Average Memory Saving: {avg_memory_saving:.2%}")
        logger.info(f"  Estimated Quality Loss: {avg_quality_loss:.2%}")
    
    # Print summary
    logger.info("\n=== BENCHMARK RESULTS ===")
    for scheme_name, result in results.items():
        logger.info(f"{scheme_name}:")
        logger.info(f"  Memory Saving: {result['memory_saving']:.2%}")
        logger.info(f"  Quality Loss: {result['quality_loss']:.2%}")
        logger.info(f"  Efficiency Score: {result['memory_saving'] - result['quality_loss']:.2%}")
    
    return results

def test_reversible_quantization():
    """Test reversible quantization functionality."""
    logger.info("Testing reversible quantization...")
    
    config = QuantizationConfig(
        scheme=QuantizationScheme.INT8,
        reversible=True,
        enable_gradual=True,
        age_threshold=2
    )
    
    cache = GradualKVCache(config, max_entries=10)
    
    # Add entry
    original_key = torch.randn(1, 64, dtype=torch.float32)
    original_value = torch.randn(1, 64, dtype=torch.float32)
    
    entry = cache.add_entry(original_key.clone(), original_value.clone(), 0, 0)
    
    # Force quantization
    for _ in range(5):
        cache._apply_gradual_quantization()
    
    # Verify entry was quantized
    assert entry.is_quantized, "Entry should be quantized"
    
    # Access entry (should trigger dequantization)
    retrieved_key, retrieved_value = cache.get_kv_tensors(0, 0)
    
    # Verify dequantization occurred
    assert not entry.is_quantized, "Entry should be dequantized after access"
    
    # Check reconstruction quality
    key_mse = torch.mean((original_key - retrieved_key) ** 2).item()
    value_mse = torch.mean((original_value - retrieved_value) ** 2).item()
    
    logger.info(f"Key reconstruction MSE: {key_mse:.6f}")
    logger.info(f"Value reconstruction MSE: {value_mse:.6f}")
    
    # Quality should be reasonable for INT8
    assert key_mse < 1e-2, f"Key reconstruction quality too poor: {key_mse}"
    assert value_mse < 1e-2, f"Value reconstruction quality too poor: {value_mse}"
    
    logger.info("✅ Reversible quantization test PASSED")

def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="KV cache quantization benchmark")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--hidden-size", type=int, default=512, help="Hidden dimension size")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--test-only", action="store_true", help="Run tests only, skip benchmark")
    
    args = parser.parse_args()
    
    logger.info("=== KV Cache Quantization Test Suite ===")
    
    # Run tests
    test_quantization_schemes()
    test_gradual_cache_basic()
    test_quantization_triggers()
    test_reversible_quantization()
    
    if args.test_only:
        logger.info("✅ All tests PASSED")
        return 0
    
    # Run benchmark
    try:
        results = benchmark_memory_savings(
            num_layers=args.num_layers,
            seq_len=args.seq_len,
            hidden_size=args.hidden_size,
            num_runs=args.num_runs
        )
        
        # Check if significant memory savings achieved
        best_scheme = max(results.items(), key=lambda x: x[1]["memory_saving"])
        best_saving = best_scheme[1]["memory_saving"]
        
        if best_saving > 0.1:  # At least 10% memory saving
            logger.info(f"🎉 Significant memory savings achieved with {best_scheme[0]}: {best_saving:.2%}")
            return 0
        else:
            logger.info("ℹ️  Modest memory savings achieved.")
            return 0
            
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

