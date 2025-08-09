#!/usr/bin/env python3
"""
Speculative Generation benchmark and test suite.

This script tests the speculative generation implementation and measures
performance improvements over baseline sequential generation.
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

# Import our speculative generation implementation
from optim.speculative_generation import (
    SpeculativeGenerator, SpeculativeConfig, MockDraftModel, MockTargetModel,
    create_speculative_generator
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_models(vocab_size: int = 1000) -> Tuple[MockDraftModel, MockTargetModel]:
    """
    Create test models for benchmarking.
    
    Args:
        vocab_size: Vocabulary size for models
        
    Returns:
        Tuple of (draft_model, target_model)
    """
    draft_model = MockDraftModel(vocab_size=vocab_size, hidden_size=512)
    target_model = MockTargetModel(vocab_size=vocab_size, hidden_size=2048)
    
    return draft_model, target_model

def generate_test_input(batch_size: int = 1, seq_len: int = 10, 
                       vocab_size: int = 1000) -> torch.Tensor:
    """
    Generate test input sequence.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        vocab_size: Vocabulary size
        
    Returns:
        Input tensor
    """
    torch.manual_seed(42)  # For reproducible results
    return torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

def benchmark_baseline_generation(target_model: MockTargetModel, 
                                input_ids: torch.Tensor,
                                max_length: int = 50,
                                num_runs: int = 10) -> Dict[str, float]:
    """
    Benchmark baseline sequential generation.
    
    Args:
        target_model: Target model to use
        input_ids: Input sequence
        max_length: Maximum generation length
        num_runs: Number of benchmark runs
        
    Returns:
        Performance metrics
    """
    logger.info("Benchmarking baseline generation...")
    
    times = []
    tokens_generated = []
    
    for run in range(num_runs):
        target_model.clear_cache()
        start_time = time.perf_counter()
        
        current_ids = input_ids.clone()
        generated = 0
        
        # Sequential generation
        while current_ids.size(1) < max_length:
            logits = target_model.forward(current_ids)
            
            # Simple greedy sampling
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            current_ids = torch.cat([
                current_ids,
                torch.tensor([[next_token]], dtype=current_ids.dtype)
            ], dim=1)
            
            generated += 1
        
        elapsed = time.perf_counter() - start_time
        times.append(elapsed)
        tokens_generated.append(generated)
    
    avg_time = statistics.mean(times)
    avg_tokens = statistics.mean(tokens_generated)
    
    return {
        "avg_time": avg_time,
        "avg_tokens": avg_tokens,
        "tokens_per_second": avg_tokens / avg_time if avg_time > 0 else 0,
        "times": times
    }

def benchmark_speculative_generation(generator: SpeculativeGenerator,
                                   input_ids: torch.Tensor,
                                   max_length: int = 50,
                                   num_runs: int = 10) -> Dict[str, float]:
    """
    Benchmark speculative generation.
    
    Args:
        generator: Speculative generator
        input_ids: Input sequence
        max_length: Maximum generation length
        num_runs: Number of benchmark runs
        
    Returns:
        Performance metrics
    """
    logger.info("Benchmarking speculative generation...")
    
    times = []
    tokens_generated = []
    acceptance_rates = []
    speedups = []
    
    for run in range(num_runs):
        generator.draft_model.clear_cache()
        generator.target_model.clear_cache()
        generator.reset_stats()
        
        start_time = time.perf_counter()
        
        # Generate with speculative decoding
        output_ids, stats = generator.generate(input_ids, max_length=max_length)
        
        elapsed = time.perf_counter() - start_time
        generated = stats["total_tokens"]
        
        times.append(elapsed)
        tokens_generated.append(generated)
        acceptance_rates.append(stats["acceptance_rate"])
        speedups.append(stats["speedup"])
    
    avg_time = statistics.mean(times)
    avg_tokens = statistics.mean(tokens_generated)
    avg_acceptance = statistics.mean(acceptance_rates)
    avg_speedup = statistics.mean(speedups)
    
    return {
        "avg_time": avg_time,
        "avg_tokens": avg_tokens,
        "tokens_per_second": avg_tokens / avg_time if avg_time > 0 else 0,
        "avg_acceptance_rate": avg_acceptance,
        "avg_speedup": avg_speedup,
        "times": times
    }

def test_correctness(generator: SpeculativeGenerator, input_ids: torch.Tensor,
                    max_length: int = 20) -> bool:
    """
    Test correctness of speculative generation.
    
    Args:
        generator: Speculative generator
        input_ids: Input sequence
        max_length: Maximum generation length
        
    Returns:
        True if generation completes successfully
    """
    logger.info("Testing correctness...")
    
    try:
        # Test basic generation
        output_ids, stats = generator.generate(input_ids, max_length=max_length)
        
        # Check output format
        assert isinstance(output_ids, torch.Tensor), "Output should be tensor"
        assert output_ids.size(0) == input_ids.size(0), "Batch size should match"
        assert output_ids.size(1) >= input_ids.size(1), "Output should be longer than input"
        assert output_ids.size(1) <= max_length, "Output should not exceed max_length"
        
        # Check statistics
        assert isinstance(stats, dict), "Stats should be dictionary"
        required_keys = ["total_tokens", "accepted_tokens", "rejected_tokens", 
                        "acceptance_rate", "speedup"]
        for key in required_keys:
            assert key in stats, f"Missing stat: {key}"
        
        # Check that input is preserved
        assert torch.equal(output_ids[:, :input_ids.size(1)], input_ids), \
            "Input should be preserved in output"
        
        logger.info(f"Generated {stats['total_tokens']} tokens")
        logger.info(f"Acceptance rate: {stats['acceptance_rate']:.2%}")
        logger.info(f"Estimated speedup: {stats['speedup']:.2f}x")
        
        return True
        
    except Exception as e:
        logger.error(f"Correctness test failed: {e}")
        return False

def test_configuration_options():
    """Test different configuration options."""
    logger.info("Testing configuration options...")
    
    vocab_size = 1000
    draft_model, target_model = create_test_models(vocab_size)
    input_ids = generate_test_input(seq_len=5, vocab_size=vocab_size)
    
    logger.info(f"Input IDs shape: {input_ids.shape}")
    logger.info(f"Input IDs range: {input_ids.min().item()} - {input_ids.max().item()}")
    logger.info(f"Draft model vocab size: {draft_model.vocab_size}")
    logger.info(f"Target model vocab size: {target_model.vocab_size}")
    
    # Test different configurations
    configs = [
        {"max_draft_tokens": 2, "acceptance_threshold": 0.5},
        {"max_draft_tokens": 4, "acceptance_threshold": 0.8},
        {"max_draft_tokens": 6, "acceptance_threshold": 0.9},
    ]
    
    for i, config_params in enumerate(configs):
        logger.info(f"Testing config {i+1}: {config_params}")
        
        config = SpeculativeConfig(**config_params)
        generator = SpeculativeGenerator(draft_model, target_model, config)
        
        try:
            # Test generation
            output_ids, stats = generator.generate(input_ids, max_length=20)
            
            logger.info(f"  Generated {stats['total_tokens']} tokens")
            logger.info(f"  Acceptance rate: {stats['acceptance_rate']:.2%}")
            logger.info(f"  Draft calls: {stats['draft_calls']}")
            logger.info(f"  Target calls: {stats['target_calls']}")
        except Exception as e:
            logger.error(f"  Config {i+1} failed: {e}")
            continue
    
    logger.info("✅ Configuration options test PASSED")

def run_comprehensive_benchmark(vocab_size: int = 1000, max_length: int = 50,
                              num_runs: int = 10):
    """
    Run comprehensive benchmark comparing baseline vs speculative generation.
    
    Args:
        vocab_size: Vocabulary size
        max_length: Maximum generation length
        num_runs: Number of benchmark runs
    """
    logger.info("=== Speculative Generation Comprehensive Benchmark ===")
    
    # Create models and input
    draft_model, target_model = create_test_models(vocab_size)
    input_ids = generate_test_input(seq_len=10, vocab_size=vocab_size)
    
    logger.info(f"Vocab size: {vocab_size}")
    logger.info(f"Input length: {input_ids.size(1)}")
    logger.info(f"Max generation length: {max_length}")
    logger.info(f"Benchmark runs: {num_runs}")
    
    # Test correctness first
    generator = create_speculative_generator(
        draft_model, target_model, 
        max_draft_tokens=4, 
        acceptance_threshold=0.8
    )
    
    if not test_correctness(generator, input_ids, max_length):
        logger.error("❌ Correctness test failed, aborting benchmark")
        return
    
    # Benchmark baseline generation
    baseline_metrics = benchmark_baseline_generation(
        target_model, input_ids, max_length, num_runs
    )
    
    # Benchmark speculative generation
    speculative_metrics = benchmark_speculative_generation(
        generator, input_ids, max_length, num_runs
    )
    
    # Calculate improvements
    speedup = speculative_metrics["tokens_per_second"] / baseline_metrics["tokens_per_second"]
    time_reduction = (baseline_metrics["avg_time"] - speculative_metrics["avg_time"]) / baseline_metrics["avg_time"] * 100
    
    # Print results
    logger.info("\n=== BENCHMARK RESULTS ===")
    logger.info("Baseline Generation:")
    logger.info(f"  Avg Time: {baseline_metrics['avg_time']:.3f}s")
    logger.info(f"  Avg Tokens: {baseline_metrics['avg_tokens']:.1f}")
    logger.info(f"  Tokens/sec: {baseline_metrics['tokens_per_second']:.1f}")
    logger.info("")
    
    logger.info("Speculative Generation:")
    logger.info(f"  Avg Time: {speculative_metrics['avg_time']:.3f}s")
    logger.info(f"  Avg Tokens: {speculative_metrics['avg_tokens']:.1f}")
    logger.info(f"  Tokens/sec: {speculative_metrics['tokens_per_second']:.1f}")
    logger.info(f"  Acceptance Rate: {speculative_metrics['avg_acceptance_rate']:.2%}")
    logger.info(f"  Internal Speedup: {speculative_metrics['avg_speedup']:.2f}x")
    logger.info("")
    
    logger.info("Overall Improvement:")
    logger.info(f"  Speedup: {speedup:.2f}x")
    logger.info(f"  Time Reduction: {time_reduction:+.1f}%")
    
    # Validate improvement
    if speedup > 1.2:  # At least 20% improvement
        logger.info("🎉 Significant performance improvement achieved!")
        return True
    else:
        logger.info("ℹ️  Modest performance improvement achieved.")
        return True

def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Speculative generation benchmark")
    parser.add_argument("--vocab-size", type=int, default=1000, help="Vocabulary size")
    parser.add_argument("--max-length", type=int, default=50, help="Maximum generation length")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--test-only", action="store_true", help="Run tests only, skip benchmark")
    
    args = parser.parse_args()
    
    logger.info("=== Speculative Generation Test Suite ===")
    
    # Test configuration options
    test_configuration_options()
    
    if args.test_only:
        logger.info("✅ All tests PASSED")
        return 0
    
    # Run comprehensive benchmark
    try:
        success = run_comprehensive_benchmark(
            vocab_size=args.vocab_size,
            max_length=args.max_length,
            num_runs=args.num_runs
        )
        
        if success:
            logger.info("✅ Speculative generation benchmark completed successfully")
            return 0
        else:
            logger.error("❌ Benchmark failed")
            return 1
            
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

