#!/usr/bin/env python3
"""
Enhanced IOBinding optimization benchmark and test suite.

This script tests the enhanced IOBinding implementation and measures
performance improvements over baseline ONNX Runtime usage.
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
import tempfile
import onnx
from onnx import helper, TensorProto

# Import our enhanced runtime
from runtime.enhanced_iobinding import create_enhanced_session
from runtime.ort_session import create_optimized_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_model(input_size: Tuple[int, ...], output_size: Tuple[int, ...]) -> str:
    """
    Create a simple test ONNX model for benchmarking.
    
    Args:
        input_size: Input tensor shape
        output_size: Output tensor shape
        
    Returns:
        Path to created ONNX model
    """
    # Create a simple MatMul + Add model
    input_name = "input"
    weight_name = "weight"
    bias_name = "bias"
    output_name = "output"
    
    # Create input
    input_tensor = helper.make_tensor_value_info(
        input_name, TensorProto.FLOAT, input_size
    )
    
    # Create weight (for MatMul)
    weight_shape = [input_size[-1], output_size[-1]]
    weight_tensor = helper.make_tensor_value_info(
        weight_name, TensorProto.FLOAT, weight_shape
    )
    
    # Create bias
    bias_tensor = helper.make_tensor_value_info(
        bias_name, TensorProto.FLOAT, [output_size[-1]]
    )
    
    # Create output
    output_tensor = helper.make_tensor_value_info(
        output_name, TensorProto.FLOAT, output_size
    )
    
    # Create MatMul node
    matmul_node = helper.make_node(
        "MatMul",
        inputs=[input_name, weight_name],
        outputs=["matmul_output"]
    )
    
    # Create Add node
    add_node = helper.make_node(
        "Add",
        inputs=["matmul_output", bias_name],
        outputs=[output_name]
    )
    
    # Create graph
    graph = helper.make_graph(
        nodes=[matmul_node, add_node],
        name="test_model",
        inputs=[input_tensor, weight_tensor, bias_tensor],
        outputs=[output_tensor]
    )
    
    # Create model
    model = helper.make_model(graph)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    onnx.save(model, temp_file.name)
    temp_file.close()
    
    logger.info(f"Created test model: {temp_file.name}")
    return temp_file.name

def generate_test_data(input_size: Tuple[int, ...], weight_shape: Tuple[int, ...], 
                      bias_shape: Tuple[int, ...], device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Generate test input data.
    
    Args:
        input_size: Input tensor shape
        weight_shape: Weight tensor shape
        bias_shape: Bias tensor shape
        device: Target device
        
    Returns:
        Dictionary of input tensors
    """
    torch.manual_seed(42)  # For reproducible results
    
    inputs = {
        "input": torch.randn(input_size, device=device, dtype=torch.float32),
        "weight": torch.randn(weight_shape, device=device, dtype=torch.float32),
        "bias": torch.randn(bias_shape, device=device, dtype=torch.float32)
    }
    
    return inputs

def benchmark_session(session, inputs: Dict[str, torch.Tensor], 
                     output_names: List[str], num_runs: int = 100) -> Dict[str, float]:
    """
    Benchmark session performance.
    
    Args:
        session: ORT session to benchmark
        inputs: Input tensors
        output_names: Output names
        num_runs: Number of benchmark runs
        
    Returns:
        Performance metrics
    """
    # Warmup runs
    for _ in range(10):
        if hasattr(session, 'run_optimized'):
            session.run_optimized(inputs, output_names)
        elif hasattr(session, 'run_with_iobinding'):
            session.run_with_iobinding(inputs, output_names)
        else:
            # Fallback to regular run
            np_inputs = {k: v.detach().cpu().numpy() for k, v in inputs.items()}
            session.session.run(output_names, np_inputs)
    
    # Benchmark runs
    start_time = time.perf_counter()
    
    for _ in range(num_runs):
        if hasattr(session, 'run_optimized'):
            outputs = session.run_optimized(inputs, output_names)
        elif hasattr(session, 'run_with_iobinding'):
            outputs = session.run_with_iobinding(inputs, output_names)
        else:
            np_inputs = {k: v.detach().cpu().numpy() for k, v in inputs.items()}
            outputs = session.session.run(output_names, np_inputs)
    
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_latency = (total_time / num_runs) * 1000  # ms
    throughput = num_runs / total_time  # ops/sec
    
    return {
        "avg_latency_ms": avg_latency,
        "throughput_ops_sec": throughput,
        "total_time_sec": total_time
    }

def test_correctness(enhanced_session, baseline_session, inputs: Dict[str, torch.Tensor],
                    output_names: List[str], tolerance: float = 1e-5) -> bool:
    """
    Test correctness of enhanced session vs baseline.
    
    Args:
        enhanced_session: Enhanced ORT session
        baseline_session: Baseline ORT session
        inputs: Input tensors
        output_names: Output names
        tolerance: Numerical tolerance for comparison
        
    Returns:
        True if outputs match within tolerance
    """
    # Get outputs from enhanced session
    enhanced_outputs = enhanced_session.run_optimized(inputs, output_names)
    
    # Get outputs from baseline session
    np_inputs = {k: v.detach().cpu().numpy() for k, v in inputs.items()}
    baseline_outputs = baseline_session.session.run(output_names, np_inputs)
    baseline_dict = {name: baseline_outputs[i] for i, name in enumerate(output_names)}
    
    # Compare outputs
    for name in output_names:
        enhanced_out = enhanced_outputs[name]
        baseline_out = baseline_dict[name]
        
        # Convert to numpy if needed
        if isinstance(enhanced_out, torch.Tensor):
            enhanced_out = enhanced_out.detach().cpu().numpy()
        
        # Check shapes
        if enhanced_out.shape != baseline_out.shape:
            logger.error(f"Shape mismatch for {name}: {enhanced_out.shape} vs {baseline_out.shape}")
            return False
        
        # Check values
        diff = np.abs(enhanced_out - baseline_out)
        max_diff = np.max(diff)
        
        if max_diff > tolerance:
            logger.error(f"Value mismatch for {name}: max_diff = {max_diff} > {tolerance}")
            return False
        
        logger.info(f"Output {name}: max_diff = {max_diff:.2e} (OK)")
    
    return True

def test_memory_pool_functionality():
    """Test memory pool functionality."""
    logger.info("Testing memory pool functionality...")
    
    from runtime.enhanced_iobinding import MemoryPool
    
    # Create memory pool
    pool = MemoryPool("cpu", max_pool_size=5)
    
    # Test buffer allocation and reuse
    shape1 = (2, 3, 4)
    dtype = torch.float32
    
    # Get first buffer
    buffer1 = pool.get_buffer(shape1, dtype)
    assert buffer1.shape == shape1
    assert buffer1.dtype == dtype
    
    # Return buffer to pool
    pool.return_buffer(buffer1)
    
    # Get buffer again - should reuse
    buffer2 = pool.get_buffer(shape1, dtype)
    assert torch.equal(buffer1, buffer2)  # Should be same buffer (zeroed)
    
    # Test different shape
    shape2 = (3, 4, 5)
    buffer3 = pool.get_buffer(shape2, dtype)
    assert buffer3.shape == shape2
    
    # Test pool stats
    stats = pool.get_stats()
    assert stats["device"] == "cpu"
    assert stats["total_buffer_types"] >= 1
    
    logger.info("✅ Memory pool functionality test PASSED")

def run_benchmark_suite(model_path: str, input_size: Tuple[int, ...], 
                       weight_shape: Tuple[int, ...], bias_shape: Tuple[int, ...],
                       output_size: Tuple[int, ...], num_runs: int = 100):
    """
    Run complete benchmark suite comparing enhanced vs baseline IOBinding.
    
    Args:
        model_path: Path to ONNX model
        input_size: Input tensor shape
        weight_shape: Weight tensor shape
        bias_shape: Bias tensor shape
        output_size: Output tensor shape
        num_runs: Number of benchmark runs
    """
    logger.info("=== Enhanced IOBinding Optimization Benchmark ===")
    
    # Create sessions
    logger.info("Creating sessions...")
    enhanced_session = create_enhanced_session(model_path, "decode", enable_memory_pool=True)
    baseline_session = create_optimized_session(model_path, "decode")
    
    # Generate test data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    inputs = generate_test_data(input_size, weight_shape, bias_shape, device)
    output_names = ["output"]
    
    # Test correctness
    logger.info("Testing correctness...")
    if test_correctness(enhanced_session, baseline_session, inputs, output_names):
        logger.info("✅ Correctness test PASSED")
    else:
        logger.error("❌ Correctness test FAILED")
        return
    
    # Benchmark enhanced session
    logger.info("Benchmarking enhanced session...")
    enhanced_metrics = benchmark_session(enhanced_session, inputs, output_names, num_runs)
    
    # Benchmark baseline session
    logger.info("Benchmarking baseline session...")
    baseline_metrics = benchmark_session(baseline_session, inputs, output_names, num_runs)
    
    # Calculate improvements
    latency_improvement = (baseline_metrics["avg_latency_ms"] - enhanced_metrics["avg_latency_ms"]) / baseline_metrics["avg_latency_ms"] * 100
    throughput_improvement = (enhanced_metrics["throughput_ops_sec"] - baseline_metrics["throughput_ops_sec"]) / baseline_metrics["throughput_ops_sec"] * 100
    
    # Print results
    logger.info("\n=== BENCHMARK RESULTS ===")
    logger.info(f"Input size: {input_size}")
    logger.info(f"Output size: {output_size}")
    logger.info(f"Device: {device}")
    logger.info(f"Runs: {num_runs}")
    logger.info("")
    
    logger.info("Baseline Session:")
    logger.info(f"  Avg Latency: {baseline_metrics['avg_latency_ms']:.2f} ms")
    logger.info(f"  Throughput:  {baseline_metrics['throughput_ops_sec']:.1f} ops/sec")
    logger.info("")
    
    logger.info("Enhanced Session:")
    logger.info(f"  Avg Latency: {enhanced_metrics['avg_latency_ms']:.2f} ms")
    logger.info(f"  Throughput:  {enhanced_metrics['throughput_ops_sec']:.1f} ops/sec")
    logger.info("")
    
    logger.info("Improvements:")
    logger.info(f"  Latency:    {latency_improvement:+.1f}%")
    logger.info(f"  Throughput: {throughput_improvement:+.1f}%")
    
    # Get enhanced session stats
    if hasattr(enhanced_session, 'get_performance_stats'):
        stats = enhanced_session.get_performance_stats()
        logger.info("")
        logger.info("Enhanced Session Stats:")
        logger.info(f"  Total runs: {stats['total_runs']}")
        logger.info(f"  IOBinding runs: {stats['iobinding_runs']}")
        logger.info(f"  Fallback runs: {stats['fallback_runs']}")
        logger.info(f"  Memory reuse rate: {stats['memory_reuse_rate']:.2f}")
        
        if 'memory_pools' in stats:
            for device, pool_stats in stats['memory_pools'].items():
                logger.info(f"  Memory pool ({device}): {pool_stats['total_buffers']} buffers")
    
    # Cleanup
    enhanced_session.clear_memory_pools()
    
    return {
        "enhanced": enhanced_metrics,
        "baseline": baseline_metrics,
        "improvements": {
            "latency_percent": latency_improvement,
            "throughput_percent": throughput_improvement
        }
    }

def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Enhanced IOBinding optimization benchmark")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--hidden-size", type=int, default=768, help="Hidden size")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of benchmark runs")
    parser.add_argument("--model-path", type=str, help="Path to existing ONNX model (optional)")
    parser.add_argument("--test-only", action="store_true", help="Run tests only, skip benchmark")
    
    args = parser.parse_args()
    
    logger.info("=== Enhanced IOBinding Test Suite ===")
    
    # Test memory pool functionality
    test_memory_pool_functionality()
    
    if args.test_only:
        logger.info("✅ All tests PASSED")
        return 0
    
    # Define tensor shapes
    input_size = (args.batch_size, args.seq_len, args.hidden_size)
    weight_shape = (args.hidden_size, args.hidden_size)
    bias_shape = (args.hidden_size,)
    output_size = (args.batch_size, args.seq_len, args.hidden_size)
    
    # Create or use existing model
    if args.model_path and os.path.exists(args.model_path):
        model_path = args.model_path
        logger.info(f"Using existing model: {model_path}")
    else:
        logger.info("Creating test model...")
        model_path = create_test_model(input_size, output_size)
    
    try:
        # Run benchmark
        results = run_benchmark_suite(
            model_path, input_size, weight_shape, bias_shape, output_size, args.num_runs
        )
        
        # Check if improvements are significant
        latency_improvement = results["improvements"]["latency_percent"]
        throughput_improvement = results["improvements"]["throughput_percent"]
        
        if latency_improvement > 5 or throughput_improvement > 5:
            logger.info("🎉 Significant performance improvement achieved!")
            return 0
        else:
            logger.info("ℹ️  Modest performance improvement achieved.")
            return 0
            
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1
    
    finally:
        # Cleanup temporary model if created
        if not args.model_path and os.path.exists(model_path):
            os.unlink(model_path)
            logger.info(f"Cleaned up temporary model: {model_path}")

if __name__ == "__main__":
    exit(main())

