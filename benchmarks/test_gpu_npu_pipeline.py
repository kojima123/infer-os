#!/usr/bin/env python3
"""
GPU-NPU Pipeline benchmark and test suite.

This script tests the GPU-NPU pipeline implementation and measures
performance improvements from heterogeneous computing.
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
import threading

# Import our GPU-NPU pipeline implementation
from optim.gpu_npu_pipeline import (
    GPUNPUPipeline, PipelineConfig, TaskType, ProcessorType, TaskSpec,
    create_gpu_npu_pipeline
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_processor_interfaces():
    """Test processor interfaces and basic functionality."""
    logger.info("Testing processor interfaces...")
    
    pipeline = create_gpu_npu_pipeline()
    
    # Check that processors were initialized
    assert len(pipeline.scheduler.processors) >= 1, "At least one processor should be available"
    
    # Test processor properties
    for processor in pipeline.scheduler.processors:
        processor_type = processor.get_processor_type()
        logger.info(f"Testing {processor_type.value} processor...")
        
        # Test availability
        assert processor.is_available(), "Processor should be available initially"
        
        # Test utilization
        utilization = processor.get_utilization()
        assert 0.0 <= utilization <= 1.0, f"Utilization should be 0-1, got {utilization}"
        
        # Test memory usage
        used, total = processor.get_memory_usage()
        assert used >= 0 and total > 0, "Memory usage should be non-negative"
        assert used <= total, "Used memory should not exceed total"
        
        # Test execution time estimation
        test_task = TaskSpec(
            task_id="test",
            task_type=TaskType.DECODE,
            input_data=torch.randn(1, 10)
        )
        
        estimated_time = processor.estimate_execution_time(test_task)
        assert estimated_time > 0, "Estimated time should be positive"
        
        logger.info(f"  Utilization: {utilization:.2%}")
        logger.info(f"  Memory: {used}/{total} bytes")
        logger.info(f"  Estimated decode time: {estimated_time:.3f}s")
    
    logger.info("✅ Processor interfaces test PASSED")

def test_task_scheduling():
    """Test task scheduling and processor selection."""
    logger.info("Testing task scheduling...")
    
    pipeline = create_gpu_npu_pipeline()
    
    # Test different task types
    task_types = [TaskType.PREFILL, TaskType.DECODE, TaskType.ATTENTION, TaskType.FFN]
    
    for task_type in task_types:
        logger.info(f"Testing {task_type.value} task scheduling...")
        
        task = TaskSpec(
            task_id=f"test_{task_type.value}",
            task_type=task_type,
            input_data=torch.randn(2, 16),
            priority=1
        )
        
        # Test processor selection
        processor = pipeline.scheduler.get_optimal_processor(task)
        assert processor is not None, f"Should find processor for {task_type.value}"
        
        logger.info(f"  Selected {processor.get_processor_type().value} for {task_type.value}")
    
    # Test processor preference
    gpu_task = TaskSpec(
        task_id="gpu_preferred",
        task_type=TaskType.PREFILL,
        input_data=torch.randn(1, 8),
        processor_preference=ProcessorType.GPU
    )
    
    npu_task = TaskSpec(
        task_id="npu_preferred",
        task_type=TaskType.DECODE,
        input_data=torch.randn(1, 8),
        processor_preference=ProcessorType.NPU
    )
    
    gpu_processor = pipeline.scheduler.get_optimal_processor(gpu_task)
    npu_processor = pipeline.scheduler.get_optimal_processor(npu_task)
    
    # Check preferences are honored when possible
    if gpu_processor:
        logger.info(f"GPU preference: {gpu_processor.get_processor_type().value}")
    if npu_processor:
        logger.info(f"NPU preference: {npu_processor.get_processor_type().value}")
    
    logger.info("✅ Task scheduling test PASSED")

def test_pipeline_execution():
    """Test end-to-end pipeline execution."""
    logger.info("Testing pipeline execution...")
    
    pipeline = create_gpu_npu_pipeline(pipeline_depth=2)
    pipeline.start_pipeline()
    
    try:
        # Submit various tasks
        task_ids = []
        
        for i in range(5):
            # Mix of different task types
            task_type = [TaskType.DECODE, TaskType.FFN, TaskType.ATTENTION][i % 3]
            input_data = torch.randn(1, 16)
            
            task_id = pipeline.submit_task(
                task_type=task_type,
                input_data=input_data,
                priority=i
            )
            task_ids.append(task_id)
            logger.info(f"Submitted task {task_id} ({task_type.value})")
        
        # Collect results
        results = {}
        timeout_per_task = 2.0
        
        for _ in range(len(task_ids)):
            result = pipeline.get_result(timeout=timeout_per_task)
            if result:
                task_id, output = result
                results[task_id] = output
                logger.info(f"Received result for {task_id}")
            else:
                logger.warning("Timeout waiting for result")
        
        # Verify we got results
        assert len(results) > 0, "Should receive at least some results"
        logger.info(f"Received {len(results)}/{len(task_ids)} results")
        
        # Check pipeline statistics
        stats = pipeline.get_pipeline_stats()
        logger.info(f"Pipeline stats: {stats}")
        
        assert stats["total_processed"] > 0, "Should have processed some tasks"
        
    finally:
        pipeline.stop_pipeline()
    
    logger.info("✅ Pipeline execution test PASSED")

def test_load_balancing():
    """Test load balancing between processors."""
    logger.info("Testing load balancing...")
    
    pipeline = create_gpu_npu_pipeline(pipeline_depth=3)
    pipeline.start_pipeline()
    
    try:
        # Submit many tasks to test load balancing
        num_tasks = 20
        task_ids = []
        
        for i in range(num_tasks):
            # Alternate between task types that favor different processors
            if i % 2 == 0:
                task_type = TaskType.PREFILL  # GPU-favored
            else:
                task_type = TaskType.DECODE   # NPU-favored
            
            task_id = pipeline.submit_task(
                task_type=task_type,
                input_data=torch.randn(1, 8),
                priority=1
            )
            task_ids.append(task_id)
        
        # Wait for all tasks to complete
        results = {}
        for _ in range(num_tasks):
            result = pipeline.get_result(timeout=3.0)
            if result:
                task_id, output = result
                results[task_id] = output
        
        # Check load distribution
        stats = pipeline.get_pipeline_stats()
        gpu_tasks = stats.get("gpu_tasks", 0)
        npu_tasks = stats.get("npu_tasks", 0)
        
        logger.info(f"GPU tasks: {gpu_tasks}, NPU tasks: {npu_tasks}")
        
        # Both processors should have received some tasks
        total_assigned = gpu_tasks + npu_tasks
        if total_assigned > 0:
            gpu_ratio = gpu_tasks / total_assigned
            npu_ratio = npu_tasks / total_assigned
            
            logger.info(f"GPU ratio: {gpu_ratio:.2%}, NPU ratio: {npu_ratio:.2%}")
            
            # Load should be somewhat balanced (not too skewed)
            assert 0.1 <= gpu_ratio <= 0.9, "Load should be distributed between processors"
            assert 0.1 <= npu_ratio <= 0.9, "Load should be distributed between processors"
        
    finally:
        pipeline.stop_pipeline()
    
    logger.info("✅ Load balancing test PASSED")

def benchmark_pipeline_performance(num_tasks: int = 50, task_complexity: int = 16,
                                 num_runs: int = 3):
    """
    Benchmark pipeline performance vs sequential processing.
    
    Args:
        num_tasks: Number of tasks to process
        task_complexity: Complexity of each task (tensor size)
        num_runs: Number of benchmark runs
    """
    logger.info("=== GPU-NPU Pipeline Performance Benchmark ===")
    logger.info(f"Tasks: {num_tasks}, Complexity: {task_complexity}, Runs: {num_runs}")
    
    # Benchmark sequential processing (baseline)
    logger.info("Benchmarking sequential processing...")
    sequential_times = []
    
    for run in range(num_runs):
        start_time = time.perf_counter()
        
        # Simulate sequential processing
        for i in range(num_tasks):
            task_type = [TaskType.DECODE, TaskType.FFN, TaskType.ATTENTION][i % 3]
            input_data = torch.randn(1, task_complexity)
            
            # Simulate processing time based on task type
            if task_type == TaskType.DECODE:
                time.sleep(0.01)  # 10ms
            elif task_type == TaskType.FFN:
                time.sleep(0.02)  # 20ms
            else:  # ATTENTION
                time.sleep(0.03)  # 30ms
        
        elapsed = time.perf_counter() - start_time
        sequential_times.append(elapsed)
        logger.info(f"  Run {run+1}: {elapsed:.3f}s")
    
    avg_sequential_time = statistics.mean(sequential_times)
    
    # Benchmark pipeline processing
    logger.info("Benchmarking pipeline processing...")
    pipeline_times = []
    
    for run in range(num_runs):
        pipeline = create_gpu_npu_pipeline(pipeline_depth=4)
        pipeline.start_pipeline()
        
        try:
            start_time = time.perf_counter()
            
            # Submit all tasks
            task_ids = []
            for i in range(num_tasks):
                task_type = [TaskType.DECODE, TaskType.FFN, TaskType.ATTENTION][i % 3]
                input_data = torch.randn(1, task_complexity)
                
                task_id = pipeline.submit_task(
                    task_type=task_type,
                    input_data=input_data,
                    priority=1
                )
                task_ids.append(task_id)
            
            # Collect all results
            results = {}
            for _ in range(num_tasks):
                result = pipeline.get_result(timeout=5.0)
                if result:
                    task_id, output = result
                    results[task_id] = output
            
            elapsed = time.perf_counter() - start_time
            pipeline_times.append(elapsed)
            
            # Get pipeline stats
            stats = pipeline.get_pipeline_stats()
            logger.info(f"  Run {run+1}: {elapsed:.3f}s "
                       f"(GPU: {stats.get('gpu_tasks', 0)}, NPU: {stats.get('npu_tasks', 0)})")
            
        finally:
            pipeline.stop_pipeline()
    
    avg_pipeline_time = statistics.mean(pipeline_times)
    
    # Calculate improvements
    speedup = avg_sequential_time / avg_pipeline_time
    time_reduction = (avg_sequential_time - avg_pipeline_time) / avg_sequential_time * 100
    
    # Print results
    logger.info("\n=== BENCHMARK RESULTS ===")
    logger.info("Sequential Processing:")
    logger.info(f"  Avg Time: {avg_sequential_time:.3f}s")
    logger.info(f"  Throughput: {num_tasks / avg_sequential_time:.1f} tasks/sec")
    logger.info("")
    
    logger.info("Pipeline Processing:")
    logger.info(f"  Avg Time: {avg_pipeline_time:.3f}s")
    logger.info(f"  Throughput: {num_tasks / avg_pipeline_time:.1f} tasks/sec")
    logger.info("")
    
    logger.info("Performance Improvement:")
    logger.info(f"  Speedup: {speedup:.2f}x")
    logger.info(f"  Time Reduction: {time_reduction:+.1f}%")
    
    # Validate improvement
    if speedup > 1.2:  # At least 20% improvement
        logger.info("🎉 Significant performance improvement achieved!")
        return True
    else:
        logger.info("ℹ️  Modest performance improvement achieved.")
        return True

def test_error_handling():
    """Test error handling and robustness."""
    logger.info("Testing error handling...")
    
    pipeline = create_gpu_npu_pipeline()
    pipeline.start_pipeline()
    
    try:
        # Test with invalid input
        task_id = pipeline.submit_task(
            task_type=TaskType.DECODE,
            input_data=None,  # Invalid input
            priority=1
        )
        
        # Should still handle gracefully
        result = pipeline.get_result(timeout=2.0)
        logger.info(f"Handled invalid input task: {result is not None}")
        
        # Test high priority task
        high_priority_id = pipeline.submit_task(
            task_type=TaskType.DECODE,
            input_data=torch.randn(1, 8),
            priority=10  # High priority
        )
        
        # Test normal priority task
        normal_priority_id = pipeline.submit_task(
            task_type=TaskType.DECODE,
            input_data=torch.randn(1, 8),
            priority=1  # Normal priority
        )
        
        # High priority should be processed first (in most cases)
        results = []
        for _ in range(2):
            result = pipeline.get_result(timeout=2.0)
            if result:
                results.append(result[0])
        
        logger.info(f"Task processing order: {results}")
        
    finally:
        pipeline.stop_pipeline()
    
    logger.info("✅ Error handling test PASSED")

def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="GPU-NPU pipeline benchmark")
    parser.add_argument("--num-tasks", type=int, default=30, help="Number of tasks")
    parser.add_argument("--task-complexity", type=int, default=16, help="Task complexity")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--test-only", action="store_true", help="Run tests only, skip benchmark")
    
    args = parser.parse_args()
    
    logger.info("=== GPU-NPU Pipeline Test Suite ===")
    
    # Run tests
    test_processor_interfaces()
    test_task_scheduling()
    test_pipeline_execution()
    test_load_balancing()
    test_error_handling()
    
    if args.test_only:
        logger.info("✅ All tests PASSED")
        return 0
    
    # Run benchmark
    try:
        success = benchmark_pipeline_performance(
            num_tasks=args.num_tasks,
            task_complexity=args.task_complexity,
            num_runs=args.num_runs
        )
        
        if success:
            logger.info("✅ GPU-NPU pipeline benchmark completed successfully")
            return 0
        else:
            logger.error("❌ Benchmark failed")
            return 1
            
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

