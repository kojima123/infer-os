"""
GPU-NPU Pipeline for Infer-OS.

This module implements efficient pipeline processing between GPU and NPU
for optimal utilization of heterogeneous computing resources.
"""

import torch
import numpy as np
import time
import logging
import threading
import queue
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import concurrent.futures
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ProcessorType(Enum):
    """Types of processors in the pipeline."""
    GPU = "gpu"
    NPU = "npu"
    CPU = "cpu"

class TaskType(Enum):
    """Types of tasks that can be processed."""
    PREFILL = "prefill"
    DECODE = "decode"
    ATTENTION = "attention"
    FFN = "ffn"
    EMBEDDING = "embedding"

@dataclass
class PipelineConfig:
    """Configuration for GPU-NPU pipeline."""
    enable_pipeline: bool = True
    gpu_batch_size: int = 8
    npu_batch_size: int = 4
    pipeline_depth: int = 3
    load_balancing: bool = True
    dynamic_scheduling: bool = True
    memory_optimization: bool = True
    overlap_computation: bool = True
    prefetch_enabled: bool = True

@dataclass
class TaskSpec:
    """Specification for a pipeline task."""
    task_id: str
    task_type: TaskType
    input_data: Any
    processor_preference: Optional[ProcessorType] = None
    priority: int = 0
    estimated_compute_time: float = 0.0
    memory_requirement: int = 0

class ProcessorInterface(ABC):
    """Abstract interface for processors in the pipeline."""
    
    @abstractmethod
    def get_processor_type(self) -> ProcessorType:
        """Get processor type."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if processor is available."""
        pass
    
    @abstractmethod
    def get_utilization(self) -> float:
        """Get current utilization (0-1)."""
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> Tuple[int, int]:
        """Get (used_memory, total_memory) in bytes."""
        pass
    
    @abstractmethod
    def execute_task(self, task: TaskSpec) -> Any:
        """Execute task and return result."""
        pass
    
    @abstractmethod
    def estimate_execution_time(self, task: TaskSpec) -> float:
        """Estimate execution time for task."""
        pass

class MockGPUProcessor(ProcessorInterface):
    """Mock GPU processor for testing and demonstration."""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.is_busy = False
        self.current_utilization = 0.0
        self.memory_used = 0
        self.memory_total = 8 * 1024 * 1024 * 1024  # 8GB
        self.performance_profile = {
            TaskType.PREFILL: 0.1,    # 100ms base time
            TaskType.DECODE: 0.02,    # 20ms base time
            TaskType.ATTENTION: 0.05, # 50ms base time
            TaskType.FFN: 0.03,       # 30ms base time
            TaskType.EMBEDDING: 0.01  # 10ms base time
        }
    
    def get_processor_type(self) -> ProcessorType:
        return ProcessorType.GPU
    
    def is_available(self) -> bool:
        return not self.is_busy
    
    def get_utilization(self) -> float:
        return self.current_utilization
    
    def get_memory_usage(self) -> Tuple[int, int]:
        return self.memory_used, self.memory_total
    
    def execute_task(self, task: TaskSpec) -> Any:
        """Execute task on GPU."""
        if self.is_busy:
            raise RuntimeError("GPU processor is busy")
        
        self.is_busy = True
        self.current_utilization = 0.8  # Simulate high utilization
        
        try:
            # Simulate execution time
            base_time = self.performance_profile.get(task.task_type, 0.05)
            execution_time = base_time * (1 + np.random.normal(0, 0.1))  # Add noise
            execution_time = max(0.001, execution_time)  # Minimum 1ms
            
            logger.debug(f"GPU executing {task.task_type.value} task {task.task_id} "
                        f"for {execution_time:.3f}s")
            
            time.sleep(execution_time)
            
            # Simulate result
            if isinstance(task.input_data, torch.Tensor):
                result = task.input_data * 2.0  # Simple transformation
            else:
                result = f"GPU_processed_{task.task_id}"
            
            return result
            
        finally:
            self.is_busy = False
            self.current_utilization = 0.1  # Low idle utilization
    
    def estimate_execution_time(self, task: TaskSpec) -> float:
        """Estimate execution time for task."""
        base_time = self.performance_profile.get(task.task_type, 0.05)
        
        # Adjust for current utilization
        utilization_factor = 1.0 + self.current_utilization * 0.5
        
        return base_time * utilization_factor

class MockNPUProcessor(ProcessorInterface):
    """Mock NPU processor for testing and demonstration."""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.is_busy = False
        self.current_utilization = 0.0
        self.memory_used = 0
        self.memory_total = 4 * 1024 * 1024 * 1024  # 4GB
        self.performance_profile = {
            TaskType.PREFILL: 0.15,   # 150ms base time (slower than GPU)
            TaskType.DECODE: 0.01,    # 10ms base time (faster than GPU)
            TaskType.ATTENTION: 0.08, # 80ms base time
            TaskType.FFN: 0.02,       # 20ms base time (faster than GPU)
            TaskType.EMBEDDING: 0.005 # 5ms base time (faster than GPU)
        }
    
    def get_processor_type(self) -> ProcessorType:
        return ProcessorType.NPU
    
    def is_available(self) -> bool:
        return not self.is_busy
    
    def get_utilization(self) -> float:
        return self.current_utilization
    
    def get_memory_usage(self) -> Tuple[int, int]:
        return self.memory_used, self.memory_total
    
    def execute_task(self, task: TaskSpec) -> Any:
        """Execute task on NPU."""
        if self.is_busy:
            raise RuntimeError("NPU processor is busy")
        
        self.is_busy = True
        self.current_utilization = 0.9  # NPU typically runs at high utilization
        
        try:
            # Simulate execution time
            base_time = self.performance_profile.get(task.task_type, 0.05)
            execution_time = base_time * (1 + np.random.normal(0, 0.05))  # Less noise than GPU
            execution_time = max(0.001, execution_time)  # Minimum 1ms
            
            logger.debug(f"NPU executing {task.task_type.value} task {task.task_id} "
                        f"for {execution_time:.3f}s")
            
            time.sleep(execution_time)
            
            # Simulate result
            if isinstance(task.input_data, torch.Tensor):
                result = task.input_data * 1.5  # Different transformation than GPU
            else:
                result = f"NPU_processed_{task.task_id}"
            
            return result
            
        finally:
            self.is_busy = False
            self.current_utilization = 0.05  # Very low idle utilization
    
    def estimate_execution_time(self, task: TaskSpec) -> float:
        """Estimate execution time for task."""
        base_time = self.performance_profile.get(task.task_type, 0.05)
        
        # NPU performance is more consistent
        utilization_factor = 1.0 + self.current_utilization * 0.2
        
        return base_time * utilization_factor

class TaskScheduler:
    """Intelligent task scheduler for GPU-NPU pipeline."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.processors: List[ProcessorInterface] = []
        self.task_queue = queue.PriorityQueue()
        self.result_queue = queue.Queue()
        self.scheduling_stats = {
            "total_tasks": 0,
            "gpu_tasks": 0,
            "npu_tasks": 0,
            "avg_wait_time": 0.0,
            "avg_execution_time": 0.0,
            "load_balance_score": 0.0
        }
    
    def add_processor(self, processor: ProcessorInterface):
        """Add processor to the scheduler."""
        self.processors.append(processor)
        logger.info(f"Added {processor.get_processor_type().value} processor")
    
    def submit_task(self, task: TaskSpec) -> str:
        """Submit task for execution."""
        # Priority queue uses negative priority for max-heap behavior
        priority = -task.priority
        self.task_queue.put((priority, time.time(), task))
        self.scheduling_stats["total_tasks"] += 1
        
        logger.debug(f"Submitted task {task.task_id} with priority {task.priority}")
        return task.task_id
    
    def get_optimal_processor(self, task: TaskSpec) -> Optional[ProcessorInterface]:
        """
        Select optimal processor for task based on various factors.
        
        Args:
            task: Task to schedule
            
        Returns:
            Best processor for the task, or None if none available
        """
        available_processors = [p for p in self.processors if p.is_available()]
        
        if not available_processors:
            return None
        
        # If task has processor preference, try to honor it
        if task.processor_preference:
            preferred = [p for p in available_processors 
                        if p.get_processor_type() == task.processor_preference]
            if preferred:
                available_processors = preferred
        
        # Score processors based on multiple factors
        best_processor = None
        best_score = float('-inf')
        
        for processor in available_processors:
            score = self._calculate_processor_score(processor, task)
            if score > best_score:
                best_score = score
                best_processor = processor
        
        return best_processor
    
    def _calculate_processor_score(self, processor: ProcessorInterface, task: TaskSpec) -> float:
        """
        Calculate score for processor-task pairing.
        
        Args:
            processor: Processor to evaluate
            task: Task to execute
            
        Returns:
            Score (higher is better)
        """
        # Base score from estimated execution time (lower is better)
        estimated_time = processor.estimate_execution_time(task)
        time_score = 1.0 / (estimated_time + 0.001)  # Avoid division by zero
        
        # Utilization score (prefer less utilized processors)
        utilization = processor.get_utilization()
        utilization_score = 1.0 - utilization
        
        # Memory score (prefer processors with more available memory)
        used_memory, total_memory = processor.get_memory_usage()
        memory_available = total_memory - used_memory
        memory_score = memory_available / total_memory
        
        # Task type affinity (some processors are better for certain tasks)
        affinity_score = self._get_task_affinity(processor.get_processor_type(), task.task_type)
        
        # Weighted combination
        total_score = (
            0.4 * time_score +
            0.3 * utilization_score +
            0.2 * memory_score +
            0.1 * affinity_score
        )
        
        return total_score
    
    def _get_task_affinity(self, processor_type: ProcessorType, task_type: TaskType) -> float:
        """Get affinity score for processor-task combination."""
        # Define affinities based on typical performance characteristics
        affinities = {
            ProcessorType.GPU: {
                TaskType.PREFILL: 0.9,    # GPU good for parallel prefill
                TaskType.DECODE: 0.6,     # GPU okay for decode
                TaskType.ATTENTION: 0.8,  # GPU good for attention
                TaskType.FFN: 0.7,        # GPU good for FFN
                TaskType.EMBEDDING: 0.5   # GPU okay for embedding
            },
            ProcessorType.NPU: {
                TaskType.PREFILL: 0.6,    # NPU okay for prefill
                TaskType.DECODE: 0.9,     # NPU excellent for decode
                TaskType.ATTENTION: 0.7,  # NPU good for attention
                TaskType.FFN: 0.8,        # NPU good for FFN
                TaskType.EMBEDDING: 0.9   # NPU excellent for embedding
            }
        }
        
        return affinities.get(processor_type, {}).get(task_type, 0.5)
    
    def execute_task_async(self, task: TaskSpec) -> concurrent.futures.Future:
        """Execute task asynchronously."""
        processor = self.get_optimal_processor(task)
        
        if processor is None:
            # No processor available, return failed future
            future = concurrent.futures.Future()
            future.set_exception(RuntimeError("No processor available"))
            return future
        
        # Update statistics
        if processor.get_processor_type() == ProcessorType.GPU:
            self.scheduling_stats["gpu_tasks"] += 1
        elif processor.get_processor_type() == ProcessorType.NPU:
            self.scheduling_stats["npu_tasks"] += 1
        
        # Execute task in thread pool
        def safe_execute():
            try:
                return processor.execute_task(task)
            except Exception as e:
                logger.warning(f"Task execution failed: {e}")
                raise
        
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(safe_execute)
        
        return future

class GPUNPUPipeline:
    """
    Main GPU-NPU pipeline orchestrator.
    
    This class manages the entire pipeline, including task scheduling,
    load balancing, and performance optimization.
    """
    
    def __init__(self, config: PipelineConfig = None):
        """
        Initialize GPU-NPU pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.scheduler = TaskScheduler(config)
        self.is_running = False
        self.worker_threads = []
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize processors
        self._initialize_processors()
        
        self.pipeline_stats = {
            "total_processed": 0,
            "total_time": 0.0,
            "throughput": 0.0,
            "avg_latency": 0.0,
            "gpu_utilization": 0.0,
            "npu_utilization": 0.0
        }
    
    def _initialize_processors(self):
        """Initialize available processors."""
        # Add mock processors for demonstration
        # In real implementation, this would detect actual hardware
        
        try:
            # Try to add GPU processor
            gpu = MockGPUProcessor(0)
            self.scheduler.add_processor(gpu)
            logger.info("GPU processor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize GPU: {e}")
        
        try:
            # Try to add NPU processor
            npu = MockNPUProcessor(0)
            self.scheduler.add_processor(npu)
            logger.info("NPU processor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize NPU: {e}")
        
        if not self.scheduler.processors:
            logger.error("No processors available!")
            raise RuntimeError("No processors available for pipeline")
    
    def start_pipeline(self):
        """Start the pipeline processing."""
        if self.is_running:
            logger.warning("Pipeline is already running")
            return
        
        self.is_running = True
        logger.info("Starting GPU-NPU pipeline")
        
        # Start worker threads for pipeline processing
        if self.config.enable_pipeline:
            for i in range(self.config.pipeline_depth):
                worker = threading.Thread(
                    target=self._pipeline_worker,
                    name=f"PipelineWorker-{i}",
                    daemon=True
                )
                worker.start()
                self.worker_threads.append(worker)
        
        # Start performance monitoring
        self.performance_monitor.start()
    
    def stop_pipeline(self):
        """Stop the pipeline processing."""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("Stopping GPU-NPU pipeline")
        
        # Wait for worker threads to finish
        for worker in self.worker_threads:
            worker.join(timeout=1.0)
        
        self.worker_threads.clear()
        self.performance_monitor.stop()
    
    def _pipeline_worker(self):
        """Worker thread for pipeline processing."""
        while self.is_running:
            try:
                # Get task from queue (with timeout)
                priority, submit_time, task = self.scheduler.task_queue.get(timeout=0.1)
                
                # Calculate wait time
                wait_time = time.time() - submit_time
                
                # Execute task
                start_time = time.time()
                future = self.scheduler.execute_task_async(task)
                result = future.result(timeout=10.0)  # 10 second timeout
                execution_time = time.time() - start_time
                
                # Update statistics
                self._update_stats(wait_time, execution_time)
                
                # Put result in result queue
                self.scheduler.result_queue.put((task.task_id, result))
                
                logger.debug(f"Completed task {task.task_id} in {execution_time:.3f}s")
                
            except queue.Empty:
                continue  # No tasks available, continue loop
            except Exception as e:
                logger.error(f"Pipeline worker error: {e}")
                continue
    
    def _update_stats(self, wait_time: float, execution_time: float):
        """Update pipeline statistics."""
        self.pipeline_stats["total_processed"] += 1
        self.pipeline_stats["total_time"] += execution_time
        
        # Update averages
        total = self.pipeline_stats["total_processed"]
        self.pipeline_stats["avg_latency"] = (
            (self.pipeline_stats["avg_latency"] * (total - 1) + execution_time) / total
        )
        
        # Update throughput (tasks per second)
        if self.pipeline_stats["total_time"] > 0:
            self.pipeline_stats["throughput"] = total / self.pipeline_stats["total_time"]
    
    def submit_task(self, task_type: TaskType, input_data: Any,
                   processor_preference: Optional[ProcessorType] = None,
                   priority: int = 0) -> str:
        """
        Submit task to pipeline.
        
        Args:
            task_type: Type of task
            input_data: Input data for task
            processor_preference: Preferred processor type
            priority: Task priority (higher = more important)
            
        Returns:
            Task ID
        """
        task_id = f"{task_type.value}_{int(time.time() * 1000000)}"
        
        task = TaskSpec(
            task_id=task_id,
            task_type=task_type,
            input_data=input_data,
            processor_preference=processor_preference,
            priority=priority
        )
        
        return self.scheduler.submit_task(task)
    
    def get_result(self, timeout: float = 1.0) -> Optional[Tuple[str, Any]]:
        """
        Get result from pipeline.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (task_id, result) or None if timeout
        """
        try:
            return self.scheduler.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics."""
        # Update processor utilizations
        gpu_utils = []
        npu_utils = []
        
        for processor in self.scheduler.processors:
            if processor.get_processor_type() == ProcessorType.GPU:
                gpu_utils.append(processor.get_utilization())
            elif processor.get_processor_type() == ProcessorType.NPU:
                npu_utils.append(processor.get_utilization())
        
        self.pipeline_stats["gpu_utilization"] = np.mean(gpu_utils) if gpu_utils else 0.0
        self.pipeline_stats["npu_utilization"] = np.mean(npu_utils) if npu_utils else 0.0
        
        # Combine with scheduler stats
        combined_stats = {
            **self.pipeline_stats,
            **self.scheduler.scheduling_stats
        }
        
        return combined_stats

class PerformanceMonitor:
    """Performance monitoring for the pipeline."""
    
    def __init__(self):
        self.is_monitoring = False
        self.monitor_thread = None
        self.metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "gpu_utilization": [],
            "npu_utilization": []
        }
    
    def start(self):
        """Start performance monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="PerformanceMonitor",
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect metrics (simplified for demo)
                self.metrics["cpu_usage"].append(np.random.uniform(0.1, 0.8))
                self.metrics["memory_usage"].append(np.random.uniform(0.3, 0.9))
                self.metrics["gpu_utilization"].append(np.random.uniform(0.2, 0.9))
                self.metrics["npu_utilization"].append(np.random.uniform(0.1, 0.8))
                
                # Keep only recent metrics
                for key in self.metrics:
                    if len(self.metrics[key]) > 100:
                        self.metrics[key] = self.metrics[key][-100:]
                
                time.sleep(0.1)  # Monitor every 100ms
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
    
    def get_metrics(self) -> Dict[str, List[float]]:
        """Get collected metrics."""
        return self.metrics.copy()

def create_gpu_npu_pipeline(enable_pipeline: bool = True,
                           gpu_batch_size: int = 8,
                           npu_batch_size: int = 4,
                           pipeline_depth: int = 3) -> GPUNPUPipeline:
    """
    Factory function to create GPU-NPU pipeline.
    
    Args:
        enable_pipeline: Enable pipeline processing
        gpu_batch_size: Batch size for GPU
        npu_batch_size: Batch size for NPU
        pipeline_depth: Number of pipeline stages
        
    Returns:
        Configured GPU-NPU pipeline
    """
    config = PipelineConfig(
        enable_pipeline=enable_pipeline,
        gpu_batch_size=gpu_batch_size,
        npu_batch_size=npu_batch_size,
        pipeline_depth=pipeline_depth
    )
    
    return GPUNPUPipeline(config)

