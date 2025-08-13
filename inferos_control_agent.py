#!/usr/bin/env python3
"""
Infer-OS Control Agent for AMD GAIA Integration
Windows Service / Sidecar implementation for dynamic optimization
"""

import asyncio
import json
import logging
import time
import threading
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from datetime import datetime
import psutil
import platform

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel

# Configuration Models
class PolicyConfig(BaseModel):
    kv: Dict[str, Any] = {"mode": "dynamic", "recent_window": 64}
    io: Dict[str, Any] = {"enable_iobinding": True}
    scheduler: Dict[str, Any] = {"mode": "hybrid"}

class RunContext(BaseModel):
    seq_len: int
    batch: int
    target_ftl_ms: int = 300
    quality_budget: float = 0.5

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float
    tps: float = 0.0
    ftl_ms: float = 0.0
    p95_ms: float = 0.0
    vram_gb: float = 0.0
    host_gb: float = 0.0
    kv_levels: Dict[str, int] = None
    npu_util: float = 0.0
    igpu_util: float = 0.0
    cpu_util: float = 0.0
    delta_ppl_est: float = 0.0
    spec_accept: float = 0.0

    def __post_init__(self):
        if self.kv_levels is None:
            self.kv_levels = {"L0": 0, "L1": 0, "L2": 0, "L3": 0}

class InferOSControlAgent:
    """
    Infer-OS Control Agent for AMD GAIA Integration
    
    Provides:
    - 1ms/10ms/100ms control loops
    - KV quantization policy management
    - IOBinding optimization
    - Telemetry collection
    - REST API for GAIA integration
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.metrics = SystemMetrics(timestamp=time.time())
        self.policy = PolicyConfig()
        self.running = False
        self.logger = self._setup_logging()
        
        # Control loop intervals (ms)
        self.fast_interval = self.config.get("loops", {}).get("fast_ms", 1) / 1000.0
        self.mid_interval = self.config.get("loops", {}).get("mid_ms", 10) / 1000.0
        self.slow_interval = self.config.get("loops", {}).get("slow_ms", 100) / 1000.0
        
        # Performance tracking
        self.performance_history = []
        self.max_history = 1000
        
        # Failsafe state
        self.baseline_mode = False
        self.error_count = 0
        self.max_errors = 5
        
        # KV quantization state
        self.kv_state = {
            "recent_window": 64,
            "level_thresholds": {
                "L1_int8": 0.7,
                "L2_int4": 0.5,
                "L3_evict": 0.3
            },
            "current_levels": {"L0": 0, "L1": 0, "L2": 0, "L3": 0}
        }
        
        self.logger.info("Infer-OS Control Agent initialized")
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file or use defaults"""
        default_config = {
            "quality": {
                "max_delta_ppl": 0.5,
                "min_accept_rate": 0.5
            },
            "kv": {
                "recent_window": 64,
                "level_thresholds": {
                    "L1_int8": 0.7,
                    "L2_int4": 0.5,
                    "L3_evict": 0.3
                }
            },
            "io": {
                "enable_iobinding": True,
                "dml_pool_bytes": "2048MiB",
                "host_pool_bytes": "4096MiB"
            },
            "scheduler": {
                "mode": "hybrid",
                "prefill_device": "dml",
                "decode_device": "npu"
            },
            "loops": {
                "fast_ms": 1,
                "mid_ms": 10,
                "slow_ms": 100
            },
            "telemetry": {
                "push_interval_ms": 1000
            }
        }
        
        if config_path:
            try:
                import yaml
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config.get("infer_os", {}))
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("InferOSAgent")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics"""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            host_gb = memory.used / (1024**3)
            
            # CPU usage
            cpu_util = psutil.cpu_percent(interval=0.1)
            
            # GPU/NPU metrics (simulated for now)
            # In real implementation, would query DirectML/NPU APIs
            npu_util = self._get_npu_utilization()
            igpu_util = self._get_igpu_utilization()
            vram_gb = self._get_vram_usage()
            
            # Performance metrics (would be provided by GAIA/Lemonade)
            tps = self._estimate_tps()
            ftl_ms = self._estimate_ftl()
            p95_ms = self._estimate_p95()
            
            # Quality metrics
            delta_ppl_est = self._estimate_delta_ppl()
            spec_accept = self._estimate_spec_accept()
            
            metrics = SystemMetrics(
                timestamp=time.time(),
                tps=tps,
                ftl_ms=ftl_ms,
                p95_ms=p95_ms,
                vram_gb=vram_gb,
                host_gb=host_gb,
                kv_levels=self.kv_state["current_levels"].copy(),
                npu_util=npu_util,
                igpu_util=igpu_util,
                cpu_util=cpu_util,
                delta_ppl_est=delta_ppl_est,
                spec_accept=spec_accept
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            return SystemMetrics(timestamp=time.time())
    
    def _get_npu_utilization(self) -> float:
        """Get NPU utilization (placeholder implementation)"""
        # In real implementation, would query NPU driver APIs
        # For now, simulate based on system load
        return min(psutil.cpu_percent() * 0.6, 100.0)
    
    def _get_igpu_utilization(self) -> float:
        """Get iGPU utilization (placeholder implementation)"""
        # In real implementation, would query DirectML APIs
        return min(psutil.cpu_percent() * 0.8, 100.0)
    
    def _get_vram_usage(self) -> float:
        """Get VRAM usage in GB (placeholder implementation)"""
        # In real implementation, would query GPU memory APIs
        return min(psutil.virtual_memory().percent / 100.0 * 8.0, 8.0)
    
    def _estimate_tps(self) -> float:
        """Estimate tokens per second (placeholder)"""
        # Would be provided by GAIA/Lemonade metrics
        base_tps = 25.0
        if self.baseline_mode:
            return base_tps
        else:
            # Simulate Infer-OS improvement
            return base_tps * 1.4
    
    def _estimate_ftl(self) -> float:
        """Estimate first token latency in ms (placeholder)"""
        base_ftl = 250.0
        if self.baseline_mode:
            return base_ftl
        else:
            return base_ftl * 0.8
    
    def _estimate_p95(self) -> float:
        """Estimate 95th percentile latency in ms (placeholder)"""
        base_p95 = 400.0
        if self.baseline_mode:
            return base_p95
        else:
            return base_p95 * 0.75
    
    def _estimate_delta_ppl(self) -> float:
        """Estimate perplexity delta from quantization"""
        # Based on current KV quantization levels
        l1_ratio = self.kv_state["current_levels"]["L1"] / 100.0
        l2_ratio = self.kv_state["current_levels"]["L2"] / 100.0
        l3_ratio = self.kv_state["current_levels"]["L3"] / 100.0
        
        # Estimate quality impact
        delta = l1_ratio * 0.1 + l2_ratio * 0.3 + l3_ratio * 0.8
        return min(delta, 1.0)
    
    def _estimate_spec_accept(self) -> float:
        """Estimate speculative decoding acceptance rate"""
        # Placeholder implementation
        return 0.65 if not self.baseline_mode else 0.0
    
    def _update_kv_quantization_policy(self, context: Optional[RunContext] = None):
        """Update KV quantization policy based on current conditions"""
        try:
            # Get current memory pressure
            memory = psutil.virtual_memory()
            mem_pressure = memory.percent / 100.0
            
            # Quality budget (from context or default)
            quality_budget = context.quality_budget if context else 0.5
            
            # Adjust KV levels based on conditions
            if mem_pressure > 0.8:
                # High memory pressure - aggressive quantization
                self.kv_state["current_levels"] = {"L0": 20, "L1": 30, "L2": 35, "L3": 15}
            elif mem_pressure > 0.6:
                # Medium memory pressure - balanced quantization
                self.kv_state["current_levels"] = {"L0": 40, "L1": 35, "L2": 20, "L3": 5}
            elif quality_budget > 0.8:
                # High quality requirement - conservative quantization
                self.kv_state["current_levels"] = {"L0": 70, "L1": 25, "L2": 5, "L3": 0}
            else:
                # Normal conditions - balanced approach
                self.kv_state["current_levels"] = {"L0": 50, "L1": 30, "L2": 15, "L3": 5}
                
        except Exception as e:
            self.logger.error(f"Error updating KV policy: {e}")
            self._enter_baseline_mode()
    
    def _optimize_iobinding(self):
        """Optimize IOBinding configuration"""
        try:
            # In real implementation, would adjust ORT IOBinding settings
            # For now, just log the optimization
            if self.policy.io.get("enable_iobinding", True):
                self.logger.debug("IOBinding optimization applied")
        except Exception as e:
            self.logger.error(f"Error optimizing IOBinding: {e}")
    
    def _adjust_hybrid_scheduler(self):
        """Adjust hybrid NPU/iGPU scheduling"""
        try:
            # Monitor device utilization and adjust scheduling
            npu_util = self.metrics.npu_util
            igpu_util = self.metrics.igpu_util
            
            # Simple load balancing logic
            if npu_util > 0.9 and igpu_util < 0.7:
                # NPU overloaded, shift some work to iGPU
                self.logger.debug("Shifting decode work from NPU to iGPU")
            elif igpu_util > 0.9 and npu_util < 0.7:
                # iGPU overloaded, shift some work to NPU
                self.logger.debug("Shifting prefill work from iGPU to NPU")
                
        except Exception as e:
            self.logger.error(f"Error adjusting scheduler: {e}")
    
    def _enter_baseline_mode(self):
        """Enter failsafe baseline mode"""
        if not self.baseline_mode:
            self.baseline_mode = True
            self.logger.warning("Entering baseline mode due to errors")
            
            # Reset to safe defaults
            self.kv_state["current_levels"] = {"L0": 100, "L1": 0, "L2": 0, "L3": 0}
    
    def _exit_baseline_mode(self):
        """Exit baseline mode if conditions are stable"""
        if self.baseline_mode and self.error_count == 0:
            self.baseline_mode = False
            self.logger.info("Exiting baseline mode - resuming optimization")
    
    async def _fast_control_loop(self):
        """Fast control loop (1ms) - Emergency response"""
        while self.running:
            try:
                # Monitor for emergency conditions
                memory = psutil.virtual_memory()
                if memory.percent > 95:
                    self.logger.warning("Emergency: High memory usage detected")
                    self._enter_baseline_mode()
                
                # Reset error count if stable
                if self.error_count > 0:
                    self.error_count = max(0, self.error_count - 1)
                
                await asyncio.sleep(self.fast_interval)
                
            except Exception as e:
                self.logger.error(f"Fast loop error: {e}")
                self.error_count += 1
                if self.error_count >= self.max_errors:
                    self._enter_baseline_mode()
    
    async def _mid_control_loop(self):
        """Mid control loop (10ms) - IOBinding and KV fine-tuning"""
        while self.running:
            try:
                if not self.baseline_mode:
                    self._optimize_iobinding()
                    # Fine-tune KV quantization levels
                    self._update_kv_quantization_policy()
                
                await asyncio.sleep(self.mid_interval)
                
            except Exception as e:
                self.logger.error(f"Mid loop error: {e}")
                self.error_count += 1
    
    async def _slow_control_loop(self):
        """Slow control loop (100ms) - High-level optimization"""
        while self.running:
            try:
                # Collect metrics
                self.metrics = self._collect_system_metrics()
                
                # Add to history
                self.performance_history.append(asdict(self.metrics))
                if len(self.performance_history) > self.max_history:
                    self.performance_history.pop(0)
                
                if not self.baseline_mode:
                    # Adjust hybrid scheduler
                    self._adjust_hybrid_scheduler()
                    
                    # Check if we can exit baseline mode
                    self._exit_baseline_mode()
                
                await asyncio.sleep(self.slow_interval)
                
            except Exception as e:
                self.logger.error(f"Slow loop error: {e}")
                self.error_count += 1
    
    async def start_control_loops(self):
        """Start all control loops"""
        self.running = True
        self.logger.info("Starting Infer-OS control loops")
        
        # Start all control loops concurrently
        await asyncio.gather(
            self._fast_control_loop(),
            self._mid_control_loop(),
            self._slow_control_loop()
        )
    
    def stop_control_loops(self):
        """Stop all control loops"""
        self.running = False
        self.logger.info("Stopping Infer-OS control loops")

# FastAPI application
app = FastAPI(title="Infer-OS Control Agent", version="1.0.0")
agent = InferOSControlAgent()

@app.post("/v1/policy")
async def set_policy(policy: PolicyConfig):
    """Set optimization policy"""
    try:
        agent.policy = policy
        agent.logger.info(f"Policy updated: {policy}")
        return JSONResponse(status_code=204, content={})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/run-context")
async def set_run_context(context: RunContext):
    """Set run context for next job"""
    try:
        agent._update_kv_quantization_policy(context)
        agent.logger.info(f"Run context updated: {context}")
        return JSONResponse(status_code=204, content={})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/metrics")
async def get_metrics():
    """Get current performance metrics"""
    try:
        return asdict(agent.metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok" if agent.running else "stopped",
        "baseline_mode": agent.baseline_mode,
        "error_count": agent.error_count,
        "platform": platform.system(),
        "version": "1.0.0"
    }

@app.get("/v1/history")
async def get_performance_history():
    """Get performance history"""
    return {
        "history": agent.performance_history[-100:],  # Last 100 entries
        "total_entries": len(agent.performance_history)
    }

@app.post("/v1/baseline")
async def toggle_baseline_mode(enable: bool = True):
    """Toggle baseline mode"""
    if enable:
        agent._enter_baseline_mode()
    else:
        agent.error_count = 0
        agent._exit_baseline_mode()
    
    return {"baseline_mode": agent.baseline_mode}

async def main():
    """Main entry point"""
    print("🚀 Starting Infer-OS Control Agent for AMD GAIA Integration")
    print(f"🪟 Platform: {platform.system()} {platform.release()}")
    print(f"🔧 Control loops: {agent.fast_interval*1000:.1f}ms / {agent.mid_interval*1000:.1f}ms / {agent.slow_interval*1000:.1f}ms")
    print(f"🌐 API server: http://127.0.0.1:7031")
    print()
    
    # Start control loops in background
    control_task = asyncio.create_task(agent.start_control_loops())
    
    # Start API server
    config = uvicorn.Config(
        app, 
        host="127.0.0.1", 
        port=7031, 
        log_level="info",
        access_log=False
    )
    server = uvicorn.Server(config)
    
    try:
        await server.serve()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Infer-OS Control Agent")
    finally:
        agent.stop_control_loops()
        control_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())

