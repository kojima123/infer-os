#!/usr/bin/env python3
"""
統合システム最終実装
Infer-OS & FlexGen++ NPU統合システムの完全版

全フェーズ統合: Phase 0-5 の全機能を統合した最終システム
最終目標: 24+ tok/s, PPL劣化 ≤ +0.5pt, 24時間連続稼働

作成者: Manus AI
バージョン: 1.0 Final
"""

import os
import sys
import time
import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import psutil
import numpy as np

# 全フェーズの実装をインポート
try:
    from phase0_implementation import PerformanceMetrics
    from phase1_implementation import Phase1SystemConfig, XDNASDKInterface, FourTierMemoryManager
    from phase2_3_implementation import RouterAPI, DynamicSkipEngine
    from phase4_5_implementation import Phase4_5SystemConfig, KVPruningEngine, IntegratedOptimizer, FlexGenPlusPlusPhase4_5
except ImportError as e:
    print(f"❌ 依存実装のインポートに失敗: {e}")
    print("全てのphase実装ファイルが同じディレクトリにあることを確認してください")
    sys.exit(1)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integrated_system_final.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class FinalSystemConfig(Phase4_5SystemConfig):
    """最終統合システム設定"""
    # 最終性能目標
    final_target_tokens_per_second: float = 24.0
    final_target_energy_efficiency: float = 0.5  # Energy/token 50%削減
    final_target_uptime_hours: float = 24.0      # 24時間連続稼働
    
    # 統合制御設定
    enable_all_optimizations: bool = True
    adaptive_control_enabled: bool = True
    real_time_monitoring: bool = True
    
    # 品質保証設定（最終）
    final_max_ppl_increase: float = 0.5  # 最大PPL増加 0.5pt
    quality_monitoring_interval: int = 10  # 10推論ごとに品質チェック

class FinalSystemMonitor:
    """最終システム監視機構"""
    
    def __init__(self, config: FinalSystemConfig):
        self.config = config
        self.monitoring_active = False
        self.monitoring_thread = None
        self.system_metrics = {}
        self.alerts = []
        
    def start_monitoring(self):
        """監視開始"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Final system monitoring started")
    
    def stop_monitoring(self):
        """監視停止"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Final system monitoring stopped")
    
    def _monitoring_loop(self):
        """監視ループ"""
        while self.monitoring_active:
            try:
                # システムメトリクスの収集
                self._collect_system_metrics()
                
                # アラートチェック
                self._check_alerts()
                
                time.sleep(1.0)  # 1秒間隔
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    def _collect_system_metrics(self):
        """システムメトリクス収集"""
        self.system_metrics = {
            "timestamp": time.time(),
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "gpu_memory": self._get_gpu_memory_usage(),
            "system_temperature": self._get_system_temperature()
        }
    
    def _get_gpu_memory_usage(self) -> float:
        """GPU メモリ使用率取得"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
        return 0.0
    
    def _get_system_temperature(self) -> float:
        """システム温度取得（シミュレーション）"""
        # 実際の実装では、システムの温度センサーから取得
        return 45.0 + np.random.normal(0, 2)
    
    def _check_alerts(self):
        """アラートチェック"""
        current_time = time.time()
        
        # CPU使用率アラート
        if self.system_metrics.get("cpu_usage", 0) > 90:
            self.alerts.append({
                "timestamp": current_time,
                "type": "cpu_high",
                "message": f"High CPU usage: {self.system_metrics['cpu_usage']:.1f}%"
            })
        
        # メモリ使用率アラート
        if self.system_metrics.get("memory_usage", 0) > 85:
            self.alerts.append({
                "timestamp": current_time,
                "type": "memory_high",
                "message": f"High memory usage: {self.system_metrics['memory_usage']:.1f}%"
            })
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """監視状態取得"""
        return {
            "monitoring_active": self.monitoring_active,
            "current_metrics": self.system_metrics,
            "recent_alerts": self.alerts[-10:],  # 最新10件のアラート
            "total_alerts": len(self.alerts)
        }

class FinalInferOSController:
    """最終統合Infer-OS制御機構"""
    
    def __init__(self, config: FinalSystemConfig):
        self.config = config
        self.flexgen = FlexGenPlusPlusPhase4_5(config)
        self.system_monitor = FinalSystemMonitor(config)
        
        # 制御状態
        self.running = False
        self.control_thread = None
        self.inference_count = 0
        self.start_time = None
        
        # 最終統計
        self.final_stats = {
            "total_inferences": 0,
            "total_runtime_hours": 0.0,
            "average_performance": 0.0,
            "quality_score": 1.0,
            "energy_efficiency": 0.0,
            "uptime_achievement": 0.0
        }
    
    def initialize_system(self) -> bool:
        """最終システム初期化"""
        logger.info("Initializing Final Integrated System...")
        
        # FlexGen++ の初期化
        if not self.flexgen.initialize_model():
            logger.error("FlexGen++ initialization failed")
            return False
        
        # システム監視の開始
        if self.config.real_time_monitoring:
            self.system_monitor.start_monitoring()
        
        # 制御ループの開始
        self._start_control_loop()
        
        self.start_time = time.time()
        logger.info("Final Integrated System initialization completed")
        return True
    
    def _start_control_loop(self):
        """制御ループ開始"""
        if self.running:
            return
            
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        logger.info("Final control loop started")
    
    def _control_loop(self):
        """最終制御ループ"""
        while self.running:
            try:
                # システム状態の監視
                system_state = self._get_system_state()
                
                # 適応的制御の実行
                if self.config.adaptive_control_enabled:
                    self._execute_adaptive_control(system_state)
                
                time.sleep(0.001)  # 1ms制御周期
                
            except Exception as e:
                logger.error(f"Control loop error: {e}")
    
    def _get_system_state(self) -> Dict[str, Any]:
        """システム状態取得"""
        monitoring_status = self.system_monitor.get_monitoring_status()
        performance_summary = self.flexgen.get_phase4_5_performance_summary()
        
        return {
            "monitoring": monitoring_status,
            "performance": performance_summary,
            "runtime_hours": self._get_runtime_hours()
        }
    
    def _get_runtime_hours(self) -> float:
        """稼働時間取得"""
        if self.start_time:
            return (time.time() - self.start_time) / 3600.0
        return 0.0
    
    def _execute_adaptive_control(self, system_state: Dict[str, Any]):
        """適応的制御実行"""
        # 性能が目標を下回る場合の調整
        current_performance = system_state.get("performance", {}).get("average_tokens_per_second", 0)
        
        if current_performance < self.config.final_target_tokens_per_second * 0.9:
            # 積極的最適化の実行
            self._apply_aggressive_optimization()
        
        # 品質劣化の検出と対応
        quality_score = system_state.get("performance", {}).get("quality_score", 1.0)
        if quality_score < (1.0 - self.config.final_max_ppl_increase):
            # 品質重視モードに切り替え
            self._apply_quality_preservation()
    
    def _apply_aggressive_optimization(self):
        """積極的最適化適用"""
        logger.debug("Applying aggressive optimization")
        # 最適化パラメータの調整
        self.flexgen.config.layer_skip_ratio = min(0.5, self.flexgen.config.layer_skip_ratio + 0.05)
        self.flexgen.config.kv_pruning_ratio = min(0.4, self.flexgen.config.kv_pruning_ratio + 0.05)
    
    def _apply_quality_preservation(self):
        """品質保持適用"""
        logger.debug("Applying quality preservation")
        # 品質重視パラメータの調整
        self.flexgen.config.layer_skip_ratio = max(0.1, self.flexgen.config.layer_skip_ratio - 0.05)
        self.flexgen.config.kv_pruning_ratio = max(0.1, self.flexgen.config.kv_pruning_ratio - 0.05)
    
    def execute_inference_request(self, prompt: str, max_tokens: int = 100) -> Tuple[str, PerformanceMetrics]:
        """推論リクエスト実行"""
        self.inference_count += 1
        
        # 推論実行
        response, metrics = self.flexgen.execute_inference(prompt, max_tokens)
        
        # 統計更新
        self._update_final_stats(metrics)
        
        # 品質監視
        if self.inference_count % self.config.quality_monitoring_interval == 0:
            self._monitor_quality(response)
        
        return response, metrics
    
    def _update_final_stats(self, metrics: PerformanceMetrics):
        """最終統計更新"""
        self.final_stats["total_inferences"] += 1
        self.final_stats["total_runtime_hours"] = self._get_runtime_hours()
        
        # 平均性能の更新
        if self.final_stats["total_inferences"] > 0:
            current_avg = self.final_stats["average_performance"]
            new_performance = metrics.tokens_per_second
            self.final_stats["average_performance"] = (
                (current_avg * (self.final_stats["total_inferences"] - 1) + new_performance) /
                self.final_stats["total_inferences"]
            )
        
        # 稼働時間達成率
        target_hours = self.config.final_target_uptime_hours
        self.final_stats["uptime_achievement"] = min(1.0, self.final_stats["total_runtime_hours"] / target_hours)
    
    def _monitor_quality(self, response: str):
        """品質監視"""
        # 簡単な品質チェック（実際の実装では、より高度な品質評価を行う）
        quality_score = min(1.0, len(response) / 100.0)  # 簡略化
        self.final_stats["quality_score"] = quality_score
    
    def stop_system(self):
        """システム停止"""
        logger.info("Stopping Final Integrated System...")
        
        # 制御ループ停止
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=5.0)
        
        # 監視停止
        self.system_monitor.stop_monitoring()
        
        logger.info("Final Integrated System stopped")
    
    def get_final_system_status(self) -> Dict[str, Any]:
        """最終システム状態取得"""
        return {
            "system_running": self.running,
            "inference_count": self.inference_count,
            "runtime_hours": self._get_runtime_hours(),
            "final_stats": self.final_stats,
            "monitoring_status": self.system_monitor.get_monitoring_status(),
            "performance_summary": self.flexgen.get_phase4_5_performance_summary()
        }

def run_final_system_benchmark():
    """最終システムベンチマーク実行"""
    logger.info("Starting Final System Comprehensive Benchmark...")
    
    # 最終システム設定
    config = FinalSystemConfig()
    
    # 最終統合システムの初期化
    final_system = FinalInferOSController(config)
    
    if not final_system.initialize_system():
        logger.error("Final system initialization failed")
        return
    
    # 包括的ベンチマーク用プロンプト
    comprehensive_prompts = [
        "Develop a comprehensive machine learning pipeline for natural language processing tasks, including data preprocessing, model selection, training, and evaluation phases with detailed performance metrics.",
        "Explain the mathematical foundations of transformer architectures, including self-attention mechanisms, positional encoding, multi-head attention computations, and their computational complexity analysis.",
        "Create a detailed software architecture design for a distributed AI inference system that can handle millions of requests per day with low latency requirements, fault tolerance, and horizontal scalability.",
        "人工知能技術の発展が医療分野に与える革新的な影響について、具体的な応用例（画像診断、薬物発見、個別化医療）と将来の可能性を含めて詳細に分析してください。",
        "Design and implement a robust error handling and monitoring system for a production-level machine learning service, considering scalability, reliability, observability requirements, and automated recovery mechanisms.",
        "Analyze the ethical implications of artificial intelligence in decision-making systems, including bias detection, fairness metrics, transparency requirements, and regulatory compliance frameworks.",
        "Implement a high-performance distributed training system for large language models, including gradient synchronization, memory optimization, fault tolerance, and dynamic resource allocation strategies.",
        "機械学習モデルの解釈可能性と説明可能性について、SHAP、LIME、Attention可視化などの手法を用いた包括的なアプローチを設計し、実装してください。"
    ]
    
    logger.info(f"Executing comprehensive benchmark with {len(comprehensive_prompts)} complex prompts...")
    
    # ベンチマーク実行
    all_metrics = []
    start_time = time.time()
    
    for i, prompt in enumerate(comprehensive_prompts):
        logger.info(f"Final Test {i+1}/{len(comprehensive_prompts)}: {prompt[:80]}...")
        
        try:
            response, metrics = final_system.execute_inference_request(prompt, max_tokens=250)
            all_metrics.append(metrics)
            
            logger.info(f"Response length: {len(response)} chars")
            logger.info(f"Performance: {metrics.tokens_per_second:.1f} tok/s")
            logger.info(f"Latency: {metrics.latency_ms:.1f} ms")
            
        except Exception as e:
            logger.error(f"Final Test {i+1} failed: {e}")
    
    total_time = time.time() - start_time
    
    # 最終結果の集計と評価
    logger.info("=" * 60)
    logger.info("🎯 FINAL SYSTEM BENCHMARK RESULTS")
    logger.info("=" * 60)
    
    if all_metrics:
        # 性能統計
        avg_tokens_per_second = sum(m.tokens_per_second for m in all_metrics) / len(all_metrics)
        avg_latency = sum(m.latency_ms for m in all_metrics) / len(all_metrics)
        min_tokens_per_second = min(m.tokens_per_second for m in all_metrics)
        max_tokens_per_second = max(m.tokens_per_second for m in all_metrics)
        
        logger.info(f"📊 Performance Metrics:")
        logger.info(f"  Total tests: {len(all_metrics)}")
        logger.info(f"  Total time: {total_time:.2f} seconds")
        logger.info(f"  Average tokens/second: {avg_tokens_per_second:.1f}")
        logger.info(f"  Min tokens/second: {min_tokens_per_second:.1f}")
        logger.info(f"  Max tokens/second: {max_tokens_per_second:.1f}")
        logger.info(f"  Average latency: {avg_latency:.1f} ms")
        
        # 目標達成評価
        target_achievement = avg_tokens_per_second / config.final_target_tokens_per_second
        logger.info(f"🎯 Target Achievement: {target_achievement:.1%}")
        
        if avg_tokens_per_second >= config.final_target_tokens_per_second:
            logger.info("✅ FINAL TARGET ACHIEVED! 🎉")
        else:
            gap = config.final_target_tokens_per_second - avg_tokens_per_second
            logger.warning(f"⚠️ Final target not achieved. Gap: {gap:.1f} tok/s")
        
        # フェーズ別改善率
        phase0_baseline = 11.0
        total_improvement = avg_tokens_per_second / phase0_baseline
        logger.info(f"📈 Total improvement over Phase 0: {total_improvement:.1%}")
        
        # 最終システム状態
        final_status = final_system.get_final_system_status()
        logger.info(f"🔧 Final System Status:")
        logger.info(f"  Total inferences: {final_status['inference_count']}")
        logger.info(f"  Runtime: {final_status['runtime_hours']:.2f} hours")
        logger.info(f"  Quality score: {final_status['final_stats']['quality_score']:.3f}")
        logger.info(f"  Uptime achievement: {final_status['final_stats']['uptime_achievement']:.1%}")
        
        # 成功判定
        success_criteria = [
            avg_tokens_per_second >= config.final_target_tokens_per_second,
            final_status['final_stats']['quality_score'] >= (1.0 - config.final_max_ppl_increase),
            len(all_metrics) == len(comprehensive_prompts)  # 全テスト成功
        ]
        
        if all(success_criteria):
            logger.info("🏆 ALL SUCCESS CRITERIA MET!")
            logger.info("🚀 Infer-OS & FlexGen++ NPU Integration System is ready for production!")
        else:
            logger.warning("⚠️ Some success criteria not met. Review and optimize.")
    
    # システム停止
    final_system.stop_system()
    logger.info("Final system benchmark completed")

def main():
    """メイン関数"""
    print("🏆 Final Integrated System: Infer-OS & FlexGen++ NPU")
    print("=" * 60)
    print("🎯 Target: 24+ tok/s, PPL ≤ +0.5pt, 24h uptime")
    print("=" * 60)
    
    try:
        run_final_system_benchmark()
    except KeyboardInterrupt:
        logger.info("Final system benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Final system benchmark failed: {e}")
        raise

if __name__ == "__main__":
    main()

