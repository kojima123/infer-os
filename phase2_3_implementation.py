#!/usr/bin/env python3
"""
Phase 2-3: Router API実装
Layer Skip、FFN Skip、Token Haltingによる動的最適化

Phase 2 目標性能: 18-19 tok/s (Layer Skip)
Phase 3 目標性能: 20-21 tok/s (FFN Skip + Token Halting)
主要実装: 統合Router API、動的スキップ機構、推論最適化

作成者: Manus AI
バージョン: 1.0
"""

import os
import sys
import time
import json
import logging
import threading
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import psutil
import numpy as np

# Phase 1 実装のインポート
try:
    from phase1_implementation import (
        Phase1SystemConfig,
        NPUMetrics,
        XDNASDKInterface,
        FourTierMemoryManager,
        FlexGenPlusPlusPhase1,
        InferOSControllerPhase1
    )
    from phase0_implementation import PerformanceMetrics
except ImportError as e:
    print(f"❌ Phase 1 実装のインポートに失敗: {e}")
    print("phase1_implementation.py が同じディレクトリにあることを確認してください")
    sys.exit(1)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase2_3_implementation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SkipType(Enum):
    """スキップタイプ"""
    LAYER_SKIP = "layer_skip"
    FFN_SKIP = "ffn_skip"
    ATTENTION_SKIP = "attention_skip"
    TOKEN_HALTING = "token_halting"

class SkipDecision(Enum):
    """スキップ判定"""
    EXECUTE = "execute"
    SKIP = "skip"
    PARTIAL = "partial"

@dataclass
class Phase2_3SystemConfig(Phase1SystemConfig):
    """Phase 2-3 システム設定"""
    # Router API設定
    enable_layer_skip: bool = True
    enable_ffn_skip: bool = True
    enable_token_halting: bool = True
    
    # Layer Skip設定
    layer_skip_threshold: float = 0.1  # スキップ閾値
    layer_skip_ratio: float = 0.3      # 最大30%の層をスキップ
    
    # FFN Skip設定
    ffn_skip_threshold: float = 0.15
    ffn_skip_ratio: float = 0.4        # 最大40%のFFNをスキップ
    
    # Token Halting設定
    token_halting_threshold: float = 0.95  # 信頼度95%で停止
    min_tokens: int = 10               # 最低生成トークン数
    max_tokens_per_step: int = 5       # ステップあたり最大トークン数
    
    # Phase 2-3 性能目標
    phase2_target_tokens_per_second: float = 18.5  # Phase 2 目標
    phase3_target_tokens_per_second: float = 20.5  # Phase 3 目標
    
    # Router API制御設定
    router_update_frequency: int = 100  # 100推論ごとに更新
    adaptive_threshold: bool = True     # 適応的閾値調整
    
    # 品質保証設定
    max_quality_degradation: float = 0.05  # 最大5%の品質劣化許容

@dataclass
class SkipMetrics:
    """スキップメトリクス"""
    layer_skips: int = 0
    ffn_skips: int = 0
    token_halts: int = 0
    total_layers: int = 0
    total_ffns: int = 0
    total_tokens: int = 0
    
    skip_ratio_layer: float = 0.0
    skip_ratio_ffn: float = 0.0
    halt_ratio_token: float = 0.0
    
    quality_score: float = 1.0
    performance_gain: float = 0.0

@dataclass
class RouterDecision:
    """Router判定結果"""
    skip_type: SkipType
    decision: SkipDecision
    confidence: float
    layer_index: Optional[int] = None
    token_index: Optional[int] = None
    reasoning: str = ""

class LayerImportanceAnalyzer:
    """層重要度分析器"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_importance_cache = {}
        self.attention_patterns = {}
        self.ffn_activation_patterns = {}
        
        # 層の重要度を事前計算
        self._precompute_layer_importance()
    
    def _precompute_layer_importance(self):
        """層重要度の事前計算"""
        logger.info("Precomputing layer importance...")
        
        # モデルの層を分析
        layer_count = 0
        for name, module in self.model.named_modules():
            if self._is_transformer_layer(module):
                importance = self._calculate_layer_importance(name, module)
                self.layer_importance_cache[layer_count] = importance
                layer_count += 1
        
        logger.info(f"Analyzed {layer_count} transformer layers")
    
    def _is_transformer_layer(self, module: nn.Module) -> bool:
        """Transformer層の判定"""
        # 一般的なTransformer層の特徴を検出
        has_attention = any("attention" in name.lower() for name, _ in module.named_modules())
        has_ffn = any("mlp" in name.lower() or "ffn" in name.lower() for name, _ in module.named_modules())
        return has_attention and has_ffn
    
    def _calculate_layer_importance(self, layer_name: str, layer_module: nn.Module) -> float:
        """層重要度の計算"""
        # パラメータ数に基づく重要度
        param_count = sum(p.numel() for p in layer_module.parameters())
        
        # 層の位置に基づく重要度（中間層ほど重要）
        layer_position_weight = self._get_position_weight(layer_name)
        
        # 正規化された重要度
        importance = (param_count / 1e6) * layer_position_weight
        return min(1.0, importance)
    
    def _get_position_weight(self, layer_name: str) -> float:
        """層位置に基づく重み"""
        # 層番号を抽出（簡略化）
        import re
        numbers = re.findall(r'\d+', layer_name)
        if numbers:
            layer_num = int(numbers[0])
            # 中間層（全体の30-70%）に高い重みを付与
            total_layers = 32  # 仮定
            position_ratio = layer_num / total_layers
            
            if 0.3 <= position_ratio <= 0.7:
                return 1.0  # 中間層は重要
            elif position_ratio < 0.3:
                return 0.7  # 初期層は中程度
            else:
                return 0.8  # 後期層は重要
        
        return 0.5  # デフォルト
    
    def get_layer_importance(self, layer_index: int) -> float:
        """層重要度の取得"""
        return self.layer_importance_cache.get(layer_index, 0.5)
    
    def analyze_attention_pattern(self, attention_weights: torch.Tensor, layer_index: int) -> Dict[str, float]:
        """Attentionパターンの分析"""
        # Attention重みの統計分析
        attention_entropy = self._calculate_attention_entropy(attention_weights)
        attention_sparsity = self._calculate_attention_sparsity(attention_weights)
        
        pattern = {
            "entropy": attention_entropy,
            "sparsity": attention_sparsity,
            "max_weight": float(torch.max(attention_weights)),
            "mean_weight": float(torch.mean(attention_weights))
        }
        
        self.attention_patterns[layer_index] = pattern
        return pattern
    
    def _calculate_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Attention重みのエントロピー計算"""
        # 正規化されたAttention重みのエントロピー
        probs = F.softmax(attention_weights.flatten(), dim=0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        return float(entropy)
    
    def _calculate_attention_sparsity(self, attention_weights: torch.Tensor) -> float:
        """Attention重みのスパース性計算"""
        threshold = 0.01
        sparse_ratio = float(torch.sum(attention_weights < threshold)) / attention_weights.numel()
        return sparse_ratio

class RouterAPI:
    """統合Router API"""
    
    def __init__(self, config: Phase2_3SystemConfig, model: nn.Module):
        self.config = config
        self.model = model
        self.layer_analyzer = LayerImportanceAnalyzer(model)
        
        # Router状態
        self.decision_history = []
        self.performance_history = []
        self.quality_history = []
        
        # 適応的閾値
        self.adaptive_thresholds = {
            SkipType.LAYER_SKIP: config.layer_skip_threshold,
            SkipType.FFN_SKIP: config.ffn_skip_threshold,
            SkipType.TOKEN_HALTING: config.token_halting_threshold
        }
        
        # 統計情報
        self.router_stats = {
            "total_decisions": 0,
            "skip_decisions": 0,
            "execute_decisions": 0,
            "partial_decisions": 0,
            "accuracy": 0.0
        }
    
    def make_layer_skip_decision(self, layer_index: int, 
                                hidden_states: torch.Tensor,
                                attention_weights: Optional[torch.Tensor] = None) -> RouterDecision:
        """Layer Skip判定"""
        
        # 層重要度の取得
        layer_importance = self.layer_analyzer.get_layer_importance(layer_index)
        
        # 隠れ状態の変化量分析
        state_change_magnitude = self._analyze_state_change(hidden_states)
        
        # Attentionパターンの分析（利用可能な場合）
        attention_score = 1.0
        if attention_weights is not None:
            attention_pattern = self.layer_analyzer.analyze_attention_pattern(attention_weights, layer_index)
            attention_score = 1.0 - attention_pattern["sparsity"]  # スパース性が高いほどスキップ候補
        
        # 総合判定スコア
        skip_score = (
            (1.0 - layer_importance) * 0.4 +
            (1.0 - state_change_magnitude) * 0.4 +
            (1.0 - attention_score) * 0.2
        )
        
        # 閾値との比較
        threshold = self.adaptive_thresholds[SkipType.LAYER_SKIP]
        
        if skip_score > threshold:
            decision = SkipDecision.SKIP
            reasoning = f"Low importance ({layer_importance:.3f}), low change ({state_change_magnitude:.3f})"
        else:
            decision = SkipDecision.EXECUTE
            reasoning = f"High importance ({layer_importance:.3f}) or significant change ({state_change_magnitude:.3f})"
        
        router_decision = RouterDecision(
            skip_type=SkipType.LAYER_SKIP,
            decision=decision,
            confidence=abs(skip_score - threshold),
            layer_index=layer_index,
            reasoning=reasoning
        )
        
        self._record_decision(router_decision)
        return router_decision
    
    def make_ffn_skip_decision(self, layer_index: int,
                              ffn_input: torch.Tensor,
                              attention_output: torch.Tensor) -> RouterDecision:
        """FFN Skip判定"""
        
        # FFN入力の分析
        input_magnitude = torch.norm(ffn_input).item()
        input_sparsity = self._calculate_tensor_sparsity(ffn_input)
        
        # Attention出力との相関分析
        attention_magnitude = torch.norm(attention_output).item()
        correlation = self._calculate_correlation(ffn_input, attention_output)
        
        # FFNスキップスコア
        skip_score = (
            (1.0 - min(1.0, input_magnitude / 10.0)) * 0.3 +  # 入力の大きさ
            input_sparsity * 0.3 +                             # 入力のスパース性
            (1.0 - abs(correlation)) * 0.4                     # Attentionとの相関
        )
        
        threshold = self.adaptive_thresholds[SkipType.FFN_SKIP]
        
        if skip_score > threshold:
            decision = SkipDecision.SKIP
            reasoning = f"Low input magnitude ({input_magnitude:.3f}), high sparsity ({input_sparsity:.3f})"
        else:
            decision = SkipDecision.EXECUTE
            reasoning = f"Significant input ({input_magnitude:.3f}) or strong correlation ({correlation:.3f})"
        
        router_decision = RouterDecision(
            skip_type=SkipType.FFN_SKIP,
            decision=decision,
            confidence=abs(skip_score - threshold),
            layer_index=layer_index,
            reasoning=reasoning
        )
        
        self._record_decision(router_decision)
        return router_decision
    
    def make_token_halting_decision(self, token_index: int,
                                   current_logits: torch.Tensor,
                                   generated_tokens: List[int],
                                   target_length: int) -> RouterDecision:
        """Token Halting判定"""
        
        # 現在のトークン予測の信頼度
        token_confidence = self._calculate_token_confidence(current_logits)
        
        # 生成済みトークンの一貫性
        sequence_consistency = self._calculate_sequence_consistency(generated_tokens)
        
        # 長さに基づく停止判定
        length_factor = min(1.0, len(generated_tokens) / target_length)
        
        # 停止スコア
        halt_score = (
            token_confidence * 0.5 +
            sequence_consistency * 0.3 +
            length_factor * 0.2
        )
        
        threshold = self.adaptive_thresholds[SkipType.TOKEN_HALTING]
        
        # 最小トークン数の確認
        if len(generated_tokens) < self.config.min_tokens:
            decision = SkipDecision.EXECUTE
            reasoning = f"Below minimum tokens ({len(generated_tokens)} < {self.config.min_tokens})"
        elif halt_score > threshold:
            decision = SkipDecision.SKIP  # 生成停止
            reasoning = f"High confidence ({token_confidence:.3f}), consistent sequence ({sequence_consistency:.3f})"
        else:
            decision = SkipDecision.EXECUTE  # 生成継続
            reasoning = f"Low confidence ({token_confidence:.3f}) or inconsistent sequence"
        
        router_decision = RouterDecision(
            skip_type=SkipType.TOKEN_HALTING,
            decision=decision,
            confidence=abs(halt_score - threshold),
            token_index=token_index,
            reasoning=reasoning
        )
        
        self._record_decision(router_decision)
        return router_decision
    
    def _analyze_state_change(self, hidden_states: torch.Tensor) -> float:
        """隠れ状態の変化量分析"""
        # 隠れ状態のノルム
        state_norm = torch.norm(hidden_states).item()
        
        # 正規化された変化量（0-1）
        normalized_change = min(1.0, state_norm / 100.0)
        return normalized_change
    
    def _calculate_tensor_sparsity(self, tensor: torch.Tensor, threshold: float = 0.01) -> float:
        """テンソルのスパース性計算"""
        sparse_elements = torch.sum(torch.abs(tensor) < threshold).item()
        total_elements = tensor.numel()
        return sparse_elements / total_elements
    
    def _calculate_correlation(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        """テンソル間の相関計算"""
        # 平坦化して相関係数を計算
        flat1 = tensor1.flatten()
        flat2 = tensor2.flatten()
        
        # 長さを合わせる
        min_len = min(len(flat1), len(flat2))
        flat1 = flat1[:min_len]
        flat2 = flat2[:min_len]
        
        # 相関係数の計算
        correlation = torch.corrcoef(torch.stack([flat1, flat2]))[0, 1]
        return float(correlation) if not torch.isnan(correlation) else 0.0
    
    def _calculate_token_confidence(self, logits: torch.Tensor) -> float:
        """トークン予測の信頼度計算"""
        # ソフトマックス確率の最大値
        probs = F.softmax(logits, dim=-1)
        max_prob = torch.max(probs).item()
        
        # エントロピーに基づく信頼度
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        normalized_entropy = entropy / math.log(probs.size(-1))  # 正規化
        
        # 信頼度スコア
        confidence = max_prob * (1.0 - normalized_entropy)
        return confidence
    
    def _calculate_sequence_consistency(self, tokens: List[int]) -> float:
        """シーケンスの一貫性計算"""
        if len(tokens) < 2:
            return 1.0
        
        # 隣接トークンの変化率
        changes = sum(1 for i in range(1, len(tokens)) if tokens[i] != tokens[i-1])
        change_rate = changes / (len(tokens) - 1)
        
        # 一貫性スコア（変化率が低いほど一貫性が高い）
        consistency = 1.0 - min(1.0, change_rate)
        return consistency
    
    def _record_decision(self, decision: RouterDecision):
        """判定記録"""
        self.decision_history.append({
            "timestamp": time.time(),
            "decision": decision
        })
        
        # 統計更新
        self.router_stats["total_decisions"] += 1
        
        if decision.decision == SkipDecision.SKIP:
            self.router_stats["skip_decisions"] += 1
        elif decision.decision == SkipDecision.EXECUTE:
            self.router_stats["execute_decisions"] += 1
        else:
            self.router_stats["partial_decisions"] += 1
    
    def update_adaptive_thresholds(self, performance_feedback: Dict[str, float]):
        """適応的閾値の更新"""
        if not self.config.adaptive_threshold:
            return
        
        # 性能フィードバックに基づく閾値調整
        performance_score = performance_feedback.get("tokens_per_second", 0.0)
        quality_score = performance_feedback.get("quality_score", 1.0)
        
        # 性能が目標を下回る場合は閾値を下げる（スキップを減らす）
        if performance_score < self.config.phase2_target_tokens_per_second:
            adjustment = -0.01
        else:
            adjustment = 0.01
        
        # 品質が劣化している場合は閾値を下げる
        if quality_score < (1.0 - self.config.max_quality_degradation):
            adjustment = -0.02
        
        # 閾値の更新
        for skip_type in self.adaptive_thresholds:
            old_threshold = self.adaptive_thresholds[skip_type]
            new_threshold = max(0.0, min(1.0, old_threshold + adjustment))
            self.adaptive_thresholds[skip_type] = new_threshold
            
            if abs(new_threshold - old_threshold) > 0.001:
                logger.debug(f"Threshold updated for {skip_type}: {old_threshold:.3f} → {new_threshold:.3f}")
    
    def get_router_statistics(self) -> Dict[str, Any]:
        """Router統計の取得"""
        stats = self.router_stats.copy()
        
        # 追加統計の計算
        if stats["total_decisions"] > 0:
            stats["skip_ratio"] = stats["skip_decisions"] / stats["total_decisions"]
            stats["execute_ratio"] = stats["execute_decisions"] / stats["total_decisions"]
        
        stats["adaptive_thresholds"] = self.adaptive_thresholds.copy()
        stats["decision_history_length"] = len(self.decision_history)
        
        return stats

class DynamicSkipEngine:
    """動的スキップエンジン"""
    
    def __init__(self, config: Phase2_3SystemConfig, router_api: RouterAPI):
        self.config = config
        self.router_api = router_api
        
        # スキップ実行統計
        self.skip_metrics = SkipMetrics()
        self.execution_history = []
        
        # 品質監視
        self.quality_monitor = QualityMonitor(config)
        
        # 性能最適化
        self.performance_optimizer = PerformanceOptimizer(config)
    
    def execute_layer_with_skip(self, layer_module: nn.Module, 
                               layer_index: int,
                               hidden_states: torch.Tensor,
                               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """スキップ機能付き層実行"""
        
        # Router判定
        decision = self.router_api.make_layer_skip_decision(layer_index, hidden_states)
        
        if decision.decision == SkipDecision.SKIP:
            # 層をスキップ
            self.skip_metrics.layer_skips += 1
            logger.debug(f"Layer {layer_index} skipped: {decision.reasoning}")
            return hidden_states  # 入力をそのまま返す
        
        elif decision.decision == SkipDecision.PARTIAL:
            # 部分実行（簡略化）
            output = self._execute_partial_layer(layer_module, hidden_states, attention_mask)
            logger.debug(f"Layer {layer_index} partially executed")
            return output
        
        else:
            # 通常実行
            output = layer_module(hidden_states, attention_mask=attention_mask)
            if isinstance(output, tuple):
                output = output[0]  # hidden_statesのみ取得
            return output
    
    def execute_ffn_with_skip(self, ffn_module: nn.Module,
                             layer_index: int,
                             ffn_input: torch.Tensor,
                             attention_output: torch.Tensor) -> torch.Tensor:
        """スキップ機能付きFFN実行"""
        
        # Router判定
        decision = self.router_api.make_ffn_skip_decision(layer_index, ffn_input, attention_output)
        
        if decision.decision == SkipDecision.SKIP:
            # FFNをスキップ
            self.skip_metrics.ffn_skips += 1
            logger.debug(f"FFN {layer_index} skipped: {decision.reasoning}")
            return ffn_input  # 入力をそのまま返す
        
        elif decision.decision == SkipDecision.PARTIAL:
            # 部分実行
            output = self._execute_partial_ffn(ffn_module, ffn_input)
            logger.debug(f"FFN {layer_index} partially executed")
            return output
        
        else:
            # 通常実行
            return ffn_module(ffn_input)
    
    def execute_generation_with_halting(self, model: nn.Module,
                                      tokenizer,
                                      input_ids: torch.Tensor,
                                      max_tokens: int) -> Tuple[torch.Tensor, List[RouterDecision]]:
        """Token Halting機能付き生成"""
        
        generated_tokens = input_ids[0].tolist()
        halting_decisions = []
        
        for step in range(max_tokens):
            # 現在の入力で推論
            with torch.no_grad():
                outputs = model(torch.tensor([generated_tokens]).to(input_ids.device))
                logits = outputs.logits[0, -1, :]
            
            # Token Halting判定
            decision = self.router_api.make_token_halting_decision(
                step, logits, generated_tokens, max_tokens
            )
            halting_decisions.append(decision)
            
            if decision.decision == SkipDecision.SKIP:
                # 生成停止
                self.skip_metrics.token_halts += 1
                logger.debug(f"Token generation halted at step {step}: {decision.reasoning}")
                break
            
            # 次のトークンを生成
            next_token = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
            generated_tokens.append(next_token)
            
            # EOS トークンで停止
            if next_token == tokenizer.eos_token_id:
                break
        
        return torch.tensor([generated_tokens]), halting_decisions
    
    def _execute_partial_layer(self, layer_module: nn.Module,
                              hidden_states: torch.Tensor,
                              attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """層の部分実行"""
        # 簡略化された層実行（例：Attentionのみ実行、FFNスキップ）
        # 実際の実装では、層の構造に応じて適切な部分実行を行う
        
        # ここでは簡単な線形変換のみ実行
        if hasattr(layer_module, 'attention'):
            attention_output = layer_module.attention(hidden_states, attention_mask=attention_mask)
            if isinstance(attention_output, tuple):
                attention_output = attention_output[0]
            return attention_output
        
        return hidden_states
    
    def _execute_partial_ffn(self, ffn_module: nn.Module, ffn_input: torch.Tensor) -> torch.Tensor:
        """FFNの部分実行"""
        # FFNの最初の層のみ実行（例）
        if hasattr(ffn_module, 'dense') or hasattr(ffn_module, 'fc1'):
            first_layer = getattr(ffn_module, 'dense', getattr(ffn_module, 'fc1', None))
            if first_layer:
                return first_layer(ffn_input)
        
        return ffn_input
    
    def update_skip_metrics(self):
        """スキップメトリクスの更新"""
        if self.skip_metrics.total_layers > 0:
            self.skip_metrics.skip_ratio_layer = self.skip_metrics.layer_skips / self.skip_metrics.total_layers
        
        if self.skip_metrics.total_ffns > 0:
            self.skip_metrics.skip_ratio_ffn = self.skip_metrics.ffn_skips / self.skip_metrics.total_ffns
        
        if self.skip_metrics.total_tokens > 0:
            self.skip_metrics.halt_ratio_token = self.skip_metrics.token_halts / self.skip_metrics.total_tokens
    
    def get_skip_statistics(self) -> Dict[str, Any]:
        """スキップ統計の取得"""
        self.update_skip_metrics()
        
        return {
            "layer_skips": self.skip_metrics.layer_skips,
            "ffn_skips": self.skip_metrics.ffn_skips,
            "token_halts": self.skip_metrics.token_halts,
            "skip_ratio_layer": self.skip_metrics.skip_ratio_layer,
            "skip_ratio_ffn": self.skip_metrics.skip_ratio_ffn,
            "halt_ratio_token": self.skip_metrics.halt_ratio_token,
            "quality_score": self.skip_metrics.quality_score,
            "performance_gain": self.skip_metrics.performance_gain
        }

class QualityMonitor:
    """品質監視機構"""
    
    def __init__(self, config: Phase2_3SystemConfig):
        self.config = config
        self.quality_history = []
        self.baseline_quality = 1.0
        
    def evaluate_quality(self, original_output: str, optimized_output: str) -> float:
        """品質評価"""
        # 簡単な品質評価（実際の実装では、より高度な評価手法を使用）
        
        # 長さの類似性
        length_similarity = min(len(optimized_output), len(original_output)) / max(len(optimized_output), len(original_output))
        
        # 文字レベルの類似性
        char_similarity = self._calculate_char_similarity(original_output, optimized_output)
        
        # 総合品質スコア
        quality_score = (length_similarity * 0.3 + char_similarity * 0.7)
        
        self.quality_history.append({
            "timestamp": time.time(),
            "quality_score": quality_score,
            "length_similarity": length_similarity,
            "char_similarity": char_similarity
        })
        
        return quality_score
    
    def _calculate_char_similarity(self, text1: str, text2: str) -> float:
        """文字レベル類似性の計算"""
        # 簡単なレーベンシュタイン距離ベースの類似性
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        # 簡略化された類似性計算
        common_chars = set(text1) & set(text2)
        total_chars = set(text1) | set(text2)
        
        if not total_chars:
            return 1.0
        
        return len(common_chars) / len(total_chars)
    
    def is_quality_acceptable(self) -> bool:
        """品質許容性の判定"""
        if not self.quality_history:
            return True
        
        recent_quality = self.quality_history[-1]["quality_score"]
        degradation = self.baseline_quality - recent_quality
        
        return degradation <= self.config.max_quality_degradation
    
    def get_quality_statistics(self) -> Dict[str, float]:
        """品質統計の取得"""
        if not self.quality_history:
            return {"average_quality": 1.0, "quality_degradation": 0.0}
        
        recent_scores = [entry["quality_score"] for entry in self.quality_history[-10:]]
        average_quality = sum(recent_scores) / len(recent_scores)
        quality_degradation = max(0.0, self.baseline_quality - average_quality)
        
        return {
            "average_quality": average_quality,
            "quality_degradation": quality_degradation,
            "quality_samples": len(self.quality_history)
        }

class PerformanceOptimizer:
    """性能最適化機構"""
    
    def __init__(self, config: Phase2_3SystemConfig):
        self.config = config
        self.performance_history = []
        self.optimization_strategies = []
        
    def optimize_skip_parameters(self, current_performance: float, target_performance: float):
        """スキップパラメータの最適化"""
        performance_gap = target_performance - current_performance
        
        if performance_gap > 0:
            # 性能向上が必要：より積極的なスキップ
            optimization = {
                "layer_skip_threshold": max(0.05, self.config.layer_skip_threshold - 0.02),
                "ffn_skip_threshold": max(0.1, self.config.ffn_skip_threshold - 0.02),
                "token_halting_threshold": min(0.98, self.config.token_halting_threshold + 0.01)
            }
        else:
            # 性能が十分：品質重視
            optimization = {
                "layer_skip_threshold": min(0.2, self.config.layer_skip_threshold + 0.01),
                "ffn_skip_threshold": min(0.25, self.config.ffn_skip_threshold + 0.01),
                "token_halting_threshold": max(0.9, self.config.token_halting_threshold - 0.01)
            }
        
        self.optimization_strategies.append({
            "timestamp": time.time(),
            "performance_gap": performance_gap,
            "optimization": optimization
        })
        
        return optimization

class FlexGenPlusPlusPhase2_3(FlexGenPlusPlusPhase1):
    """FlexGen++ Phase 2-3 実装"""
    
    def __init__(self, config: Phase2_3SystemConfig):
        # Phase 1 の初期化
        super().__init__(config)
        
        # Phase 2-3 固有の設定
        self.config = config
        self.router_api = RouterAPI(config, self.model)
        self.skip_engine = DynamicSkipEngine(config, self.router_api)
        
        # Phase 2-3 メトリクス
        self.phase2_3_metrics = {
            "router_decisions": 0,
            "skip_efficiency": 0.0,
            "quality_maintenance": 1.0,
            "adaptive_optimizations": 0
        }
        
        # 最適化カウンター
        self.optimization_counter = 0
    
    def initialize_model(self) -> bool:
        """Phase 2-3 モデル初期化"""
        logger.info("Initializing FlexGen++ Phase 2-3...")
        
        # Phase 1 の初期化
        if not super().initialize_model():
            return False
        
        # Router API の初期化
        self._initialize_router_optimizations()
        
        logger.info("FlexGen++ Phase 2-3 initialization completed")
        return True
    
    def _initialize_router_optimizations(self):
        """Router最適化の初期化"""
        logger.info("Initializing Router API optimizations...")
        
        # モデル構造の分析
        self._analyze_model_structure()
        
        # 最適化ポイントの特定
        self._identify_optimization_points()
        
        logger.info("Router API optimizations initialized")
    
    def _analyze_model_structure(self):
        """モデル構造の分析"""
        layer_count = 0
        ffn_count = 0
        
        for name, module in self.model.named_modules():
            if "layer" in name.lower() and hasattr(module, 'attention'):
                layer_count += 1
            if "mlp" in name.lower() or "ffn" in name.lower():
                ffn_count += 1
        
        self.skip_engine.skip_metrics.total_layers = layer_count
        self.skip_engine.skip_metrics.total_ffns = ffn_count
        
        logger.info(f"Model structure: {layer_count} layers, {ffn_count} FFNs")
    
    def _identify_optimization_points(self):
        """最適化ポイントの特定"""
        # 重要度の低い層を特定
        low_importance_layers = []
        
        for layer_idx in range(self.skip_engine.skip_metrics.total_layers):
            importance = self.router_api.layer_analyzer.get_layer_importance(layer_idx)
            if importance < 0.3:  # 重要度30%未満
                low_importance_layers.append(layer_idx)
        
        logger.info(f"Identified {len(low_importance_layers)} low-importance layers for optimization")
    
    def execute_inference(self, prompt: str, max_tokens: int = 100) -> Tuple[str, PerformanceMetrics]:
        """Phase 2-3 推論実行"""
        start_time = time.time()
        
        # 入力のトークン化
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=self.config.max_sequence_length, truncation=True)
        input_length = len(inputs["input_ids"][0])
        
        # Router API を使用した最適化推論
        outputs = self._execute_optimized_inference(inputs, max_tokens)
        
        # 結果のデコード
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_tokens = len(outputs[0]) - input_length
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        # 性能メトリクスの計算
        tokens_per_second = generated_tokens / inference_time if inference_time > 0 else 0
        
        metrics = PerformanceMetrics(
            tokens_per_second=tokens_per_second,
            latency_ms=inference_time * 1000,
            memory_usage_gb=self._get_memory_usage(),
            gpu_utilization=self._get_gpu_utilization(),
            throughput_efficiency=tokens_per_second / self.config.phase3_target_tokens_per_second
        )
        
        # Phase 2-3 メトリクスの更新
        self._update_phase2_3_metrics(metrics)
        
        # 適応的最適化
        self._perform_adaptive_optimization(metrics)
        
        logger.info(f"Phase 2-3 inference: {tokens_per_second:.1f} tok/s")
        
        return response, metrics
    
    def _execute_optimized_inference(self, inputs: Dict[str, torch.Tensor], max_tokens: int) -> torch.Tensor:
        """最適化された推論実行"""
        
        if self.config.enable_token_halting:
            # Token Halting機能付き生成
            outputs, halting_decisions = self.skip_engine.execute_generation_with_halting(
                self.model, self.tokenizer, inputs["input_ids"], max_tokens
            )
            
            # Halting決定の記録
            self.phase2_3_metrics["router_decisions"] += len(halting_decisions)
            
        else:
            # 通常の生成（Layer Skip、FFN Skipのみ）
            with torch.no_grad():
                outputs = self._generate_with_layer_ffn_skip(inputs, max_tokens)
        
        return outputs
    
    def _generate_with_layer_ffn_skip(self, inputs: Dict[str, torch.Tensor], max_tokens: int) -> torch.Tensor:
        """Layer Skip、FFN Skip機能付き生成"""
        
        # カスタム生成ループ（簡略化）
        input_ids = inputs["input_ids"]
        
        for step in range(max_tokens):
            # モデルの順伝播をカスタマイズ
            outputs = self._custom_forward_pass(input_ids)
            
            # 次のトークンを選択
            next_token_logits = outputs[:, -1, :]
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1)
            
            # 入力に追加
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # EOS トークンで停止
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return input_ids
    
    def _custom_forward_pass(self, input_ids: torch.Tensor) -> torch.Tensor:
        """カスタム順伝播（Skip機能付き）"""
        
        # 埋め込み層
        hidden_states = self.model.get_input_embeddings()(input_ids)
        
        # Transformer層の処理
        for layer_idx, layer in enumerate(self.model.transformer.h if hasattr(self.model, 'transformer') else []):
            
            if self.config.enable_layer_skip:
                # Layer Skip判定と実行
                hidden_states = self.skip_engine.execute_layer_with_skip(
                    layer, layer_idx, hidden_states
                )
            else:
                # 通常実行
                layer_output = layer(hidden_states)
                if isinstance(layer_output, tuple):
                    hidden_states = layer_output[0]
                else:
                    hidden_states = layer_output
        
        # 最終層の正規化
        if hasattr(self.model, 'ln_f'):
            hidden_states = self.model.ln_f(hidden_states)
        
        # 言語モデルヘッド
        logits = self.model.lm_head(hidden_states)
        
        return logits
    
    def _update_phase2_3_metrics(self, metrics: PerformanceMetrics):
        """Phase 2-3 メトリクスの更新"""
        
        # スキップ効率の計算
        skip_stats = self.skip_engine.get_skip_statistics()
        self.phase2_3_metrics["skip_efficiency"] = (
            skip_stats["skip_ratio_layer"] * 0.4 +
            skip_stats["skip_ratio_ffn"] * 0.4 +
            skip_stats["halt_ratio_token"] * 0.2
        )
        
        # 品質維持の評価
        quality_stats = self.skip_engine.quality_monitor.get_quality_statistics()
        self.phase2_3_metrics["quality_maintenance"] = quality_stats["average_quality"]
        
        # Router決定数の更新
        router_stats = self.router_api.get_router_statistics()
        self.phase2_3_metrics["router_decisions"] = router_stats["total_decisions"]
    
    def _perform_adaptive_optimization(self, metrics: PerformanceMetrics):
        """適応的最適化の実行"""
        self.optimization_counter += 1
        
        # 定期的な最適化
        if self.optimization_counter % self.config.router_update_frequency == 0:
            
            # 性能フィードバック
            performance_feedback = {
                "tokens_per_second": metrics.tokens_per_second,
                "quality_score": self.phase2_3_metrics["quality_maintenance"]
            }
            
            # Router閾値の適応的更新
            self.router_api.update_adaptive_thresholds(performance_feedback)
            
            # 性能最適化
            target_performance = self.config.phase3_target_tokens_per_second
            optimization = self.skip_engine.performance_optimizer.optimize_skip_parameters(
                metrics.tokens_per_second, target_performance
            )
            
            self.phase2_3_metrics["adaptive_optimizations"] += 1
            
            logger.debug(f"Adaptive optimization performed: {optimization}")
    
    def get_phase2_3_performance_summary(self) -> Dict[str, Any]:
        """Phase 2-3 性能サマリーの取得"""
        base_summary = super().get_phase1_performance_summary()
        
        if "error" in base_summary:
            return base_summary
        
        # Phase 2-3 固有メトリクスの追加
        phase2_3_summary = base_summary.copy()
        phase2_3_summary.update({
            "phase2_3_metrics": self.phase2_3_metrics,
            "skip_statistics": self.skip_engine.get_skip_statistics(),
            "router_statistics": self.router_api.get_router_statistics(),
            "quality_statistics": self.skip_engine.quality_monitor.get_quality_statistics(),
            "improvement_over_phase1": phase2_3_summary["average_tokens_per_second"] / 13.5,
            "phase2_target_achievement": phase2_3_summary["average_tokens_per_second"] / self.config.phase2_target_tokens_per_second,
            "phase3_target_achievement": phase2_3_summary["average_tokens_per_second"] / self.config.phase3_target_tokens_per_second
        })
        
        return phase2_3_summary

def run_phase2_3_benchmark():
    """Phase 2-3 ベンチマークの実行"""
    logger.info("Starting Phase 2-3 benchmark...")
    
    # Phase 2-3 システム設定
    config = Phase2_3SystemConfig()
    
    # FlexGen++ Phase 2-3 の初期化
    flexgen = FlexGenPlusPlusPhase2_3(config)
    
    if not flexgen.initialize_model():
        logger.error("Phase 2-3 system initialization failed")
        return
    
    # ベンチマーク用プロンプト（より複雑で多様）
    test_prompts = [
        "Develop a comprehensive machine learning pipeline for natural language processing tasks, including data preprocessing, model selection, training, and evaluation phases.",
        "Explain the mathematical foundations of transformer architectures, including self-attention mechanisms, positional encoding, and multi-head attention computations.",
        "Create a detailed software architecture design for a distributed AI inference system that can handle millions of requests per day with low latency requirements.",
        "人工知能技術の発展が医療分野に与える革新的な影響について、具体的な応用例と将来の可能性を含めて詳細に分析してください。",
        "Design and implement a robust error handling and monitoring system for a production-level machine learning service, considering scalability, reliability, and observability requirements."
    ]
    
    logger.info("Executing Phase 2-3 benchmark tests...")
    
    # Phase 2 テスト（Layer Skip中心）
    logger.info("=== Phase 2 Testing (Layer Skip) ===")
    config.enable_layer_skip = True
    config.enable_ffn_skip = False
    config.enable_token_halting = False
    
    phase2_metrics = []
    for i, prompt in enumerate(test_prompts):
        logger.info(f"Phase 2 Test {i+1}/{len(test_prompts)}")
        
        try:
            response, metrics = flexgen.execute_inference(prompt, max_tokens=200)
            phase2_metrics.append(metrics)
            logger.info(f"Phase 2 Performance: {metrics.tokens_per_second:.1f} tok/s")
            
        except Exception as e:
            logger.error(f"Phase 2 Test {i+1} failed: {e}")
    
    # Phase 3 テスト（全機能有効）
    logger.info("=== Phase 3 Testing (All Features) ===")
    config.enable_layer_skip = True
    config.enable_ffn_skip = True
    config.enable_token_halting = True
    
    phase3_metrics = []
    for i, prompt in enumerate(test_prompts):
        logger.info(f"Phase 3 Test {i+1}/{len(test_prompts)}")
        
        try:
            response, metrics = flexgen.execute_inference(prompt, max_tokens=200)
            phase3_metrics.append(metrics)
            logger.info(f"Phase 3 Performance: {metrics.tokens_per_second:.1f} tok/s")
            
        except Exception as e:
            logger.error(f"Phase 3 Test {i+1} failed: {e}")
    
    # 結果の集計と表示
    logger.info("=== Phase 2-3 Benchmark Results ===")
    
    if phase2_metrics:
        phase2_avg = sum(m.tokens_per_second for m in phase2_metrics) / len(phase2_metrics)
        logger.info(f"Phase 2 Average: {phase2_avg:.1f} tok/s")
        logger.info(f"Phase 2 Target Achievement: {phase2_avg/config.phase2_target_tokens_per_second:.1%}")
        
        if phase2_avg >= config.phase2_target_tokens_per_second:
            logger.info("✅ Phase 2 target achieved!")
        else:
            gap = config.phase2_target_tokens_per_second - phase2_avg
            logger.warning(f"⚠️ Phase 2 target not achieved. Gap: {gap:.1f} tok/s")
    
    if phase3_metrics:
        phase3_avg = sum(m.tokens_per_second for m in phase3_metrics) / len(phase3_metrics)
        logger.info(f"Phase 3 Average: {phase3_avg:.1f} tok/s")
        logger.info(f"Phase 3 Target Achievement: {phase3_avg/config.phase3_target_tokens_per_second:.1%}")
        
        if phase3_avg >= config.phase3_target_tokens_per_second:
            logger.info("✅ Phase 3 target achieved!")
        else:
            gap = config.phase3_target_tokens_per_second - phase3_avg
            logger.warning(f"⚠️ Phase 3 target not achieved. Gap: {gap:.1f} tok/s")
    
    # 改善率の表示
    if phase2_metrics and phase3_metrics:
        phase1_baseline = 13.5  # Phase 1 目標
        phase2_improvement = phase2_avg / phase1_baseline
        phase3_improvement = phase3_avg / phase1_baseline
        
        logger.info(f"Phase 2 improvement over Phase 1: {phase2_improvement:.1%}")
        logger.info(f"Phase 3 improvement over Phase 1: {phase3_improvement:.1%}")
    
    # 詳細統計の表示
    performance_summary = flexgen.get_phase2_3_performance_summary()
    logger.info("Phase 2-3 System Statistics:")
    logger.info(f"  Router decisions: {performance_summary['phase2_3_metrics']['router_decisions']}")
    logger.info(f"  Skip efficiency: {performance_summary['phase2_3_metrics']['skip_efficiency']:.1%}")
    logger.info(f"  Quality maintenance: {performance_summary['phase2_3_metrics']['quality_maintenance']:.1%}")
    logger.info(f"  Adaptive optimizations: {performance_summary['phase2_3_metrics']['adaptive_optimizations']}")
    
    logger.info("Phase 2-3 benchmark completed")

def main():
    """メイン関数"""
    print("🚀 Phase 2-3: Router API実装")
    print("=" * 50)
    
    try:
        run_phase2_3_benchmark()
    except KeyboardInterrupt:
        logger.info("Phase 2-3 benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Phase 2-3 benchmark failed: {e}")
        raise

if __name__ == "__main__":
    main()

