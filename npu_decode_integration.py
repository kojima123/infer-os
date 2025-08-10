# -*- coding: utf-8 -*-
"""
🚀 NPU Decode統合システム (v1.0)

仕様書に基づく「DecodeのみNPU」統合実装
- 既存のPyTorchモデルとNPU Runtimeの統合
- 段階的NPU移行（Phase 1: RMSNormのみ）
- スマートフォールバック機能
"""

import os
import sys
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
import traceback

# NPU Runtime API
from npu_runtime_api import (
    NPURuntime, NPUStatus, NPUQuantType, 
    NPUModelDesc, NPUQuantProfile, NPUDecodeArgs
)

class NPUDecodeIntegrator:
    """NPU Decode統合クラス"""
    
    def __init__(self, pytorch_model, tokenizer):
        """NPU統合デコード初期化"""
        self.pytorch_model = pytorch_model
        self.model = pytorch_model  # NPUデコードで使用するため
        self.tokenizer = tokenizer
        self.npu_runtime = None
        self.npu_graph = None
        self.npu_available = False
        
        # 統計情報
        self.stats = {
            "total_tokens": 0,
            "npu_tokens": 0,
            "cpu_tokens": 0,
            "npu_time": 0.0,
            "cpu_time": 0.0,
            "npu_errors": 0,
            "fallback_count": 0
        }
        
    def initialize_npu(self) -> bool:
        """NPU初期化"""
        try:
            print("🚀 NPU Decode統合初期化開始...")
            
            # NPU Runtime初期化
            self.npu_runtime = NPURuntime()
            status = self.npu_runtime.init()
            
            if status != NPUStatus.NPU_OK:
                print("⚠️ NPU Runtime初期化失敗、CPUフォールバック")
                return False
            
            # モデル記述子作成（PyTorchモデルから推定）
            model_desc = self._create_model_desc()
            
            # 量子化プロファイル作成
            quant_profile = NPUQuantProfile(
                weights=NPUQuantType.NPU_QUANT_W8A8,
                kv_level_near=64,
                kv_level_mid=1024,
                kv_block=32
            )
            
            # NPUグラフ構築
            status, graph = self.npu_runtime.build_graph(model_desc, quant_profile)
            if status != NPUStatus.NPU_OK or graph is None:
                print("⚠️ NPUグラフ構築失敗、CPUフォールバック")
                return False
            
            self.npu_graph = graph
            self.npu_available = True
            
            print("✅ NPU Decode統合初期化完了")
            print(f"  📊 NPU対応レイヤー: {list(graph.sessions.keys())}")
            return True
            
        except Exception as e:
            print(f"❌ NPU初期化エラー: {e}")
            return False
    
    def _create_model_desc(self) -> NPUModelDesc:
        """PyTorchモデルからNPUモデル記述子を作成"""
        try:
            # モデル設定から情報を抽出
            config = getattr(self.pytorch_model, 'config', None)
            
            if config:
                return NPUModelDesc(
                    max_ctx=getattr(config, 'max_position_embeddings', 8192),
                    heads=getattr(config, 'num_attention_heads', 32),
                    head_dim=getattr(config, 'hidden_size', 4096) // getattr(config, 'num_attention_heads', 32),
                    layers=getattr(config, 'num_hidden_layers', 32),
                    gqa_group=getattr(config, 'num_key_value_heads', 32) // getattr(config, 'num_attention_heads', 32) if hasattr(config, 'num_key_value_heads') else 1,
                    vocab_size=getattr(config, 'vocab_size', 32000),
                    hidden_dim=getattr(config, 'hidden_size', 4096)
                )
            else:
                # デフォルト値（7Bモデル想定）
                return NPUModelDesc(
                    max_ctx=8192,
                    heads=32,
                    head_dim=128,
                    layers=32,
                    gqa_group=1,
                    vocab_size=32000,
                    hidden_dim=4096
                )
                
        except Exception as e:
            print(f"⚠️ モデル記述子作成エラー: {e}")
            # フォールバック
            return NPUModelDesc()
    
    def generate_with_npu_decode(
        self, 
        input_text: str, 
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """NPU統合デコード生成"""
        try:
            start_time = time.time()
            
            print("🎯 NPU統合デコード生成開始")
            print(f"  📝 入力: {input_text[:50]}...")
            
            # トークン化
            inputs = self.tokenizer(input_text, return_tensors="pt")
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)
            
            print(f"  📊 入力トークン数: {input_ids.shape[1]}")
            
            # 生成ループ
            generated_tokens = []
            current_ids = input_ids
            
            for step in range(max_new_tokens):
                # NPU統合デコード実行
                next_token_logits = self._decode_step_with_npu(
                    current_ids, 
                    attention_mask,
                    step
                )
                
                if next_token_logits is None:
                    print("❌ デコードステップ失敗")
                    break
                
                # 次トークン選択
                if do_sample and temperature > 0:
                    # 温度サンプリング
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy選択
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated_tokens.append(next_token.item())
                
                # 入力更新
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                # 終了条件チェック
                if next_token.item() == self.tokenizer.eos_token_id:
                    # EOS検出時も最小限の生成を保証
                    if step < 5:  # 最初の5ステップではEOSを無視
                        print(f"  ⚠️ 早期EOS検出 (step {step})、生成継続")
                        continue
                    else:
                        print(f"  🏁 EOS検出、生成終了 (step {step})")
                        break
                
                # 進捗表示
                if (step + 1) % 10 == 0:
                    print(f"  📊 生成進捗: {step + 1}/{max_new_tokens} tokens")
            
            # 結果デコード
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            full_text = input_text + generated_text
            
            # 統計情報更新
            generation_time = time.time() - start_time
            tokens_generated = len(generated_tokens)
            
            self.stats["total_tokens"] += tokens_generated
            
            # 結果返却
            result = {
                "generated_text": full_text,
                "new_text": generated_text,
                "input_tokens": input_ids.shape[1],
                "output_tokens": tokens_generated,
                "generation_time": generation_time,
                "tokens_per_sec": tokens_generated / generation_time if generation_time > 0 else 0,
                "npu_utilization": self._calculate_npu_utilization(),
                "stats": self.stats.copy()
            }
            
            print(f"✅ NPU統合デコード生成完了")
            print(f"  📊 生成トークン数: {tokens_generated}")
            print(f"  ⏱️ 生成時間: {generation_time:.2f}秒")
            print(f"  🚀 速度: {result['tokens_per_sec']:.1f} tokens/sec")
            print(f"  ⚡ NPU利用率: {result['npu_utilization']:.1f}%")
            
            return result
            
        except Exception as e:
            print(f"❌ NPU統合デコード生成エラー: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    def _decode_step_with_npu(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor],
        step: int
    ) -> Optional[torch.Tensor]:
        """NPU統合デコードステップ"""
        try:
            step_start = time.time()
            
            # Phase 1: 部分的NPU実行
            if self.npu_available and self.npu_graph:
                try:
                    # NPUデコード試行
                    decode_args = NPUDecodeArgs(
                        kv_handle=None,
                        t_new=1,
                        ctx_len=input_ids.shape[1]
                    )
                    
                    # PyTorchモデルと入力データを渡す
                    status, npu_logits = self.npu_runtime.decode(
                        self.npu_graph, 
                        decode_args,
                        pytorch_model=self.model,  # 実際のPyTorchモデル
                        input_ids=input_ids,       # 入力トークン
                        attention_mask=attention_mask  # アテンションマスク
                    )
                    
                    if status == NPUStatus.NPU_OK and npu_logits is not None:
                        # NPU成功
                        npu_time = time.time() - step_start
                        self.stats["npu_tokens"] += 1
                        self.stats["npu_time"] += npu_time
                        
                        # NumPy -> PyTorch変換
                        logits_tensor = torch.from_numpy(npu_logits)
                        
                        if step % 20 == 0:  # 定期的に表示
                            print(f"    ⚡ NPU実行成功 (step {step}): {npu_time*1000:.1f}ms")
                        
                        return logits_tensor
                    else:
                        # NPU失敗、CPUフォールバック
                        self.stats["fallback_count"] += 1
                        if step % 20 == 0:
                            print(f"    ⚠️ NPU実行失敗、CPUフォールバック (step {step})")
                        
                except Exception as e:
                    # NPUエラー、CPUフォールバック
                    self.stats["fallback_count"] += 1
                    if step % 20 == 0:
                        print(f"    ⚠️ NPUエラー、CPUフォールバック (step {step}): {e}")
            
            # CPUフォールバック実行
            cpu_start = time.time()
            
            with torch.no_grad():
                outputs = self.pytorch_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True
                )
                
                logits = outputs.logits[:, -1, :]  # 最後のトークンのlogits
                
                cpu_time = time.time() - cpu_start
                self.stats["cpu_tokens"] += 1
                self.stats["cpu_time"] += cpu_time
                
                if step % 20 == 0:  # 定期的に表示
                    print(f"    🖥️ CPU実行 (step {step}): {cpu_time*1000:.1f}ms")
                
                return logits
                
        except Exception as e:
            print(f"❌ デコードステップエラー: {e}")
            return None
    
    def _calculate_npu_utilization(self) -> float:
        """NPU利用率計算"""
        total_tokens = self.stats["npu_tokens"] + self.stats["cpu_tokens"]
        if total_tokens == 0:
            return 0.0
        return (self.stats["npu_tokens"] / total_tokens) * 100
    
    def get_performance_report(self) -> Dict[str, Any]:
        """性能レポート取得"""
        total_tokens = self.stats["npu_tokens"] + self.stats["cpu_tokens"]
        total_time = self.stats["npu_time"] + self.stats["cpu_time"]
        
        report = {
            "total_tokens": total_tokens,
            "npu_tokens": self.stats["npu_tokens"],
            "cpu_tokens": self.stats["cpu_tokens"],
            "npu_utilization_percent": self._calculate_npu_utilization(),
            "total_time_sec": total_time,
            "average_tokens_per_sec": total_tokens / total_time if total_time > 0 else 0,
            "npu_average_time_ms": (self.stats["npu_time"] / self.stats["npu_tokens"] * 1000) if self.stats["npu_tokens"] > 0 else 0,
            "cpu_average_time_ms": (self.stats["cpu_time"] / self.stats["cpu_tokens"] * 1000) if self.stats["cpu_tokens"] > 0 else 0,
            "fallback_count": self.stats["fallback_count"],
            "fallback_rate_percent": (self.stats["fallback_count"] / total_tokens * 100) if total_tokens > 0 else 0
        }
        
        # NPU Runtime性能レポート統合
        if self.npu_runtime:
            npu_report = self.npu_runtime.get_performance_report()
            report.update({f"npu_{k}": v for k, v in npu_report.items()})
        
        return report
    
    def cleanup(self):
        """リソース解放"""
        try:
            print("🔄 NPU Decode統合リソース解放中...")
            
            if self.npu_runtime:
                self.npu_runtime.teardown()
                self.npu_runtime = None
            
            self.npu_graph = None
            self.npu_available = False
            
            print("✅ NPU Decode統合リソース解放完了")
            
        except Exception as e:
            print(f"❌ リソース解放エラー: {e}")

# 使用例とテスト
def test_npu_decode_integration():
    """NPU Decode統合テスト"""
    print("🧪 NPU Decode統合テスト開始...")
    
    # ダミーモデルとトークナイザー（実際の使用では実モデルを使用）
    class DummyModel:
        def __init__(self):
            self.config = type('Config', (), {
                'max_position_embeddings': 8192,
                'num_attention_heads': 32,
                'hidden_size': 4096,
                'num_hidden_layers': 32,
                'vocab_size': 32000
            })()
        
        def __call__(self, input_ids, attention_mask=None, use_cache=False):
            # ダミーlogits
            batch_size, seq_len = input_ids.shape
            vocab_size = 32000
            logits = torch.randn(batch_size, seq_len, vocab_size)
            return type('Output', (), {'logits': logits})()
    
    class DummyTokenizer:
        def __init__(self):
            self.eos_token_id = 2
        
        def __call__(self, text, return_tensors=None):
            # ダミートークン化
            tokens = [1, 2, 3, 4, 5]  # ダミートークン
            return {"input_ids": torch.tensor([tokens])}
        
        def decode(self, tokens, skip_special_tokens=False):
            return f"Generated text with {len(tokens)} tokens"
    
    # 統合システム初期化
    dummy_model = DummyModel()
    dummy_tokenizer = DummyTokenizer()
    
    integrator = NPUDecodeIntegrator(dummy_model, dummy_tokenizer, "test-model")
    
    # NPU初期化
    npu_success = integrator.initialize_npu()
    print(f"NPU初期化結果: {'成功' if npu_success else '失敗（CPUフォールバック）'}")
    
    # 生成テスト
    result = integrator.generate_with_npu_decode(
        "テストプロンプト",
        max_new_tokens=20,
        temperature=0.7
    )
    
    if "error" not in result:
        print("✅ 生成テスト成功")
        print(f"  📊 結果: {result}")
    else:
        print(f"❌ 生成テスト失敗: {result['error']}")
    
    # 性能レポート
    report = integrator.get_performance_report()
    print(f"📊 性能レポート: {report}")
    
    # リソース解放
    integrator.cleanup()
    print("✅ NPU Decode統合テスト完了")

if __name__ == "__main__":
    test_npu_decode_integration()

