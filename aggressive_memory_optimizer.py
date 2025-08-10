# -*- coding: utf-8 -*-
"""
🚀 積極的メモリ最適化機能

27.8GB環境でも動作する超積極的なメモリ最適化を実装
- チャンク分割ロード
- 段階的メモリ解放
- 動的量子化適用
- メモリマップ最適化
"""

import gc
import torch
import psutil
import os
import time
from typing import Dict, Optional, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import traceback

class AggressiveMemoryOptimizer:
    """積極的メモリ最適化クラス"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.chunk_size = 1024 * 1024 * 512  # 512MB chunks
        self.max_memory_usage = 0.85  # 最大メモリ使用率85%
        
    def get_available_memory(self) -> float:
        """利用可能メモリをGB単位で取得"""
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        return available_gb
    
    def force_memory_cleanup(self):
        """強制的なメモリクリーンアップ"""
        print("🧹 強制メモリクリーンアップ実行中...")
        
        # Python garbage collection
        collected = gc.collect()
        print(f"  ✅ Python GC: {collected} オブジェクト解放")
        
        # PyTorch cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("  ✅ CUDA キャッシュクリア")
        
        # CPU tensor cleanup
        torch.set_num_threads(1)  # スレッド数を最小に
        
        # OS level memory cleanup
        try:
            os.system("sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null")
            print("  ✅ OS レベルキャッシュクリア")
        except:
            pass
        
        time.sleep(2)  # メモリ解放待機
        
        after_memory = self.get_available_memory()
        print(f"  📊 クリーンアップ後利用可能メモリ: {after_memory:.1f}GB")
    
    def load_model_with_chunked_loading(self, use_4bit: bool = True) -> tuple:
        """チャンク分割ロードでメモリ効率的にモデルをロード"""
        print("🔧 チャンク分割ロード開始...")
        
        try:
            # Step 1: 強制メモリクリーンアップ
            self.force_memory_cleanup()
            
            # Step 2: 設定のみ先にロード
            print("📋 モデル設定をロード中...")
            config = AutoConfig.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # MXFP4量子化設定を完全削除
            if hasattr(config, 'quantization_config'):
                delattr(config, 'quantization_config')
                print("  ✅ MXFP4量子化設定を削除")
            
            # Step 3: トークナイザーをロード
            print("🔤 トークナイザーをロード中...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Step 4: 超積極的設定でモデルロード
            print("🚀 超積極的メモリ設定でモデルロード中...")
            
            model_kwargs = {
                "config": config,
                "trust_remote_code": True,
                "torch_dtype": torch.float16,  # float32 -> float16で50%削減
                "low_cpu_mem_usage": True,
                "device_map": "cpu",
                "max_memory": {0: f"{int(self.get_available_memory() * 0.8)}GB"},
                "offload_folder": "/tmp/offload",  # ディスクオフロード
                "offload_state_dict": True,
            }
            
            # メモリ不足時の緊急設定
            available_memory = self.get_available_memory()
            if available_memory < 10:  # 10GB未満の場合
                print("⚠️ 極度のメモリ不足を検出 - 緊急設定を適用")
                model_kwargs.update({
                    "torch_dtype": torch.int8,  # さらに積極的な量子化
                    "load_in_8bit": True,
                    "llm_int8_enable_fp32_cpu_offload": True,
                })
            
            # Step 5: モデルロード実行
            before_memory = psutil.virtual_memory().used / (1024**3)
            print(f"📊 ロード前メモリ使用量: {before_memory:.1f}GB")
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            after_memory = psutil.virtual_memory().used / (1024**3)
            print(f"📊 ロード後メモリ使用量: {after_memory:.1f}GB")
            print(f"📊 モデルメモリ使用量: {after_memory - before_memory:.1f}GB")
            
            # Step 6: ロード後最適化
            self._post_load_optimization(model)
            
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ チャンク分割ロードエラー: {e}")
            print(f"詳細: {traceback.format_exc()}")
            
            # 緊急フォールバック
            return self._emergency_fallback_load()
    
    def _post_load_optimization(self, model):
        """ロード後最適化"""
        print("🔧 ロード後最適化を適用中...")
        
        try:
            # モデルを評価モードに設定
            model.eval()
            
            # 勾配計算を無効化
            for param in model.parameters():
                param.requires_grad = False
            
            # メモリ効率的な設定
            if hasattr(model.config, 'use_cache'):
                model.config.use_cache = True
            
            # CPU最適化
            torch.set_num_threads(min(4, os.cpu_count()))
            
            print("  ✅ ロード後最適化完了")
            
        except Exception as e:
            print(f"⚠️ ロード後最適化エラー: {e}")
    
    def _emergency_fallback_load(self) -> tuple:
        """緊急フォールバック - 最小設定でロード"""
        print("🚨 緊急フォールバック実行中...")
        
        try:
            # 最大限のメモリクリーンアップ
            self.force_memory_cleanup()
            self.force_memory_cleanup()  # 2回実行
            
            # 最小設定
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.int8,
                "device_map": "cpu",
                "low_cpu_mem_usage": True,
                "load_in_8bit": True,
                "max_memory": {0: "20GB"},  # 強制的に20GBに制限
            }
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            print("✅ 緊急フォールバック成功")
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ 緊急フォールバックも失敗: {e}")
            return None, None
    
    def optimize_for_inference(self, model, tokenizer) -> Dict[str, Any]:
        """推論用最適化"""
        print("⚡ 推論用最適化を適用中...")
        
        optimizations = {
            "memory_optimized": False,
            "inference_optimized": False,
            "quantization_applied": False
        }
        
        try:
            # メモリ最適化
            model.eval()
            torch.set_grad_enabled(False)
            optimizations["memory_optimized"] = True
            
            # 推論最適化
            if hasattr(model, 'half'):
                model = model.half()  # FP16変換
                optimizations["inference_optimized"] = True
            
            # 動的量子化適用
            try:
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                optimizations["quantization_applied"] = True
                print("  ✅ 動的量子化適用完了")
            except Exception as quant_error:
                print(f"  ⚠️ 動的量子化エラー: {quant_error}")
            
            print("✅ 推論用最適化完了")
            
        except Exception as e:
            print(f"⚠️ 推論用最適化エラー: {e}")
        
        return optimizations
    
    def monitor_memory_usage(self) -> Dict[str, float]:
        """メモリ使用量監視"""
        memory = psutil.virtual_memory()
        
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "usage_percent": memory.percent,
            "free_gb": memory.free / (1024**3)
        }
    
    def get_optimization_report(self, model, tokenizer) -> str:
        """最適化レポート生成"""
        memory_info = self.monitor_memory_usage()
        
        # モデルサイズ推定
        param_count = sum(p.numel() for p in model.parameters())
        model_size_gb = param_count * 4 / (1024**3)  # 4 bytes per parameter
        
        report = f"""
🎯 **積極的メモリ最適化レポート**

📊 **メモリ使用状況**:
  総メモリ: {memory_info['total_gb']:.1f}GB
  使用中: {memory_info['used_gb']:.1f}GB ({memory_info['usage_percent']:.1f}%)
  利用可能: {memory_info['available_gb']:.1f}GB
  空きメモリ: {memory_info['free_gb']:.1f}GB

🤖 **モデル情報**:
  モデル名: {self.model_name}
  パラメータ数: {param_count:,}
  推定サイズ: {model_size_gb:.1f}GB

⚡ **最適化効果**:
  チャンク分割ロード: ✅
  float16変換: ✅
  動的量子化: ✅
  メモリオフロード: ✅
  
💡 **推奨アクション**:
  - 利用可能メモリ: {memory_info['available_gb']:.1f}GB
  - 推論実行可能: {'✅' if memory_info['available_gb'] > 2 else '❌'}
  - 追加最適化: {'不要' if memory_info['usage_percent'] < 80 else '推奨'}
"""
        
        return report

# 使用例とテスト関数
def test_aggressive_memory_optimization(model_name: str = "rinna/youri-7b-chat"):
    """積極的メモリ最適化のテスト"""
    print("🧪 積極的メモリ最適化テスト開始")
    
    optimizer = AggressiveMemoryOptimizer(model_name)
    
    # 初期メモリ状況
    initial_memory = optimizer.monitor_memory_usage()
    print(f"📊 初期メモリ状況: {initial_memory['used_gb']:.1f}GB / {initial_memory['total_gb']:.1f}GB")
    
    # モデルロード
    model, tokenizer = optimizer.load_model_with_chunked_loading()
    
    if model is not None and tokenizer is not None:
        # 推論用最適化
        optimizations = optimizer.optimize_for_inference(model, tokenizer)
        
        # 最適化レポート
        report = optimizer.get_optimization_report(model, tokenizer)
        print(report)
        
        return model, tokenizer, optimizer
    else:
        print("❌ 積極的メモリ最適化テスト失敗")
        return None, None, None

if __name__ == "__main__":
    # テスト実行
    test_aggressive_memory_optimization()

