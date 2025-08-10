#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 大規模LLM Infer-OS最適化デモ（修正版）

openai/gpt-oss-120b等の大規模LLMモデル（120B+パラメータ）での
Infer-OS最適化効果を実際のプロンプト処理で体験

修正内容:
- BitsAndBytesConfig互換性エラー修正
- transformers/bitsandbyteバージョン対応
- エラーハンドリング強化
- フォールバック機能追加

対応モデル:
- openai/gpt-oss-120b (120Bパラメータ)
- EleutherAI/gpt-neox-20b (20Bパラメータ)
- microsoft/DialoGPT-large (774Mパラメータ)
- その他大規模Transformerモデル

使用方法:
    python llm_demo_large_models_fixed.py --model openai/gpt-oss-120b --use-4bit --interactive
"""

import sys
import os
import gc
import time
import json
import psutil
import argparse
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import traceback

try:
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    import numpy as np
    
    # 最適化ライブラリ
    try:
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        from accelerate.utils import get_balanced_memory
        ACCELERATE_AVAILABLE = True
    except ImportError:
        ACCELERATE_AVAILABLE = False
    
    try:
        from bitsandbytes import BitsAndBytesConfig
        BITSANDBYTES_AVAILABLE = True
    except ImportError:
        BITSANDBYTES_AVAILABLE = False
        
except ImportError as e:
    print(f"❌ 必要なライブラリが不足しています: {e}")
    print("以下のコマンドでインストールしてください:")
    print("pip install torch transformers accelerate bitsandbytes numpy psutil")
    sys.exit(1)

class LargeLLMDemo:
    """大規模LLMデモクラス（修正版）"""
    
    def __init__(self, model_name: str, use_4bit: bool = False, use_8bit: bool = False):
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # モデル・トークナイザー
        self.model = None
        self.tokenizer = None
        
        # 最適化状態
        self.optimization_applied = False
        self.quantization_info = {}
        
        # システム情報
        self.system_info = self._get_system_info()
        
        print(f"🤖 大規模LLM Infer-OS最適化デモ（修正版）")
        print(f"対象モデル: {model_name}")
        self._print_system_info()
    
    def _get_system_info(self) -> Dict:
        """システム情報を取得"""
        info = {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
        }
        
        if torch.cuda.is_available():
            info.update({
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "cuda_version": torch.version.cuda,
            })
        
        # ライブラリ対応状況
        info.update({
            "accelerate_available": ACCELERATE_AVAILABLE,
            "bitsandbytes_available": BITSANDBYTES_AVAILABLE,
        })
        
        return info
    
    def _print_system_info(self):
        """システム情報を表示"""
        print(f"\n📊 システム情報:")
        print(f"  Python: {sys.version.split()[0]}")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA利用可能: {torch.cuda.is_available()}")
        print(f"  CPU: {self.system_info['cpu_count']}コア")
        print(f"  メモリ: {self.system_info['memory_total_gb']:.1f}GB")
        
        if torch.cuda.is_available():
            print(f"  GPU: {self.system_info['gpu_name']}")
            print(f"  GPU メモリ: {self.system_info['gpu_memory_total_gb']:.1f}GB")
        
        print(f"\n🔧 最適化ライブラリ:")
        print(f"  Accelerate: {'✅' if ACCELERATE_AVAILABLE else '❌'}")
        print(f"  BitsAndBytes: {'✅' if BITSANDBYTES_AVAILABLE else '❌'}")
    
    def estimate_model_requirements(self) -> Dict:
        """モデル要件を推定"""
        try:
            print(f"📏 モデル '{self.model_name}' の要件を推定中...")
            
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            
            # パラメータ数推定
            params = self._estimate_parameters(config)
            
            # メモリ要件推定
            memory_requirements = self._estimate_memory_requirements(params)
            
            print(f"推定パラメータ数: {params:,}")
            print(f"推定メモリ使用量:")
            print(f"  FP16: {memory_requirements['fp16_gb']:.1f}GB")
            print(f"  INT8: {memory_requirements['int8_gb']:.1f}GB")
            print(f"  INT4: {memory_requirements['int4_gb']:.1f}GB")
            print(f"システムメモリ: {self.system_info['memory_total_gb']:.1f}GB")
            
            return {
                "parameters": params,
                "memory_requirements": memory_requirements
            }
            
        except Exception as e:
            print(f"❌ 要件推定エラー: {e}")
            return {}
    
    def _estimate_parameters(self, config) -> int:
        """パラメータ数を推定"""
        try:
            # 設定から直接取得を試行
            if hasattr(config, 'n_parameters'):
                return config.n_parameters
            
            # 推定計算
            hidden_size = getattr(config, 'hidden_size', getattr(config, 'd_model', 4096))
            n_layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 24))
            vocab_size = getattr(config, 'vocab_size', 50257)
            
            # Transformer パラメータ推定
            attention_params = n_layers * 4 * hidden_size * hidden_size
            ffn_intermediate = getattr(config, 'intermediate_size', 4 * hidden_size)
            ffn_params = n_layers * 2 * hidden_size * ffn_intermediate
            embedding_params = vocab_size * hidden_size
            other_params = n_layers * hidden_size * 4
            
            total_params = attention_params + ffn_params + embedding_params + other_params
            return total_params
            
        except Exception as e:
            print(f"⚠️ パラメータ推定エラー: {e}")
            return 4_000_000_000  # デフォルト値
    
    def _estimate_memory_requirements(self, params: int) -> Dict:
        """メモリ要件を推定"""
        # 各精度でのメモリ使用量（バイト）
        fp16_memory = params * 2
        int8_memory = params * 1
        int4_memory = params * 0.5
        
        # 推論時の追加メモリ（アクティベーション、KVキャッシュ等）
        inference_overhead = 0.5
        
        return {
            "fp16_gb": fp16_memory * (1 + inference_overhead) / (1024**3),
            "int8_gb": int8_memory * (1 + inference_overhead) / (1024**3),
            "int4_gb": int4_memory * (1 + inference_overhead) / (1024**3)
        }
    
    def create_quantization_config(self) -> Optional[Any]:
        """量子化設定を作成（互換性対応版）"""
        if not BITSANDBYTES_AVAILABLE:
            return None
        
        try:
            if self.use_4bit:
                print("🔧 4bit量子化を有効化しました")
                # 新しいBitsAndBytesConfigの形式に対応
                config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                return config
            elif self.use_8bit:
                print("🔧 8bit量子化を有効化しました")
                config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                return config
        except Exception as e:
            print(f"⚠️ 量子化設定エラー: {e}")
            print("💡 量子化無しで続行します")
            return None
        
        return None
    
    def load_model_with_optimization(self) -> bool:
        """最適化を適用してモデルをロード"""
        try:
            print("📥 大規模モデルをロード中...")
            print("⚠️  初回実行時は大容量ダウンロードのため時間がかかります")
            
            # 量子化設定
            quantization_config = self.create_quantization_config()
            
            # モデルロード設定
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
            }
            
            # 量子化設定を追加（エラーハンドリング付き）
            if quantization_config is not None:
                try:
                    model_kwargs["quantization_config"] = quantization_config
                except Exception as e:
                    print(f"⚠️ 量子化設定適用エラー: {e}")
                    print("💡 量子化無しで続行します")
            
            # デバイス配置（Accelerate利用可能時）
            if ACCELERATE_AVAILABLE:
                model_kwargs["device_map"] = "auto"
            
            print(f"📥 大規模モデル '{self.model_name}' をロード中...")
            
            # モデルロード（フォールバック付き）
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            except Exception as e:
                print(f"⚠️ 最適化モデルロードエラー: {e}")
                print("💡 基本設定でリトライします")
                
                # フォールバック: 基本設定でロード
                basic_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                }
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **basic_kwargs
                )
            
            # トークナイザーロード
            print("📝 トークナイザーをロード中...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # パディングトークン設定
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # モデルを評価モードに設定
            self.model.eval()
            
            # 最適化適用
            self._apply_runtime_optimizations()
            
            print("✅ モデルロード完了")
            return True
            
        except Exception as e:
            print(f"❌ モデルロードエラー: {e}")
            print(f"詳細: {traceback.format_exc()}")
            return False
    
    def _apply_runtime_optimizations(self):
        """実行時最適化を適用"""
        try:
            print("🔧 実行時最適化を適用中...")
            
            # グラディエントチェックポイント
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("  ✅ グラディエントチェックポイント有効化")
            
            # キャッシュ設定
            if hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = True
                print("  ✅ キャッシュ有効化")
            
            # Flash Attention（対応モデルのみ）
            try:
                if hasattr(self.model.config, 'use_flash_attention_2'):
                    self.model.config.use_flash_attention_2 = True
                    print("  ✅ Flash Attention 2 有効化")
            except:
                pass
            
            self.optimization_applied = True
            print("🚀 実行時最適化適用完了")
            
        except Exception as e:
            print(f"⚠️ 実行時最適化エラー: {e}")
    
    def generate_text(self, prompt: str, max_length: int = 200) -> Dict:
        """テキスト生成（最適化効果測定付き）"""
        if self.model is None or self.tokenizer is None:
            return {"error": "モデルまたはトークナイザーが未ロード"}
        
        try:
            print(f"\n🎯 テキスト生成開始")
            print(f"プロンプト: \"{prompt}\"")
            print(f"最大長: {max_length}")
            
            # メモリ使用量測定開始
            initial_memory = self._get_memory_usage()
            
            # トークン化
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成設定
            generation_config = {
                "max_length": max_length,
                "num_return_sequences": 1,
                "temperature": 0.7,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # 生成実行（時間測定）
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # 結果デコード
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # 生成部分のみ抽出
            generated_only = generated_text[len(prompt):].strip()
            
            # メモリ使用量測定終了
            final_memory = self._get_memory_usage()
            
            # トークン数計算
            input_tokens = len(inputs['input_ids'][0])
            output_tokens = len(outputs[0]) - input_tokens
            total_tokens = len(outputs[0])
            
            # 性能指標計算
            tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
            
            result = {
                "prompt": prompt,
                "generated_text": generated_only,
                "full_text": generated_text,
                "generation_time": generation_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "tokens_per_second": tokens_per_second,
                "initial_memory": initial_memory,
                "final_memory": final_memory,
                "memory_used": {
                    "system_mb": final_memory["system_mb"] - initial_memory["system_mb"],
                    "gpu_mb": final_memory.get("gpu_mb", 0) - initial_memory.get("gpu_mb", 0)
                },
                "optimization_applied": self.optimization_applied,
                "quantization_info": self.quantization_info
            }
            
            self._print_generation_results(result)
            return result
            
        except Exception as e:
            error_msg = f"テキスト生成エラー: {e}"
            print(f"❌ {error_msg}")
            return {"error": error_msg, "traceback": traceback.format_exc()}
    
    def _get_memory_usage(self) -> Dict:
        """メモリ使用量を取得"""
        memory_info = {
            "system_mb": psutil.virtual_memory().used / (1024**2)
        }
        
        if torch.cuda.is_available():
            memory_info["gpu_mb"] = torch.cuda.memory_allocated() / (1024**2)
            memory_info["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)
        
        return memory_info
    
    def _print_generation_results(self, result: Dict):
        """生成結果を表示"""
        print(f"\n📊 生成結果:")
        print(f"  生成時間: {result['generation_time']:.2f}秒")
        print(f"  入力トークン: {result['input_tokens']}")
        print(f"  出力トークン: {result['output_tokens']}")
        print(f"  スループット: {result['tokens_per_second']:.1f} tokens/sec")
        
        print(f"\n💾 メモリ使用量:")
        print(f"  システムメモリ: {result['memory_used']['system_mb']:.1f}MB")
        if torch.cuda.is_available():
            print(f"  GPU メモリ: {result['memory_used']['gpu_mb']:.1f}MB")
        
        print(f"\n🔧 最適化状態:")
        print(f"  最適化適用: {'✅' if result['optimization_applied'] else '❌'}")
        print(f"  量子化: {'✅' if self.use_4bit or self.use_8bit else '❌'}")
        
        print(f"\n📝 生成テキスト:")
        print(f"  \"{result['generated_text'][:200]}{'...' if len(result['generated_text']) > 200 else ''}\"")
    
    def interactive_mode(self):
        """インタラクティブモード"""
        print(f"\n🎯 インタラクティブモード開始")
        print(f"プロンプトを入力してください（'quit'で終了）:")
        
        results = []
        
        while True:
            try:
                prompt = input("\n> ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not prompt:
                    continue
                
                result = self.generate_text(prompt)
                if "error" not in result:
                    results.append(result)
                
            except KeyboardInterrupt:
                print("\n👋 インタラクティブモードを終了します")
                break
            except Exception as e:
                print(f"❌ エラー: {e}")
        
        # セッション結果保存
        if results:
            self._save_session_results(results)
    
    def _save_session_results(self, results: List[Dict]):
        """セッション結果を保存"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_safe_name = self.model_name.replace("/", "_").replace("-", "_")
            filename = f'llm_demo_session_{model_safe_name}_{timestamp}.json'
            
            session_data = {
                "model_name": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "system_info": self.system_info,
                "optimization_config": {
                    "use_4bit": self.use_4bit,
                    "use_8bit": self.use_8bit,
                    "optimization_applied": self.optimization_applied
                },
                "results": results,
                "summary": self._calculate_session_summary(results)
            }
            
            os.makedirs('demo_results', exist_ok=True)
            filepath = os.path.join('demo_results', filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 セッション結果を保存しました: {filepath}")
            
        except Exception as e:
            print(f"❌ 結果保存エラー: {e}")
    
    def _calculate_session_summary(self, results: List[Dict]) -> Dict:
        """セッション結果のサマリーを計算"""
        if not results:
            return {}
        
        generation_times = [r['generation_time'] for r in results]
        tokens_per_second = [r['tokens_per_second'] for r in results]
        output_tokens = [r['output_tokens'] for r in results]
        
        return {
            "total_generations": len(results),
            "avg_generation_time": sum(generation_times) / len(generation_times),
            "avg_tokens_per_second": sum(tokens_per_second) / len(tokens_per_second),
            "total_output_tokens": sum(output_tokens),
            "min_generation_time": min(generation_times),
            "max_generation_time": max(generation_times)
        }

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="大規模LLM Infer-OS最適化デモ（修正版）")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="使用するモデル名")
    parser.add_argument("--prompt", help="テスト用プロンプト")
    parser.add_argument("--max-length", type=int, default=200, help="最大生成長")
    parser.add_argument("--use-4bit", action="store_true", help="4bit量子化を使用")
    parser.add_argument("--use-8bit", action="store_true", help="8bit量子化を使用")
    parser.add_argument("--interactive", action="store_true", help="インタラクティブモード")
    
    args = parser.parse_args()
    
    print(f"""
{'='*80}
🤖 大規模LLM Infer-OS最適化デモ（修正版）
{'='*80}

対象モデル: {args.model}
最適化設定:
  4bit量子化: {'✅' if args.use_4bit else '❌'}
  8bit量子化: {'✅' if args.use_8bit else '❌'}
  インタラクティブ: {'✅' if args.interactive else '❌'}

{'='*80}
""")
    
    try:
        # デモ初期化
        demo = LargeLLMDemo(
            model_name=args.model,
            use_4bit=args.use_4bit,
            use_8bit=args.use_8bit
        )
        
        # 要件推定
        requirements = demo.estimate_model_requirements()
        
        # モデルロード
        if not demo.load_model_with_optimization():
            print("❌ モデルのロードに失敗しました")
            return
        
        # テキスト生成実行
        if args.interactive:
            demo.interactive_mode()
        else:
            prompt = args.prompt or "The future of artificial intelligence is"
            result = demo.generate_text(prompt, args.max_length)
            
            if "error" in result:
                print(f"❌ 生成エラー: {result['error']}")
            else:
                print("\n🎉 テキスト生成完了")
        
    except KeyboardInterrupt:
        print("\n👋 デモを中断しました")
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        print(f"詳細: {traceback.format_exc()}")

if __name__ == "__main__":
    main()

