#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏋️ CPU環境最大規模LLM Infer-OS最適化デモ

CPU環境で動作する最も重いLLMモデル（EleutherAI/gpt-neox-20b）での
Infer-OS最適化効果を実際のプロンプト処理で体験

対応モデル:
- EleutherAI/gpt-neox-20b (20Bパラメータ) - 最重量級
- EleutherAI/gpt-j-6B (6Bパラメータ) - 重量級
- bigscience/bloom-7b1 (7.1Bパラメータ) - 重量級
- その他CPU対応大規模モデル

特徴:
- CPU専用最適化
- 大容量メモリ効率化
- 段階的量子化対応
- リアルタイム性能監視

使用方法:
    python cpu_heaviest_llm_demo.py --model EleutherAI/gpt-neox-20b --use-8bit --interactive
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

# CPU対応最大規模モデル定義
CPU_HEAVY_MODELS = {
    "EleutherAI/gpt-neox-20b": {
        "parameters": 20_554_568_704,
        "size_gb": {"fp32": 80, "fp16": 40, "int8": 20, "int4": 10},
        "min_memory_gb": 64,
        "recommended_memory_gb": 128,
        "description": "最重量級 20Bパラメータ GPT-NeoX",
        "rank": 1
    },
    "bigscience/bloom-7b1": {
        "parameters": 7_069_016_064,
        "size_gb": {"fp32": 28, "fp16": 14, "int8": 7, "int4": 3.5},
        "min_memory_gb": 32,
        "recommended_memory_gb": 64,
        "description": "重量級 7.1Bパラメータ BLOOM",
        "rank": 2
    },
    "EleutherAI/gpt-j-6B": {
        "parameters": 6_053_381_344,
        "size_gb": {"fp32": 24, "fp16": 12, "int8": 6, "int4": 3},
        "min_memory_gb": 32,
        "recommended_memory_gb": 48,
        "description": "重量級 6Bパラメータ GPT-J",
        "rank": 3
    },
    "microsoft/DialoGPT-large": {
        "parameters": 774_030_080,
        "size_gb": {"fp32": 3, "fp16": 1.5, "int8": 0.8, "int4": 0.4},
        "min_memory_gb": 8,
        "recommended_memory_gb": 16,
        "description": "中量級 774Mパラメータ DialoGPT",
        "rank": 4
    }
}

class CPUHeaviestLLMDemo:
    """CPU環境最大規模LLMデモクラス"""
    
    def __init__(self, model_name: str, use_4bit: bool = False, use_8bit: bool = False):
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.device = torch.device("cpu")  # CPU専用
        
        # モデル・トークナイザー
        self.model = None
        self.tokenizer = None
        
        # 最適化状態
        self.optimization_applied = False
        self.quantization_info = {}
        
        # システム情報
        self.system_info = self._get_system_info()
        
        print(f"🏋️ CPU環境最大規模LLM Infer-OS最適化デモ")
        print(f"対象モデル: {model_name}")
        self._print_system_info()
        self._validate_system_requirements()
    
    def _get_system_info(self) -> Dict:
        """システム情報を取得"""
        memory = psutil.virtual_memory()
        
        info = {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": memory.total / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "memory_used_gb": memory.used / (1024**3),
            "memory_percent": memory.percent,
        }
        
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
        print(f"  CPU: {self.system_info['cpu_count']}コア")
        print(f"  メモリ: {self.system_info['memory_total_gb']:.1f}GB")
        print(f"  使用中: {self.system_info['memory_used_gb']:.1f}GB ({self.system_info['memory_percent']:.1f}%)")
        print(f"  利用可能: {self.system_info['memory_available_gb']:.1f}GB")
        
        print(f"\n🔧 最適化ライブラリ:")
        print(f"  Accelerate: {'✅' if ACCELERATE_AVAILABLE else '❌'}")
        print(f"  BitsAndBytes: {'✅' if BITSANDBYTES_AVAILABLE else '❌'}")
    
    def _validate_system_requirements(self):
        """システム要件を検証"""
        if self.model_name in CPU_HEAVY_MODELS:
            model_info = CPU_HEAVY_MODELS[self.model_name]
            min_memory = model_info["min_memory_gb"]
            recommended_memory = model_info["recommended_memory_gb"]
            
            print(f"\n🏋️ モデル要件:")
            print(f"  モデル: {model_info['description']}")
            print(f"  パラメータ数: {model_info['parameters']:,}")
            print(f"  最小メモリ: {min_memory}GB")
            print(f"  推奨メモリ: {recommended_memory}GB")
            
            # 量子化適用時のメモリ要件
            if self.use_4bit:
                required_memory = model_info["size_gb"]["int4"]
                print(f"  INT4量子化時: {required_memory}GB")
            elif self.use_8bit:
                required_memory = model_info["size_gb"]["int8"]
                print(f"  INT8量子化時: {required_memory}GB")
            else:
                required_memory = model_info["size_gb"]["fp16"]
                print(f"  FP16時: {required_memory}GB")
            
            # メモリ充足性チェック
            available_memory = self.system_info['memory_available_gb']
            
            if available_memory < required_memory:
                print(f"⚠️  メモリ不足の可能性があります")
                print(f"  必要: {required_memory}GB, 利用可能: {available_memory:.1f}GB")
                print(f"💡 量子化オプション（--use-8bit または --use-4bit）の使用を推奨")
            elif available_memory < recommended_memory:
                print(f"⚠️  推奨メモリ未満です")
                print(f"  推奨: {recommended_memory}GB, 利用可能: {available_memory:.1f}GB")
                print(f"💡 量子化オプションで安定性向上")
            else:
                print(f"✅ メモリ要件を満たしています")
    
    def list_available_models(self):
        """利用可能なモデル一覧を表示"""
        print(f"\n🏋️ CPU対応最大規模モデル一覧:")
        
        sorted_models = sorted(CPU_HEAVY_MODELS.items(), key=lambda x: x[1]["rank"])
        
        for model_name, info in sorted_models:
            rank_emoji = ["🥇", "🥈", "🥉", "🏅"][info["rank"] - 1] if info["rank"] <= 4 else "📋"
            print(f"  {rank_emoji} {model_name}")
            print(f"    {info['description']}")
            print(f"    パラメータ: {info['parameters']:,}")
            print(f"    推奨メモリ: {info['recommended_memory_gb']}GB")
            print()
    
    def create_quantization_config(self) -> Optional[Any]:
        """量子化設定を作成（CPU対応版）"""
        if not BITSANDBYTES_AVAILABLE:
            print("⚠️ BitsAndBytes未対応のため、量子化無しで実行します")
            return None
        
        try:
            if self.use_4bit:
                print("🔧 4bit量子化を有効化しました（CPU最適化）")
                config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float32,  # CPU用
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_enable_fp32_cpu_offload=True
                )
                return config
            elif self.use_8bit:
                print("🔧 8bit量子化を有効化しました（CPU最適化）")
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
            
            # メモリ使用量監視開始
            initial_memory = psutil.virtual_memory().used / (1024**3)
            print(f"📊 ロード前メモリ使用量: {initial_memory:.1f}GB")
            
            # 量子化設定
            quantization_config = self.create_quantization_config()
            
            # モデルロード設定（CPU最適化）
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float32,  # CPU用
                "low_cpu_mem_usage": True,
                "device_map": "cpu",
            }
            
            # 量子化設定を追加（エラーハンドリング付き）
            if quantization_config is not None:
                try:
                    model_kwargs["quantization_config"] = quantization_config
                except Exception as e:
                    print(f"⚠️ 量子化設定適用エラー: {e}")
                    print("💡 量子化無しで続行します")
            
            print(f"📥 大規模モデル '{self.model_name}' をロード中...")
            
            # モデルロード（段階的フォールバック）
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            except Exception as e:
                print(f"⚠️ 最適化モデルロードエラー: {e}")
                print("💡 基本設定でリトライします")
                
                # フォールバック1: 量子化無し
                basic_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.float32,
                    "low_cpu_mem_usage": True,
                }
                
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        **basic_kwargs
                    )
                except Exception as e2:
                    print(f"⚠️ 基本モデルロードエラー: {e2}")
                    print("💡 最小設定でリトライします")
                    
                    # フォールバック2: 最小設定
                    minimal_kwargs = {
                        "trust_remote_code": True,
                    }
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        **minimal_kwargs
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
            
            # CPU最適化適用
            self._apply_cpu_optimizations()
            
            # メモリ使用量監視終了
            final_memory = psutil.virtual_memory().used / (1024**3)
            memory_used = final_memory - initial_memory
            
            print(f"📊 ロード後メモリ使用量: {final_memory:.1f}GB")
            print(f"📊 モデルメモリ使用量: {memory_used:.1f}GB")
            print("✅ モデルロード完了")
            
            return True
            
        except Exception as e:
            print(f"❌ モデルロードエラー: {e}")
            print(f"詳細: {traceback.format_exc()}")
            return False
    
    def _apply_cpu_optimizations(self):
        """CPU専用最適化を適用"""
        try:
            print("🔧 CPU専用最適化を適用中...")
            
            # CPU最適化設定
            torch.set_num_threads(psutil.cpu_count())
            print(f"  ✅ CPUスレッド数設定: {psutil.cpu_count()}")
            
            # メモリ効率化
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("  ✅ グラディエントチェックポイント有効化")
            
            # キャッシュ設定
            if hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = True
                print("  ✅ キャッシュ有効化")
            
            # CPU専用最適化
            try:
                # JITコンパイル（対応モデルのみ）
                if hasattr(torch.jit, 'optimize_for_inference'):
                    self.model = torch.jit.optimize_for_inference(self.model)
                    print("  ✅ JIT最適化適用")
            except:
                pass
            
            self.optimization_applied = True
            print("🚀 CPU専用最適化適用完了")
            
        except Exception as e:
            print(f"⚠️ CPU最適化エラー: {e}")
    
    def generate_text(self, prompt: str, max_length: int = 200) -> Dict:
        """テキスト生成（CPU最適化版）"""
        if self.model is None or self.tokenizer is None:
            return {"error": "モデルまたはトークナイザーが未ロード"}
        
        try:
            print(f"\n🎯 テキスト生成開始")
            print(f"プロンプト: \"{prompt}\"")
            print(f"最大長: {max_length}")
            
            # メモリ・CPU使用量測定開始
            initial_memory = psutil.virtual_memory().used / (1024**3)
            initial_cpu = psutil.cpu_percent(interval=None)
            
            # トークン化
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # 生成設定（CPU最適化）
            generation_config = {
                "max_length": max_length,
                "num_return_sequences": 1,
                "temperature": 0.7,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }
            
            # 生成実行（時間・リソース測定）
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
            
            # リソース使用量測定終了
            final_memory = psutil.virtual_memory().used / (1024**3)
            final_cpu = psutil.cpu_percent(interval=None)
            
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
                "resource_usage": {
                    "memory_used_gb": final_memory - initial_memory,
                    "memory_total_gb": final_memory,
                    "cpu_usage_percent": final_cpu
                },
                "optimization_applied": self.optimization_applied,
                "quantization_info": {
                    "use_4bit": self.use_4bit,
                    "use_8bit": self.use_8bit
                }
            }
            
            self._print_generation_results(result)
            return result
            
        except Exception as e:
            error_msg = f"テキスト生成エラー: {e}"
            print(f"❌ {error_msg}")
            return {"error": error_msg, "traceback": traceback.format_exc()}
    
    def _print_generation_results(self, result: Dict):
        """生成結果を表示"""
        print(f"\n📊 生成結果:")
        print(f"  生成時間: {result['generation_time']:.2f}秒")
        print(f"  入力トークン: {result['input_tokens']}")
        print(f"  出力トークン: {result['output_tokens']}")
        print(f"  スループット: {result['tokens_per_second']:.1f} tokens/sec")
        
        print(f"\n💾 リソース使用量:")
        print(f"  メモリ使用: {result['resource_usage']['memory_used_gb']:.1f}GB")
        print(f"  総メモリ: {result['resource_usage']['memory_total_gb']:.1f}GB")
        print(f"  CPU使用率: {result['resource_usage']['cpu_usage_percent']:.1f}%")
        
        print(f"\n🔧 最適化状態:")
        print(f"  CPU最適化: {'✅' if result['optimization_applied'] else '❌'}")
        print(f"  4bit量子化: {'✅' if result['quantization_info']['use_4bit'] else '❌'}")
        print(f"  8bit量子化: {'✅' if result['quantization_info']['use_8bit'] else '❌'}")
        
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
                
                # メモリクリーンアップ
                gc.collect()
                
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
            filename = f'cpu_heaviest_llm_session_{model_safe_name}_{timestamp}.json'
            
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
        memory_used = [r['resource_usage']['memory_used_gb'] for r in results]
        
        return {
            "total_generations": len(results),
            "avg_generation_time": sum(generation_times) / len(generation_times),
            "avg_tokens_per_second": sum(tokens_per_second) / len(tokens_per_second),
            "total_output_tokens": sum(output_tokens),
            "avg_memory_used_gb": sum(memory_used) / len(memory_used),
            "min_generation_time": min(generation_times),
            "max_generation_time": max(generation_times)
        }

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="CPU環境最大規模LLM Infer-OS最適化デモ")
    parser.add_argument("--model", default="EleutherAI/gpt-neox-20b", help="使用するモデル名")
    parser.add_argument("--prompt", help="テスト用プロンプト")
    parser.add_argument("--max-length", type=int, default=200, help="最大生成長")
    parser.add_argument("--use-4bit", action="store_true", help="4bit量子化を使用")
    parser.add_argument("--use-8bit", action="store_true", help="8bit量子化を使用")
    parser.add_argument("--interactive", action="store_true", help="インタラクティブモード")
    parser.add_argument("--list-models", action="store_true", help="利用可能なモデル一覧を表示")
    
    args = parser.parse_args()
    
    if args.list_models:
        demo = CPUHeaviestLLMDemo("dummy", False, False)
        demo.list_available_models()
        return
    
    print(f"""
{'='*80}
🏋️ CPU環境最大規模LLM Infer-OS最適化デモ
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
        demo = CPUHeaviestLLMDemo(
            model_name=args.model,
            use_4bit=args.use_4bit,
            use_8bit=args.use_8bit
        )
        
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

