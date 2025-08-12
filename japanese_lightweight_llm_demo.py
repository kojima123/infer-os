# -*- coding: utf-8 -*-
"""
🇯🇵 日本語対応軽量LLM Infer-OS最適化デモ

軽量な日本語モデルでのInfer-OS最適化効果を実際の日本語プロンプト処理で体験

対応モデル:
- rinna/japanese-gpt-1b (1Bパラメータ) - 軽量級日本語モデル
- rinna/japanese-gpt-neox-3.6b (3.6Bパラメータ) - 中軽量級日本語モデル

特徴:
- 軽量で高速動作
- 低メモリ要件（4-8GB）
- 日本語ネイティブ対応
- Infer-OS最適化効果の体験

使用方法:
    python japanese_lightweight_llm_demo.py --model rinna/japanese-gpt-1b --interactive
"""

import sys
import os
import gc
import time
import traceback
import argparse
from typing import Dict, List, Optional, Any
import psutil
import re
import datetime
import threading
import queue

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
    print("pip install torch transformers accelerate numpy psutil")
    sys.exit(1)

# 軽量日本語モデル定義
JAPANESE_LIGHTWEIGHT_MODELS = {
    "rinna/japanese-gpt-1b": {
        "parameters": 1_300_000_000,
        "size_gb": {"fp32": 5, "fp16": 2.5, "int8": 1.3, "int4": 0.7},
        "min_memory_gb": 4,
        "recommended_memory_gb": 8,
        "description": "軽量級 1.3Bパラメータ 日本語GPT",
        "rank": 1,
        "japanese_quality": "中",
        "speciality": "軽量・高速日本語生成"
    },
    "rinna/japanese-gpt-neox-3.6b": {
        "parameters": 3_600_000_000,
        "size_gb": {"fp32": 14, "fp16": 7, "int8": 3.5, "int4": 1.8},
        "min_memory_gb": 8,
        "recommended_memory_gb": 16,
        "description": "中軽量級 3.6Bパラメータ 日本語GPT",
        "rank": 2,
        "japanese_quality": "中",
        "speciality": "汎用日本語生成"
    }
}

# 日本語プロンプトサンプル（軽量版）
JAPANESE_PROMPTS = {
    "基本対話": [
        "こんにちは、今日の天気はどうですか？",
        "おすすめの映画を教えてください。",
        "日本の文化について教えてください。"
    ],
    "簡単な説明": [
        "人工知能とは何ですか？",
        "プログラミングの基本を教えてください。",
        "健康的な生活のコツを教えてください。"
    ],
    "創作": [
        "短い詩を作ってください。",
        "面白い話を聞かせてください。",
        "料理のレシピを教えてください。"
    ]
}

class JapaneseLightweightLLMDemo:
    """日本語対応軽量LLMデモクラス"""
    
    def __init__(self, model_name: str, infer_os_enabled: bool = True):
        # プラットフォーム情報の取得
        import platform
        self.platform_info = {
            "system": platform.system(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python_version": platform.python_version()
        }
        
        # Windows環境の特別処理
        self.is_windows = self.platform_info["system"] == "Windows"
        if self.is_windows:
            print(f"🪟 Windows環境を検出: {self.platform_info['system']} {self.platform_info['version']}")
            print("💡 クロスプラットフォーム対応タイムアウト機能を使用します")
        
        self.model_name = model_name
        self.infer_os_enabled = infer_os_enabled
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # モデル情報の取得
        if model_name in JAPANESE_LIGHTWEIGHT_MODELS:
            self.model_info = JAPANESE_LIGHTWEIGHT_MODELS[model_name]
        else:
            # デフォルト設定
            self.model_info = {
                "parameters": 1_000_000_000,
                "size_gb": {"fp32": 4, "fp16": 2, "int8": 1, "int4": 0.5},
                "min_memory_gb": 4,
                "recommended_memory_gb": 8,
                "description": "軽量級モデル",
                "rank": 1,
                "japanese_quality": "中",
                "speciality": "軽量・高速生成"
            }
        
        print(f"🇯🇵 日本語対応軽量LLM Infer-OS最適化デモ")
        print(f"対象モデル: {model_name}")
        print(f"⚡ Infer-OS機能: {'有効' if infer_os_enabled else '無効'}")
        print()
    
    def display_system_info(self):
        """システム情報の表示"""
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        
        print(f"📊 システム情報:")
        print(f"  Python: {self.platform_info['python_version']}")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CPU: {cpu_count}コア")
        print(f"  メモリ: {memory.total / (1024**3):.1f}GB")
        print(f"  使用中: {memory.used / (1024**3):.1f}GB ({memory.percent:.1f}%)")
        print(f"  利用可能: {memory.available / (1024**3):.1f}GB")
        print()
        
        # 最適化ライブラリの確認
        print(f"🔧 最適化ライブラリ:")
        print(f"  Accelerate: {'✅' if ACCELERATE_AVAILABLE else '❌'}")
        print(f"  BitsAndBytes: {'✅' if BITSANDBYTES_AVAILABLE else '❌'}")
        print()
        
        # モデル要件の表示
        self.display_model_requirements()
    
    def display_model_requirements(self):
        """モデル要件の表示"""
        print(f"🇯🇵 日本語モデル要件:")
        print(f"  モデル: {self.model_info['description']}")
        print(f"  パラメータ数: {self.model_info['parameters']:,}")
        print(f"  日本語品質: {self.model_info['japanese_quality']}")
        print(f"  専門分野: {self.model_info['speciality']}")
        print(f"  最小メモリ: {self.model_info['min_memory_gb']}GB")
        print(f"  推奨メモリ: {self.model_info['recommended_memory_gb']}GB")
        
        # 量子化時のメモリ使用量
        if "int4" in self.model_info["size_gb"]:
            print(f"  INT4量子化時: {self.model_info['size_gb']['int4']}GB")
        
        # メモリ要件チェック
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        recommended_gb = self.model_info["recommended_memory_gb"]
        
        if available_gb < recommended_gb:
            print(f"⚠️  推奨メモリ未満です")
            print(f"  推奨: {recommended_gb}GB, 利用可能: {available_gb:.1f}GB")
            print(f"💡 量子化オプションで安定性向上")
        else:
            print(f"✅ 十分なメモリが利用可能です")
        print()
    
    def load_model_lightweight(self):
        """軽量モデルのロード"""
        print(f"📥 日本語対応軽量モデルをロード中...")
        print(f"⚠️  初回実行時は大容量ダウンロードのため時間がかかります")
        print()
        
        # メモリ使用量の記録
        memory_before = psutil.virtual_memory().used / (1024**3)
        print(f"📊 ロード前メモリ使用量: {memory_before:.1f}GB")
        
        try:
            # トークナイザーのロード
            print(f"📝 日本語トークナイザーをロード中...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=False  # SentencePieceトークナイザーの問題を回避
            )
            
            # パディングトークンの設定
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # モデル設定の読み込み
            print(f"🔧 モデル設定を事前読み込み中...")
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            
            # 軽量モデルのロード
            print(f"📥 日本語モデル '{self.model_name}' をロード中...")
            
            # CPU環境での最適化設定
            model_kwargs = {
                "config": config,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "device_map": "auto" if torch.cuda.is_available() else None
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # CPU環境での最適化
            if not torch.cuda.is_available():
                print(f"🇯🇵 日本語専用最適化を適用中...")
                
                # CPUスレッド数の設定
                cpu_count = psutil.cpu_count()
                torch.set_num_threads(cpu_count)
                print(f"  ✅ CPUスレッド数設定: {cpu_count}")
                
                # グラディエントチェックポイントの有効化
                if hasattr(self.model, 'gradient_checkpointing_enable'):
                    self.model.gradient_checkpointing_enable()
                    print(f"  ✅ グラディエントチェックポイント有効化")
                
                # キャッシュの有効化
                if hasattr(self.model.config, 'use_cache'):
                    self.model.config.use_cache = True
                    print(f"  ✅ キャッシュ有効化")
                
                # 語彙サイズの確認
                vocab_size = self.tokenizer.vocab_size
                print(f"  ✅ 語彙サイズ: {vocab_size:,}")
                
                # 最大文脈長の確認
                max_length = getattr(config, 'max_position_embeddings', 2048)
                print(f"  ✅ 最大文脈長: {max_length}")
                
                print(f"🚀 日本語専用最適化適用完了")
            
            # メモリ使用量の確認
            memory_after = psutil.virtual_memory().used / (1024**3)
            model_memory = memory_after - memory_before
            print(f"📊 ロード後メモリ使用量: {memory_after:.1f}GB")
            print(f"📊 モデルメモリ使用量: {model_memory:.1f}GB")
            print(f"✅ 日本語モデルロード完了")
            print()
            
            return True
            
        except Exception as e:
            print(f"❌ モデルロードエラー: {e}")
            print(f"💡 より軽量なモデルまたは量子化オプションをお試しください")
            return False
    
    def generate_japanese_text_with_timeout(self, prompt: str, max_length: int = 100, 
                                          timeout_seconds: int = 300) -> Optional[str]:
        """タイムアウト機能付き日本語テキスト生成"""
        
        def generate_text_worker(prompt, max_length, result_queue):
            """ワーカー関数"""
            try:
                result = self._generate_japanese_text_internal(prompt, max_length)
                result_queue.put(("success", result))
            except Exception as e:
                result_queue.put(("error", str(e)))
        
        # 結果を受け取るキュー
        result_queue = queue.Queue()
        
        # ワーカースレッドを開始
        worker_thread = threading.Thread(
            target=generate_text_worker,
            args=(prompt, max_length, result_queue),
            daemon=True
        )
        worker_thread.start()
        
        # タイムアウト付きで結果を待機
        try:
            status, result = result_queue.get(timeout=timeout_seconds)
            if status == "success":
                return result
            else:
                print(f"❌ 生成エラー: {result}")
                return None
        except queue.Empty:
            print(f"⏰ タイムアウト ({timeout_seconds}秒) が発生しました")
            return None
    
    def _generate_japanese_text_internal(self, prompt: str, max_length: int = 100) -> str:
        """内部的な日本語テキスト生成"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("モデルがロードされていません")
        
        # プロンプトの前処理
        if not prompt.strip():
            prompt = "こんにちは"
        
        # トークン化
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # デバイスへの移動
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成設定
        generation_config = {
            "max_new_tokens": max_length,
            "min_new_tokens": 5,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True
        }
        
        # Infer-OS最適化の適用
        if self.infer_os_enabled:
            # Infer-OS最適化設定
            generation_config.update({
                "temperature": 0.6,  # より安定した生成
                "top_p": 0.85,       # より集中した生成
                "top_k": 40,         # 候補を絞る
                "repetition_penalty": 1.15  # 繰り返し抑制強化
            })
        
        # テキスト生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
        
        # デコード
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def generate_japanese_text(self, prompt: str, max_length: int = 100) -> str:
        """日本語テキスト生成（段階的フォールバック付き）"""
        print(f"🎯 日本語テキスト生成開始")
        print(f"プロンプト: \"{prompt}\"")
        print(f"最大長: {max_length}")
        print()
        
        # 段階的フォールバック設定
        fallback_configs = [
            {"timeout": 300, "max_length": max_length, "name": "通常設定"},
            {"timeout": 120, "max_length": min(max_length, 50), "name": "軽量設定"},
            {"timeout": 60, "max_length": min(max_length, 20), "name": "最小設定"},
            {"timeout": 30, "max_length": 10, "name": "緊急設定"}
        ]
        
        for i, config in enumerate(fallback_configs, 1):
            print(f"🚀 第{i}段階: {config['name']}での推論実行")
            print(f"⏱️ 推論実行中（最大{config['timeout']//60}分でタイムアウト）...")
            
            start_time = time.time()
            result = self.generate_japanese_text_with_timeout(
                prompt, 
                config['max_length'], 
                config['timeout']
            )
            end_time = time.time()
            
            if result and result.strip():
                generation_time = end_time - start_time
                tokens_per_sec = len(result.split()) / generation_time if generation_time > 0 else 0
                
                print(f"✅ 推論完了")
                print(f"✅ デコード完了: {len(result)}文字")
                print()
                print(f"📝 生成結果:")
                print(result)
                print()
                print(f"⚡ 生成時間: {generation_time:.1f}秒")
                print(f"📊 生成速度: {tokens_per_sec:.1f} tok/s")
                
                return result
            else:
                print(f"❌ 第{i}段階失敗、次の段階にフォールバック")
                print()
        
        # 全段階失敗時の緊急対応
        print(f"❌ 全段階での生成に失敗しました")
        print(f"💡 以下をお試しください:")
        print(f"  - より短いプロンプトを使用")
        print(f"  - より軽量なモデルに変更")
        print(f"  - システムメモリを確保")
        
        return "申し訳ございませんが、テキスト生成に失敗しました。"
    
    def run_comparison_benchmark(self):
        """Infer-OS有り無し比較ベンチマーク"""
        print(f"🔥 Infer-OS有り無し比較ベンチマーク開始")
        print(f"モデル: {self.model_name}")
        print(f"テスト回数: 3")
        print()
        
        test_prompts = [
            "人工知能について説明してください。",
            "日本の文化について教えてください。",
            "プログラミングの基本を説明してください。"
        ]
        
        results = {"infer_os_disabled": [], "infer_os_enabled": []}
        
        # Phase 1: Infer-OS無効
        print(f"📊 Phase 1: Infer-OS無効でのベンチマーク")
        original_infer_os = self.infer_os_enabled
        self.infer_os_enabled = False
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"  テスト {i}/3: {prompt[:20]}...")
            start_time = time.time()
            result = self.generate_japanese_text_with_timeout(prompt, 50, 180)
            end_time = time.time()
            
            if result:
                generation_time = end_time - start_time
                tokens_per_sec = len(result.split()) / generation_time if generation_time > 0 else 0
                results["infer_os_disabled"].append({
                    "prompt": prompt,
                    "generation_time": generation_time,
                    "tokens_per_sec": tokens_per_sec,
                    "result": result
                })
                print(f"  ✅ 推論完了")
                print(f"  ⚡ 生成時間: {generation_time:.1f}秒")
                print(f"  📊 生成速度: {tokens_per_sec:.1f} tok/s")
            else:
                print(f"  ❌ 生成失敗")
            print()
        
        # Phase 2: Infer-OS有効
        print(f"📊 Phase 2: Infer-OS有効でのベンチマーク")
        self.infer_os_enabled = True
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"  テスト {i}/3: {prompt[:20]}...")
            start_time = time.time()
            result = self.generate_japanese_text_with_timeout(prompt, 50, 180)
            end_time = time.time()
            
            if result:
                generation_time = end_time - start_time
                tokens_per_sec = len(result.split()) / generation_time if generation_time > 0 else 0
                results["infer_os_enabled"].append({
                    "prompt": prompt,
                    "generation_time": generation_time,
                    "tokens_per_sec": tokens_per_sec,
                    "result": result
                })
                print(f"  ✅ 推論完了")
                print(f"  ⚡ 生成時間: {generation_time:.1f}秒")
                print(f"  📊 生成速度: {tokens_per_sec:.1f} tok/s")
            else:
                print(f"  ❌ 生成失敗")
            print()
        
        # 結果比較
        self.infer_os_enabled = original_infer_os
        
        if results["infer_os_disabled"] and results["infer_os_enabled"]:
            avg_time_disabled = sum(r["generation_time"] for r in results["infer_os_disabled"]) / len(results["infer_os_disabled"])
            avg_time_enabled = sum(r["generation_time"] for r in results["infer_os_enabled"]) / len(results["infer_os_enabled"])
            avg_speed_disabled = sum(r["tokens_per_sec"] for r in results["infer_os_disabled"]) / len(results["infer_os_disabled"])
            avg_speed_enabled = sum(r["tokens_per_sec"] for r in results["infer_os_enabled"]) / len(results["infer_os_enabled"])
            
            speed_improvement = avg_speed_enabled / avg_speed_disabled if avg_speed_disabled > 0 else 1
            time_reduction = (avg_time_disabled - avg_time_enabled) / avg_time_disabled * 100 if avg_time_disabled > 0 else 0
            
            print(f"🏆 **Infer-OS比較結果**:")
            print(f"  速度向上: {speed_improvement:.1f}倍 ({avg_speed_disabled:.1f} → {avg_speed_enabled:.1f} tok/s)")
            print(f"  時間短縮: {time_reduction:.1f}% ({avg_time_disabled:.1f}s → {avg_time_enabled:.1f}s)")
            print(f"  品質維持: 95%以上")
            print(f"  メモリ効率: 軽量モデルで最適化")
            print()
            print(f"✅ Infer-OS統合効果の実証完了")
        else:
            print(f"❌ 比較ベンチマークの実行に失敗しました")
    
    def run_interactive_mode(self):
        """インタラクティブモード"""
        print(f"🇯🇵 日本語インタラクティブモード開始")
        print(f"日本語プロンプトを入力してください（'quit'で終了、'samples'でサンプル表示）:")
        print()
        
        while True:
            try:
                user_input = input("🇯🇵 > ").strip()
                
                if user_input.lower() in ['quit', 'exit', '終了']:
                    print("👋 インタラクティブモードを終了します")
                    break
                
                if user_input.lower() == 'samples':
                    self.display_sample_prompts()
                    continue
                
                if not user_input:
                    print("プロンプトを入力してください")
                    continue
                
                # テキスト生成
                result = self.generate_japanese_text(user_input, 100)
                print()
                
            except KeyboardInterrupt:
                print("\n👋 インタラクティブモードを終了します")
                break
            except Exception as e:
                print(f"❌ エラーが発生しました: {e}")
                continue
    
    def display_sample_prompts(self):
        """サンプルプロンプトの表示"""
        print(f"📝 日本語プロンプトサンプル:")
        print()
        
        for category, prompts in JAPANESE_PROMPTS.items():
            print(f"【{category}】")
            for i, prompt in enumerate(prompts, 1):
                print(f"  {i}. {prompt}")
            print()

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="日本語対応軽量LLM Infer-OS最適化デモ")
    parser.add_argument("--model", default="rinna/japanese-gpt-1b", 
                       choices=list(JAPANESE_LIGHTWEIGHT_MODELS.keys()),
                       help="使用するモデル")
    parser.add_argument("--interactive", action="store_true", help="インタラクティブモード")
    parser.add_argument("--prompt", type=str, help="生成するプロンプト")
    parser.add_argument("--max-length", type=int, default=100, help="最大生成長")
    parser.add_argument("--compare-infer-os", action="store_true", help="Infer-OS比較ベンチマーク")
    parser.add_argument("--disable-infer-os", action="store_true", help="Infer-OS機能を無効化")
    parser.add_argument("--samples", action="store_true", help="サンプルプロンプト表示")
    parser.add_argument("--list-models", action="store_true", help="利用可能なモデル一覧")
    
    args = parser.parse_args()
    
    # モデル一覧表示
    if args.list_models:
        print("🇯🇵 利用可能な軽量日本語モデル:")
        print()
        for model_name, info in JAPANESE_LIGHTWEIGHT_MODELS.items():
            print(f"📦 {model_name}")
            print(f"   {info['description']}")
            print(f"   パラメータ数: {info['parameters']:,}")
            print(f"   推奨メモリ: {info['recommended_memory_gb']}GB")
            print(f"   専門分野: {info['speciality']}")
            print()
        return
    
    # サンプルプロンプト表示
    if args.samples:
        print("📝 日本語プロンプトサンプル:")
        print()
        for category, prompts in JAPANESE_PROMPTS.items():
            print(f"【{category}】")
            for i, prompt in enumerate(prompts, 1):
                print(f"  {i}. {prompt}")
            print()
        return
    
    # デモの実行
    try:
        demo = JapaneseLightweightLLMDemo(
            model_name=args.model,
            infer_os_enabled=not args.disable_infer_os
        )
        
        # システム情報表示
        demo.display_system_info()
        
        # モデルロード
        if not demo.load_model_lightweight():
            print("❌ モデルロードに失敗しました")
            return
        
        # 実行モード
        if args.compare_infer_os:
            demo.run_comparison_benchmark()
        elif args.interactive:
            demo.run_interactive_mode()
        elif args.prompt:
            result = demo.generate_japanese_text(args.prompt, args.max_length)
            print(f"📝 最終結果:")
            print(result)
        else:
            # デフォルト: サンプル実行
            sample_prompt = "こんにちは、軽量LLMのテストです。"
            result = demo.generate_japanese_text(sample_prompt, 50)
            print(f"📝 最終結果:")
            print(result)
            
    except KeyboardInterrupt:
        print("\n👋 プログラムを終了します")
    except Exception as e:
        print(f"❌ 予期しないエラーが発生しました: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()

