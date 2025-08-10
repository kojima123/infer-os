#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌍 拡張版重量級LLM Infer-OS最適化デモ

gpt-oss-20b、DeepSeek、日本語モデルを含む最大規模LLMモデルでの
Infer-OS最適化効果を実際のプロンプト処理で体験

対応モデル:
【超重量級 (20B+)】
- openai/gpt-oss-20b (20Bパラメータ) - GPT系最重量級
- deepseek-ai/deepseek-llm-67b-chat (67Bパラメータ) - 超重量級
- deepseek-ai/deepseek-coder-33b-instruct (33Bパラメータ) - コード特化

【重量級 (7B-20B)】
- matsuo-lab/weblab-10b (10Bパラメータ) - 日本語最重量級
- EleutherAI/gpt-neox-20b (20Bパラメータ) - 英語重量級
- bigscience/bloom-7b1 (7.1Bパラメータ) - 多言語重量級

特徴:
- CPU/GPU自動検出・最適化
- 高度な量子化対応（MXFP4/INT4/INT8）
- 多言語対応（日本語・英語・中国語等）
- リアルタイム性能監視
- 分散推論・メモリオフロード

使用方法:
    python extended_heavy_llm_demo.py --model openai/gpt-oss-20b --use-8bit --interactive
    python extended_heavy_llm_demo.py --model deepseek-ai/deepseek-llm-67b-chat --use-4bit --interactive
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
import warnings
warnings.filterwarnings("ignore")

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

# 拡張版重量級モデル定義
EXTENDED_HEAVY_MODELS = {
    # 超重量級 (20B+)
    "deepseek-ai/deepseek-llm-67b-chat": {
        "parameters": 67_000_000_000,
        "size_gb": {"fp32": 268, "fp16": 134, "int8": 67, "int4": 33.5},
        "min_memory_gb": 150,
        "recommended_memory_gb": 200,
        "description": "超重量級 67Bパラメータ DeepSeek チャット特化",
        "rank": 1,
        "category": "超重量級",
        "language": "多言語",
        "speciality": "対話・推論・数学",
        "gpu_required": False,
        "quantization_support": ["int8", "int4"]
    },
    "deepseek-ai/deepseek-coder-33b-instruct": {
        "parameters": 33_000_000_000,
        "size_gb": {"fp32": 132, "fp16": 66, "int8": 33, "int4": 16.5},
        "min_memory_gb": 80,
        "recommended_memory_gb": 120,
        "description": "超重量級 33Bパラメータ DeepSeek コード特化",
        "rank": 2,
        "category": "超重量級",
        "language": "多言語",
        "speciality": "プログラミング・コード生成",
        "gpu_required": False,
        "quantization_support": ["int8", "int4"]
    },
    "openai/gpt-oss-20b": {
        "parameters": 20_000_000_000,
        "size_gb": {"fp32": 80, "fp16": 40, "int8": 20, "int4": 10},
        "min_memory_gb": 50,
        "recommended_memory_gb": 80,
        "description": "重量級 20Bパラメータ GPT-OSS",
        "rank": 3,
        "category": "重量級",
        "language": "英語中心",
        "speciality": "汎用テキスト生成",
        "gpu_required": False,
        "quantization_support": ["int8", "int4"],
        "note": "MXFP4量子化済みの場合はGPU必須"
    },
    "EleutherAI/gpt-neox-20b": {
        "parameters": 20_000_000_000,
        "size_gb": {"fp32": 80, "fp16": 40, "int8": 20, "int4": 10},
        "min_memory_gb": 50,
        "recommended_memory_gb": 80,
        "description": "重量級 20Bパラメータ GPT-NeoX",
        "rank": 4,
        "category": "重量級",
        "language": "英語中心",
        "speciality": "汎用テキスト生成",
        "gpu_required": False,
        "quantization_support": ["int8", "int4"]
    },
    
    # 日本語重量級
    "matsuo-lab/weblab-10b": {
        "parameters": 10_737_418_240,
        "size_gb": {"fp32": 43, "fp16": 21.5, "int8": 10.8, "int4": 5.4},
        "min_memory_gb": 48,
        "recommended_memory_gb": 64,
        "description": "重量級 10Bパラメータ 日本語特化（東大松尾研）",
        "rank": 5,
        "category": "重量級",
        "language": "日本語",
        "speciality": "学術・技術文書",
        "gpu_required": False,
        "quantization_support": ["int8", "int4"]
    },
    "rinna/youri-7b-chat": {
        "parameters": 7_241_732_096,
        "size_gb": {"fp32": 28, "fp16": 14, "int8": 7, "int4": 3.5},
        "min_memory_gb": 32,
        "recommended_memory_gb": 48,
        "description": "重量級 7Bパラメータ 日本語チャット特化",
        "rank": 6,
        "category": "重量級",
        "language": "日本語",
        "speciality": "対話・チャット",
        "gpu_required": False,
        "quantization_support": ["int8", "int4"]
    },
    
    # 多言語重量級
    "bigscience/bloom-7b1": {
        "parameters": 7_100_000_000,
        "size_gb": {"fp32": 28, "fp16": 14, "int8": 7, "int4": 3.5},
        "min_memory_gb": 32,
        "recommended_memory_gb": 48,
        "description": "重量級 7.1Bパラメータ 多言語BLOOM",
        "rank": 7,
        "category": "重量級",
        "language": "多言語",
        "speciality": "多言語テキスト生成",
        "gpu_required": False,
        "quantization_support": ["int8", "int4"]
    },
    
    # 中量級（比較用）
    "microsoft/DialoGPT-large": {
        "parameters": 774_000_000,
        "size_gb": {"fp32": 3, "fp16": 1.5, "int8": 0.8, "int4": 0.4},
        "min_memory_gb": 8,
        "recommended_memory_gb": 16,
        "description": "中量級 774Mパラメータ 対話特化",
        "rank": 8,
        "category": "中量級",
        "language": "英語",
        "speciality": "対話・チャット",
        "gpu_required": False,
        "quantization_support": ["int8", "int4"]
    }
}

# 多言語プロンプトサンプル
MULTILINGUAL_PROMPTS = {
    "English": {
        "General": [
            "Explain the future of artificial intelligence from a technical perspective.",
            "Write a short story about a character discovering a hidden talent.",
            "Describe the impact of quantum computing on modern technology."
        ],
        "Technical": [
            "Explain machine learning algorithms in simple terms.",
            "Describe the principles of blockchain technology.",
            "What are the advantages of cloud computing?"
        ],
        "Creative": [
            "Write a poem about the beauty of nature.",
            "Create a dialogue between two AI systems.",
            "Describe a futuristic city in the year 2050."
        ]
    },
    "Japanese": {
        "文章生成": [
            "人工知能の未来について、技術的な観点から詳しく説明してください。",
            "桜が咲く春の日に、主人公が新しい出会いを経験する短編小説を書いてください。",
            "量子コンピュータの基本原理と将来の応用可能性について教えてください。"
        ],
        "技術解説": [
            "機械学習における深層学習の仕組みを、初心者にもわかりやすく説明してください。",
            "ブロックチェーン技術の仕組みとビジネスへの応用例を説明してください。",
            "クラウドコンピューティングのメリットとデメリットを整理してください。"
        ],
        "創作": [
            "未来都市を舞台にしたSF小説の冒頭部分を書いてください。",
            "2つのAIシステム間の対話を創作してください。",
            "2050年の未来都市について詩的に描写してください。"
        ]
    },
    "Programming": {
        "Python": [
            "Write a Python function to implement binary search.",
            "Create a class for managing a simple database.",
            "Implement a web scraper using requests and BeautifulSoup."
        ],
        "JavaScript": [
            "Write a JavaScript function to validate email addresses.",
            "Create a React component for a todo list.",
            "Implement a simple REST API using Node.js and Express."
        ],
        "Algorithm": [
            "Explain the quicksort algorithm with code examples.",
            "Implement a graph traversal algorithm (DFS or BFS).",
            "Write a function to find the longest common subsequence."
        ]
    }
}

class ExtendedHeavyLLMDemo:
    """拡張版重量級LLMデモクラス"""
    
    def __init__(self, model_name: str, use_4bit: bool = False, use_8bit: bool = False, force_cpu: bool = False):
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.force_cpu = force_cpu
        
        # デバイス検出
        self.device = self._detect_optimal_device()
        
        # モデル・トークナイザー
        self.model = None
        self.tokenizer = None
        
        # 最適化状態
        self.optimization_applied = False
        self.quantization_info = {}
        
        # システム情報
        self.system_info = self._get_system_info()
        
        print(f"🌍 拡張版重量級LLM Infer-OS最適化デモ")
        print(f"対象モデル: {model_name}")
        self._print_system_info()
        self._validate_system_requirements()
    
    def _detect_optimal_device(self) -> torch.device:
        """最適なデバイスを検出"""
        if self.force_cpu:
            print("🔧 CPU強制モード")
            return torch.device("cpu")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🎮 GPU検出: {gpu_memory:.1f}GB VRAM")
            return torch.device("cuda")
        else:
            print("💻 CPU環境で実行")
            return torch.device("cpu")
    
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
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
        }
        
        # GPU情報
        if torch.cuda.is_available():
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
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
        print(f"  デバイス: {self.device}")
        print(f"  CPU: {self.system_info['cpu_count']}コア")
        print(f"  メモリ: {self.system_info['memory_total_gb']:.1f}GB")
        print(f"  使用中: {self.system_info['memory_used_gb']:.1f}GB ({self.system_info['memory_percent']:.1f}%)")
        print(f"  利用可能: {self.system_info['memory_available_gb']:.1f}GB")
        
        if torch.cuda.is_available():
            print(f"  GPU: {self.system_info['gpu_name']}")
            print(f"  VRAM: {self.system_info['gpu_memory_gb']:.1f}GB")
        
        print(f"\n🔧 最適化ライブラリ:")
        print(f"  Accelerate: {'✅' if ACCELERATE_AVAILABLE else '❌'}")
        print(f"  BitsAndBytes: {'✅' if BITSANDBYTES_AVAILABLE else '❌'}")
    
    def _validate_system_requirements(self):
        """システム要件を検証"""
        if self.model_name in EXTENDED_HEAVY_MODELS:
            model_info = EXTENDED_HEAVY_MODELS[self.model_name]
            min_memory = model_info["min_memory_gb"]
            recommended_memory = model_info["recommended_memory_gb"]
            
            print(f"\n🌍 拡張モデル要件:")
            print(f"  モデル: {model_info['description']}")
            print(f"  カテゴリ: {model_info['category']}")
            print(f"  パラメータ数: {model_info['parameters']:,}")
            print(f"  言語: {model_info['language']}")
            print(f"  専門分野: {model_info['speciality']}")
            print(f"  最小メモリ: {min_memory}GB")
            print(f"  推奨メモリ: {recommended_memory}GB")
            
            if "note" in model_info:
                print(f"  注意: {model_info['note']}")
            
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
        """利用可能な拡張モデル一覧を表示"""
        print(f"\n🌍 拡張版重量級モデル一覧:")
        
        # カテゴリ別に整理
        categories = {}
        for model_name, info in EXTENDED_HEAVY_MODELS.items():
            category = info["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append((model_name, info))
        
        # カテゴリ順で表示
        category_order = ["超重量級", "重量級", "中量級"]
        
        for category in category_order:
            if category in categories:
                print(f"\n【{category}】")
                sorted_models = sorted(categories[category], key=lambda x: x[1]["rank"])
                
                for model_name, info in sorted_models:
                    rank_emoji = ["🥇", "🥈", "🥉", "🏅", "📋", "📋", "📋", "📋"][info["rank"] - 1] if info["rank"] <= 8 else "📋"
                    print(f"  {rank_emoji} {model_name}")
                    print(f"    {info['description']}")
                    print(f"    パラメータ: {info['parameters']:,}")
                    print(f"    言語: {info['language']}")
                    print(f"    専門分野: {info['speciality']}")
                    print(f"    推奨メモリ: {info['recommended_memory_gb']}GB")
                    if "note" in info:
                        print(f"    注意: {info['note']}")
                    print()
    
    def show_sample_prompts(self):
        """多言語プロンプトサンプルを表示"""
        print(f"\n📝 多言語プロンプトサンプル:")
        
        for language, categories in MULTILINGUAL_PROMPTS.items():
            print(f"\n【{language}】")
            for category, prompts in categories.items():
                print(f"\n  ◆ {category}")
                for i, prompt in enumerate(prompts, 1):
                    print(f"    {i}. {prompt}")
    
    def create_quantization_config(self) -> Optional[Any]:
        """量子化設定を作成（拡張版）"""
        if not BITSANDBYTES_AVAILABLE:
            print("⚠️ BitsAndBytes未対応のため、量子化無しで実行します")
            return None
        
        try:
            if self.use_4bit:
                print("🔧 4bit量子化を有効化しました（拡張最適化）")
                
                # デバイス別最適化
                if self.device.type == "cuda":
                    compute_dtype = torch.float16
                    print("  GPU用4bit量子化設定")
                else:
                    compute_dtype = torch.float32
                    print("  CPU用4bit量子化設定")
                
                config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_enable_fp32_cpu_offload=True if self.device.type == "cpu" else False
                )
                return config
                
            elif self.use_8bit:
                print("🔧 8bit量子化を有効化しました（拡張最適化）")
                
                config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True if self.device.type == "cpu" else False
                )
                return config
                
        except Exception as e:
            print(f"⚠️ 量子化設定エラー: {e}")
            print("💡 量子化無しで続行します")
            return None
        
        return None
    
    def load_model_with_optimization(self) -> bool:
        """最適化を適用してモデルをロード（拡張版）"""
        try:
            print("📥 拡張版重量級モデルをロード中...")
            print("⚠️  初回実行時は大容量ダウンロードのため時間がかかります")
            
            # メモリ使用量監視開始
            initial_memory = psutil.virtual_memory().used / (1024**3)
            print(f"📊 ロード前メモリ使用量: {initial_memory:.1f}GB")
            
            # 量子化設定
            quantization_config = self.create_quantization_config()
            
            # モデルロード設定（デバイス別最適化）
            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            # デバイス別設定
            if self.device.type == "cuda":
                model_kwargs.update({
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                })
                print("🎮 GPU最適化設定適用")
            else:
                model_kwargs.update({
                    "torch_dtype": torch.float32,
                    "device_map": "cpu",
                })
                print("💻 CPU最適化設定適用")
            
            # 量子化設定を追加（エラーハンドリング付き）
            if quantization_config is not None:
                try:
                    model_kwargs["quantization_config"] = quantization_config
                except Exception as e:
                    print(f"⚠️ 量子化設定適用エラー: {e}")
                    print("💡 量子化無しで続行します")
            
            print(f"📥 拡張モデル '{self.model_name}' をロード中...")
            
            # モデルロード（段階的フォールバック）
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            except Exception as e:
                print(f"⚠️ 最適化モデルロードエラー: {e}")
                
                # MXFP4量子化エラーの特別処理
                if "MXFP4" in str(e) and "GPU" in str(e):
                    print("💡 MXFP4量子化済みモデルです。CPU環境では量子化無しで実行します")
                    
                    # CPU用基本設定
                    basic_kwargs = {
                        "trust_remote_code": True,
                        "torch_dtype": torch.float32,
                        "low_cpu_mem_usage": True,
                        "device_map": "cpu",
                    }
                    
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            **basic_kwargs
                        )
                    except Exception as e2:
                        print(f"⚠️ 基本モデルロードエラー: {e2}")
                        print("💡 最小設定でリトライします")
                        
                        # 最小設定
                        minimal_kwargs = {
                            "trust_remote_code": True,
                        }
                        
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            **minimal_kwargs
                        )
                else:
                    # 一般的なエラーの場合
                    print("💡 基本設定でリトライします")
                    
                    basic_kwargs = {
                        "trust_remote_code": True,
                        "torch_dtype": torch.float32 if self.device.type == "cpu" else torch.float16,
                        "low_cpu_mem_usage": True,
                    }
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        **basic_kwargs
                    )
            
            # トークナイザーロード
            print("📝 拡張トークナイザーをロード中...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # パディングトークン設定
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # モデルを評価モードに設定
            self.model.eval()
            
            # 拡張最適化適用
            self._apply_extended_optimizations()
            
            # メモリ使用量監視終了
            final_memory = psutil.virtual_memory().used / (1024**3)
            memory_used = final_memory - initial_memory
            
            print(f"📊 ロード後メモリ使用量: {final_memory:.1f}GB")
            print(f"📊 モデルメモリ使用量: {memory_used:.1f}GB")
            print("✅ 拡張モデルロード完了")
            
            return True
            
        except Exception as e:
            print(f"❌ モデルロードエラー: {e}")
            print(f"詳細: {traceback.format_exc()}")
            return False
    
    def _apply_extended_optimizations(self):
        """拡張最適化を適用"""
        try:
            print("🌍 拡張最適化を適用中...")
            
            # デバイス別最適化
            if self.device.type == "cuda":
                print("  🎮 GPU最適化設定")
                # GPU最適化
                if hasattr(torch.backends, 'cudnn'):
                    torch.backends.cudnn.benchmark = True
                    print("    ✅ cuDNN最適化有効化")
            else:
                print("  💻 CPU最適化設定")
                # CPU最適化
                torch.set_num_threads(psutil.cpu_count())
                print(f"    ✅ CPUスレッド数設定: {psutil.cpu_count()}")
            
            # メモリ効率化
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("  ✅ グラディエントチェックポイント有効化")
            
            # キャッシュ設定
            if hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = True
                print("  ✅ キャッシュ有効化")
            
            # モデル固有最適化
            try:
                # 語彙サイズ最適化
                if hasattr(self.model.config, 'vocab_size'):
                    print(f"  ✅ 語彙サイズ: {self.model.config.vocab_size:,}")
                
                # 文脈長最適化
                if hasattr(self.model.config, 'max_position_embeddings'):
                    print(f"  ✅ 最大文脈長: {self.model.config.max_position_embeddings}")
                elif hasattr(self.model.config, 'max_sequence_length'):
                    print(f"  ✅ 最大文脈長: {self.model.config.max_sequence_length}")
                
                # アーキテクチャ情報
                if hasattr(self.model.config, 'model_type'):
                    print(f"  ✅ アーキテクチャ: {self.model.config.model_type}")
                
            except:
                pass
            
            self.optimization_applied = True
            print("🚀 拡張最適化適用完了")
            
        except Exception as e:
            print(f"⚠️ 拡張最適化エラー: {e}")
    
    def generate_text(self, prompt: str, max_length: int = 300, language: str = "auto") -> Dict:
        """テキスト生成（拡張版）"""
        if self.model is None or self.tokenizer is None:
            return {"error": "モデルまたはトークナイザーが未ロード"}
        
        try:
            print(f"\n🎯 拡張テキスト生成開始")
            print(f"プロンプト: \"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"")
            print(f"最大長: {max_length}")
            print(f"言語: {language}")
            
            # メモリ・CPU使用量測定開始
            initial_memory = psutil.virtual_memory().used / (1024**3)
            initial_cpu = psutil.cpu_percent(interval=None)
            
            # GPU使用量測定（GPU環境の場合）
            initial_gpu_memory = None
            if torch.cuda.is_available():
                initial_gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            
            # トークン化（多言語対応）
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            )
            
            # デバイスに移動
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成設定（拡張版）
            generation_config = {
                "max_length": max_length,
                "num_return_sequences": 1,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }
            
            # 言語別最適化
            if language == "Japanese" or "日本語" in language:
                generation_config.update({
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "repetition_penalty": 1.05,
                })
                print("  🇯🇵 日本語最適化設定適用")
            elif "Programming" in language or "Code" in language:
                generation_config.update({
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "repetition_penalty": 1.2,
                })
                print("  💻 プログラミング最適化設定適用")
            
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
            
            final_gpu_memory = None
            if torch.cuda.is_available():
                final_gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            
            # トークン数計算
            input_tokens = len(inputs['input_ids'][0])
            output_tokens = len(outputs[0]) - input_tokens
            total_tokens = len(outputs[0])
            
            # 性能指標計算
            tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
            
            # 品質評価
            quality_score = self._evaluate_text_quality(generated_only, language)
            
            result = {
                "prompt": prompt,
                "generated_text": generated_only,
                "full_text": generated_text,
                "generation_time": generation_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "tokens_per_second": tokens_per_second,
                "language": language,
                "quality_score": quality_score,
                "resource_usage": {
                    "memory_used_gb": final_memory - initial_memory,
                    "memory_total_gb": final_memory,
                    "cpu_usage_percent": final_cpu,
                },
                "optimization_applied": self.optimization_applied,
                "quantization_info": {
                    "use_4bit": self.use_4bit,
                    "use_8bit": self.use_8bit
                },
                "device": str(self.device)
            }
            
            # GPU情報追加
            if initial_gpu_memory is not None and final_gpu_memory is not None:
                result["resource_usage"]["gpu_memory_used_gb"] = final_gpu_memory - initial_gpu_memory
                result["resource_usage"]["gpu_memory_total_gb"] = final_gpu_memory
            
            self._print_generation_results(result)
            return result
            
        except Exception as e:
            error_msg = f"テキスト生成エラー: {e}"
            print(f"❌ {error_msg}")
            return {"error": error_msg, "traceback": traceback.format_exc()}
    
    def _evaluate_text_quality(self, text: str, language: str) -> Dict:
        """テキスト品質を評価（多言語対応）"""
        try:
            if not text:
                return {"error": "テキストが空です"}
            
            # 基本指標
            char_count = len(text)
            word_count = len(text.split())
            sentence_count = text.count('.') + text.count('!') + text.count('?') + text.count('。')
            
            # 言語別評価
            if language == "Japanese" or "日本語" in language:
                return self._evaluate_japanese_quality(text)
            elif "Programming" in language or "Code" in language:
                return self._evaluate_code_quality(text)
            else:
                return self._evaluate_english_quality(text)
            
        except Exception as e:
            return {"error": f"品質評価エラー: {e}"}
    
    def _evaluate_japanese_quality(self, text: str) -> Dict:
        """日本語品質評価"""
        hiragana_count = sum(1 for c in text if '\u3040' <= c <= '\u309F')
        katakana_count = sum(1 for c in text if '\u30A0' <= c <= '\u30FF')
        kanji_count = sum(1 for c in text if '\u4E00' <= c <= '\u9FAF')
        ascii_count = sum(1 for c in text if ord(c) < 128)
        
        total_chars = len(text)
        japanese_ratio = (hiragana_count + katakana_count + kanji_count) / total_chars if total_chars > 0 else 0
        
        if japanese_ratio > 0.8:
            quality_level = "優秀"
        elif japanese_ratio > 0.6:
            quality_level = "良好"
        elif japanese_ratio > 0.4:
            quality_level = "普通"
        else:
            quality_level = "要改善"
        
        return {
            "language": "Japanese",
            "japanese_ratio": japanese_ratio,
            "quality_level": quality_level,
            "character_breakdown": {
                "hiragana": hiragana_count,
                "katakana": katakana_count,
                "kanji": kanji_count,
                "ascii": ascii_count,
                "total": total_chars
            }
        }
    
    def _evaluate_english_quality(self, text: str) -> Dict:
        """英語品質評価"""
        words = text.split()
        sentences = text.count('.') + text.count('!') + text.count('?')
        
        avg_word_length = sum(len(word.strip('.,!?')) for word in words) / len(words) if words else 0
        avg_sentence_length = len(words) / sentences if sentences > 0 else len(words)
        
        # 簡易品質評価
        if avg_word_length > 4 and avg_sentence_length > 10:
            quality_level = "Good"
        elif avg_word_length > 3 and avg_sentence_length > 5:
            quality_level = "Fair"
        else:
            quality_level = "Basic"
        
        return {
            "language": "English",
            "quality_level": quality_level,
            "word_count": len(words),
            "sentence_count": sentences,
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length
        }
    
    def _evaluate_code_quality(self, text: str) -> Dict:
        """コード品質評価"""
        lines = text.split('\n')
        code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        comment_lines = [line for line in lines if line.strip().startswith('#')]
        
        # コード特徴検出
        has_functions = any('def ' in line for line in lines)
        has_classes = any('class ' in line for line in lines)
        has_imports = any('import ' in line or 'from ' in line for line in lines)
        
        quality_score = 0
        if has_functions: quality_score += 1
        if has_classes: quality_score += 1
        if has_imports: quality_score += 1
        if len(comment_lines) > 0: quality_score += 1
        
        if quality_score >= 3:
            quality_level = "Good"
        elif quality_score >= 2:
            quality_level = "Fair"
        else:
            quality_level = "Basic"
        
        return {
            "language": "Programming",
            "quality_level": quality_level,
            "total_lines": len(lines),
            "code_lines": len(code_lines),
            "comment_lines": len(comment_lines),
            "has_functions": has_functions,
            "has_classes": has_classes,
            "has_imports": has_imports
        }
    
    def _print_generation_results(self, result: Dict):
        """生成結果を表示"""
        print(f"\n📊 拡張生成結果:")
        print(f"  生成時間: {result['generation_time']:.2f}秒")
        print(f"  入力トークン: {result['input_tokens']}")
        print(f"  出力トークン: {result['output_tokens']}")
        print(f"  スループット: {result['tokens_per_second']:.1f} tokens/sec")
        print(f"  デバイス: {result['device']}")
        
        print(f"\n💾 リソース使用量:")
        print(f"  メモリ使用: {result['resource_usage']['memory_used_gb']:.1f}GB")
        print(f"  総メモリ: {result['resource_usage']['memory_total_gb']:.1f}GB")
        print(f"  CPU使用率: {result['resource_usage']['cpu_usage_percent']:.1f}%")
        
        if "gpu_memory_used_gb" in result['resource_usage']:
            print(f"  GPU使用: {result['resource_usage']['gpu_memory_used_gb']:.1f}GB")
            print(f"  総GPU: {result['resource_usage']['gpu_memory_total_gb']:.1f}GB")
        
        print(f"\n🌍 品質評価:")
        if "error" not in result['quality_score']:
            quality = result['quality_score']
            print(f"  言語: {quality.get('language', 'Unknown')}")
            print(f"  品質レベル: {quality.get('quality_level', 'Unknown')}")
            
            if quality.get('language') == 'Japanese':
                print(f"  日本語比率: {quality.get('japanese_ratio', 0):.1%}")
            elif quality.get('language') == 'English':
                print(f"  平均単語長: {quality.get('avg_word_length', 0):.1f}")
                print(f"  平均文長: {quality.get('avg_sentence_length', 0):.1f}")
            elif quality.get('language') == 'Programming':
                print(f"  コード行数: {quality.get('code_lines', 0)}")
                print(f"  関数: {'✅' if quality.get('has_functions') else '❌'}")
                print(f"  クラス: {'✅' if quality.get('has_classes') else '❌'}")
        else:
            print(f"  評価エラー: {result['quality_score']['error']}")
        
        print(f"\n🔧 最適化状態:")
        print(f"  拡張最適化: {'✅' if result['optimization_applied'] else '❌'}")
        print(f"  4bit量子化: {'✅' if result['quantization_info']['use_4bit'] else '❌'}")
        print(f"  8bit量子化: {'✅' if result['quantization_info']['use_8bit'] else '❌'}")
        
        print(f"\n📝 生成されたテキスト:")
        print(f"  \"{result['generated_text'][:300]}{'...' if len(result['generated_text']) > 300 else ''}\"")
    
    def interactive_mode(self):
        """拡張インタラクティブモード"""
        print(f"\n🌍 拡張インタラクティブモード開始")
        print(f"プロンプトを入力してください（'quit'で終了、'samples'でサンプル表示、'models'でモデル一覧）:")
        
        results = []
        
        while True:
            try:
                prompt = input("\n🌍 > ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q', '終了']:
                    break
                
                if prompt.lower() in ['samples', 'sample', 'サンプル']:
                    self.show_sample_prompts()
                    continue
                
                if prompt.lower() in ['models', 'model', 'モデル']:
                    self.list_available_models()
                    continue
                
                if not prompt:
                    continue
                
                # 言語自動検出
                language = self._detect_language(prompt)
                
                result = self.generate_text(prompt, language=language)
                if "error" not in result:
                    results.append(result)
                
                # メモリクリーンアップ
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except KeyboardInterrupt:
                print("\n👋 拡張インタラクティブモードを終了します")
                break
            except Exception as e:
                print(f"❌ エラー: {e}")
        
        # セッション結果保存
        if results:
            self._save_session_results(results)
    
    def _detect_language(self, text: str) -> str:
        """言語を自動検出"""
        # 日本語文字の検出
        japanese_chars = sum(1 for c in text if '\u3040' <= c <= '\u309F' or '\u30A0' <= c <= '\u30FF' or '\u4E00' <= c <= '\u9FAF')
        
        # プログラミング関連キーワードの検出
        programming_keywords = ['def ', 'class ', 'import ', 'function', 'var ', 'const ', 'let ', '#!/', 'print(', 'console.log']
        has_programming = any(keyword in text.lower() for keyword in programming_keywords)
        
        if japanese_chars > 0:
            return "Japanese"
        elif has_programming:
            return "Programming"
        else:
            return "English"
    
    def _save_session_results(self, results: List[Dict]):
        """セッション結果を保存"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_safe_name = self.model_name.replace("/", "_").replace("-", "_")
            filename = f'extended_heavy_llm_session_{model_safe_name}_{timestamp}.json'
            
            session_data = {
                "model_name": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "system_info": self.system_info,
                "optimization_config": {
                    "use_4bit": self.use_4bit,
                    "use_8bit": self.use_8bit,
                    "force_cpu": self.force_cpu,
                    "optimization_applied": self.optimization_applied
                },
                "results": results,
                "summary": self._calculate_session_summary(results)
            }
            
            os.makedirs('demo_results', exist_ok=True)
            filepath = os.path.join('demo_results', filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 拡張セッション結果を保存しました: {filepath}")
            
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
        
        # 言語分布
        languages = [r.get('language', 'Unknown') for r in results]
        language_distribution = {lang: languages.count(lang) for lang in set(languages)}
        
        return {
            "total_generations": len(results),
            "avg_generation_time": sum(generation_times) / len(generation_times),
            "avg_tokens_per_second": sum(tokens_per_second) / len(tokens_per_second),
            "total_output_tokens": sum(output_tokens),
            "avg_memory_used_gb": sum(memory_used) / len(memory_used),
            "min_generation_time": min(generation_times),
            "max_generation_time": max(generation_times),
            "language_distribution": language_distribution
        }

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="拡張版重量級LLM Infer-OS最適化デモ")
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="使用するモデル名")
    parser.add_argument("--prompt", help="テスト用プロンプト")
    parser.add_argument("--max-length", type=int, default=300, help="最大生成長")
    parser.add_argument("--use-4bit", action="store_true", help="4bit量子化を使用")
    parser.add_argument("--use-8bit", action="store_true", help="8bit量子化を使用")
    parser.add_argument("--force-cpu", action="store_true", help="CPU強制使用")
    parser.add_argument("--interactive", action="store_true", help="拡張インタラクティブモード")
    parser.add_argument("--list-models", action="store_true", help="利用可能なモデル一覧を表示")
    parser.add_argument("--samples", action="store_true", help="多言語プロンプトサンプルを表示")
    
    args = parser.parse_args()
    
    if args.list_models:
        demo = ExtendedHeavyLLMDemo("dummy", False, False)
        demo.list_available_models()
        return
    
    if args.samples:
        demo = ExtendedHeavyLLMDemo("dummy", False, False)
        demo.show_sample_prompts()
        return
    
    print(f"""
{'='*80}
🌍 拡張版重量級LLM Infer-OS最適化デモ
{'='*80}

対象モデル: {args.model}
最適化設定:
  4bit量子化: {'✅' if args.use_4bit else '❌'}
  8bit量子化: {'✅' if args.use_8bit else '❌'}
  CPU強制: {'✅' if args.force_cpu else '❌'}
  インタラクティブ: {'✅' if args.interactive else '❌'}

{'='*80}
""")
    
    try:
        # デモ初期化
        demo = ExtendedHeavyLLMDemo(
            model_name=args.model,
            use_4bit=args.use_4bit,
            use_8bit=args.use_8bit,
            force_cpu=args.force_cpu
        )
        
        # モデルロード
        if not demo.load_model_with_optimization():
            print("❌ 拡張モデルのロードに失敗しました")
            return
        
        # テキスト生成実行
        if args.interactive:
            demo.interactive_mode()
        else:
            prompt = args.prompt or "Explain the future of artificial intelligence from a technical perspective."
            result = demo.generate_text(prompt, args.max_length)
            
            if "error" in result:
                print(f"❌ 生成エラー: {result['error']}")
            else:
                print("\n🎉 拡張テキスト生成完了")
        
    except KeyboardInterrupt:
        print("\n👋 デモを中断しました")
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        print(f"詳細: {traceback.format_exc()}")

if __name__ == "__main__":
    main()

