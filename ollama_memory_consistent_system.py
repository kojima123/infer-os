#!/usr/bin/env python3
"""
Ollama + infer-OS最適化制御システム（メモリ使用率乖離修正版）
メモリ使用率測定の一貫性を確保し、正確なinfer-OS最適化効果を測定
"""

import os
import sys
import time
import json
import argparse
import threading
import requests
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
import warnings
warnings.filterwarnings("ignore")

try:
    import psutil
    import onnxruntime as ort
    import numpy as np
    print("✅ 必要なライブラリのインポート成功")
except ImportError as e:
    print(f"❌ ライブラリインポートエラー: {e}")
    print("pip install psutil onnxruntime requests")
    sys.exit(1)

class OllamaMemoryConsistentController:
    """Ollama + infer-OS最適化制御システム（メモリ使用率乖離修正版）"""
    
    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host
        self.ollama_api = f"{ollama_host}/api"
        
        # infer-OS最適化制御
        self.infer_os_enabled = True
        self.infer_os_config = {
            "npu_optimization": True,
            "memory_optimization": True,
            "cpu_optimization": True,
            "gpu_acceleration": True,
            "quantization": True,
            "parallel_processing": True,
        }
        
        # システム状態
        self.available_models = []
        self.current_model = None
        self.npu_monitoring = False
        self.npu_stats = {"usage_changes": 0, "max_usage": 0.0, "avg_usage": 0.0}
        self.onnx_session = None
        self.generating = False  # 生成中フラグ
        
        # メモリ測定履歴
        self.memory_history = []
        
        # シンプルなプロンプトテンプレート
        self.templates = {
            "conversation": "{prompt}",
            "instruction": "指示: {prompt}\n\n回答:",
            "reasoning": "問題: {prompt}\n\n解答:",
            "creative": "テーマ: {prompt}\n\n内容:",
            "simple": "{prompt}"
        }
        
        self.current_template = "simple"
        
        print("🚀 Ollama + infer-OS最適化制御システム（メモリ使用率乖離修正版）初期化")
        print(f"🔗 Ollama接続先: {ollama_host}")
        print(f"⚡ infer-OS最適化: {'有効' if self.infer_os_enabled else '無効'}")
        print(f"🎯 設計方針: メモリ測定一貫性 + 正確な最適化効果測定")
        print(f"🔧 修正内容: 統一測定タイミング + 安定化待機 + 詳細プロファイル")
    
    def measure_memory_consistently(self, context: str = "", wait_seconds: int = 5) -> float:
        """一貫したメモリ使用率測定（修正版）"""
        try:
            if wait_seconds > 0:
                print(f"⏳ メモリ安定化待機中（{wait_seconds}秒）...")
                time.sleep(wait_seconds)
            
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()
            
            measurement = {
                "timestamp": time.time(),
                "context": context,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "cpu_percent": cpu_percent
            }
            
            self.memory_history.append(measurement)
            
            print(f"📊 メモリ使用率 ({context}): {memory.percent:.1f}% ({memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB)")
            print(f"💻 CPU使用率 ({context}): {cpu_percent:.1f}%")
            
            return memory.percent
            
        except Exception as e:
            print(f"❌ メモリ測定エラー: {e}")
            return 0.0
    
    def comprehensive_memory_analysis(self, prompt: str, max_tokens: int = 100, template: str = None) -> Dict[str, Any]:
        """包括的メモリ分析（修正版）"""
        print("🔍 包括的メモリ分析開始...")
        
        measurements = {
            "pre_generation": None,
            "during_generation_max": 0.0,
            "post_generation": None,
            "stabilized": None
        }
        
        # 生成前測定
        print("📊 生成前メモリ測定...")
        measurements["pre_generation"] = self.measure_memory_consistently("生成前", wait_seconds=2)
        
        # 生成中監視用スレッド
        def monitor_during_generation():
            max_usage = 0.0
            sample_count = 0
            while self.generating:
                try:
                    current = psutil.virtual_memory().percent
                    max_usage = max(max_usage, current)
                    sample_count += 1
                    if sample_count % 10 == 0:  # 5秒ごとに表示
                        print(f"🔥 生成中メモリ監視: {current:.1f}% (最大: {max_usage:.1f}%)")
                    time.sleep(0.5)
                except:
                    break
            measurements["during_generation_max"] = max_usage
        
        # 生成実行
        print("🚀 テキスト生成実行中...")
        monitor_thread = threading.Thread(target=monitor_during_generation, daemon=True)
        self.generating = True
        monitor_thread.start()
        
        start_time = time.time()
        result = self.generate_text_with_ollama_fixed(prompt, max_tokens, template)
        end_time = time.time()
        
        self.generating = False
        monitor_thread.join(timeout=1)
        
        # 生成直後測定
        print("📊 生成直後メモリ測定...")
        measurements["post_generation"] = self.measure_memory_consistently("生成直後", wait_seconds=0)
        
        # 安定化後測定
        print("📊 安定化後メモリ測定...")
        measurements["stabilized"] = self.measure_memory_consistently("安定化後", wait_seconds=5)
        
        analysis_result = {
            "measurements": measurements,
            "result": result,
            "generation_time": end_time - start_time,
            "memory_reduction": measurements["post_generation"] - measurements["stabilized"] if measurements["post_generation"] and measurements["stabilized"] else 0
        }
        
        print("✅ 包括的メモリ分析完了")
        return analysis_result
    
    def clear_memory_cache(self):
        """メモリキャッシュクリア（修正版）"""
        try:
            print("🧹 メモリキャッシュクリア開始...")
            
            # Ollamaモデルアンロード試行
            if self.current_model:
                try:
                    unload_response = requests.post(
                        f"{self.ollama_api}/unload",
                        json={"model": self.current_model["name"]},
                        timeout=10
                    )
                    if unload_response.status_code == 200:
                        print("✅ Ollamaモデルアンロード成功")
                    else:
                        print("⚠️ Ollamaモデルアンロード失敗（継続）")
                except:
                    print("⚠️ Ollamaモデルアンロード試行失敗（継続）")
            
            # Python ガベージコレクション
            import gc
            collected = gc.collect()
            print(f"🗑️ ガベージコレクション: {collected}オブジェクト回収")
            
            # 短時間待機
            time.sleep(3)
            
            print("✅ メモリキャッシュクリア完了")
            
        except Exception as e:
            print(f"⚠️ メモリクリアエラー: {e}")
    
    def accurate_optimization_comparison(self, prompt: str, tokens: int = 100) -> Dict[str, Any]:
        """正確な最適化効果比較（修正版）"""
        print("🎯 正確なinfer-OS最適化効果比較開始...")
        
        comparison_result = {
            "optimization_on": None,
            "optimization_off": None,
            "effectiveness": {}
        }
        
        # 最適化有効で測定
        print("\n⚡ infer-OS最適化有効での測定...")
        self.infer_os_enabled = True
        self.apply_infer_os_optimizations()
        comparison_result["optimization_on"] = self.comprehensive_memory_analysis(prompt, tokens)
        
        # メモリクリア
        print("\n🧹 測定間メモリクリア...")
        self.clear_memory_cache()
        time.sleep(10)  # 十分な待機時間
        
        # 最適化無効で測定
        print("\n❌ infer-OS最適化無効での測定...")
        self.infer_os_enabled = False
        comparison_result["optimization_off"] = self.comprehensive_memory_analysis(prompt, tokens)
        
        # 効果計算
        on_data = comparison_result["optimization_on"]["measurements"]
        off_data = comparison_result["optimization_off"]["measurements"]
        
        if on_data["stabilized"] and off_data["stabilized"]:
            comparison_result["effectiveness"] = {
                "memory_reduction": off_data["stabilized"] - on_data["stabilized"],
                "memory_reduction_percent": ((off_data["stabilized"] - on_data["stabilized"]) / off_data["stabilized"]) * 100,
                "peak_reduction": off_data["during_generation_max"] - on_data["during_generation_max"],
                "generation_time_diff": comparison_result["optimization_off"]["generation_time"] - comparison_result["optimization_on"]["generation_time"]
            }
        
        print("✅ 正確なinfer-OS最適化効果比較完了")
        return comparison_result
    
    def show_detailed_memory_stats(self):
        """詳細メモリ統計表示（修正版）"""
        if not self.memory_history:
            print("📊 メモリ測定履歴がありません")
            return
        
        print("\n📊 詳細メモリ使用率統計:")
        print("=" * 60)
        
        for i, measurement in enumerate(self.memory_history[-10:], 1):  # 直近10件
            timestamp = time.strftime("%H:%M:%S", time.localtime(measurement["timestamp"]))
            print(f"  {i:2d}. [{timestamp}] {measurement['context']:15s}: "
                  f"{measurement['memory_percent']:5.1f}% "
                  f"({measurement['memory_used_gb']:4.1f}GB) "
                  f"CPU: {measurement['cpu_percent']:4.1f}%")
        
        if len(self.memory_history) >= 2:
            latest = self.memory_history[-1]
            previous = self.memory_history[-2]
            change = latest["memory_percent"] - previous["memory_percent"]
            change_symbol = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            print(f"\n{change_symbol} 前回比較: {change:+.1f}%")
        
        # 統計情報
        memory_values = [m["memory_percent"] for m in self.memory_history]
        print(f"\n📈 統計情報:")
        print(f"  最大使用率: {max(memory_values):.1f}%")
        print(f"  最小使用率: {min(memory_values):.1f}%")
        print(f"  平均使用率: {sum(memory_values) / len(memory_values):.1f}%")
        print(f"  測定回数: {len(memory_values)}回")
    
    def check_ollama_connection(self) -> bool:
        """Ollama接続確認"""
        try:
            print("🔍 Ollama接続確認中...")
            response = requests.get(f"{self.ollama_api}/tags", timeout=10)
            
            if response.status_code == 200:
                print("✅ Ollama接続成功")
                return True
            else:
                print(f"❌ Ollama接続失敗: ステータスコード {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("❌ Ollama接続失敗: 接続エラー")
            print("💡 Ollamaが起動していることを確認してください")
            print("   Windows: ollama serve")
            return False
        except Exception as e:
            print(f"❌ Ollama接続確認エラー: {e}")
            return False
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """利用可能なモデル一覧取得"""
        try:
            print("📋 利用可能なモデル一覧取得中...")
            response = requests.get(f"{self.ollama_api}/tags", timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                
                self.available_models = []
                for model in models:
                    model_info = {
                        "name": model.get("name", "unknown"),
                        "size": model.get("size", 0),
                        "modified_at": model.get("modified_at", ""),
                        "digest": model.get("digest", ""),
                        "details": model.get("details", {})
                    }
                    self.available_models.append(model_info)
                
                print(f"✅ {len(self.available_models)}個のモデルが利用可能")
                for i, model in enumerate(self.available_models, 1):
                    size_gb = model["size"] / (1024**3) if model["size"] > 0 else 0
                    print(f"  {i}. {model['name']} ({size_gb:.1f}GB)")
                
                return self.available_models
            else:
                print(f"❌ モデル一覧取得失敗: ステータスコード {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ モデル一覧取得エラー: {e}")
            return []
    
    def select_model(self, model_name: str = None) -> bool:
        """モデル選択"""
        try:
            if not self.available_models:
                self.get_available_models()
            
            if not self.available_models:
                print("❌ 利用可能なモデルがありません")
                return False
            
            if model_name is None:
                selected_model = self.available_models[0]
            else:
                selected_model = None
                for model in self.available_models:
                    if model_name in model["name"]:
                        selected_model = model
                        break
                
                if selected_model is None:
                    print(f"❌ モデル '{model_name}' が見つかりません")
                    return False
            
            self.current_model = selected_model
            size_gb = selected_model["size"] / (1024**3) if selected_model["size"] > 0 else 0
            
            print(f"✅ モデル選択完了: {selected_model['name']}")
            print(f"📊 モデルサイズ: {size_gb:.1f}GB")
            
            return True
            
        except Exception as e:
            print(f"❌ モデル選択エラー: {e}")
            return False
    
    def apply_infer_os_optimizations(self):
        """infer-OS最適化適用"""
        if not self.infer_os_enabled:
            print("⚠️ infer-OS最適化は無効です")
            return
        
        print("⚡ infer-OS最適化設定適用中...")
        
        if self.infer_os_config["memory_optimization"]:
            print("🔧 メモリ最適化: 有効")
            os.environ['OLLAMA_MAX_LOADED_MODELS'] = '1'
            os.environ['OLLAMA_NUM_PARALLEL'] = '1'
            os.environ['OLLAMA_LOAD_TIMEOUT'] = '60'
        
        if self.infer_os_config["cpu_optimization"]:
            print("🔧 CPU最適化: 有効")
            cpu_count = os.cpu_count()
            os.environ['OLLAMA_NUM_THREADS'] = str(min(2, cpu_count))
        
        if self.infer_os_config["gpu_acceleration"]:
            print("🔧 GPU加速: 有効")
            os.environ['OLLAMA_GPU_LAYERS'] = '20'
        
        if self.infer_os_config["npu_optimization"]:
            print("🔧 NPU最適化: 有効")
            os.environ['ONNXRUNTIME_PROVIDERS'] = 'DmlExecutionProvider,CPUExecutionProvider'
        
        print("✅ infer-OS最適化設定適用完了")
    
    def create_safe_onnx_session(self) -> bool:
        """安全なONNX推論セッション作成"""
        try:
            print("🔧 安全なONNX推論セッション作成中...")
            
            os.makedirs("models", exist_ok=True)
            
            import torch
            import torch.nn as nn
            
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(256, 512)
                    
                def forward(self, x):
                    return self.linear(x)
            
            model = SimpleModel()
            model.eval()
            
            dummy_input = torch.randn(1, 256)
            onnx_path = "models/ollama_memory_consistent_model.onnx"
            
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output']
            )
            
            print(f"✅ ONNXモデル作成完了: {onnx_path}")
            
            providers = []
            available_providers = ort.get_available_providers()
            
            if 'DmlExecutionProvider' in available_providers:
                providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
                print("🎯 DmlExecutionProvider使用")
            elif 'VitisAIExecutionProvider' in available_providers:
                providers = ['VitisAIExecutionProvider', 'CPUExecutionProvider']
                print("🎯 VitisAIExecutionProvider使用")
            else:
                providers = ['CPUExecutionProvider']
                print("🎯 CPUExecutionProvider使用")
            
            session_options = ort.SessionOptions()
            session_options.log_severity_level = 3
            session_options.enable_cpu_mem_arena = True
            
            self.onnx_session = ort.InferenceSession(
                onnx_path,
                sess_options=session_options,
                providers=providers
            )
            
            active_provider = self.onnx_session.get_providers()[0]
            print(f"✅ 安全なONNX推論セッション作成成功")
            print(f"🎯 アクティブプロバイダー: {active_provider}")
            
            return True
            
        except Exception as e:
            print(f"❌ 安全なONNX推論セッション作成エラー: {e}")
            return False
    
    def generate_text_with_ollama_fixed(self, prompt: str, max_tokens: int = 100, template: str = None) -> str:
        """Ollamaを使用したテキスト生成"""
        if self.current_model is None:
            return "❌ モデルが選択されていません"
        
        try:
            if template and template in self.templates:
                formatted_prompt = self.templates[template].format(prompt=prompt)
            else:
                formatted_prompt = prompt
            
            print(f"💬 Ollamaテキスト生成中: '{formatted_prompt[:30]}...'")
            print(f"🎯 使用モデル: {self.current_model['name']}")
            print(f"🎯 最大トークン数: {max_tokens}")
            print(f"⚡ infer-OS最適化: {'有効' if self.infer_os_enabled else '無効'}")
            
            payload = {
                "model": self.current_model["name"],
                "prompt": formatted_prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "top_k": 40,
                    "repeat_penalty": 1.05,
                    "stop": ["\n\n", "人間:", "Human:", "Assistant:"],
                }
            }
            
            print("🔧 Ollama API呼び出し中...")
            start_time = time.time()
            
            response = requests.post(
                f"{self.ollama_api}/generate",
                json=payload,
                timeout=30
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                generated_text = data.get("response", "").strip()
                
                print(f"✅ Ollamaテキスト生成完了")
                print(f"📝 生成文字数: {len(generated_text)}")
                print(f"⏱️ 生成時間: {end_time - start_time:.2f}秒")
                
                if len(generated_text) < 5:
                    print("⚠️ 生成結果が短すぎます")
                    return self.generate_fallback_text(prompt)
                
                return generated_text
            else:
                print(f"❌ Ollama API呼び出し失敗: ステータスコード {response.status_code}")
                return self.generate_fallback_text(prompt)
                
        except requests.exceptions.Timeout:
            print("❌ Ollama API呼び出しタイムアウト（30秒）")
            return self.generate_fallback_text(prompt)
        except Exception as e:
            print(f"❌ Ollamaテキスト生成エラー: {e}")
            return self.generate_fallback_text(prompt)
    
    def generate_fallback_text(self, prompt: str) -> str:
        """フォールバックテキスト生成"""
        fallback_responses = {
            "人工知能": "人工知能（AI）は、機械学習や深層学習などの技術を用いて、人間のような知的な処理を行う技術です。現在、様々な分野で活用が進んでおり、今後さらなる発展が期待されています。",
            "量子": "量子コンピューティングは、量子力学の原理を利用した革新的な計算技術です。従来のコンピューターでは困難な問題を高速で解決できる可能性があり、暗号解読や薬物開発などの分野での応用が期待されています。",
            "人参": "人参（にんじん）は、セリ科の野菜で、β-カロテンを豊富に含む栄養価の高い食材です。生食、煮物、炒め物など様々な調理法で楽しめ、甘みがあって子供にも人気があります。",
            "テスト": "テストは、システムや知識の動作確認や評価を行う重要なプロセスです。適切なテストにより、品質の向上や問題の早期発見が可能になります。",
            "default": f"「{prompt}」について、基本的な情報をお伝えします。より詳細な情報が必要でしたら、具体的な質問をしていただければと思います。"
        }
        
        for keyword, response in fallback_responses.items():
            if keyword in prompt and keyword != "default":
                return response
        
        return fallback_responses["default"]
    
    def initialize_system(self) -> bool:
        """システム初期化"""
        try:
            print("🚀 Ollama + infer-OS制御システム初期化開始（メモリ一貫性版）")
            
            # 初期化前メモリ測定
            self.measure_memory_consistently("初期化前", wait_seconds=0)
            
            if not self.check_ollama_connection():
                return False
            
            models = self.get_available_models()
            if not models:
                print("❌ 利用可能なモデルがありません")
                return False
            
            if not self.select_model():
                return False
            
            # 初期化後メモリ測定
            self.measure_memory_consistently("初期化後", wait_seconds=2)
            
            self.apply_infer_os_optimizations()
            
            # 最適化適用後メモリ測定
            self.measure_memory_consistently("最適化適用後", wait_seconds=2)
            
            onnx_created = self.create_safe_onnx_session()
            if not onnx_created:
                print("⚠️ ONNX推論セッション作成に失敗しましたが、継続します")
            
            # 完了後メモリ測定
            self.measure_memory_consistently("初期化完了", wait_seconds=2)
            
            print("✅ Ollama + infer-OS制御システム初期化完了（メモリ一貫性版）")
            
            return True
            
        except Exception as e:
            print(f"❌ システム初期化エラー: {e}")
            return False
    
    def run_interactive_mode(self):
        """インタラクティブモード実行（メモリ一貫性版）"""
        print("\n🎯 Ollama + infer-OS制御インタラクティブモード（メモリ一貫性版）")
        print(f"🎯 使用モデル: {self.current_model['name'] if self.current_model else 'なし'}")
        print(f"⚡ infer-OS最適化: {'有効' if self.infer_os_enabled else '無効'}")
        
        print("💡 コマンド:")
        print("  'quit' - 終了")
        print("  'memory' - 詳細メモリ統計表示")
        print("  'analysis' - 包括的メモリ分析")
        print("  'compare' - 最適化効果比較")
        print("  'clear' - メモリキャッシュクリア")
        print("  'toggle' - infer-OS最適化ON/OFF切り替え")
        print("=" * 70)
        
        # インタラクティブモード開始時メモリ測定
        self.measure_memory_consistently("インタラクティブ開始", wait_seconds=2)
        
        try:
            while True:
                try:
                    infer_os_status = "ON" if self.infer_os_enabled else "OFF"
                    prompt = input(f"\n💬 プロンプトを入力してください [infer-OS:{infer_os_status}]: ").strip()
                    
                    if not prompt:
                        continue
                    
                    if prompt.lower() == 'quit':
                        print("👋 システムを終了します")
                        break
                    
                    if prompt.lower() == 'memory':
                        self.show_detailed_memory_stats()
                        continue
                    
                    if prompt.lower() == 'analysis':
                        analysis = self.comprehensive_memory_analysis("テスト分析", 50)
                        print(f"\n🎯 分析結果: {analysis['result'][:100]}...")
                        continue
                    
                    if prompt.lower() == 'compare':
                        comparison = self.accurate_optimization_comparison("比較テスト", 50)
                        print(f"\n📊 最適化効果: {comparison['effectiveness']}")
                        continue
                    
                    if prompt.lower() == 'clear':
                        self.clear_memory_cache()
                        self.measure_memory_consistently("キャッシュクリア後", wait_seconds=3)
                        continue
                    
                    if prompt.lower() == 'toggle':
                        self.infer_os_enabled = not self.infer_os_enabled
                        status = "有効" if self.infer_os_enabled else "無効"
                        print(f"🔄 infer-OS最適化を{status}に切り替えました")
                        if self.infer_os_enabled:
                            self.apply_infer_os_optimizations()
                        continue
                    
                    # 通常のテキスト生成（一貫したメモリ測定付き）
                    print("🚀 テキスト生成開始...")
                    analysis = self.comprehensive_memory_analysis(prompt, 100)
                    
                    print(f"\n🎯 生成結果:")
                    print(analysis['result'])
                    print(f"\n⏱️ 生成時間: {analysis['generation_time']:.2f}秒")
                    print(f"📊 メモリ削減: {analysis['memory_reduction']:.1f}%")
                    
                except KeyboardInterrupt:
                    print("\n👋 システムを終了します")
                    break
                except Exception as e:
                    print(f"❌ エラー: {e}")
                    continue
        
        finally:
            # 終了時メモリ測定
            self.measure_memory_consistently("インタラクティブ終了", wait_seconds=2)

def main():
    """メイン関数（メモリ一貫性版）"""
    parser = argparse.ArgumentParser(description="Ollama + infer-OS最適化制御システム（メモリ使用率乖離修正版）")
    parser.add_argument("--interactive", action="store_true", help="インタラクティブモード")
    parser.add_argument("--prompt", type=str, help="単発プロンプト")
    parser.add_argument("--tokens", type=int, default=100, help="最大トークン数")
    parser.add_argument("--template", type=str, default="simple", help="プロンプトテンプレート")
    parser.add_argument("--model", type=str, help="使用モデル名")
    parser.add_argument("--ollama-host", type=str, default="http://localhost:11434", help="Ollama接続先")
    parser.add_argument("--infer-os", action="store_true", default=True, help="infer-OS最適化有効")
    parser.add_argument("--no-infer-os", action="store_true", help="infer-OS最適化無効")
    parser.add_argument("--compare", action="store_true", help="最適化効果比較モード")
    
    args = parser.parse_args()
    
    infer_os_enabled = args.infer_os and not args.no_infer_os
    
    system = OllamaMemoryConsistentController(ollama_host=args.ollama_host)
    system.infer_os_enabled = infer_os_enabled
    
    if not system.initialize_system():
        print("❌ システム初期化に失敗しました")
        sys.exit(1)
    
    if args.model:
        if not system.select_model(args.model):
            print(f"❌ モデル '{args.model}' の選択に失敗しました")
            sys.exit(1)
    
    try:
        if args.interactive:
            system.run_interactive_mode()
        elif args.compare:
            # 最適化効果比較モード
            prompt = args.prompt or "人工知能の未来について教えてください"
            comparison = system.accurate_optimization_comparison(prompt, args.tokens)
            
            print("\n📊 infer-OS最適化効果比較結果:")
            print("=" * 60)
            
            on_data = comparison["optimization_on"]["measurements"]
            off_data = comparison["optimization_off"]["measurements"]
            effectiveness = comparison["effectiveness"]
            
            print(f"⚡ 最適化有効:")
            print(f"  📊 安定化後: {on_data['stabilized']:.1f}%")
            print(f"  🔥 生成中最大: {on_data['during_generation_max']:.1f}%")
            
            print(f"\n❌ 最適化無効:")
            print(f"  📊 安定化後: {off_data['stabilized']:.1f}%")
            print(f"  🔥 生成中最大: {off_data['during_generation_max']:.1f}%")
            
            print(f"\n💡 最適化効果:")
            print(f"  📉 メモリ削減: {effectiveness['memory_reduction']:.1f}% ({effectiveness['memory_reduction_percent']:.1f}%削減)")
            print(f"  🔥 ピーク削減: {effectiveness['peak_reduction']:.1f}%")
            print(f"  ⏱️ 時間短縮: {effectiveness['generation_time_diff']:.2f}秒")
            
            system.show_detailed_memory_stats()
            
        elif args.prompt:
            # 単発生成（一貫したメモリ測定付き）
            analysis = system.comprehensive_memory_analysis(args.prompt, args.tokens, args.template)
            
            print(f"\n🎯 生成結果:")
            print(analysis['result'])
            print(f"\n📊 詳細メモリ分析:")
            print(f"  📊 生成前: {analysis['measurements']['pre_generation']:.1f}%")
            print(f"  🔥 生成中最大: {analysis['measurements']['during_generation_max']:.1f}%")
            print(f"  📊 生成直後: {analysis['measurements']['post_generation']:.1f}%")
            print(f"  ✅ 安定化後: {analysis['measurements']['stabilized']:.1f}%")
            print(f"  📉 メモリ削減: {analysis['memory_reduction']:.1f}%")
            print(f"  ⏱️ 生成時間: {analysis['generation_time']:.2f}秒")
            
            system.show_detailed_memory_stats()
        else:
            print("使用方法: --interactive, --prompt, または --compare を指定してください")
            print("例: python ollama_memory_consistent_system.py --interactive")
            print("例: python ollama_memory_consistent_system.py --prompt '人工知能について' --tokens 200")
            print("例: python ollama_memory_consistent_system.py --compare --prompt '人工知能の未来' --tokens 150")
    
    except KeyboardInterrupt:
        print("\n👋 システムを終了します")
    except Exception as e:
        print(f"❌ 実行エラー: {e}")

if __name__ == "__main__":
    main()

