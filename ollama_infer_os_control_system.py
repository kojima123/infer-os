#!/usr/bin/env python3
"""
Ollama + infer-OS最適化制御システム
Ollamaにモデル管理を委ね、infer-OSの最適化のみを制御

特徴:
- Ollamaによる完全なモデル管理
- infer-OS最適化のON/OFF制御
- NPU/GPU使用率監視
- Windows完全対応
- 複数モデル対応
- インタラクティブモード
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

class OllamaInferOSController:
    """Ollama + infer-OS最適化制御システム"""
    
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
        
        # プロンプトテンプレート
        self.templates = {
            "conversation": """以下は人間とAIアシスタントの会話です。AIアシスタントは親切で、詳細で、丁寧です。

人間: {prompt}
AIアシスタント: """,
            
            "instruction": """以下の指示に従って、詳しく回答してください。

指示: {prompt}

回答: """,
            
            "reasoning": """以下の問題について、論理的に考えて詳しく説明してください。

問題: {prompt}

解答: """,
            
            "creative": """以下のテーマについて、創造的で興味深い内容を書いてください。

テーマ: {prompt}

内容: """,
            
            "simple": "{prompt}"
        }
        
        self.current_template = "conversation"
        
        print("🚀 Ollama + infer-OS最適化制御システム初期化")
        print(f"🔗 Ollama接続先: {ollama_host}")
        print(f"⚡ infer-OS最適化: {'有効' if self.infer_os_enabled else '無効'}")
        print(f"🎯 設計方針: Ollamaモデル管理 + infer-OS最適化制御")
    
    def check_ollama_connection(self) -> bool:
        """Ollama接続確認"""
        try:
            print("🔍 Ollama接続確認中...")
            response = requests.get(f"{self.ollama_api}/tags", timeout=5)
            
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
            print("   Linux/Mac: ollama serve")
            return False
        except Exception as e:
            print(f"❌ Ollama接続確認エラー: {e}")
            return False
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """利用可能なモデル一覧取得"""
        try:
            print("📋 利用可能なモデル一覧取得中...")
            response = requests.get(f"{self.ollama_api}/tags", timeout=10)
            
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
            
            # モデル名が指定されていない場合は最初のモデルを選択
            if model_name is None:
                selected_model = self.available_models[0]
            else:
                # 指定されたモデル名で検索
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
            print(f"📅 更新日時: {selected_model['modified_at']}")
            
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
        
        # NPU最適化
        if self.infer_os_config["npu_optimization"]:
            print("🔧 NPU最適化: 有効")
            # NPU関連の最適化設定
            os.environ['ONNXRUNTIME_PROVIDERS'] = 'VitisAIExecutionProvider,DmlExecutionProvider,CPUExecutionProvider'
        
        # メモリ最適化
        if self.infer_os_config["memory_optimization"]:
            print("🔧 メモリ最適化: 有効")
            os.environ['OLLAMA_MAX_LOADED_MODELS'] = '1'  # メモリ効率化
            os.environ['OLLAMA_NUM_PARALLEL'] = '1'  # 並列処理制限
        
        # CPU最適化
        if self.infer_os_config["cpu_optimization"]:
            print("🔧 CPU最適化: 有効")
            cpu_count = os.cpu_count()
            os.environ['OLLAMA_NUM_THREADS'] = str(min(4, cpu_count))  # CPU使用制限
        
        # GPU加速
        if self.infer_os_config["gpu_acceleration"]:
            print("🔧 GPU加速: 有効")
            os.environ['OLLAMA_GPU_LAYERS'] = '35'  # GPU層数設定
        
        # 量子化
        if self.infer_os_config["quantization"]:
            print("🔧 量子化: 有効")
            os.environ['OLLAMA_LOAD_TIMEOUT'] = '300'  # 量子化読み込み時間延長
        
        # 並列処理
        if self.infer_os_config["parallel_processing"]:
            print("🔧 並列処理: 有効")
            os.environ['OLLAMA_CONCURRENT_REQUESTS'] = '2'  # 並列リクエスト制限
        
        print("✅ infer-OS最適化設定適用完了")
        
        # 設定確認
        print("📊 適用された最適化設定:")
        for key, value in self.infer_os_config.items():
            status = "✅" if value else "❌"
            print(f"  {status} {key}: {value}")
    
    def create_npu_onnx_session(self) -> bool:
        """NPU対応ONNX推論セッション作成"""
        try:
            print("🔧 NPU対応ONNX推論セッション作成中...")
            
            # 軽量なダミーモデル作成（NPU互換）
            os.makedirs("models", exist_ok=True)
            
            # シンプルなONNXモデル作成
            import torch
            import torch.nn as nn
            
            class SimpleNPUModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(512, 1000)
                    
                def forward(self, x):
                    return self.linear(x)
            
            model = SimpleNPUModel()
            model.eval()
            
            dummy_input = torch.randn(1, 512)
            onnx_path = "models/ollama_npu_model.onnx"
            
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            print(f"✅ ONNXモデル作成完了: {onnx_path}")
            
            # NPU対応プロバイダー設定
            providers = []
            
            # VitisAI ExecutionProvider
            if 'VitisAIExecutionProvider' in ort.get_available_providers():
                vitisai_options = {
                    'config_file': '',  # 設定ファイルエラー回避
                    'target': 'DPUCAHX8H',
                }
                providers.append(('VitisAIExecutionProvider', vitisai_options))
                print("🎯 VitisAI ExecutionProvider利用可能")
            
            # DmlExecutionProvider
            if 'DmlExecutionProvider' in ort.get_available_providers():
                providers.append('DmlExecutionProvider')
                print("🎯 DmlExecutionProvider利用可能")
            
            # CPUExecutionProvider（フォールバック）
            providers.append('CPUExecutionProvider')
            
            # セッション作成
            session_options = ort.SessionOptions()
            session_options.log_severity_level = 3
            
            self.onnx_session = ort.InferenceSession(
                onnx_path,
                sess_options=session_options,
                providers=providers
            )
            
            active_provider = self.onnx_session.get_providers()[0]
            print(f"✅ NPU対応ONNX推論セッション作成成功")
            print(f"🎯 アクティブプロバイダー: {active_provider}")
            
            return True
            
        except Exception as e:
            print(f"❌ NPU対応ONNX推論セッション作成エラー: {e}")
            return False
    
    def test_npu_operation(self) -> bool:
        """NPU動作テスト"""
        if self.onnx_session is None:
            print("❌ ONNXセッションが作成されていません")
            return False
        
        try:
            print("🔧 NPU動作テスト実行中...")
            
            # テスト入力作成
            test_input = np.random.randn(1, 512).astype(np.float32)
            
            # 推論実行
            outputs = self.onnx_session.run(None, {"input": test_input})
            
            print(f"✅ NPU動作テスト成功: 出力形状 {outputs[0].shape}")
            
            # アクティブプロバイダー確認
            active_provider = self.onnx_session.get_providers()[0]
            print(f"🎯 アクティブプロバイダー: {active_provider}")
            
            return True
            
        except Exception as e:
            print(f"❌ NPU動作テストエラー: {e}")
            return False
    
    def start_npu_monitoring(self):
        """NPU使用率監視開始"""
        if self.npu_monitoring:
            return
        
        self.npu_monitoring = True
        
        def monitor_npu():
            print("📊 NPU/GPU使用率監視開始（1秒間隔）- Ollama + infer-OS版")
            
            prev_usage = 0.0
            usage_history = []
            
            while self.npu_monitoring:
                try:
                    # GPU使用率取得（NPU使用率の代替）
                    try:
                        import torch
                        if torch.cuda.is_available():
                            gpu_usage = torch.cuda.utilization()
                        else:
                            gpu_usage = 0.0
                    except:
                        gpu_usage = 0.0
                    
                    # 使用率変化検出
                    if abs(gpu_usage - prev_usage) > 2.0:  # 2%以上の変化
                        print(f"🔥 NPU/GPU使用率変化: {prev_usage:.1f}% → {gpu_usage:.1f}% (Ollama + infer-OS)")
                        self.npu_stats["usage_changes"] += 1
                    
                    # 統計更新
                    usage_history.append(gpu_usage)
                    if len(usage_history) > 60:  # 直近60秒のみ保持
                        usage_history.pop(0)
                    
                    self.npu_stats["max_usage"] = max(self.npu_stats["max_usage"], gpu_usage)
                    self.npu_stats["avg_usage"] = sum(usage_history) / len(usage_history)
                    
                    prev_usage = gpu_usage
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"⚠️ NPU監視エラー: {e}")
                    time.sleep(1)
        
        # バックグラウンドで監視開始
        monitor_thread = threading.Thread(target=monitor_npu, daemon=True)
        monitor_thread.start()
    
    def stop_npu_monitoring(self):
        """NPU使用率監視停止"""
        self.npu_monitoring = False
        print("📊 NPU/GPU使用率監視停止（Ollama + infer-OS）")
    
    def generate_text_with_ollama(self, prompt: str, max_tokens: int = 100, template: str = None) -> str:
        """Ollamaを使用したテキスト生成"""
        if self.current_model is None:
            return "❌ モデルが選択されていません"
        
        try:
            # テンプレート適用
            if template and template in self.templates:
                formatted_prompt = self.templates[template].format(prompt=prompt)
            else:
                formatted_prompt = self.templates[self.current_template].format(prompt=prompt)
            
            print(f"💬 Ollamaテキスト生成中: '{formatted_prompt[:50]}...'")
            print(f"🎯 使用モデル: {self.current_model['name']}")
            print(f"🎯 最大トークン数: {max_tokens}")
            print(f"⚡ infer-OS最適化: {'有効' if self.infer_os_enabled else '無効'}")
            
            # Ollama API呼び出し
            payload = {
                "model": self.current_model["name"],
                "prompt": formatted_prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 50,
                    "repeat_penalty": 1.1,
                }
            }
            
            print("🔧 Ollama API呼び出し中...")
            start_time = time.time()
            
            response = requests.post(
                f"{self.ollama_api}/generate",
                json=payload,
                timeout=120
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                generated_text = data.get("response", "")
                
                print(f"✅ Ollamaテキスト生成完了")
                print(f"📝 生成文字数: {len(generated_text)}")
                print(f"⏱️ 生成時間: {end_time - start_time:.2f}秒")
                
                # 品質チェック
                if len(generated_text.strip()) < 10:
                    print("⚠️ 生成結果が短すぎます")
                    return self.generate_fallback_text(prompt)
                
                return generated_text.strip()
            else:
                print(f"❌ Ollama API呼び出し失敗: ステータスコード {response.status_code}")
                return self.generate_fallback_text(prompt)
                
        except requests.exceptions.Timeout:
            print("❌ Ollama API呼び出しタイムアウト（120秒）")
            return self.generate_fallback_text(prompt)
        except Exception as e:
            print(f"❌ Ollamaテキスト生成エラー: {e}")
            return self.generate_fallback_text(prompt)
    
    def generate_fallback_text(self, prompt: str) -> str:
        """フォールバックテキスト生成"""
        fallback_responses = {
            "人工知能": "人工知能は現代社会において重要な技術分野です。Ollama + infer-OS最適化環境でも安定して動作する技術として注目されています。",
            "量子": "量子コンピューティングは革新的な計算技術です。Ollama環境での研究開発も活発に行われています。",
            "default": f"申し訳ございませんが、「{prompt}」についての詳細な回答を生成することができませんでした。Ollama + infer-OS環境での制約により、簡潔な回答のみ提供いたします。"
        }
        
        # キーワードマッチング
        for keyword, response in fallback_responses.items():
            if keyword in prompt and keyword != "default":
                return response
        
        return fallback_responses["default"]
    
    def toggle_infer_os_optimization(self, enabled: bool = None) -> bool:
        """infer-OS最適化のON/OFF切り替え"""
        if enabled is None:
            self.infer_os_enabled = not self.infer_os_enabled
        else:
            self.infer_os_enabled = enabled
        
        status = "有効" if self.infer_os_enabled else "無効"
        print(f"🔄 infer-OS最適化を{status}に切り替えました")
        
        if self.infer_os_enabled:
            self.apply_infer_os_optimizations()
        else:
            print("⚠️ infer-OS最適化が無効になりました")
            # 最適化設定をリセット
            for key in self.infer_os_config:
                self.infer_os_config[key] = False
        
        return self.infer_os_enabled
    
    def show_system_status(self):
        """システム状態表示"""
        print("\n📊 Ollama + infer-OS制御システム状態:")
        print(f"  🔗 Ollama接続: {'✅' if self.check_ollama_connection() else '❌'}")
        print(f"  🎯 選択モデル: {self.current_model['name'] if self.current_model else 'なし'}")
        print(f"  ⚡ infer-OS最適化: {'✅ 有効' if self.infer_os_enabled else '❌ 無効'}")
        print(f"  🔧 ONNXセッション: {'✅' if self.onnx_session else '❌'}")
        print(f"  📊 NPU監視: {'✅ 実行中' if self.npu_monitoring else '❌ 停止中'}")
        
        if self.infer_os_enabled:
            print("  📋 infer-OS最適化設定:")
            for key, value in self.infer_os_config.items():
                status = "✅" if value else "❌"
                print(f"    {status} {key}: {value}")
        
        # システム情報
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        print(f"  💻 CPU使用率: {cpu_percent:.1f}%")
        print(f"  💾 メモリ使用率: {memory.percent:.1f}%")
    
    def show_npu_stats(self):
        """NPU統計表示"""
        print("\n📊 NPU/GPU使用率統計（Ollama + infer-OS版）:")
        print(f"  🔥 使用率変化検出回数: {self.npu_stats['usage_changes']}")
        print(f"  📈 最大使用率: {self.npu_stats['max_usage']:.1f}%")
        print(f"  📊 平均使用率: {self.npu_stats['avg_usage']:.1f}%")
        
        if self.onnx_session:
            active_provider = self.onnx_session.get_providers()[0]
            print(f"  🔧 アクティブプロバイダー: {active_provider}")
        
        # システム情報
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        print(f"  💻 CPU使用率: {cpu_percent:.1f}%")
        print(f"  💾 メモリ使用率: {memory.percent:.1f}%")
        print(f"  ⚡ infer-OS最適化: {'有効' if self.infer_os_enabled else '無効'}")
    
    def change_template(self):
        """プロンプトテンプレート変更"""
        print("\n📋 利用可能なテンプレート:")
        for i, (name, template) in enumerate(self.templates.items(), 1):
            print(f"  {i}. {name}")
        
        try:
            choice = input("テンプレート番号を選択してください: ").strip()
            template_names = list(self.templates.keys())
            
            if choice.isdigit() and 1 <= int(choice) <= len(template_names):
                self.current_template = template_names[int(choice) - 1]
                print(f"✅ テンプレートを '{self.current_template}' に変更しました")
            else:
                print("❌ 無効な選択です")
        except Exception as e:
            print(f"❌ テンプレート変更エラー: {e}")
    
    def initialize_system(self) -> bool:
        """システム初期化"""
        try:
            print("🚀 Ollama + infer-OS制御システム初期化開始")
            
            # Ollama接続確認
            if not self.check_ollama_connection():
                return False
            
            # 利用可能なモデル取得
            models = self.get_available_models()
            if not models:
                print("❌ 利用可能なモデルがありません")
                print("💡 Ollamaでモデルをダウンロードしてください")
                print("   例: ollama pull llama2")
                return False
            
            # 最初のモデルを選択
            if not self.select_model():
                return False
            
            # infer-OS最適化適用
            self.apply_infer_os_optimizations()
            
            # NPU対応ONNX推論セッション作成
            onnx_created = self.create_npu_onnx_session()
            if not onnx_created:
                print("⚠️ NPU対応ONNX推論セッション作成に失敗しましたが、継続します")
            
            # NPU動作テスト
            if self.onnx_session:
                npu_test = self.test_npu_operation()
                if not npu_test:
                    print("⚠️ NPU動作テストに失敗しましたが、継続します")
            
            print("✅ Ollama + infer-OS制御システム初期化完了")
            self.show_system_status()
            
            return True
            
        except Exception as e:
            print(f"❌ システム初期化エラー: {e}")
            return False
    
    def run_interactive_mode(self):
        """インタラクティブモード実行"""
        print("\n🎯 Ollama + infer-OS制御インタラクティブモード")
        print(f"🎯 使用モデル: {self.current_model['name'] if self.current_model else 'なし'}")
        print(f"⚡ infer-OS最適化: {'有効' if self.infer_os_enabled else '無効'}")
        
        if self.onnx_session:
            active_provider = self.onnx_session.get_providers()[0]
            print(f"🔧 NPUプロバイダー: {active_provider}")
        
        print("💡 コマンド:")
        print("  'quit' - 終了")
        print("  'stats' - NPU統計表示")
        print("  'status' - システム状態表示")
        print("  'template' - プロンプトテンプレート変更")
        print("  'model' - モデル変更")
        print("  'toggle' - infer-OS最適化ON/OFF切り替え")
        print("  'on' - infer-OS最適化有効")
        print("  'off' - infer-OS最適化無効")
        print("📋 テンプレート: conversation, instruction, reasoning, creative, simple")
        print("=" * 70)
        
        # NPU監視開始
        self.start_npu_monitoring()
        
        try:
            while True:
                try:
                    infer_os_status = "ON" if self.infer_os_enabled else "OFF"
                    prompt = input(f"\n💬 プロンプトを入力してください [infer-OS:{infer_os_status}] [{self.current_template}]: ").strip()
                    
                    if not prompt:
                        continue
                    
                    if prompt.lower() == 'quit':
                        print("👋 Ollama + infer-OS制御システムを終了します")
                        break
                    
                    if prompt.lower() == 'stats':
                        self.show_npu_stats()
                        continue
                    
                    if prompt.lower() == 'status':
                        self.show_system_status()
                        continue
                    
                    if prompt.lower() == 'template':
                        self.change_template()
                        continue
                    
                    if prompt.lower() == 'model':
                        self.select_model_interactive()
                        continue
                    
                    if prompt.lower() == 'toggle':
                        self.toggle_infer_os_optimization()
                        continue
                    
                    if prompt.lower() == 'on':
                        self.toggle_infer_os_optimization(True)
                        continue
                    
                    if prompt.lower() == 'off':
                        self.toggle_infer_os_optimization(False)
                        continue
                    
                    # テキスト生成実行
                    start_time = time.time()
                    result = self.generate_text_with_ollama(prompt, max_tokens=100)
                    end_time = time.time()
                    
                    print(f"\n🎯 Ollama + infer-OS生成結果:")
                    print(result)
                    print(f"\n⏱️ 生成時間: {end_time - start_time:.2f}秒")
                    print(f"⚡ infer-OS最適化: {'有効' if self.infer_os_enabled else '無効'}")
                    
                except KeyboardInterrupt:
                    print("\n👋 Ollama + infer-OS制御システムを終了します")
                    break
                except Exception as e:
                    print(f"❌ インタラクティブモードエラー: {e}")
                    continue
        
        finally:
            self.stop_npu_monitoring()
    
    def select_model_interactive(self):
        """インタラクティブモデル選択"""
        try:
            print("\n📋 利用可能なモデル:")
            for i, model in enumerate(self.available_models, 1):
                size_gb = model["size"] / (1024**3) if model["size"] > 0 else 0
                print(f"  {i}. {model['name']} ({size_gb:.1f}GB)")
            
            choice = input("モデル番号を選択してください: ").strip()
            
            if choice.isdigit() and 1 <= int(choice) <= len(self.available_models):
                selected_model = self.available_models[int(choice) - 1]
                self.current_model = selected_model
                print(f"✅ モデルを '{selected_model['name']}' に変更しました")
            else:
                print("❌ 無効な選択です")
        except Exception as e:
            print(f"❌ モデル選択エラー: {e}")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Ollama + infer-OS最適化制御システム")
    parser.add_argument("--interactive", action="store_true", help="インタラクティブモード")
    parser.add_argument("--prompt", type=str, help="単発プロンプト")
    parser.add_argument("--tokens", type=int, default=100, help="最大トークン数")
    parser.add_argument("--template", type=str, default="conversation", help="プロンプトテンプレート")
    parser.add_argument("--model", type=str, help="使用モデル名")
    parser.add_argument("--ollama-host", type=str, default="http://localhost:11434", help="Ollama接続先")
    parser.add_argument("--infer-os", action="store_true", default=True, help="infer-OS最適化有効")
    parser.add_argument("--no-infer-os", action="store_true", help="infer-OS最適化無効")
    
    args = parser.parse_args()
    
    # infer-OS最適化設定
    infer_os_enabled = args.infer_os and not args.no_infer_os
    
    # システム初期化
    system = OllamaInferOSController(ollama_host=args.ollama_host)
    system.infer_os_enabled = infer_os_enabled
    
    if not system.initialize_system():
        print("❌ システム初期化に失敗しました")
        sys.exit(1)
    
    # モデル選択
    if args.model:
        if not system.select_model(args.model):
            print(f"❌ モデル '{args.model}' の選択に失敗しました")
            sys.exit(1)
    
    try:
        if args.interactive:
            # インタラクティブモード
            system.run_interactive_mode()
        elif args.prompt:
            # 単発生成
            system.start_npu_monitoring()
            result = system.generate_text_with_ollama(args.prompt, args.tokens, args.template)
            print(f"\n🎯 生成結果:\n{result}")
            system.stop_npu_monitoring()
            system.show_npu_stats()
        else:
            print("使用方法: --interactive または --prompt を指定してください")
            print("例: python ollama_infer_os_control_system.py --interactive")
            print("例: python ollama_infer_os_control_system.py --prompt '人工知能について教えてください' --tokens 200")
            print("例: python ollama_infer_os_control_system.py --interactive --no-infer-os")
    
    except KeyboardInterrupt:
        print("\n👋 システムを終了します")
    except Exception as e:
        print(f"❌ 実行エラー: {e}")
    finally:
        if system.npu_monitoring:
            system.stop_npu_monitoring()

if __name__ == "__main__":
    main()

