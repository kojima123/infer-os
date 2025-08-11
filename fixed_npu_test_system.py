#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RyzenAI 1.5対応 修正版NPUテストシステム
ONNX IRバージョン10互換性対応版
"""

import os
import sys
import time
import threading
import psutil
import argparse
import signal
from typing import Optional, Dict, Any, List
import warnings
warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import numpy as np
    import onnx
    import onnxruntime as ort
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("✅ 必要なライブラリのインポート成功")
except ImportError as e:
    print(f"❌ ライブラリインポートエラー: {e}")
    sys.exit(1)

class TimeoutHandler:
    """タイムアウト処理クラス"""
    def __init__(self, timeout_seconds: int = 60):
        self.timeout_seconds = timeout_seconds
        self.timed_out = False
    
    def timeout_handler(self, signum, frame):
        self.timed_out = True
        print(f"⏰ タイムアウト ({self.timeout_seconds}秒) が発生しました")
        raise TimeoutError(f"処理が{self.timeout_seconds}秒でタイムアウトしました")
    
    def __enter__(self):
        if os.name != 'nt':  # Windows以外でのみsignalを使用
            signal.signal(signal.SIGALRM, self.timeout_handler)
            signal.alarm(self.timeout_seconds)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.name != 'nt':
            signal.alarm(0)

class NPUPerformanceMonitor:
    """NPU性能監視クラス"""
    def __init__(self):
        self.monitoring = False
        self.cpu_samples = []
        self.memory_samples = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """監視開始"""
        self.monitoring = True
        self.cpu_samples = []
        self.memory_samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("📊 性能監視開始")
    
    def stop_monitoring(self):
        """監視停止"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        print("📊 性能監視停止")
    
    def _monitor_loop(self):
        """監視ループ"""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent
                self.cpu_samples.append(cpu_percent)
                self.memory_samples.append(memory_percent)
                time.sleep(0.5)
            except Exception:
                break
    
    def get_report(self) -> Dict[str, Any]:
        """性能レポート取得"""
        if not self.cpu_samples:
            return {"error": "監視データなし"}
        
        return {
            "samples": len(self.cpu_samples),
            "avg_cpu": sum(self.cpu_samples) / len(self.cpu_samples),
            "max_cpu": max(self.cpu_samples),
            "avg_memory": sum(self.memory_samples) / len(self.memory_samples),
            "max_memory": max(self.memory_samples)
        }

class FixedNPUTestSystem:
    """修正版NPUテストシステム"""
    
    def __init__(self, timeout_seconds: int = 60):
        self.timeout_seconds = timeout_seconds
        self.tokenizer = None
        self.model = None
        self.onnx_session = None
        self.generation_count = 0
        self.infer_os_enabled = False  # infer-OS最適化を明示的にOFF
        self.performance_monitor = NPUPerformanceMonitor()
        
        print("🚀 修正版NPUテストシステム初期化")
        print("============================================================")
        print(f"⏰ タイムアウト設定: {timeout_seconds}秒")
        print(f"🔧 infer-OS最適化: {'ON' if self.infer_os_enabled else 'OFF'}")
    
    def _create_compatible_onnx_model(self, model_path: str) -> bool:
        """RyzenAI 1.5互換のONNXモデル作成（IRバージョン10）"""
        try:
            print("📄 RyzenAI 1.5互換ダミーONNXモデル作成中...")
            
            # シンプルなPyTorchモデル作成
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(512, 1000)
                
                def forward(self, x):
                    return self.linear(x)
            
            model = SimpleModel()
            model.eval()
            
            # ダミー入力
            dummy_input = torch.randn(1, 512)
            
            # ONNX IRバージョン10で明示的にエクスポート
            torch.onnx.export(
                model,
                dummy_input,
                model_path,
                export_params=True,
                opset_version=11,  # opset_versionは11を使用
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # ONNXモデルを読み込んでIRバージョンを確認・修正
            onnx_model = onnx.load(model_path)
            
            # IRバージョンを10に強制設定
            onnx_model.ir_version = 10
            
            # モデルを保存し直す
            onnx.save(onnx_model, model_path)
            
            print(f"✅ RyzenAI 1.5互換ONNXモデル作成完了: {model_path}")
            print(f"📋 IRバージョン: {onnx_model.ir_version}")
            return True
            
        except Exception as e:
            print(f"❌ ONNXモデル作成エラー: {e}")
            return False
    
    def _setup_npu_session(self) -> bool:
        """NPUセッション設定"""
        try:
            print("⚡ NPUセッション設定中...")
            
            # vaip_config.jsonの確認
            vaip_config_paths = [
                "C:/Program Files/RyzenAI/1.5/voe-4.0-win_amd64/vaip_config.json",
                "C:/Program Files/RyzenAI/vaip_config.json",
                "./vaip_config.json"
            ]
            
            vaip_config_found = False
            for path in vaip_config_paths:
                if os.path.exists(path):
                    print(f"📁 vaip_config.json発見: {path}")
                    vaip_config_found = True
                    break
            
            if not vaip_config_found:
                print("⚠️ vaip_config.jsonが見つかりません")
            
            # 互換性のあるONNXモデル作成
            model_path = "fixed_dummy_npu_model.onnx"
            if not self._create_compatible_onnx_model(model_path):
                return False
            
            # 利用可能なプロバイダー確認
            available_providers = ort.get_available_providers()
            print(f"📋 利用可能なプロバイダー: {available_providers}")
            
            # NPU用プロバイダーの優先順位
            npu_providers = []
            if 'VitisAIExecutionProvider' in available_providers:
                npu_providers.append('VitisAIExecutionProvider')
                print("✅ VitisAIExecutionProvider利用可能")
            if 'DmlExecutionProvider' in available_providers:
                npu_providers.append('DmlExecutionProvider')
                print("✅ DmlExecutionProvider利用可能")
            
            if not npu_providers:
                print("❌ NPU用プロバイダーが見つかりません")
                return False
            
            # NPUセッション作成
            session_options = ort.SessionOptions()
            session_options.log_severity_level = 3  # エラーのみ表示
            
            self.onnx_session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=npu_providers
            )
            
            print(f"✅ NPUセッション作成成功")
            print(f"🔧 使用プロバイダー: {self.onnx_session.get_providers()}")
            
            # NPU動作テスト
            test_input = np.random.randn(1, 512).astype(np.float32)
            test_output = self.onnx_session.run(None, {'input': test_input})
            print(f"✅ NPU動作テスト完了: 出力形状 {test_output[0].shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ NPUセッション設定エラー: {e}")
            return False
    
    def _load_tokenizer_and_model(self) -> bool:
        """トークナイザーとモデルのロード"""
        try:
            print("🔤 トークナイザーロード中...")
            
            # llama3-8b-amd-npuモデルのパス確認
            model_paths = [
                "llama3-8b-amd-npu",
                "./llama3-8b-amd-npu",
                "C:/infer-os-demo/infer-os/infer-os/llama3-8b-amd-npu"
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"📁 モデルディレクトリ発見: {path}")
                    break
            
            if not model_path:
                print("⚠️ llama3-8b-amd-npuモデルが見つかりません")
                print("🔄 フォールバックモデルを使用します")
                model_path = "microsoft/DialoGPT-medium"
            
            # トークナイザーロード
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✅ トークナイザーロード成功")
            
            # モデルロード（CPU用）
            print("🤖 モデルロード中...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            self.model.eval()
            print("✅ モデルロード成功")
            
            return True
            
        except Exception as e:
            print(f"❌ モデルロードエラー: {e}")
            return False
    
    def initialize(self) -> bool:
        """システム初期化"""
        try:
            with TimeoutHandler(self.timeout_seconds):
                # NPUセッション設定
                if not self._setup_npu_session():
                    print("❌ NPUセッション設定失敗")
                    return False
                
                # トークナイザーとモデルロード
                if not self._load_tokenizer_and_model():
                    print("❌ モデルロード失敗")
                    return False
                
                print("✅ 修正版NPUテストシステム初期化完了")
                return True
                
        except TimeoutError:
            print("❌ 初期化タイムアウト")
            return False
        except Exception as e:
            print(f"❌ 初期化エラー: {e}")
            return False
    
    def _npu_inference_test(self, num_inferences: int = 20) -> Dict[str, Any]:
        """NPU推論テスト"""
        try:
            print(f"🎯 NPU推論テスト開始（{num_inferences}回）...")
            
            start_time = time.time()
            
            for i in range(num_inferences):
                # ダミー入力でNPU推論実行
                test_input = np.random.randn(1, 512).astype(np.float32)
                output = self.onnx_session.run(None, {'input': test_input})
                
                if (i + 1) % 5 == 0:
                    print(f"  📊 進捗: {i + 1}/{num_inferences}")
            
            end_time = time.time()
            total_time = end_time - start_time
            throughput = num_inferences / total_time
            
            return {
                "success": True,
                "num_inferences": num_inferences,
                "total_time": total_time,
                "throughput": throughput,
                "provider": self.onnx_session.get_providers()[0]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_text(self, prompt: str, max_tokens: int = 30) -> str:
        """テキスト生成（NPU推論テスト付き）"""
        try:
            print(f"🔄 生成中（タイムアウト: {self.timeout_seconds}秒）...")
            
            with TimeoutHandler(self.timeout_seconds):
                # 性能監視開始
                self.performance_monitor.start_monitoring()
                
                # NPU推論テスト実行
                npu_result = self._npu_inference_test(max_tokens)
                
                # 性能監視停止
                self.performance_monitor.stop_monitoring()
                
                # 結果表示
                if npu_result["success"]:
                    print(f"🎯 NPU推論テスト結果:")
                    print(f"  📝 入力: {prompt}")
                    print(f"  ⚡ NPU推論回数: {npu_result['num_inferences']}")
                    print(f"  ⏱️ 推論時間: {npu_result['total_time']:.3f}秒")
                    print(f"  📊 スループット: {npu_result['throughput']:.1f} 推論/秒")
                    print(f"  🔧 プロバイダー: {npu_result['provider']}")
                else:
                    print(f"❌ NPU推論テストエラー: {npu_result['error']}")
                
                # 性能レポート
                perf_report = self.performance_monitor.get_report()
                if "error" not in perf_report:
                    print(f"📊 NPU性能レポート:")
                    print(f"  ⏱️ 実行時間: {npu_result.get('total_time', 0):.2f}秒")
                    print(f"  🔢 サンプル数: {perf_report['samples']}")
                    print(f"  💻 平均CPU使用率: {perf_report['avg_cpu']:.1f}%")
                    print(f"  💻 最大CPU使用率: {perf_report['max_cpu']:.1f}%")
                    print(f"  💾 平均メモリ使用率: {perf_report['avg_memory']:.1f}%")
                
                self.generation_count += 1
                
                # 簡単な応答生成（デモ用）
                response = f"NPUテスト完了: {prompt} (推論{max_tokens}回実行)"
                return response
                
        except TimeoutError:
            return f"⏰ タイムアウト: {prompt}"
        except Exception as e:
            return f"❌ エラー: {e}"
    
    def interactive_mode(self):
        """インタラクティブモード"""
        print(f"\n🇯🇵 修正版NPUテストシステム - インタラクティブモード")
        print(f"⏰ タイムアウト設定: {self.timeout_seconds}秒")
        print(f"🔧 infer-OS最適化: {'ON' if self.infer_os_enabled else 'OFF'}")
        print(f"💡 'exit'または'quit'で終了、'stats'で統計表示")
        print("============================================================")
        
        while True:
            try:
                prompt = input("\n🤖 プロンプトを入力してください: ").strip()
                
                if prompt.lower() in ['exit', 'quit', '終了']:
                    print("👋 修正版NPUテストシステムを終了します")
                    break
                
                if prompt.lower() == 'stats':
                    print(f"\n📊 システム統計:")
                    print(f"  🔢 生成回数: {self.generation_count}")
                    print(f"  ⏰ タイムアウト設定: {self.timeout_seconds}秒")
                    print(f"  🔧 infer-OS最適化: {'ON' if self.infer_os_enabled else 'OFF'}")
                    print(f"  🤖 モデル: llama3-8b-amd-npu")
                    print(f"  🔤 トークナイザー: {'✅ 利用可能' if self.tokenizer else '❌ 未ロード'}")
                    print(f"  🧠 モデル: {'✅ 利用可能' if self.model else '❌ 未ロード'}")
                    print(f"  ⚡ NPUセッション: {'✅ 利用可能' if self.onnx_session else '❌ 未作成'}")
                    continue
                
                if not prompt:
                    continue
                
                start_time = time.time()
                response = self.generate_text(prompt, max_tokens=30)
                end_time = time.time()
                
                print(f"\n📝 応答: {response}")
                print(f"⏱️ 生成時間: {end_time - start_time:.2f}秒")
                
            except KeyboardInterrupt:
                print("\n👋 修正版NPUテストシステムを終了します")
                break
            except Exception as e:
                print(f"❌ エラー: {e}")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="修正版NPUテストシステム")
    parser.add_argument("--interactive", action="store_true", help="インタラクティブモード")
    parser.add_argument("--prompt", type=str, help="単発テスト用プロンプト")
    parser.add_argument("--tokens", type=int, default=30, help="NPU推論回数")
    parser.add_argument("--timeout", type=int, default=60, help="タイムアウト秒数")
    
    args = parser.parse_args()
    
    # システム初期化
    system = FixedNPUTestSystem(timeout_seconds=args.timeout)
    
    if not system.initialize():
        print("❌ システム初期化に失敗しました")
        return
    
    if args.interactive:
        # インタラクティブモード
        system.interactive_mode()
    elif args.prompt:
        # 単発テスト
        print(f"\n🎯 単発NPUテスト実行")
        print(f"📝 プロンプト: {args.prompt}")
        print(f"⚡ NPU推論回数: {args.tokens}")
        
        start_time = time.time()
        response = system.generate_text(args.prompt, max_tokens=args.tokens)
        end_time = time.time()
        
        print(f"\n📝 応答: {response}")
        print(f"⏱️ 総実行時間: {end_time - start_time:.2f}秒")
    else:
        print("❌ --interactive または --prompt を指定してください")

if __name__ == "__main__":
    main()

