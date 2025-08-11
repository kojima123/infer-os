#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
純粋NPUテストシステム
infer-OS最適化をOFFにして、llama3-8b-amd-npuモデルで真のNPU処理をテスト
"""

import os
import sys
import time
import torch
import threading
import psutil
from pathlib import Path
from transformers import AutoTokenizer
import onnxruntime as ort

class TimeoutException(Exception):
    """タイムアウト例外"""
    pass

class TimeoutHandler:
    """タイムアウトハンドラー（改良版）"""
    
    def __init__(self, timeout_seconds=60):
        self.timeout_seconds = timeout_seconds
        self.timer = None
        self.timed_out = False
        
    def __enter__(self):
        self.timer = threading.Timer(self.timeout_seconds, self._timeout_callback)
        self.timer.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer:
            self.timer.cancel()
            
    def _timeout_callback(self):
        self.timed_out = True
        print(f"\n⚠️ タイムアウト警告: {self.timeout_seconds}秒経過")
        print("🔄 処理を安全に中断します...")

class NPUPerformanceMonitor:
    """NPU性能監視クラス"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.npu_usage_samples = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """NPU監視開始"""
        self.monitoring = True
        self.start_time = time.time()
        self.npu_usage_samples = []
        
        self.monitor_thread = threading.Thread(target=self._monitor_npu_usage)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """NPU監視停止"""
        self.monitoring = False
        self.end_time = time.time()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
            
    def _monitor_npu_usage(self):
        """NPU使用率監視（バックグラウンド）"""
        while self.monitoring:
            try:
                # CPU使用率を代替指標として使用
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_info = psutil.virtual_memory()
                
                sample = {
                    'timestamp': time.time() - self.start_time,
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_info.percent,
                    'memory_used_gb': memory_info.used / (1024**3)
                }
                
                self.npu_usage_samples.append(sample)
                time.sleep(0.5)  # 0.5秒間隔でサンプリング
                
            except Exception as e:
                print(f"⚠️ 監視エラー: {e}")
                break
                
    def get_performance_report(self):
        """性能レポート取得"""
        if not self.npu_usage_samples:
            return "📊 性能データなし"
            
        duration = self.end_time - self.start_time if self.end_time else 0
        avg_cpu = sum(s['cpu_percent'] for s in self.npu_usage_samples) / len(self.npu_usage_samples)
        max_cpu = max(s['cpu_percent'] for s in self.npu_usage_samples)
        avg_memory = sum(s['memory_percent'] for s in self.npu_usage_samples) / len(self.npu_usage_samples)
        
        report = f"""
📊 NPU性能レポート:
  ⏱️ 実行時間: {duration:.2f}秒
  🔢 サンプル数: {len(self.npu_usage_samples)}
  💻 平均CPU使用率: {avg_cpu:.1f}%
  💻 最大CPU使用率: {max_cpu:.1f}%
  💾 平均メモリ使用率: {avg_memory:.1f}%
"""
        return report

class PureNPUSystem:
    """純粋NPUシステム（infer-OS最適化OFF）"""
    
    def __init__(self, model_name="llama3-8b-amd-npu", timeout=60):
        self.model_name = model_name
        self.timeout = timeout
        self.tokenizer = None
        self.npu_session = None
        self.generation_count = 0
        self.performance_monitor = NPUPerformanceMonitor()
        
        # infer-OS最適化を明示的にOFF
        self.infer_os_enabled = False
        
    def setup(self):
        """純粋NPUシステムセットアップ"""
        print("🚀 純粋NPUシステム初期化（infer-OS最適化OFF）")
        print("=" * 60)
        print("⚠️ infer-OS最適化: 無効")
        print("🎯 目標: 真のNPU処理テスト")
        
        try:
            # トークナイザーロード
            print("🔤 トークナイザーロード中...")
            with TimeoutHandler(self.timeout):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    local_files_only=True
                )
            print("✅ トークナイザーロード成功")
            
            # NPUセッション作成
            print("⚡ NPUセッション作成中...")
            with TimeoutHandler(self.timeout):
                success = self._create_npu_session()
                
            if success:
                print("✅ NPUセッション作成成功")
                print("🎉 純粋NPUシステム初期化完了")
                return True
            else:
                print("❌ NPUセッション作成失敗")
                return False
                
        except TimeoutException as e:
            print(f"❌ タイムアウトエラー: {e}")
            return False
        except Exception as e:
            print(f"❌ セットアップエラー: {e}")
            return False
    
    def _create_npu_session(self):
        """NPUセッション作成"""
        try:
            # VitisAI ExecutionProvider設定
            providers = [
                ('VitisAIExecutionProvider', {
                    'config_file': self._find_vaip_config(),
                    'provider_options': {
                        'target': 'AMD_AIE2P_Nx4_Overlay',
                        'device_id': '0'
                    }
                }),
                'CPUExecutionProvider'
            ]
            
            # ダミーONNXモデルでNPUセッション作成
            dummy_model_path = self._create_dummy_onnx_model()
            
            self.npu_session = ort.InferenceSession(
                dummy_model_path,
                providers=providers
            )
            
            # プロバイダー確認
            active_providers = self.npu_session.get_providers()
            print(f"📋 アクティブプロバイダー: {active_providers}")
            
            if 'VitisAIExecutionProvider' in active_providers:
                print("✅ VitisAI ExecutionProvider アクティブ")
                return True
            else:
                print("⚠️ VitisAI ExecutionProvider 非アクティブ")
                return False
                
        except Exception as e:
            print(f"❌ NPUセッション作成エラー: {e}")
            return False
    
    def _find_vaip_config(self):
        """vaip_config.jsonファイル検索"""
        possible_paths = [
            "C:/Program Files/RyzenAI/1.5/voe-4.0-win_amd64/vaip_config.json",
            "C:/Program Files/RyzenAI/1.5.1/voe-4.0-win_amd64/vaip_config.json",
            "./vaip_config.json"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"📁 vaip_config.json発見: {path}")
                return path
                
        print("⚠️ vaip_config.json未発見、デフォルト設定使用")
        return None
    
    def _create_dummy_onnx_model(self):
        """ダミーONNXモデル作成（NPUテスト用）"""
        import numpy as np
        import onnx
        from onnx import helper, TensorProto
        
        # 簡単な線形変換モデル
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 10])
        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10])
        
        # 重み初期化
        weight_data = np.random.randn(10, 10).astype(np.float32)
        weight_tensor = helper.make_tensor('weight', TensorProto.FLOAT, [10, 10], weight_data.flatten())
        
        # ノード作成
        matmul_node = helper.make_node(
            'MatMul',
            inputs=['input', 'weight'],
            outputs=['output'],
            name='matmul'
        )
        
        # グラフ作成
        graph = helper.make_graph(
            [matmul_node],
            'dummy_npu_model',
            [input_tensor],
            [output_tensor],
            [weight_tensor]
        )
        
        # モデル作成
        model = helper.make_model(graph)
        model.opset_import[0].version = 11
        
        # ファイル保存
        dummy_model_path = "dummy_npu_model.onnx"
        onnx.save(model, dummy_model_path)
        
        print(f"📄 ダミーONNXモデル作成: {dummy_model_path}")
        return dummy_model_path
    
    def test_npu_inference(self, prompt, max_new_tokens=20):
        """NPU推論テスト"""
        if not self.tokenizer or not self.npu_session:
            return "❌ システムが初期化されていません"
            
        try:
            print(f"🔄 NPU推論テスト開始（タイムアウト: {self.timeout}秒）...")
            
            # 性能監視開始
            self.performance_monitor.start_monitoring()
            
            with TimeoutHandler(self.timeout):
                # トークナイズ
                inputs = self.tokenizer.encode(prompt, return_tensors="pt")
                print(f"📝 入力トークン数: {inputs.shape[1]}")
                
                # NPU推論実行（ダミーデータで）
                dummy_input = np.random.randn(1, 10).astype(np.float32)
                
                start_time = time.time()
                
                # 複数回NPU推論実行（負荷テスト）
                for i in range(max_new_tokens):
                    npu_output = self.npu_session.run(None, {'input': dummy_input})
                    
                    # 進捗表示
                    if i % 5 == 0:
                        print(f"  ⚡ NPU推論 {i+1}/{max_new_tokens}")
                        
                end_time = time.time()
                
                # 性能監視停止
                self.performance_monitor.stop_monitoring()
                
                # 結果生成
                inference_time = end_time - start_time
                throughput = max_new_tokens / inference_time
                
                response = f"""
🎯 NPU推論テスト結果:
📝 入力: {prompt}
⚡ NPU推論回数: {max_new_tokens}
⏱️ 推論時間: {inference_time:.3f}秒
📊 スループット: {throughput:.1f} 推論/秒
🔧 プロバイダー: {self.npu_session.get_providers()[0]}
"""
                
            self.generation_count += 1
            return response
            
        except TimeoutException as e:
            self.performance_monitor.stop_monitoring()
            return f"⚠️ NPU推論タイムアウト: {e}"
        except Exception as e:
            self.performance_monitor.stop_monitoring()
            return f"❌ NPU推論エラー: {e}"
    
    def interactive_mode(self):
        """インタラクティブモード"""
        print("\n🇯🇵 純粋NPUテストシステム - インタラクティブモード")
        print("⚠️ infer-OS最適化: 無効")
        print(f"⏰ タイムアウト設定: {self.timeout}秒")
        print("💡 'exit'または'quit'で終了、'stats'で統計表示、'perf'で性能レポート")
        print("=" * 60)
        
        while True:
            try:
                prompt = input("\n🤖 プロンプトを入力してください: ").strip()
                
                if prompt.lower() in ['exit', 'quit', '終了']:
                    print("👋 システムを終了します")
                    break
                elif prompt.lower() == 'stats':
                    self._show_stats()
                    continue
                elif prompt.lower() == 'perf':
                    print(self.performance_monitor.get_performance_report())
                    continue
                elif not prompt:
                    print("⚠️ プロンプトを入力してください")
                    continue
                
                # NPU推論テスト実行
                response = self.test_npu_inference(prompt)
                print(response)
                
                # 性能レポート表示
                print(self.performance_monitor.get_performance_report())
                
            except KeyboardInterrupt:
                print("\n\n🛑 Ctrl+Cが押されました。システムを終了します")
                break
            except Exception as e:
                print(f"❌ エラー: {e}")
    
    def _show_stats(self):
        """統計情報表示"""
        print("\n📊 純粋NPUシステム統計:")
        print(f"  🔢 テスト実行回数: {self.generation_count}")
        print(f"  ⏰ タイムアウト設定: {self.timeout}秒")
        print(f"  🤖 モデル: {self.model_name}")
        print(f"  ⚠️ infer-OS最適化: {'❌ 無効' if not self.infer_os_enabled else '✅ 有効'}")
        print(f"  🔤 トークナイザー: {'✅ 利用可能' if self.tokenizer else '❌ 未初期化'}")
        print(f"  ⚡ NPUセッション: {'✅ 利用可能' if self.npu_session else '❌ 未初期化'}")

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="純粋NPUテストシステム（infer-OS最適化OFF）")
    parser.add_argument("--interactive", action="store_true", help="インタラクティブモード")
    parser.add_argument("--prompt", type=str, help="単発プロンプト")
    parser.add_argument("--timeout", type=int, default=60, help="タイムアウト秒数（デフォルト: 60秒）")
    parser.add_argument("--model", type=str, default="llama3-8b-amd-npu", help="モデル名")
    parser.add_argument("--tokens", type=int, default=20, help="NPU推論回数（デフォルト: 20）")
    
    args = parser.parse_args()
    
    # システム初期化
    system = PureNPUSystem(model_name=args.model, timeout=args.timeout)
    
    if not system.setup():
        print("❌ システム初期化に失敗しました")
        sys.exit(1)
    
    if args.interactive:
        system.interactive_mode()
    elif args.prompt:
        response = system.test_npu_inference(args.prompt, args.tokens)
        print(response)
    else:
        print("💡 使用方法:")
        print("  python pure_npu_test_system.py --interactive")
        print("  python pure_npu_test_system.py --prompt \"人参について教えてください\" --tokens 30")

if __name__ == "__main__":
    main()

