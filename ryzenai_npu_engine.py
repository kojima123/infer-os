#!/usr/bin/env python3
"""
RyzenAI 1.5.1 NPU推論エンジン
AMD Ryzen AI 9 365専用NPU推論実装
"""

import os
import sys
import time
import json
import numpy as np
from typing import Optional, List, Dict, Any
import logging

# RyzenAI 1.5.1 SDK インポート
try:
    # RyzenAI SDK の主要モジュール
    import ryzenai
    from ryzenai import ops
    from ryzenai.runtime import Runtime
    from ryzenai.quantization import Quantizer
    from ryzenai.optimization import Optimizer
    RYZENAI_AVAILABLE = True
    print("✅ RyzenAI 1.5.1 SDK インポート成功")
except ImportError as e:
    RYZENAI_AVAILABLE = False
    print(f"⚠️ RyzenAI SDK インポートエラー: {e}")
    print("💡 RyzenAI 1.5.1 SDKをインストールしてください")

class RyzenAINPUEngine:
    """RyzenAI 1.5.1 NPU推論エンジン"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        RyzenAI NPU推論エンジン初期化
        
        Args:
            model_path: モデルファイルパス（オプション）
        """
        self.model_path = model_path
        self.runtime = None
        self.quantizer = None
        self.optimizer = None
        self.npu_available = False
        self.performance_stats = {}
        
        # ログ設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        print("🚀 RyzenAI 1.5.1 NPU推論エンジン初期化開始")
        
        if RYZENAI_AVAILABLE:
            self._initialize_ryzenai()
        else:
            print("❌ RyzenAI SDK が利用できません")
    
    def _initialize_ryzenai(self):
        """RyzenAI SDK初期化"""
        try:
            print("🔧 RyzenAI SDK初期化中...")
            
            # NPUデバイス確認
            available_devices = ryzenai.get_available_devices()
            print(f"📱 利用可能デバイス: {len(available_devices)}個")
            
            npu_devices = [dev for dev in available_devices if 'NPU' in dev.get('type', '').upper()]
            
            if npu_devices:
                print(f"🎯 NPUデバイス発見: {len(npu_devices)}個")
                for i, device in enumerate(npu_devices):
                    print(f"  📱 NPU {i}: {device.get('name', 'Unknown')}")
                
                # 最初のNPUデバイスを使用
                self.npu_device = npu_devices[0]
                print(f"✅ 使用NPUデバイス: {self.npu_device.get('name', 'Unknown')}")
                
                # RyzenAI Runtime初期化
                self._initialize_runtime()
                
            else:
                print("❌ NPUデバイスが見つかりません")
                self._list_device_details(available_devices)
                
        except Exception as e:
            print(f"❌ RyzenAI SDK初期化エラー: {e}")
            self.logger.error(f"RyzenAI初期化失敗: {e}")
    
    def _initialize_runtime(self):
        """RyzenAI Runtime初期化"""
        try:
            print("⚡ RyzenAI Runtime初期化中...")
            
            # Runtime設定
            runtime_config = {
                'device': self.npu_device,
                'precision': 'int8',  # NPU最適化のためint8量子化
                'optimization_level': 'high',
                'memory_optimization': True,
                'batch_size': 1,
            }
            
            # Runtime作成
            self.runtime = Runtime(config=runtime_config)
            
            # Quantizer初期化（int8量子化）
            self.quantizer = Quantizer(
                precision='int8',
                calibration_method='entropy',
                optimization_target='npu'
            )
            
            # Optimizer初期化
            self.optimizer = Optimizer(
                target='npu',
                optimization_level='aggressive',
                memory_optimization=True
            )
            
            print("✅ RyzenAI Runtime初期化成功")
            self.npu_available = True
            
            # NPU性能テスト
            self._npu_performance_test()
            
        except Exception as e:
            print(f"❌ RyzenAI Runtime初期化エラー: {e}")
            self.logger.error(f"Runtime初期化失敗: {e}")
    
    def _npu_performance_test(self):
        """NPU性能テスト"""
        try:
            print("🧪 NPU性能テスト実行中...")
            
            # テスト用データ作成
            test_input = np.random.randn(1, 512, 4096).astype(np.float32)
            
            # NPU推論テスト
            start_time = time.time()
            
            # RyzenAI ops使用
            linear_op = ops.Linear(
                in_features=4096,
                out_features=4096,
                device=self.npu_device,
                precision='int8'
            )
            
            # NPU実行
            for i in range(10):
                output = linear_op(test_input)
                if i % 3 == 0:
                    print(f"  🔄 NPU性能テスト {i+1}/10")
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 性能統計
            self.performance_stats = {
                'npu_execution_time': execution_time,
                'iterations': 10,
                'avg_time_per_iteration': execution_time / 10,
                'throughput': 10 / execution_time,
                'input_shape': test_input.shape,
                'output_shape': output.shape if 'output' in locals() else None,
                'device': self.npu_device.get('name', 'Unknown'),
                'precision': 'int8',
                'success': True
            }
            
            print("✅ NPU性能テスト完了")
            print(f"  ⏱️ 実行時間: {execution_time:.3f}秒")
            print(f"  🚀 スループット: {self.performance_stats['throughput']:.1f}回/秒")
            print(f"  📊 平均実行時間: {self.performance_stats['avg_time_per_iteration']*1000:.1f}ms/回")
            
        except Exception as e:
            print(f"❌ NPU性能テストエラー: {e}")
            self.performance_stats = {'success': False, 'error': str(e)}
    
    def _list_device_details(self, devices: List[Dict]):
        """デバイス詳細一覧表示"""
        print("📋 利用可能デバイス詳細:")
        for i, device in enumerate(devices):
            print(f"  {i}: {device}")
    
    def load_model(self, model_path: str) -> bool:
        """
        モデルロード
        
        Args:
            model_path: モデルファイルパス
            
        Returns:
            bool: ロード成功フラグ
        """
        if not self.npu_available:
            print("❌ NPUが利用できません")
            return False
        
        try:
            print(f"📦 モデルロード開始: {model_path}")
            
            # モデルファイル確認
            if not os.path.exists(model_path):
                print(f"❌ モデルファイルが見つかりません: {model_path}")
                return False
            
            # RyzenAI用モデル最適化
            print("🔧 RyzenAI用モデル最適化中...")
            
            # 量子化
            quantized_model = self.quantizer.quantize(model_path)
            
            # 最適化
            optimized_model = self.optimizer.optimize(quantized_model)
            
            # Runtime にロード
            self.runtime.load_model(optimized_model)
            
            print("✅ モデルロード完了")
            return True
            
        except Exception as e:
            print(f"❌ モデルロードエラー: {e}")
            self.logger.error(f"モデルロード失敗: {e}")
            return False
    
    def inference(self, input_data: np.ndarray) -> Optional[np.ndarray]:
        """
        NPU推論実行
        
        Args:
            input_data: 入力データ
            
        Returns:
            Optional[np.ndarray]: 推論結果
        """
        if not self.npu_available:
            print("❌ NPUが利用できません")
            return None
        
        try:
            print("⚡ RyzenAI NPU推論実行中...")
            
            start_time = time.time()
            
            # NPU推論実行
            output = self.runtime.run(input_data)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            print(f"✅ NPU推論完了: {execution_time:.3f}秒")
            print(f"  📊 入力形状: {input_data.shape}")
            print(f"  📊 出力形状: {output.shape}")
            
            return output
            
        except Exception as e:
            print(f"❌ NPU推論エラー: {e}")
            self.logger.error(f"NPU推論失敗: {e}")
            return None
    
    def get_npu_status(self) -> Dict[str, Any]:
        """NPU状態取得"""
        return {
            'npu_available': self.npu_available,
            'device': self.npu_device.get('name', 'Unknown') if hasattr(self, 'npu_device') else None,
            'performance_stats': self.performance_stats,
            'ryzenai_version': getattr(ryzenai, '__version__', 'Unknown') if RYZENAI_AVAILABLE else None,
            'sdk_available': RYZENAI_AVAILABLE
        }
    
    def create_simple_llm_inference(self, vocab_size: int = 32000, hidden_dim: int = 4096):
        """シンプルなLLM推論レイヤー作成"""
        if not self.npu_available:
            print("❌ NPUが利用できません")
            return None
        
        try:
            print("🔧 RyzenAI LLM推論レイヤー作成中...")
            
            # Linear層（言語モデルヘッド）
            lm_head = ops.Linear(
                in_features=hidden_dim,
                out_features=vocab_size,
                device=self.npu_device,
                precision='int8'
            )
            
            # RMSNorm層
            rms_norm = ops.RMSNorm(
                normalized_shape=hidden_dim,
                device=self.npu_device,
                precision='int8'
            )
            
            print("✅ RyzenAI LLM推論レイヤー作成完了")
            
            return {
                'lm_head': lm_head,
                'rms_norm': rms_norm,
                'vocab_size': vocab_size,
                'hidden_dim': hidden_dim
            }
            
        except Exception as e:
            print(f"❌ LLM推論レイヤー作成エラー: {e}")
            return None

def main():
    """メイン実行関数"""
    print("🎯 RyzenAI 1.5.1 NPU推論エンジンテスト")
    print("=" * 50)
    
    # RyzenAI NPU推論エンジン初期化
    engine = RyzenAINPUEngine()
    
    # NPU状態確認
    status = engine.get_npu_status()
    print("\n📊 NPU状態:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    if engine.npu_available:
        print("\n🎉 RyzenAI NPU推論エンジン準備完了！")
        
        # シンプルLLM推論テスト
        llm_layers = engine.create_simple_llm_inference()
        
        if llm_layers:
            print("\n🚀 LLM推論レイヤーテスト実行中...")
            
            # テスト用隠れ状態
            hidden_state = np.random.randn(1, 4096).astype(np.float32)
            
            try:
                # RMSNorm実行
                normalized = llm_layers['rms_norm'](hidden_state)
                print(f"  ✅ RMSNorm実行成功: {normalized.shape}")
                
                # Linear層実行
                logits = llm_layers['lm_head'](normalized)
                print(f"  ✅ Linear層実行成功: {logits.shape}")
                
                print("🎉 RyzenAI LLM推論テスト成功！")
                
            except Exception as e:
                print(f"❌ LLM推論テストエラー: {e}")
    else:
        print("\n❌ RyzenAI NPU推論エンジンが利用できません")
        print("💡 RyzenAI 1.5.1 SDKのインストールを確認してください")

if __name__ == "__main__":
    main()

