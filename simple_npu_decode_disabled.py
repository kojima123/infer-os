"""
シンプルNPUデコーダー（無効化版）
ダミー処理を削除し、真のNPU処理のみを実行
"""

import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto
import time
from typing import Dict, Any, Optional, Tuple

class SimpleNPUDecoder:
    """シンプルNPUデコーダー（無効化版）"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.npu_session = None
        # セットアップを無効化
        print("🚫 シンプルNPUデコーダーは無効化されています")
        print("💡 真のNPU処理は修正版メインファイルで実行されます")
    
    def setup_npu(self):
        """NPUセットアップ（無効化）"""
        print("🚫 シンプルNPUデコーダーのセットアップはスキップされました")
        print("💡 ダミー処理による偽のNPU負荷を防ぐため無効化")
        return False
    
    def decode_with_npu(self, hidden_state: np.ndarray, temperature: float = 0.7) -> Tuple[int, float]:
        """NPUデコード（無効化）"""
        print("🚫 シンプルNPUデコードは無効化されています")
        print("💡 真のNPU処理を使用してください")
        return None, 0.0
    
    def create_simple_onnx_model(self):
        """ONNXモデル作成（無効化）"""
        print("🚫 ダミーONNXモデル作成は無効化されています")
        pass
    
    def load_npu_config(self) -> int:
        """NPU設定読み込み（無効化）"""
        print("🚫 NPU設定読み込みは無効化されています")
        return 0

def _setup_npu_decode_integration(demo_instance):
    """NPUデコード統合セットアップ（無効化版）"""
    print("🚫 シンプルNPUデコード統合は無効化されています")
    print("💡 理由: ダミー処理による偽のNPU負荷を防ぐため")
    print("✅ 真のNPU処理は修正版メインファイルで実行されます")
    
    # ダミーのNPUデコーダーを設定（実際には何もしない）
    demo_instance.simple_npu_decoder = None
    
    print("🎯 シンプルNPUデコーダー統合: 無効化完了")
    print("📊 NPU利用率: 0% (ダミー処理なし)")
    print("🔧 処理モード: 無効化")

