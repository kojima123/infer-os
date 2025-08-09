# DirectML インストールガイド

AMD GPU/NPU最適化のためのDirectMLセットアップ完全ガイド

## 🎯 DirectMLとは

DirectMLは、MicrosoftのWindows Machine Learning (WinML) の一部として開発された、DirectX 12上で動作する機械学習ライブラリです。AMD、Intel、NVIDIAのGPUに対応し、特にAMD環境でのNPU/GPU最適化に重要な役割を果たします。

### 主な特徴
- **クロスベンダー対応**: AMD、Intel、NVIDIA GPU対応
- **NPU最適化**: AMD Ryzen AI NPUの活用
- **高性能**: DirectX 12による低レベル最適化
- **PyTorch/ONNX Runtime統合**: 既存のMLフレームワークとの連携

## 📋 前提条件

### システム要件
- **OS**: Windows 10 (1903以降) または Windows 11
- **DirectX**: DirectX 12対応GPU
- **Python**: 3.8-3.11 (3.9-3.11推奨)
- **メモリ**: 8GB以上 (16GB推奨)

### 対応ハードウェア
- **AMD**: Radeon RX 5000シリーズ以降、Ryzen AI NPU
- **Intel**: Arc GPU、統合GPU (Iris Xe以降)
- **NVIDIA**: GTX 1060以降、RTX全シリーズ

## 🚀 インストール手順

### Step 1: Python環境確認

```bash
# Pythonバージョン確認
python --version

# pipアップデート
python -m pip install --upgrade pip
```

### Step 2: PyTorch DirectMLインストール

```bash
# PyTorch DirectMLインストール
pip install torch-directml

# インストール確認
python -c "import torch_directml; print('DirectML available:', torch_directml.is_available())"
```

### Step 3: ONNX Runtime DirectMLインストール

```bash
# ONNX Runtime DirectMLインストール
pip install onnxruntime-directml

# インストール確認
python -c "import onnxruntime as ort; print('DirectML Provider:', 'DmlExecutionProvider' in ort.get_available_providers())"
```

### Step 4: 動作確認

```python
# DirectML動作確認スクリプト
import torch
import torch_directml
import onnxruntime as ort

print("=== DirectML動作確認 ===")

# PyTorch DirectML確認
print(f"torch-directml available: {torch_directml.is_available()}")
if torch_directml.is_available():
    device = torch_directml.device()
    print(f"DirectML device: {device}")
    
    # 簡単なテンソル演算
    x = torch.randn(100, 100, device=device)
    y = torch.randn(100, 100, device=device)
    z = torch.mm(x, y)
    print("PyTorch DirectML演算成功!")

# ONNX Runtime DirectML確認
providers = ort.get_available_providers()
print(f"ONNX Runtime providers: {providers}")
if 'DmlExecutionProvider' in providers:
    print("ONNX Runtime DirectML利用可能!")
else:
    print("ONNX Runtime DirectML未対応")
```

## 🔧 トラブルシューティング

### 問題1: torch-directmlインストールエラー

**症状**:
```
ERROR: Could not find a version that satisfies the requirement torch-directml
```

**解決策**:
```bash
# PyTorchを先にインストール
pip install torch torchvision torchaudio

# その後torch-directmlをインストール
pip install torch-directml
```

### 問題2: DirectMLデバイスが見つからない

**症状**:
```python
RuntimeError: DirectML device not found
```

**解決策**:
1. **GPU ドライバー更新**:
   - AMD: https://www.amd.com/support
   - Intel: https://www.intel.com/content/www/us/en/support/
   - NVIDIA: https://www.nvidia.com/drivers/

2. **DirectX 12確認**:
   ```bash
   # DirectX診断ツール実行
   dxdiag
   ```

3. **Windows更新**:
   - 設定 → Windows Update → 更新プログラムのチェック

### 問題3: ONNX Runtime DirectMLプロバイダーが利用できない

**症状**:
```python
'DmlExecutionProvider' not in ort.get_available_providers()
```

**解決策**:
```bash
# ONNX Runtime完全アンインストール
pip uninstall onnxruntime onnxruntime-directml

# DirectML版のみ再インストール
pip install onnxruntime-directml
```

### 問題4: パフォーマンスが期待より低い

**解決策**:
1. **電源設定を高性能モードに変更**
2. **AMD Software設定**:
   - AMD Software起動
   - 性能 → GPU調整 → 最大性能
3. **Windows グラフィック設定**:
   - 設定 → システム → ディスプレイ → グラフィックの設定
   - アプリを高性能GPUに割り当て

## 📊 性能ベンチマーク

### ベンチマークスクリプト

```python
import time
import torch
import torch_directml
import numpy as np

def benchmark_directml():
    print("=== DirectML性能ベンチマーク ===")
    
    # CPU vs DirectML比較
    sizes = [500, 1000, 2000]
    
    for size in sizes:
        print(f"\n行列サイズ: {size}x{size}")
        
        # CPU演算
        start_time = time.time()
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        c_cpu = torch.mm(a_cpu, b_cpu)
        cpu_time = time.time() - start_time
        print(f"CPU時間: {cpu_time:.4f}秒")
        
        # DirectML演算
        if torch_directml.is_available():
            device = torch_directml.device()
            
            start_time = time.time()
            a_dml = torch.randn(size, size, device=device)
            b_dml = torch.randn(size, size, device=device)
            c_dml = torch.mm(a_dml, b_dml)
            # 結果をCPUに戻す（同期のため）
            c_dml_cpu = c_dml.cpu()
            dml_time = time.time() - start_time
            
            speedup = cpu_time / dml_time
            print(f"DirectML時間: {dml_time:.4f}秒")
            print(f"高速化: {speedup:.2f}x")
        else:
            print("DirectML利用不可")

if __name__ == "__main__":
    benchmark_directml()
```

## 🎯 最適化設定

### PyTorch DirectML最適化

```python
import torch
import torch_directml

# DirectMLデバイス設定
device = torch_directml.device()

# メモリ効率化
torch.backends.directml.allow_reduced_precision = True

# 自動混合精度（AMP）使用
from torch.cuda.amp import autocast, GradScaler

# DirectMLでもAMPが使用可能
scaler = GradScaler()

def optimized_training_step(model, data, target):
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### ONNX Runtime DirectML最適化

```python
import onnxruntime as ort

# DirectML最適化設定
session_options = ort.SessionOptions()
session_options.enable_mem_pattern = False
session_options.enable_mem_reuse = False

# DirectMLプロバイダー設定
providers = [
    ('DmlExecutionProvider', {
        'device_id': 0,
        'enable_dynamic_graph_fusion': True,
    }),
    'CPUExecutionProvider'
]

# セッション作成
session = ort.InferenceSession(
    model_path,
    sess_options=session_options,
    providers=providers
)
```

## 🔍 検証方法

### 完全検証スクリプト

```python
#!/usr/bin/env python3
"""DirectML完全検証スクリプト"""

import sys
import time
import torch
import torch_directml
import onnxruntime as ort
import numpy as np

def test_pytorch_directml():
    """PyTorch DirectMLテスト"""
    print("=== PyTorch DirectML テスト ===")
    
    if not torch_directml.is_available():
        print("❌ DirectML利用不可")
        return False
    
    try:
        device = torch_directml.device()
        print(f"✅ DirectMLデバイス: {device}")
        
        # 基本演算テスト
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        z = torch.mm(x, y)
        
        print("✅ 基本演算成功")
        
        # 性能テスト
        size = 1000
        start_time = time.time()
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        c = torch.mm(a, b)
        c_cpu = c.cpu()  # 同期
        elapsed = time.time() - start_time
        
        print(f"✅ 性能テスト: {size}x{size}行列演算 {elapsed:.4f}秒")
        return True
        
    except Exception as e:
        print(f"❌ PyTorch DirectMLエラー: {e}")
        return False

def test_onnx_directml():
    """ONNX Runtime DirectMLテスト"""
    print("\n=== ONNX Runtime DirectML テスト ===")
    
    providers = ort.get_available_providers()
    print(f"利用可能プロバイダー: {providers}")
    
    if 'DmlExecutionProvider' not in providers:
        print("❌ DirectMLプロバイダー利用不可")
        return False
    
    try:
        # 簡単なモデルでテスト
        session_options = ort.SessionOptions()
        test_providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        
        print("✅ DirectMLプロバイダー利用可能")
        return True
        
    except Exception as e:
        print(f"❌ ONNX Runtime DirectMLエラー: {e}")
        return False

def main():
    """メイン検証"""
    print("DirectML完全検証開始\n")
    
    # システム情報
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"ONNX Runtime: {ort.__version__}")
    
    # テスト実行
    pytorch_ok = test_pytorch_directml()
    onnx_ok = test_onnx_directml()
    
    # 結果サマリー
    print("\n=== 検証結果サマリー ===")
    print(f"PyTorch DirectML: {'✅ 成功' if pytorch_ok else '❌ 失敗'}")
    print(f"ONNX Runtime DirectML: {'✅ 成功' if onnx_ok else '❌ 失敗'}")
    
    if pytorch_ok and onnx_ok:
        print("\n🎉 DirectMLセットアップ完了!")
        print("Infer-OSでAMD GPU/NPU最適化が利用可能です。")
    else:
        print("\n⚠️ 一部問題があります。トラブルシューティングを確認してください。")

if __name__ == "__main__":
    main()
```

## 📚 参考資料

### 公式ドキュメント
- **DirectML**: https://docs.microsoft.com/en-us/windows/ai/directml/
- **PyTorch DirectML**: https://pytorch.org/blog/directml-backend/
- **ONNX Runtime DirectML**: https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html

### AMD関連リソース
- **AMD GPU開発者ガイド**: https://gpuopen.com/
- **Radeon Software**: https://www.amd.com/support/graphics/amd-radeon-6000-series

### トラブルシューティングリソース
- **DirectML GitHub**: https://github.com/microsoft/DirectML
- **PyTorch Issues**: https://github.com/pytorch/pytorch/issues

---

このガイドに従ってDirectMLをセットアップすることで、AMD GPU/NPU環境でのInfer-OS最適化が可能になります。

