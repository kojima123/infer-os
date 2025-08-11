# 🚀 真のNPU動作のための環境構築ガイド

## 🔍 現在の問題分析

### ❌ 現在の状況
- **DirectML**: GPU（統合グラフィックス）で実行中
- **VitisAI ExecutionProvider**: ❌ 未インストール
- **真のNPU**: 全く使用されていない
- **デバイス**: `CPU-DML`（NPUではない）

### ✅ 目標
- **VitisAI ExecutionProvider**: NPU専用プロバイダー有効化
- **Ryzen AI NPU**: 真のNPU処理実現
- **NPU負荷率**: タスクマネージャーで確認可能

---

## 🛠️ 真のNPU環境構築手順

### ステップ1: Ryzen AI SDK インストール

#### 1.1 前提条件確認
```powershell
# システム情報確認
systeminfo | findstr "Processor"
dxdiag
```

**必要な条件:**
- AMD Ryzen AI プロセッサー（7040/8040シリーズ以降）
- Windows 11 22H2以降
- 16GB以上のRAM

#### 1.2 AMD Ryzen AI SDK ダウンロード
```powershell
# AMD公式サイトからダウンロード
# https://www.amd.com/en/products/software/ryzen-ai.html
```

**ダウンロードファイル:**
- `ryzen-ai-sw-1.5.1.msi`（最新版）
- `ryzen-ai-sdk-1.5.1.zip`

#### 1.3 SDK インストール
```powershell
# MSIインストーラー実行
ryzen-ai-sw-1.5.1.msi

# 環境変数設定
set RYZEN_AI_INSTALLATION_PATH=C:\AMD\RyzenAI\1.5.1
set PATH=%PATH%;%RYZEN_AI_INSTALLATION_PATH%\bin
```

### ステップ2: VitisAI ExecutionProvider インストール

#### 2.1 専用ONNXRuntimeインストール
```powershell
# 現在のONNXRuntimeアンインストール
pip uninstall onnxruntime onnxruntime-directml

# VitisAI対応版インストール
pip install onnxruntime-vitisai
# または
pip install onnxruntime-vitisai-1.18.0-cp311-cp311-win_amd64.whl
```

#### 2.2 VitisAI依存関係インストール
```powershell
# VitisAI Python パッケージ
pip install vitis-ai-runtime
pip install xir
pip install vai_q_pytorch

# 追加依存関係
pip install protobuf==3.20.3
pip install numpy==1.24.3
```

### ステップ3: NPUドライバー更新

#### 3.1 AMD Adrenalin ドライバー
```powershell
# AMD公式サイトから最新ドライバーダウンロード
# https://www.amd.com/support/download-center
```

#### 3.2 NPU専用ドライバー
```powershell
# デバイスマネージャーで確認
# "Neural Processing Unit" または "AMD NPU" デバイス
```

### ステップ4: 環境確認

#### 4.1 VitisAI ExecutionProvider確認
```python
import onnxruntime as ort
providers = ort.get_available_providers()
print("VitisAIExecutionProvider" in providers)
```

**期待結果:** `True`

#### 4.2 NPUデバイス確認
```python
# NPUデバイス情報取得
import subprocess
result = subprocess.run(['xdputil', 'query'], capture_output=True, text=True)
print(result.stdout)
```

---

## 🔧 現在の環境での最善策

### 真のNPU環境構築が困難な場合の対応

#### アプローチ1: DirectML最適化
```python
# 現在のDirectMLを最大限活用
providers = [
    ('DmlExecutionProvider', {
        'device_id': 0,
        'enable_dynamic_graph_fusion': True,
        'enable_graph_optimization': True,
        'disable_memory_arena': False,
        'memory_limit_mb': 8192,
    })
]
```

#### アプローチ2: CPU並列処理最適化
```python
# CPU処理の並列化
session_options = ort.SessionOptions()
session_options.inter_op_num_threads = 8
session_options.intra_op_num_threads = 8
session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
```

#### アプローチ3: モデル量子化
```python
# INT8量子化でメモリ効率向上
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="model.onnx",
    model_output="model_quantized.onnx",
    weight_type=QuantType.QInt8
)
```

---

## 🎯 真のNPU動作確認方法

### 成功指標

#### 1. プロバイダー確認
```python
active_providers = session.get_providers()
assert 'VitisAIExecutionProvider' in active_providers
```

#### 2. デバイス情報
```
デバイス: NPU-0 (期待値)
現在: CPU-DML (問題)
```

#### 3. タスクマネージャー
- **NPU使用率**: 50-80%に上昇
- **GPU使用率**: 低下（NPUに移行）

#### 4. 処理速度
- **推論時間**: 大幅短縮
- **電力効率**: 向上

---

## 🚨 トラブルシューティング

### よくある問題

#### 問題1: VitisAI ExecutionProvider未検出
```powershell
# 解決策
pip install --force-reinstall onnxruntime-vitisai
set PYTHONPATH=%PYTHONPATH%;%RYZEN_AI_INSTALLATION_PATH%\python
```

#### 問題2: NPUデバイス未認識
```powershell
# デバイスマネージャーで確認
devmgmt.msc

# ドライバー再インストール
# AMD公式サイトから最新版ダウンロード
```

#### 問題3: 権限エラー
```powershell
# 管理者権限でPowerShell実行
# UAC設定確認
```

---

## 📊 性能比較

### 期待される改善

| 項目 | DirectML | 真のNPU | 改善率 |
|------|----------|---------|--------|
| 推論速度 | 1.0x | 3-5x | 300-500% |
| 電力効率 | 1.0x | 2-3x | 200-300% |
| 並列処理 | 制限あり | 最適化 | 大幅向上 |
| メモリ使用 | 高い | 効率的 | 30-50%削減 |

---

## 💡 重要なポイント

### 1. ハードウェア要件
- **必須**: AMD Ryzen AI プロセッサー
- **推奨**: 32GB RAM、NVMe SSD

### 2. ソフトウェア要件
- **OS**: Windows 11 22H2以降
- **Python**: 3.8-3.11
- **ONNX**: 1.14以降

### 3. 開発環境
- **IDE**: Visual Studio Code推奨
- **デバッグ**: AMD ROCm Profiler
- **監視**: AMD Ryzen Master

---

## 🔗 参考リンク

- [AMD Ryzen AI 公式サイト](https://www.amd.com/en/products/software/ryzen-ai.html)
- [VitisAI Documentation](https://xilinx.github.io/Vitis-AI/)
- [ONNXRuntime VitisAI Provider](https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html)
- [AMD ROCm Platform](https://rocm.docs.amd.com/)

---

## 📞 サポート

### 技術サポート
- **AMD Developer Support**: developer.amd.com
- **GitHub Issues**: github.com/microsoft/onnxruntime
- **Community Forum**: community.amd.com

**真のNPU動作には、適切な環境構築が不可欠です。**

