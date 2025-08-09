# AMD NPU専用セットアップガイド

Windows 11 AMD環境でのInfer-OS NPU最適化セットアップ完全ガイド

## 🎯 対象環境

- **OS**: Windows 11 (22H2以降推奨)
- **CPU**: AMD Ryzen AI対応プロセッサー
  - Phoenix世代 (Ryzen 7040シリーズ)
  - Hawk Point世代 (Ryzen 8040シリーズ)
- **メモリ**: 16GB以上推奨
- **ストレージ**: 10GB以上の空き容量

## 🔍 NPU検出エラーの解決

### 問題の症状
```
=== NPU検出テスト ===
検出エラー: [WinError 2] 指定されたファイルが見つかりません。
```

### 解決方法

#### 1. AMD NPU専用検出ツールの使用

```bash
# リポジトリクローン
git clone https://github.com/kojima123/infer-os.git
cd infer-os

# AMD NPU専用検出ツール実行
python amd_npu_detector.py
```

#### 2. 修正版クイックテストの使用

```bash
# 修正版クイックテスト実行
python quick_test_amd_fixed.py
```

## 🛠️ 必須ソフトウェアのインストール

### 1. AMD Software (最重要)

**ダウンロード**: https://www.amd.com/support

1. AMD公式サイトにアクセス
2. 「ドライバーとサポート」を選択
3. プロセッサーまたはグラフィックスを選択
4. 最新のAMD Softwareをダウンロード・インストール

**確認方法**:
```powershell
# PowerShellで確認
Get-WmiObject -Class Win32_PnPEntity | Where-Object { $_.Name -match "AMD" }
```

### 2. AMD Ryzen AI Software (NPU最適化)

**入手方法**:
- AMD開発者サイト
- OEMメーカー（HP、Lenovo、ASUS等）のサポートページ

**インストール後の確認**:
```bash
# インストール確認
python amd_npu_detector.py
```

### 3. Python環境セットアップ

#### Python 3.9-3.11のインストール

```bash
# Python公式サイトからダウンロード・インストール
# https://www.python.org/downloads/

# バージョン確認
python --version
```

#### 必須パッケージのインストール

```bash
# 基本パッケージ
pip install numpy pandas matplotlib requests psutil

# PyTorch (CPU版)
pip install torch torchvision torchaudio

# PyTorch DirectML (AMD GPU/NPU対応)
pip install torch-directml

# ONNX Runtime DirectML
pip install onnxruntime-directml

# Flask (Webデモ用)
pip install flask

# その他
pip install beautifulsoup4 pillow
```

## 🔧 DirectML対応確認

### PyTorch DirectMLテスト

```python
import torch
import torch_directml

# DirectMLデバイス確認
device = torch_directml.device()
print(f"DirectMLデバイス: {device}")

# 簡単なテンソル演算
x = torch.randn(100, 100, device=device)
y = torch.randn(100, 100, device=device)
z = torch.mm(x, y)
print("DirectML演算成功!")
```

### ONNX Runtime DirectMLテスト

```python
import onnxruntime as ort

# 利用可能プロバイダー確認
providers = ort.get_available_providers()
print(f"利用可能プロバイダー: {providers}")

if 'DmlExecutionProvider' in providers:
    print("✅ DirectMLプロバイダー利用可能")
else:
    print("❌ DirectMLプロバイダー未対応")
```

## 🚀 Infer-OS セットアップ

### 1. リポジトリクローン

```bash
git clone https://github.com/kojima123/infer-os.git
cd infer-os
```

### 2. 依存関係インストール

```bash
# requirements.txtがある場合
pip install -r requirements.txt

# 手動インストール
pip install torch onnxruntime numpy flask requests psutil torch-directml onnxruntime-directml
```

### 3. 環境テスト

```bash
# AMD専用検出ツール
python amd_npu_detector.py

# 修正版クイックテスト
python quick_test_amd_fixed.py

# 統合テスト
python infer_os_npu_test.py --mode basic
```

## 📊 期待される結果

### 正常な検出結果例

```
=== AMD NPU検出結果 ===
✅ NPU検出: AMD Ryzen AI (Phoenix)
🎮 GPU検出: AMD Radeon Graphics
💻 CPU検出: AMD Ryzen 7 7840U

=== DirectML対応 ===
✅ torch-directml: 利用可能
✅ ONNX Runtime DirectML: 利用可能
✅ DirectMLデバイス: DirectML device (AMD Radeon Graphics)

=== 最適化効果 ===
- 総合高速化: 1.14-1.25x
- メモリ削減: 75% (KV量子化)
- NPU活用: 自動検出・最適化
```

## 🔧 トラブルシューティング

### 1. NPUが検出されない場合

**原因**: AMD Ryzen AI非対応プロセッサー
**解決策**: 
- プロセッサー仕様を確認
- Phoenix/Hawk Point世代のRyzen AIプロセッサーが必要

### 2. DirectMLが利用できない場合

**原因**: ドライバー不足またはパッケージ未インストール
**解決策**:
```bash
# AMD Softwareを最新版に更新
# DirectMLパッケージ再インストール
pip uninstall torch-directml onnxruntime-directml
pip install torch-directml onnxruntime-directml
```

### 3. PowerShell実行エラー

**原因**: PowerShell実行ポリシー制限
**解決策**:
```powershell
# 管理者権限でPowerShell実行
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 4. メモリ不足エラー

**原因**: システムメモリ不足
**解決策**:
- 16GB以上のメモリを推奨
- 他のアプリケーションを終了
- 仮想メモリ設定を増加

## 📈 性能最適化設定

### 1. AMD Software設定

1. AMD Softwareを起動
2. 「性能」タブを選択
3. 「GPU調整」で最大性能に設定
4. 「電源管理」で高性能モードに設定

### 2. Windows電源設定

1. 設定 → システム → 電源とバッテリー
2. 電源モードを「最適なパフォーマンス」に設定
3. 追加の電源設定 → 高パフォーマンスを選択

### 3. Infer-OS最適化設定

```python
# 設定例 (config.py)
OPTIMIZATION_CONFIG = {
    'kv_quantization': {
        'enabled': True,
        'method': 'int8',
        'compression_ratio': 4.0
    },
    'speculative_generation': {
        'enabled': True,
        'draft_length': 4,
        'acceptance_threshold': 0.8
    },
    'gpu_npu_pipeline': {
        'enabled': True,
        'device_allocation': 'auto'
    }
}
```

## 🧪 検証手順

### 1. 基本検証

```bash
# 1. AMD NPU検出
python amd_npu_detector.py

# 2. クイックテスト
python quick_test_amd_fixed.py

# 3. 基本統合テスト
python infer_os_npu_test.py --mode basic
```

### 2. 性能検証

```bash
# 包括的性能テスト
python infer_os_npu_test.py --mode comprehensive --iterations 10
```

### 3. Webデモ検証

```bash
# Webデモ起動
python infer-os-demo/src/main.py

# ブラウザでアクセス
# http://localhost:5000
```

## 📞 サポート情報

### AMD公式サポート
- **AMD サポート**: https://www.amd.com/support
- **AMD開発者フォーラム**: https://community.amd.com/

### Infer-OSサポート
- **GitHub Issues**: https://github.com/kojima123/infer-os/issues
- **ドキュメント**: プロジェクトREADME参照

### よくある質問

**Q: Ryzen 5000シリーズでNPUは使えますか？**
A: Ryzen 5000シリーズにはNPUが搭載されていません。Phoenix世代（7040シリーズ）以降が必要です。

**Q: DirectMLとCUDAの違いは？**
A: DirectMLはMicrosoft製のAMD/Intel GPU対応ライブラリ、CUDAはNVIDIA専用です。

**Q: NPU最適化の効果はどの程度ですか？**
A: 1.14-1.25倍の高速化と75%のメモリ削減が期待できます。

## 🎉 セットアップ完了確認

全ての手順が完了したら、以下のコマンドで最終確認を行ってください：

```bash
# 最終確認テスト
python amd_npu_detector.py
python quick_test_amd_fixed.py
python infer_os_npu_test.py --mode basic
```

成功すれば、AMD NPU環境でのInfer-OS最適化が利用可能になります！

---

*このガイドは Windows 11 AMD環境専用です。他の環境では標準のセットアップガイドをご利用ください。*

