# NumPyインポートエラー解決ガイド

Infer-OS NPUテスト実行時のNumPyエラー完全解決ガイド

## 🚨 発生しているエラー

```
ImportError: Error importing numpy: you should not try to import numpy from
        its source directory; please exit the numpy source tree, and relaunch
        your python interpreter from there.

ModuleNotFoundError: No module named 'numpy._core._multiarray_umath'
```

## 🎯 エラーの原因

1. **NumPyバージョン競合**: NumPy 2.1.2とPython 3.12の互換性問題
2. **仮想環境の依存関係破損**: パッケージインストール時の競合
3. **C拡張モジュールの破損**: numpy._core._multiarray_umathが見つからない
4. **ディレクトリ構造問題**: NumPyソースディレクトリからのインポート試行

## 🔧 解決方法

### 方法1: NumPy完全再インストール（推奨）

```powershell
# 現在の仮想環境で実行
# 破損したNumPyを完全削除
pip uninstall numpy -y

# キャッシュクリア
pip cache purge

# 安定版NumPyを再インストール
pip install numpy==1.24.3

# 依存関係確認
pip install scipy pandas matplotlib

# インストール確認
python -c "import numpy as np; print('NumPy version:', np.__version__); print('NumPy working:', np.array([1,2,3]))"
```

### 方法2: 仮想環境完全再構築

```powershell
# 現在の仮想環境を無効化
deactivate

# 古い仮想環境削除
Remove-Item -Recurse -Force "C:\infer-os-test\infer-os\venv"

# 新しい仮想環境作成
cd "C:\infer-os-test\infer-os"
python -m venv venv

# 新しい仮想環境有効化
.\venv\Scripts\Activate.ps1

# 基本パッケージインストール
pip install --upgrade pip
pip install numpy==1.24.3 scipy pandas matplotlib requests psutil
```

### 方法3: Conda環境使用（Ryzen AI Softwareと統合）

```powershell
# Condaが利用可能な場合
conda create -n infer_os_npu python=3.11
conda activate infer_os_npu

# 安定版パッケージインストール
conda install numpy=1.24.3 scipy pandas matplotlib
pip install requests psutil

# Infer-OS依存関係
pip install torch onnxruntime
```

## 🚀 推奨解決手順（段階的アプローチ）

### Step 1: クイック修正（5分）

```powershell
# 現在の仮想環境で実行
pip uninstall numpy -y
pip install numpy==1.24.3

# テスト
python -c "import numpy as np; print('NumPy OK:', np.__version__)"
```

### Step 2: 依存関係修復（10分）

```powershell
# 関連パッケージも再インストール
pip uninstall numpy scipy pandas matplotlib -y
pip install numpy==1.24.3 scipy==1.10.1 pandas==2.0.3 matplotlib==3.7.2

# Infer-OS依存関係
pip install torch onnxruntime requests psutil
```

### Step 3: 完全環境再構築（15分）

```powershell
# 仮想環境完全再作成
deactivate
Remove-Item -Recurse -Force venv
python -m venv venv
.\venv\Scripts\Activate.ps1

# 必要パッケージのみインストール
pip install --upgrade pip
pip install numpy==1.24.3 scipy pandas matplotlib torch onnxruntime requests psutil
```

## 🧪 動作確認手順

### 基本確認

```powershell
# NumPy動作確認
python -c "import numpy as np; print('NumPy version:', np.__version__); print('Test array:', np.array([1,2,3]))"

# 依存関係確認
python -c "import scipy, pandas, matplotlib; print('All packages OK')"

# Infer-OS依存関係確認
python -c "import torch, onnxruntime; print('ML packages OK')"
```

### Infer-OSテスト実行

```powershell
# 基本テスト
python infer_os_npu_test.py --mode basic

# 包括的テスト（NumPy修復後）
python infer_os_npu_test.py --mode comprehensive
```

## 🔍 トラブルシューティング

### 問題1: pip uninstallでもエラー

```powershell
# 強制削除
pip uninstall numpy --yes --break-system-packages

# または手動削除
Remove-Item -Recurse -Force "venv\Lib\site-packages\numpy*"
pip install numpy==1.24.3
```

### 問題2: 他のパッケージとの競合

```powershell
# 競合パッケージ確認
pip list | Select-String "numpy\|scipy\|pandas"

# 競合解決
pip install --force-reinstall numpy==1.24.3 scipy==1.10.1
```

### 問題3: Python 3.12互換性問題

```powershell
# Python 3.11使用を推奨
python --version

# 必要に応じてPython 3.11環境作成
conda create -n infer_os_py311 python=3.11
conda activate infer_os_py311
```

## 📊 バージョン互換性表

| Python | NumPy | SciPy | 推奨度 | 備考 |
|--------|-------|-------|--------|------|
| 3.12 | 1.24.3 | 1.10.1 | ⭐⭐⭐ | 安定 |
| 3.12 | 2.1.2 | 最新 | ⭐ | 不安定 |
| 3.11 | 1.24.3 | 1.10.1 | ⭐⭐⭐⭐⭐ | 最推奨 |
| 3.11 | 2.0.x | 最新 | ⭐⭐⭐⭐ | 良好 |

## 💡 予防策

### 安定した環境維持

```powershell
# requirements.txt作成
pip freeze > requirements.txt

# 特定バージョン固定
echo "numpy==1.24.3" > requirements_stable.txt
echo "scipy==1.10.1" >> requirements_stable.txt
echo "pandas==2.0.3" >> requirements_stable.txt
echo "matplotlib==3.7.2" >> requirements_stable.txt

# 安定環境復元
pip install -r requirements_stable.txt
```

### 定期的な環境確認

```powershell
# 週次確認スクリプト
python -c "
import numpy as np
import scipy
import pandas as pd
import matplotlib
print('Environment Check:')
print(f'NumPy: {np.__version__}')
print(f'SciPy: {scipy.__version__}')
print(f'Pandas: {pd.__version__}')
print(f'Matplotlib: {matplotlib.__version__}')
print('All packages working correctly!')
"
```

## 🎯 最終確認

### 成功の指標

```powershell
# 以下が全て成功すれば修復完了
python -c "import numpy as np; print('✅ NumPy:', np.__version__)"
python -c "import scipy; print('✅ SciPy OK')"
python -c "import pandas; print('✅ Pandas OK')"
python -c "import torch; print('✅ PyTorch OK')"
python -c "import onnxruntime; print('✅ ONNX Runtime OK')"

# Infer-OSテスト実行
python infer_os_npu_test.py --mode basic
```

### 期待される出力

```
✅ NumPy: 1.24.3
✅ SciPy OK
✅ Pandas OK
✅ PyTorch OK
✅ ONNX Runtime OK

🚀 Infer-OS NPUテスト開始...
✅ NPU検出: AMD Ryzen AI 9 365 w/ Radeon 880M
✅ 最適化機能: 全て有効
✅ テスト成功
```

## 🚀 次のステップ

NumPyエラー解決後：

1. **DirectMLインストール**
   ```powershell
   pip install torch-directml onnxruntime-directml
   ```

2. **統合最適化テスト**
   ```powershell
   python infer_os_npu_test.py --mode comprehensive
   ```

3. **性能ベンチマーク**
   ```powershell
   python benchmarks/integrated_performance_test.py
   ```

---

**結論**: NumPyエラーは仮想環境の依存関係問題です。NumPy 1.24.3への降格で確実に解決できます。

