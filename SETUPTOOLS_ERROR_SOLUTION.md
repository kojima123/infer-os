# setuptools/pipエラー解決ガイド

新しい仮想環境でのsetuptools関連エラー完全解決ガイド

## 🚨 発生しているエラー

```
pip._vendor.pyproject_hooks._impl.BackendUnavailable: Cannot import 'setuptools.build_meta'
```

## 🎯 エラーの原因

1. **setuptools未インストール**: 新しい仮想環境にsetuptoolsが含まれていない
2. **pip古いバージョン**: pip 25.0.1が古く、最新のsetuptoolsと互換性がない
3. **ビルド依存関係不足**: NumPyのソースからのビルドに必要な依存関係が不足
4. **Python 3.12環境問題**: 新しいPython環境での互換性問題

## 🔧 解決方法

### 方法1: 段階的インストール（推奨）

```powershell
# Step 1: 基本ツールのアップグレード
pip install --upgrade pip setuptools wheel

# Step 2: NumPyをバイナリ版でインストール（ソースビルド回避）
pip install --only-binary=all numpy==1.24.3

# Step 3: 他のパッケージを順次インストール
pip install scipy
pip install pandas
pip install matplotlib
pip install torch
pip install onnxruntime
pip install requests psutil
```

### 方法2: 事前ビルド版使用

```powershell
# 基本ツール更新
pip install --upgrade pip setuptools wheel

# 事前ビルド版のみ使用（コンパイル回避）
pip install --only-binary=:all: numpy==1.24.3 scipy pandas matplotlib

# ML関連パッケージ
pip install torch onnxruntime

# ユーティリティ
pip install requests psutil
```

### 方法3: conda-forge使用（最も安全）

```powershell
# Minicondaがインストールされている場合
conda install -c conda-forge numpy=1.24.3 scipy pandas matplotlib
pip install torch onnxruntime requests psutil
```

## 🚀 推奨解決手順（段階的アプローチ）

### Step 1: 基本環境修復（5分）

```powershell
# 現在の仮想環境で実行
pip install --upgrade pip
pip install --upgrade setuptools wheel

# 確認
pip --version
python -c "import setuptools; print('setuptools:', setuptools.__version__)"
```

### Step 2: NumPy単体インストール（5分）

```powershell
# バイナリ版NumPyインストール（ソースビルド回避）
pip install --only-binary=all numpy==1.24.3

# 確認
python -c "import numpy as np; print('NumPy:', np.__version__)"
```

### Step 3: 依存関係順次インストール（10分）

```powershell
# 科学計算ライブラリ
pip install --only-binary=all scipy
pip install pandas
pip install matplotlib

# 機械学習ライブラリ
pip install torch
pip install onnxruntime

# ユーティリティ
pip install requests psutil

# 全体確認
python -c "import numpy, scipy, pandas, matplotlib, torch, onnxruntime; print('All packages OK')"
```

## 🧪 動作確認手順

### 基本確認

```powershell
# 基本ツール確認
pip --version
python -c "import setuptools; print('setuptools:', setuptools.__version__)"
python -c "import wheel; print('wheel:', wheel.__version__)"

# NumPy確認
python -c "import numpy as np; print('NumPy:', np.__version__); print('Test:', np.array([1,2,3]))"

# 依存関係確認
python -c "
import numpy as np
import scipy
import pandas as pd
import matplotlib
import torch
import onnxruntime as ort
print('✅ All packages imported successfully')
print(f'NumPy: {np.__version__}')
print(f'SciPy: {scipy.__version__}')
print(f'Pandas: {pd.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'ONNX Runtime: {ort.__version__}')
"
```

### Infer-OSテスト実行

```powershell
# 基本テスト
python infer_os_npu_test.py --mode basic

# 包括的テスト
python infer_os_npu_test.py --mode comprehensive
```

## 🔍 トラブルシューティング

### 問題1: pip upgradeでもエラー

```powershell
# Python内蔵のensurepip使用
python -m ensurepip --upgrade

# 手動pip更新
python -m pip install --upgrade pip

# 確認
pip --version
```

### 問題2: setuptools importエラー継続

```powershell
# setuptools強制再インストール
pip uninstall setuptools -y
pip install setuptools

# 古いキャッシュクリア
pip cache purge

# 再試行
pip install --upgrade setuptools wheel
```

### 問題3: NumPyソースビルドエラー

```powershell
# 事前ビルド版強制使用
pip install --only-binary=:all: numpy==1.24.3

# または異なるバージョン試行
pip install --only-binary=:all: numpy==1.23.5
```

### 問題4: 仮想環境自体の問題

```powershell
# 仮想環境完全再作成
deactivate
Remove-Item -Recurse -Force venv
python -m venv venv --upgrade-deps
.\venv\Scripts\Activate.ps1

# 基本ツール最初にインストール
pip install --upgrade pip setuptools wheel
```

## 📊 推奨パッケージバージョン

| パッケージ | 推奨バージョン | 理由 |
|------------|----------------|------|
| pip | 25.2+ | 最新機能・バグ修正 |
| setuptools | 69.0+ | Python 3.12対応 |
| wheel | 0.42+ | ビルド最適化 |
| numpy | 1.24.3 | 安定・互換性 |
| scipy | 1.10.1 | NumPy 1.24.3対応 |
| pandas | 2.0.3 | 安定版 |
| matplotlib | 3.7.2 | 互換性良好 |

## 💡 予防策

### 仮想環境作成時のベストプラクティス

```powershell
# 依存関係込みで仮想環境作成
python -m venv venv --upgrade-deps

# 有効化
.\venv\Scripts\Activate.ps1

# 最初に基本ツール更新
pip install --upgrade pip setuptools wheel

# requirements.txt作成
echo "pip>=25.2" > requirements_base.txt
echo "setuptools>=69.0" >> requirements_base.txt
echo "wheel>=0.42" >> requirements_base.txt
echo "numpy==1.24.3" >> requirements_base.txt
echo "scipy==1.10.1" >> requirements_base.txt
echo "pandas==2.0.3" >> requirements_base.txt
echo "matplotlib==3.7.2" >> requirements_base.txt
echo "torch" >> requirements_base.txt
echo "onnxruntime" >> requirements_base.txt
echo "requests" >> requirements_base.txt
echo "psutil" >> requirements_base.txt

# 一括インストール
pip install -r requirements_base.txt
```

### 定期的な環境確認

```powershell
# 環境健全性チェック
pip check

# パッケージリスト確認
pip list

# 古いパッケージ確認
pip list --outdated
```

## 🎯 最終確認

### 成功の指標

```powershell
# 以下が全て成功すれば修復完了
pip --version                    # pip 25.2+
python -c "import setuptools; print('setuptools:', setuptools.__version__)"  # 69.0+
python -c "import numpy as np; print('NumPy:', np.__version__)"              # 1.24.3
python -c "import scipy; print('SciPy OK')"                                  # OK
python -c "import pandas; print('Pandas OK')"                                # OK
python -c "import torch; print('PyTorch OK')"                                # OK
python -c "import onnxruntime; print('ONNX Runtime OK')"                     # OK

# Infer-OSテスト実行
python infer_os_npu_test.py --mode basic
```

### 期待される出力

```
pip 25.2
setuptools: 69.0.3
NumPy: 1.24.3
SciPy OK
Pandas OK
PyTorch OK
ONNX Runtime OK

🚀 Infer-OS NPUテスト開始...
✅ NPU検出: AMD Ryzen AI 9 365 w/ Radeon 880M
✅ 最適化機能: 全て有効
✅ テスト成功
```

## 🚀 次のステップ

setuptools/pipエラー解決後：

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

**結論**: setuptools/pipエラーは新しい仮想環境でよくある問題です。段階的インストールで確実に解決できます。

