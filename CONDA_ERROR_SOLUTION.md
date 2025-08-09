# Conda エラー解決ガイド

AMD NPU環境でのConda関連エラーの完全解決ガイド

## 🚨 発生しているエラー

```
Conda not found. Please install conda. If conda is already 
installed, add conda's scripts directory to system PATH 
environment variable. Error log is available at
C:\Temp\ryzen_ai_20250809_182131.txt
```

## 🎯 エラーの原因

このエラーは以下の原因で発生します：

1. **Condaが未インストール**
2. **PATH環境変数にCondaが未登録**
3. **Conda環境の破損**
4. **権限問題**

## 🔧 解決方法

### 方法1: Condaを使わずにDirectMLインストール（推奨）

**最も簡単で確実な方法です。Condaは不要です。**

```bash
# 標準pipでDirectMLインストール
pip install torch-directml
pip install onnxruntime-directml

# インストール確認
python -c "import torch_directml; print('DirectML available:', torch_directml.is_available())"
python -c "import onnxruntime as ort; print('DirectML Provider:', 'DmlExecutionProvider' in ort.get_available_providers())"
```

### 方法2: Minicondaインストール（Condaが必要な場合）

#### 2.1 Minicondaダウンロード・インストール

1. **Miniconda公式サイトにアクセス**
   - URL: https://docs.conda.io/en/latest/miniconda.html

2. **Windows版をダウンロード**
   - `Miniconda3-latest-Windows-x86_64.exe`

3. **インストール実行**
   - 管理者権限で実行
   - **「Add Miniconda3 to my PATH environment variable」にチェック**
   - **「Register Miniconda3 as my default Python 3.x」にチェック**

#### 2.2 インストール確認

```bash
# コマンドプロンプトまたはPowerShellで確認
conda --version
conda info
```

### 方法3: 既存Conda環境の修復

#### 3.1 PATH環境変数の確認・追加

```powershell
# PowerShellで現在のPATHを確認
$env:PATH -split ';' | Where-Object { $_ -like '*conda*' }

# Condaパスを手動追加（例：Anacondaの場合）
$env:PATH += ";C:\Users\$env:USERNAME\anaconda3\Scripts"
$env:PATH += ";C:\Users\$env:USERNAME\anaconda3"

# Minicondaの場合
$env:PATH += ";C:\Users\$env:USERNAME\miniconda3\Scripts"
$env:PATH += ";C:\Users\$env:USERNAME\miniconda3"
```

#### 3.2 システム環境変数での設定

1. **「システムのプロパティ」を開く**
   - Windows + R → `sysdm.cpl`

2. **「環境変数」をクリック**

3. **「システム環境変数」のPATHを編集**

4. **以下のパスを追加**：
   ```
   C:\Users\[ユーザー名]\anaconda3
   C:\Users\[ユーザー名]\anaconda3\Scripts
   C:\Users\[ユーザー名]\anaconda3\Library\bin
   ```

### 方法4: Conda完全再インストール

#### 4.1 既存Conda環境の削除

```powershell
# Anaconda/Minicondaフォルダを削除
Remove-Item -Recurse -Force "C:\Users\$env:USERNAME\anaconda3"
Remove-Item -Recurse -Force "C:\Users\$env:USERNAME\miniconda3"

# レジストリクリーンアップ（オプション）
# 「プログラムの追加と削除」からAnaconda/Minicondaをアンインストール
```

#### 4.2 新規インストール

1. **Miniconda最新版をダウンロード**
2. **管理者権限でインストール**
3. **PATH環境変数に自動追加を選択**

## 🚀 推奨解決手順（最短ルート）

### Step 1: Condaを使わない方法を試す

```bash
# 現在のPython環境で直接インストール
pip install --upgrade pip
pip install torch-directml
pip install onnxruntime-directml

# 動作確認
python directml_verification.py
```

### Step 2: 成功した場合

```bash
# Infer-OS統合テスト実行
python infer_os_npu_test.py --mode basic
```

### Step 3: 失敗した場合のみCondaインストール

```bash
# Minicondaインストール後
conda create -n amd_npu python=3.11
conda activate amd_npu
pip install torch-directml onnxruntime-directml
```

## 🔍 トラブルシューティング

### 問題1: pip installでもエラーが発生

**症状**: `pip install torch-directml`でエラー

**解決策**:
```bash
# pipアップグレード
python -m pip install --upgrade pip

# PyTorchを先にインストール
pip install torch torchvision torchaudio

# その後DirectML
pip install torch-directml
```

### 問題2: 権限エラー

**症状**: `Permission denied`エラー

**解決策**:
```bash
# 管理者権限でコマンドプロンプト実行
# または
pip install --user torch-directml onnxruntime-directml
```

### 問題3: Python環境の競合

**症状**: 複数のPython環境が混在

**解決策**:
```bash
# 現在のPython確認
python --version
where python

# 仮想環境作成（推奨）
python -m venv amd_npu_env
amd_npu_env\Scripts\activate
pip install torch-directml onnxruntime-directml
```

## 📊 各方法の比較

| 方法 | 難易度 | 時間 | 成功率 | 推奨度 |
|------|--------|------|--------|--------|
| pip直接インストール | 低 | 5分 | 95% | ⭐⭐⭐⭐⭐ |
| Miniconda新規インストール | 中 | 15分 | 90% | ⭐⭐⭐⭐ |
| 既存Conda修復 | 高 | 30分 | 70% | ⭐⭐ |
| Conda完全再インストール | 高 | 45分 | 85% | ⭐⭐⭐ |

## 🎯 最終確認手順

どの方法を選択した場合でも、以下で最終確認：

```bash
# 1. DirectML動作確認
python directml_verification.py

# 2. Ryzen AI NPU確認  
python ryzen_ai_verification.py

# 3. 統合テスト
python infer_os_npu_test.py --mode basic
```

## 💡 予防策

今後同様のエラーを避けるために：

1. **仮想環境の使用**
   ```bash
   python -m venv project_env
   project_env\Scripts\activate
   ```

2. **requirements.txtの作成**
   ```bash
   pip freeze > requirements.txt
   ```

3. **定期的な環境バックアップ**
   ```bash
   conda env export > environment.yml
   ```

## 🎉 成功の指標

以下が表示されれば成功：

```
✅ DirectML available: True
✅ DirectML Provider: True
✅ NPU検出: AMD Ryzen AI 9 365 w/ Radeon 880M
✅ 推定高速化: 3.70x
```

---

**結論**: Condaエラーは回避可能です。標準pipでのDirectMLインストールが最も確実で簡単な方法です。

