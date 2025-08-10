# AMD公式RyzenAI セットアップガイド

## 🎯 概要

AMD公式ドキュメントに基づいたRyzenAI 1.5.1の正確なセットアップガイドです。

**公式ドキュメント**: https://ryzenai.docs.amd.com/en/latest/inst.html

## 📋 前提条件

### ハードウェア要件
- **CPU**: AMD Ryzen AI 9 365 (NPU搭載)
- **メモリ**: 16GB以上推奨
- **OS**: Windows 11 (build >= 22621.3527)

### ソフトウェア要件
- **Visual Studio 2022 Community**: Desktop Development with C++
- **Miniforge**: Conda環境管理
- **cmake**: version >= 3.26

## 🚀 インストール手順

### ステップ1: 前提条件インストール

#### 1.1 Visual Studio 2022 Community
```powershell
# Visual Studio 2022 Community をダウンロード・インストール
# 重要: "Desktop Development with C++" を必ずインストール
```

#### 1.2 Miniforge
```powershell
# Miniforge をダウンロード・インストール
# https://github.com/conda-forge/miniforge

# システムPATH変数に以下を追加:
# path\to\miniforge3\condabin
# または path\to\miniforge3\Scripts\
# または path\to\miniforge3\

# ターミナルで初期化
conda init
```

### ステップ2: NPUドライバーインストール

#### 2.1 NPUドライバーダウンロード
```powershell
# AMD公式サイトからNPUドライバーをダウンロード
# ファイル名: NPU Driver (ZIP形式)
```

#### 2.2 NPUドライバーインストール
```powershell
# 1. ダウンロードしたZIPファイルを展開
# 2. 管理者権限でターミナルを開く
# 3. インストーラー実行
.\npu_sw_installer.exe
```

#### 2.3 NPUドライバー確認
```powershell
# タスクマネージャーで確認
# タスクマネージャー -> パフォーマンス -> NPU0
# NPU MCDM driver (Version:32.0.203.280, Date:5/16/2025) が表示されることを確認
```

### ステップ3: RyzenAI ソフトウェアインストール

#### 3.1 RyzenAI MSIインストーラーダウンロード
```powershell
# AMD公式サイトから最新版をダウンロード
# 推奨: ryzen-ai-1.5.1.msi (LLM性能改善版)
```

#### 3.2 MSIインストーラー実行
```powershell
# 1. ryzen-ai-1.5.1.msi を実行
# 2. ライセンス条項に同意
# 3. インストール先指定 (デフォルト: C:\Program Files\RyzenAI\1.5.1)
# 4. Conda環境名指定 (デフォルト: ryzen-ai-1.5.1)
```

### ステップ4: 高レベルPython SDK環境セットアップ

#### 4.1 専用Conda環境作成
```powershell
# Miniforge Prompt を開く (スタートメニューから検索)

# 専用環境作成
conda create -n ryzenai-llm python=3.10
conda activate ryzenai-llm
```

#### 4.2 Lemonade SDK インストール
```powershell
# Lemonade SDK (公式高レベルAPI) インストール
pip install lemonade-sdk[llm-oga-hybrid]

# RyzenAI ハイブリッドモード設定
lemonade-install --ryzenai hybrid
```

### ステップ5: インストール確認

#### 5.1 基本インストール確認
```powershell
# RyzenAI インストールフォルダーのクイックテスト
cd %RYZEN_AI_INSTALLATION_PATH%/quicktest
python quicktest.py
```

**期待される出力**:
```
[Vitis AI EP] No. of Operators :   NPU   398 VITIS_EP_CPU     2
[Vitis AI EP] No. of Subgraphs :   NPU     1 Actually running on NPU     1
Test Passed
```

#### 5.2 Lemonade SDK確認
```powershell
# 環境確認
python ryzenai_official_implementation.py --check-env

# 公式検証コマンド実行
python ryzenai_official_implementation.py --validate
```

#### 5.3 LLM動作確認
```powershell
# 公式検証コマンド (コマンドライン)
lemonade -i amd/Llama-3.2-1B-Instruct-awq-g128-int4-asym-fp16-onnx-hybrid oga-load --device hybrid --dtype int4 llm-prompt --max-new-tokens 64 -p "Hello, how are you?"

# Python API確認
python ryzenai_official_implementation.py --prompt "Hello, how are you?"
```

## 🎯 使用方法

### 基本的な使用方法

#### コマンドライン使用
```powershell
# 環境確認
python ryzenai_official_implementation.py --check-env

# 単発プロンプト実行
python ryzenai_official_implementation.py --prompt "人参について教えてください。"

# インタラクティブモード
python ryzenai_official_implementation.py --interactive
```

#### Python API使用
```python
from ryzenai_official_implementation import RyzenAIOfficialLLM

# LLM初期化
llm = RyzenAIOfficialLLM()

# テキスト生成
result = llm.generate_text("Hello, how are you?")
print(result)

# 日本語生成
result = llm.generate_japanese("人参について教えてください。")
print(result)
```

### 高度な使用方法

#### 公式Lemonade API直接使用
```python
from lemonade.api import from_pretrained

# モデル初期化
model, tokenizer = from_pretrained(
    "amd/Llama-3.2-1B-Instruct-awq-g128-int4-asym-fp16-onnx-hybrid", 
    recipe="oga-hybrid"
)

# 推論実行
input_ids = tokenizer("This is my prompt", return_tensors="pt").input_ids
response = model.generate(input_ids, max_new_tokens=30)
print(tokenizer.decode(response[0]))
```

## 📊 パフォーマンス期待値

### NPU + CPU ハイブリッドモード
- **推論速度**: 30-80トークン/秒
- **NPU使用率**: 50-80% (タスクマネージャー確認)
- **メモリ使用量**: 4-8GB
- **消費電力**: CPU単体比30-50%削減

### 対応モデル
- **Llama-3.2-1B-Instruct**: 軽量、高速
- **Llama-3.2-3B-Instruct**: バランス型
- **その他AMD最適化モデル**: AMD公式リポジトリ参照

## 🔧 トラブルシューティング

### 問題1: Lemonade SDK インポートエラー

**症状**:
```
ImportError: No module named 'lemonade'
```

**解決方法**:
```powershell
# 1. 正しいConda環境確認
conda activate ryzenai-llm

# 2. Lemonade SDK再インストール
pip uninstall lemonade-sdk
pip install lemonade-sdk[llm-oga-hybrid]
lemonade-install --ryzenai hybrid
```

### 問題2: NPU認識エラー

**症状**:
```
NPU device not found
```

**解決方法**:
```powershell
# 1. NPUドライバー確認
# タスクマネージャー -> パフォーマンス -> NPU0

# 2. NPUドライバー再インストール
# 管理者権限で npu_sw_installer.exe 実行

# 3. システム再起動
shutdown /r /t 0
```

### 問題3: モデルダウンロードエラー

**症状**:
```
Model download failed
```

**解決方法**:
```powershell
# 1. インターネット接続確認
# 2. プロキシ設定確認
# 3. 手動モデルダウンロード
lemonade -i amd/Llama-3.2-1B-Instruct-awq-g128-int4-asym-fp16-onnx-hybrid oga-load --device hybrid --dtype int4
```

### 問題4: 権限エラー

**症状**:
```
PermissionError: Access denied
```

**解決方法**:
```powershell
# 管理者権限でMiniforge Prompt実行
# スタートメニュー -> Miniforge Prompt -> 右クリック -> 管理者として実行
```

## 📋 環境変数

### 重要な環境変数
```powershell
# RyzenAI インストールパス
RYZEN_AI_INSTALLATION_PATH=C:\Program Files\RyzenAI\1.5.1

# Conda環境パス
CONDA_DEFAULT_ENV=ryzenai-llm

# システムPATH (Miniforge)
PATH=%PATH%;path\to\miniforge3\condabin
```

## 🔗 参考リンク

### 公式ドキュメント
- [RyzenAI インストール](https://ryzenai.docs.amd.com/en/latest/inst.html)
- [High-Level Python SDK](https://ryzenai.docs.amd.com/en/latest/llm/high_level_python.html)
- [Featured LLMs](https://ryzenai.docs.amd.com/en/latest/llm/examples.html)

### ダウンロードリンク
- [NPU Driver](https://www.amd.com/en/support/download/drivers.html)
- [RyzenAI 1.5.1 MSI](https://www.amd.com/en/developer/ryzen-ai.html)
- [Miniforge](https://github.com/conda-forge/miniforge)

### サポート
- [AMD Developer Community](https://community.amd.com/t5/ai-ml/ct-p/ai-ml)
- [GitHub Issues](https://github.com/amd/RyzenAI-SW)

## ✅ チェックリスト

### インストール完了確認
- [ ] Visual Studio 2022 Community (C++開発環境)
- [ ] Miniforge インストール・PATH設定
- [ ] NPUドライバー インストール・認識確認
- [ ] RyzenAI 1.5.1 MSI インストール
- [ ] ryzenai-llm Conda環境作成
- [ ] Lemonade SDK インストール
- [ ] quicktest.py 実行成功
- [ ] 公式検証コマンド実行成功

### 動作確認
- [ ] NPU使用率上昇 (タスクマネージャー)
- [ ] LLM推論実行成功
- [ ] 日本語生成品質確認
- [ ] エラーなしでの継続実行

この手順に従うことで、AMD公式仕様に完全準拠したRyzenAI環境が構築できます。

