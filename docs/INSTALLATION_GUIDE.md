# 📦 Infer-OS 日本語重量級LLM インストールガイド

このガイドでは、Infer-OS日本語重量級LLMデモの詳細なインストール手順を説明します。

## 🎯 インストール概要

### 必要な手順
1. **システム要件の確認**
2. **Python環境の準備**
3. **リポジトリのクローン**
4. **基本依存関係のインストール**
5. **オプション機能のインストール**
6. **動作確認**

## 📋 システム要件

### 最小要件
- **OS**: Windows 10/11, Linux (Ubuntu 18.04+), macOS 10.15+
- **Python**: 3.8以上
- **メモリ**: 16GB以上
- **ストレージ**: 50GB以上の空き容量
- **インターネット**: モデルダウンロード用

### 推奨要件
- **OS**: Windows 11 (NPU対応)
- **Python**: 3.10以上
- **メモリ**: 32GB以上
- **ストレージ**: 100GB以上の空き容量
- **GPU**: NVIDIA GPU (CUDA対応) または AMD GPU
- **NPU**: AMD Ryzen AI, Intel NPU, Qualcomm NPU

### 環境別推奨設定

#### 標準PC環境 (32GB+ メモリ)
```bash
# 基本設定
--use-8bit --interactive
```

#### 限定メモリ環境 (27.8GB)
```bash
# 積極的メモリ最適化
--use-aggressive-memory --interactive
```

#### NPU搭載PC
```bash
# NPU最適化有効
--enable-npu --use-advanced-quant --interactive
```

## 🛠️ インストール手順

### Step 1: Python環境の準備

#### Windows
```powershell
# Python 3.10のインストール（推奨）
# https://www.python.org/downloads/ からダウンロード

# pipの更新
python -m pip install --upgrade pip

# 仮想環境の作成（推奨）
python -m venv infer_os_env
infer_os_env\Scripts\activate
```

#### Linux/macOS
```bash
# Python 3.10のインストール（Ubuntu）
sudo apt update
sudo apt install python3.10 python3.10-pip python3.10-venv

# pipの更新
python3.10 -m pip install --upgrade pip

# 仮想環境の作成（推奨）
python3.10 -m venv infer_os_env
source infer_os_env/bin/activate
```

### Step 2: リポジトリのクローン

```bash
# GitHubからクローン
git clone https://github.com/kojima123/infer-os.git
cd infer-os

# 最新の統合ブランチに切り替え
git checkout aggressive-memory-optimization
```

### Step 3: 基本依存関係のインストール

```bash
# 必須ライブラリのインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Transformersとその他の基本ライブラリ
pip install transformers accelerate
pip install psutil argparse typing-extensions

# 進捗表示用
pip install tqdm
```

### Step 4: オプション機能のインストール

#### 4.1 量子化サポート (推奨)
```bash
# BitsAndBytesのインストール
pip install bitsandbytes

# CUDA対応の場合
pip install bitsandbytes-cuda
```

#### 4.2 ONNX Runtime (高速化)
```bash
# CPU版
pip install onnxruntime

# GPU版 (NVIDIA)
pip install onnxruntime-gpu

# DirectML版 (Windows NPU対応)
pip install onnxruntime-directml
```

#### 4.3 Windows NPU対応
```bash
# Windows環境でのNPU対応
pip install onnxruntime-directml
pip install numpy

# オプション: DirectML開発ツール
pip install directml
```

#### 4.4 高度な最適化ライブラリ
```bash
# Flash Attention (高速化)
pip install flash-attn --no-build-isolation

# xFormers (メモリ効率化)
pip install xformers

# DeepSpeed (分散処理)
pip install deepspeed
```

### Step 5: 動作確認

#### 5.1 基本動作確認
```bash
# システム情報表示
python infer_os_japanese_llm_demo.py --list-models

# サンプルプロンプト表示
python infer_os_japanese_llm_demo.py --samples
```

#### 5.2 軽量モデルでのテスト
```bash
# 軽量モデルでの動作確認
python infer_os_japanese_llm_demo.py --model rinna/youri-7b-chat --prompt "こんにちは"
```

#### 5.3 インタラクティブモードテスト
```bash
# インタラクティブモード開始
python infer_os_japanese_llm_demo.py --model rinna/youri-7b-chat --interactive
```

## 🔧 環境別セットアップ

### Windows 11 + NPU環境

#### 前提条件
- AMD Ryzen AI搭載PC または Intel NPU搭載PC
- Windows 11 22H2以降
- 最新のNPUドライバー

#### セットアップ
```powershell
# DirectML環境の準備
pip install onnxruntime-directml
pip install torch-directml

# NPU対応確認
python -c "import onnxruntime as ort; print('DirectML Providers:', ort.get_available_providers())"

# NPU有効で実行
python infer_os_japanese_llm_demo.py --enable-npu --model rinna/youri-7b-chat --interactive
```

### Linux + CUDA環境

#### 前提条件
- NVIDIA GPU (Compute Capability 6.0以上)
- CUDA 11.8以降
- cuDNN 8.0以降

#### セットアップ
```bash
# CUDA対応PyTorchのインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA対応ライブラリ
pip install bitsandbytes-cuda
pip install flash-attn --no-build-isolation

# GPU確認
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

# GPU有効で実行
python infer_os_japanese_llm_demo.py --use-8bit --model rinna/youri-7b-chat --interactive
```

### macOS + Apple Silicon環境

#### 前提条件
- Apple M1/M2/M3チップ
- macOS 12.0以降

#### セットアップ
```bash
# Apple Silicon対応PyTorchのインストール
pip install torch torchvision torchaudio

# Metal Performance Shaders対応
pip install torch-audio

# MPS確認
python -c "import torch; print('MPS Available:', torch.backends.mps.is_available())"

# MPS有効で実行
python infer_os_japanese_llm_demo.py --model rinna/youri-7b-chat --interactive
```

## 🚨 トラブルシューティング

### よくあるインストールエラー

#### エラー: "Microsoft Visual C++ 14.0 is required"
```bash
# Windows: Visual Studio Build Toolsをインストール
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

# または、事前コンパイル済みパッケージを使用
pip install --only-binary=all bitsandbytes
```

#### エラー: "CUDA out of memory"
```bash
# より軽量な設定を使用
python infer_os_japanese_llm_demo.py --use-aggressive-memory --quantization-profile safe
```

#### エラー: "No module named 'torch'"
```bash
# PyTorchの再インストール
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### エラー: "NPU not detected"
```bash
# NPUドライバーの確認
# Windows: デバイスマネージャーでNPUデバイスを確認
# DirectML依存関係の再インストール
pip uninstall onnxruntime-directml
pip install onnxruntime-directml
```

### パフォーマンス最適化

#### メモリ使用量の削減
```bash
# 積極的メモリ最適化
python infer_os_japanese_llm_demo.py --use-aggressive-memory

# 量子化レベルの調整
python infer_os_japanese_llm_demo.py --quantization-profile aggressive
```

#### 推論速度の向上
```bash
# 高度な量子化最適化
python infer_os_japanese_llm_demo.py --use-advanced-quant

# NPU最適化（Windows）
python infer_os_japanese_llm_demo.py --enable-npu
```

## 📊 インストール確認チェックリスト

### 基本環境
- [ ] Python 3.8以上がインストール済み
- [ ] pip が最新版に更新済み
- [ ] 仮想環境が作成・有効化済み
- [ ] リポジトリがクローン済み

### 必須ライブラリ
- [ ] PyTorch がインストール済み
- [ ] Transformers がインストール済み
- [ ] psutil がインストール済み
- [ ] 基本動作確認が完了

### オプション機能
- [ ] BitsAndBytes (量子化サポート)
- [ ] ONNX Runtime (高速化)
- [ ] DirectML (Windows NPU対応)
- [ ] Flash Attention (高速化)

### 動作確認
- [ ] モデル一覧表示が正常
- [ ] サンプルプロンプト表示が正常
- [ ] 軽量モデルでの生成が正常
- [ ] インタラクティブモードが正常

## 🔗 参考リンク

- [PyTorch公式インストールガイド](https://pytorch.org/get-started/locally/)
- [Transformers公式ドキュメント](https://huggingface.co/docs/transformers)
- [BitsAndBytes GitHub](https://github.com/TimDettmers/bitsandbytes)
- [ONNX Runtime公式サイト](https://onnxruntime.ai/)
- [DirectML公式ドキュメント](https://docs.microsoft.com/en-us/windows/ai/directml/)

---

**インストールが完了したら、[使用方法ガイド](USAGE_GUIDE.md)に進んでください！**

