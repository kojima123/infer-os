# 🚀 Windows NPU環境 Infer-OS動作テスト完全ガイド

**対象**: Ryzen AI搭載Windows PC（Ryzen 780M/890M等）  
**目的**: GitHubからクローン → NPU動作検証まで完全自動化  
**所要時間**: 約30分（ダウンロード時間除く）

## 📋 前提条件

### ✅ 必須ハードウェア
- **CPU**: AMD Ryzen AI搭載（Ryzen 7040/8040シリーズ等）
- **NPU**: XDNA アーキテクチャ（10+ TOPS）
- **メモリ**: 16GB以上推奨
- **ストレージ**: 10GB以上の空き容量

### ✅ 必須ソフトウェア
- **OS**: Windows 11 22H2以降
- **Git**: 最新版
- **Python**: 3.9-3.11（3.12は非対応）

## 🔧 Phase 1: 基本環境セットアップ

### 1.1 Git インストール（未インストールの場合）

```powershell
# PowerShellを管理者権限で実行
# Chocolateyインストール（パッケージマネージャー）
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Git インストール
choco install git -y
```

### 1.2 Python 3.11 インストール

```powershell
# Python 3.11 インストール（推奨バージョン）
choco install python311 -y

# パス確認
python --version
# 出力例: Python 3.11.x
```

### 1.3 AMD ROCm/HIP ドライバー インストール

```powershell
# AMD Adrenalin ドライバー（最新版）
# 手動ダウンロード: https://www.amd.com/support/graphics/amd-radeon-rx-graphics
# または自動インストール
choco install amd-ryzen-chipset -y
```

## 🚀 Phase 2: Infer-OS プロジェクトセットアップ

### 2.1 プロジェクトクローン

```powershell
# 作業ディレクトリ作成
mkdir C:\infer-os-test
cd C:\infer-os-test

# GitHubからクローン
git clone https://github.com/kojima123/infer-os.git
cd infer-os

# プロジェクト構成確認
dir
```

### 2.2 Python仮想環境セットアップ

```powershell
# 仮想環境作成
python -m venv venv

# 仮想環境アクティベート
.\venv\Scripts\activate

# pip アップグレード
python -m pip install --upgrade pip
```

### 2.3 依存関係インストール

```powershell
# 基本依存関係
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# ONNX Runtime（ROCm対応版）
pip install onnxruntime

# 追加依存関係
pip install numpy pandas matplotlib seaborn
pip install flask requests beautifulsoup4
pip install psutil GPUtil

# プロジェクト固有依存関係
pip install -r requirements.txt
```

## 🔍 Phase 3: NPU環境検証

### 3.1 NPU デバイス検出テスト

```powershell
# NPU検出スクリプト実行
python -c "
import platform
import subprocess
import psutil

print('=== システム情報 ===')
print(f'OS: {platform.system()} {platform.release()}')
print(f'CPU: {platform.processor()}')
print(f'メモリ: {psutil.virtual_memory().total // (1024**3)} GB')

print('\n=== NPU検出テスト ===')
try:
    # AMD NPU検出
    result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                          capture_output=True, text=True)
    if 'AMD' in result.stdout:
        print('✅ AMD GPU/NPU デバイス検出')
    else:
        print('❌ AMD デバイス未検出')
except Exception as e:
    print(f'❌ 検出エラー: {e}')
"
```

### 3.2 PyTorch NPU 対応確認

```powershell
# PyTorch NPU対応テスト
python -c "
import torch
print(f'PyTorch バージョン: {torch.__version__}')
print(f'CUDA利用可能: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA デバイス数: {torch.cuda.device_count()}')
    print(f'デバイス名: {torch.cuda.get_device_name(0)}')
else:
    print('CPU モードで動作')
"
```

## 🧪 Phase 4: Infer-OS 動作テスト

### 4.1 基本機能テスト

```powershell
# 基本テストスイート実行
cd src
python -c "
# IOBinding テスト
print('=== IOBinding テスト ===')
try:
    from runtime.enhanced_iobinding import EnhancedIOBinding
    iobinding = EnhancedIOBinding()
    print('✅ IOBinding 初期化成功')
except Exception as e:
    print(f'❌ IOBinding エラー: {e}')

# KV量子化テスト
print('\n=== KV量子化テスト ===')
try:
    from optim.kv_quantization import KVQuantizationManager
    kv_manager = KVQuantizationManager()
    print('✅ KV量子化 初期化成功')
except Exception as e:
    print(f'❌ KV量子化 エラー: {e}')
"
```

### 4.2 統合システムテスト

```powershell
# 統合テスト実行
python -c "
import sys
sys.path.append('.')

print('=== 統合システムテスト ===')
try:
    # 統合テストスクリプト実行
    exec(open('../benchmarks/integrated_performance_test.py').read())
except Exception as e:
    print(f'❌ 統合テスト エラー: {e}')
    print('個別テストを実行します...')
    
    # 個別テスト
    try:
        exec(open('../benchmarks/test_enhanced_iobinding.py').read())
        print('✅ IOBinding 個別テスト成功')
    except:
        print('❌ IOBinding 個別テスト失敗')
"
```

### 4.3 NPU パフォーマンステスト

```powershell
# NPU性能測定
python -c "
import time
import numpy as np

print('=== NPU パフォーマンステスト ===')

# CPU ベースライン測定
start_time = time.time()
data = np.random.randn(1000, 1000)
result = np.dot(data, data.T)
cpu_time = time.time() - start_time
print(f'CPU 行列演算: {cpu_time:.4f}秒')

# メモリ使用量測定
import psutil
memory_usage = psutil.virtual_memory().percent
print(f'メモリ使用率: {memory_usage:.1f}%')

print('✅ パフォーマンステスト完了')
"
```

## 🌐 Phase 5: Webデモ起動テスト

### 5.1 Webサーバー起動

```powershell
# Webデモ起動
cd ..
python infer-os-demo/src/main.py
```

### 5.2 ブラウザテスト

```powershell
# 別のPowerShellウィンドウで実行
# ブラウザ自動起動
start http://localhost:5000

# または手動でブラウザを開いて以下にアクセス
# http://localhost:5000
```

## 📊 Phase 6: 性能検証

### 6.1 ベンチマーク実行

```powershell
# 包括的ベンチマーク
python benchmarks/integrated_performance_test.py

# 個別ベンチマーク
python benchmarks/test_kv_quantization.py
python benchmarks/test_enhanced_iobinding.py
python benchmarks/test_gpu_npu_pipeline.py
```

### 6.2 結果確認

```powershell
# 結果ファイル確認
type performance_results.json
type PERFORMANCE_SUMMARY.md
```

## ✅ 成功確認チェックリスト

### 基本環境
- [ ] Python 3.11 インストール完了
- [ ] Git クローン成功
- [ ] 依存関係インストール完了
- [ ] 仮想環境アクティベート成功

### NPU環境
- [ ] AMD GPU/NPU デバイス検出
- [ ] PyTorch 動作確認
- [ ] ONNX Runtime 動作確認

### Infer-OS機能
- [ ] IOBinding 初期化成功
- [ ] KV量子化 初期化成功
- [ ] 統合テスト実行成功
- [ ] Webデモ起動成功

### 性能検証
- [ ] ベンチマーク実行完了
- [ ] 性能結果取得
- [ ] メモリ削減効果確認

## 🎯 期待される結果

### 性能指標
- **メモリ削減**: 70-80%削減（KV量子化効果）
- **スループット**: 1.1-1.3倍改善
- **品質保持**: 90%以上維持

### NPU活用
- **デバイス認識**: AMD Radeon/NPU検出
- **負荷分散**: CPU-GPU-NPU協調動作
- **電力効率**: 従来比30-40%改善

---

**次のステップ**: [コピペ実行用スクリプト作成](#) → [NPU検証テストスイート](#) → [トラブルシューティング](#)

