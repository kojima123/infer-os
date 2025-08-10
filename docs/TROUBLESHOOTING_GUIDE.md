# 🚨 Infer-OS 日本語重量級LLM トラブルシューティングガイド

このガイドでは、Infer-OS日本語重量級LLMデモで発生する可能性のある問題と解決方法を説明します。

## 🎯 よくある問題と解決方法

### 1. インストール関連の問題

#### 問題: "Microsoft Visual C++ 14.0 is required"
**症状**: Windows環境でbitsandbytesインストール時にエラー
```
error: Microsoft Visual C++ 14.0 is required. Get it with "Microsoft Visual C++ Build Tools"
```

**解決方法**:
```bash
# 方法1: Visual Studio Build Toolsをインストール
# https://visualstudio.microsoft.com/visual-cpp-build-tools/ からダウンロード

# 方法2: 事前コンパイル済みパッケージを使用
pip install --only-binary=all bitsandbytes

# 方法3: conda環境を使用
conda install -c conda-forge bitsandbytes
```

#### 問題: "No module named 'torch'"
**症状**: PyTorchが正しくインストールされていない
```
ModuleNotFoundError: No module named 'torch'
```

**解決方法**:
```bash
# PyTorchの再インストール
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CUDA対応版（GPU環境）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 問題: "ImportError: cannot import name 'platform'"
**症状**: platformモジュールのインポートエラー
```
NameError: name 'platform' is not defined
```

**解決方法**:
```bash
# 最新版を取得
git pull origin aggressive-memory-optimization

# または手動でplatformインポートを追加
python -c "import platform; print('Platform check OK')"
```

### 2. メモリ関連の問題

#### 問題: "CUDA out of memory" / "RuntimeError: out of memory"
**症状**: メモリ不足によるモデルロード失敗
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**解決方法**:
```bash
# 段階的対策

# レベル1: 積極的メモリ最適化
python infer_os_japanese_llm_demo.py --use-aggressive-memory --interactive

# レベル2: 4bit量子化追加
python infer_os_japanese_llm_demo.py --use-aggressive-memory --use-4bit --interactive

# レベル3: 安全プロファイル使用
python infer_os_japanese_llm_demo.py --use-aggressive-memory --use-4bit --quantization-profile safe --interactive

# レベル4: より軽量なモデル使用
python infer_os_japanese_llm_demo.py --model rinna/youri-7b-chat --use-aggressive-memory --use-4bit --interactive
```

#### 問題: メモリ使用量が27.8GBで張り付く
**症状**: 限定メモリ環境でのメモリ不足
```
Memory usage stuck at 27.8GB
```

**解決方法**:
```bash
# 27.8GB環境専用設定
python infer_os_japanese_llm_demo.py \
  --model rinna/youri-7b-chat \
  --use-aggressive-memory \
  --use-4bit \
  --quantization-profile safe \
  --interactive

# 緊急時最小設定
python infer_os_japanese_llm_demo.py \
  --model rinna/youri-7b-chat \
  --use-aggressive-memory \
  --use-4bit \
  --quantization-profile safe \
  --max-length 100 \
  --interactive
```

### 3. NPU関連の問題

#### 問題: "NPU not detected"
**症状**: Windows NPUが検出されない
```
⚠️ NPUが検出されませんでした
```

**解決方法**:
```bash
# 1. DirectML依存関係の確認・再インストール
pip uninstall onnxruntime-directml
pip install onnxruntime-directml

# 2. NPU検出確認
python -c "
from windows_npu_optimizer import WindowsNPUOptimizer
optimizer = WindowsNPUOptimizer()
print(optimizer.detect_npu_hardware())
"

# 3. NPU無効で実行
python infer_os_japanese_llm_demo.py --disable-npu --interactive

# 4. BIOSでNPU有効化確認
# BIOS設定でNPU/AI Accelerationが有効になっているか確認
```

#### 問題: "DirectML provider not available"
**症状**: DirectMLプロバイダーが利用できない
```
DirectML provider not available in ONNX Runtime
```

**解決方法**:
```bash
# DirectML対応ONNX Runtimeの再インストール
pip uninstall onnxruntime onnxruntime-directml
pip install onnxruntime-directml

# 利用可能プロバイダーの確認
python -c "
import onnxruntime as ort
print('Available providers:', ort.get_available_providers())
"

# CPUモードで実行
python infer_os_japanese_llm_demo.py --disable-npu --interactive
```

### 4. 生成品質の問題

#### 問題: 生成結果が短い・「。」のみ
**症状**: プロンプトに対して極端に短い回答
```
🤖 プロンプト: 日本の四季について説明してください
📝 生成結果: 。
```

**解決方法**:
```bash
# 1. より緩い設定で実行
python infer_os_japanese_llm_demo.py --quantization-profile safe --interactive

# 2. 温度設定の調整
python infer_os_japanese_llm_demo.py --interactive
# インタラクティブモード内で温度を調整

# 3. 8bit量子化使用（4bitより高品質）
python infer_os_japanese_llm_demo.py --use-8bit --interactive

# 4. より大きなモデル使用
python infer_os_japanese_llm_demo.py --model matsuo-lab/weblab-10b --interactive
```

#### 問題: 日本語品質が低い
**症状**: 日本語比率が低い、文字化けが発生
```
🇯🇵 日本語品質:
  品質レベル: 要改善
  日本語比率: 25.3%
```

**解決方法**:
```bash
# 1. 日本語特化モデル使用
python infer_os_japanese_llm_demo.py --model rinna/youri-7b-chat --interactive

# 2. 安全プロファイル使用
python infer_os_japanese_llm_demo.py --quantization-profile safe --interactive

# 3. より大きなモデル使用
python infer_os_japanese_llm_demo.py --model matsuo-lab/weblab-10b --interactive

# 4. 量子化レベルを下げる
python infer_os_japanese_llm_demo.py --use-8bit --interactive  # 4bitより高品質
```

### 5. 性能関連の問題

#### 問題: 推論速度が遅い
**症状**: tokens/secが期待値より低い
```
📊 統計: 42トークン, 15.2秒, 2.8トークン/秒
```

**解決方法**:
```bash
# 1. NPU最適化有効化
python infer_os_japanese_llm_demo.py --enable-npu --interactive

# 2. 高度な量子化最適化
python infer_os_japanese_llm_demo.py --use-advanced-quant --interactive

# 3. 積極的プロファイル使用
python infer_os_japanese_llm_demo.py --quantization-profile aggressive --interactive

# 4. ONNX Runtime使用
python infer_os_japanese_llm_demo.py --use-onnx-runtime --onnx-optimization-level 2 --interactive

# 5. 全最適化有効
python infer_os_japanese_llm_demo.py \
  --enable-npu \
  --use-advanced-quant \
  --quantization-profile aggressive \
  --use-onnx-runtime \
  --interactive
```

#### 問題: CPU使用率が高い
**症状**: CPU使用率が90%以上で推論が遅い
```
💾 リソース使用量:
  CPU使用率: 95.2%
```

**解決方法**:
```bash
# 1. NPU最適化でCPU負荷軽減
python infer_os_japanese_llm_demo.py --enable-npu --interactive

# 2. 量子化でCPU負荷軽減
python infer_os_japanese_llm_demo.py --use-4bit --interactive

# 3. より軽量なモデル使用
python infer_os_japanese_llm_demo.py --model rinna/youri-7b-chat --interactive
```

### 6. モデルロード関連の問題

#### 問題: "HuggingFace Hub connection error"
**症状**: モデルダウンロード時のネットワークエラー
```
ConnectionError: Unable to connect to HuggingFace Hub
```

**解決方法**:
```bash
# 1. ネットワーク接続確認
ping huggingface.co

# 2. プロキシ設定（必要に応じて）
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080

# 3. キャッシュクリア
rm -rf ~/.cache/huggingface/

# 4. 手動ダウンロード
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('rinna/youri-7b-chat')
model = AutoModelForCausalLM.from_pretrained('rinna/youri-7b-chat')
"
```

#### 問題: "Model loading timeout"
**症状**: モデルロードが長時間完了しない
```
⏳ モデルロード中... (10分経過)
```

**解決方法**:
```bash
# 1. より軽量なモデルでテスト
python infer_os_japanese_llm_demo.py --model rinna/youri-7b-chat --interactive

# 2. 積極的メモリ最適化使用
python infer_os_japanese_llm_demo.py --use-aggressive-memory --interactive

# 3. 段階的ロード確認
python -c "
from transformers import AutoTokenizer
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('rinna/youri-7b-chat')
print('Tokenizer loaded successfully')
"
```

### 7. インタラクティブモード関連の問題

#### 問題: "'JapaneseHeavyLLMDemo' object has no attribute 'interactive_mode'"
**症状**: インタラクティブモードメソッドが見つからない
```
AttributeError: 'JapaneseHeavyLLMDemo' object has no attribute 'interactive_mode'
```

**解決方法**:
```bash
# 最新版を取得
git pull origin aggressive-memory-optimization

# ファイルの整合性確認
python -c "
import infer_os_japanese_llm_demo
demo = infer_os_japanese_llm_demo.InferOSJapaneseLLMDemo()
print(hasattr(demo, 'interactive_mode'))
"
```

#### 問題: インタラクティブモードで応答しない
**症状**: プロンプト入力後に処理が停止
```
🤖 プロンプトを入力してください: テスト
🔄 生成中...
(応答なし)
```

**解決方法**:
```bash
# 1. Ctrl+Cで中断後、より軽量な設定で再試行
python infer_os_japanese_llm_demo.py --use-4bit --quantization-profile safe --interactive

# 2. より短いプロンプトでテスト
# インタラクティブモードで「テスト」など短いプロンプトを試行

# 3. デバッグモードで実行
python infer_os_japanese_llm_demo.py --interactive --verbose
```

## 🔍 診断ツール

### システム診断
```bash
# システム情報確認
python -c "
import psutil, torch, platform
print(f'OS: {platform.system()} {platform.release()}')
print(f'Python: {platform.python_version()}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'Memory: {psutil.virtual_memory().total / (1024**3):.1f}GB')
print(f'CPU Cores: {psutil.cpu_count()}')
"
```

### NPU診断
```bash
# NPU検出確認
python -c "
try:
    from windows_npu_optimizer import WindowsNPUOptimizer
    optimizer = WindowsNPUOptimizer()
    npu_info = optimizer.detect_npu_hardware()
    print('NPU Detection Result:', npu_info)
except Exception as e:
    print('NPU Detection Error:', e)
"
```

### メモリ診断
```bash
# メモリ使用量監視
python -c "
import psutil
import time
for i in range(5):
    mem = psutil.virtual_memory()
    print(f'Memory: {mem.used / (1024**3):.1f}GB / {mem.total / (1024**3):.1f}GB ({mem.percent:.1f}%)')
    time.sleep(1)
"
```

### 依存関係診断
```bash
# 必要ライブラリの確認
python -c "
import sys
required_packages = ['torch', 'transformers', 'accelerate', 'psutil', 'bitsandbytes']
for package in required_packages:
    try:
        __import__(package)
        print(f'✅ {package}: OK')
    except ImportError:
        print(f'❌ {package}: Missing')
"
```

## 🚨 緊急時対応

### 最小動作設定
```bash
# 最も軽量な設定で動作確認
python infer_os_japanese_llm_demo.py \
  --model rinna/youri-7b-chat \
  --use-aggressive-memory \
  --use-4bit \
  --quantization-profile safe \
  --disable-npu \
  --max-length 50 \
  --prompt "テスト"
```

### 段階的復旧手順
1. **最小設定で動作確認**
2. **メモリ最適化追加**
3. **量子化レベル向上**
4. **NPU最適化有効化**
5. **高度な最適化追加**

### ログ収集
```bash
# 詳細ログ出力
python infer_os_japanese_llm_demo.py --interactive --verbose 2>&1 | tee debug.log

# エラー情報のみ抽出
grep -i error debug.log
grep -i warning debug.log
```

## 📞 サポート情報

### GitHub Issues
問題が解決しない場合は、以下の情報と共にGitHub Issuesに報告してください：

1. **環境情報**:
   - OS（Windows/Linux/macOS）
   - Python バージョン
   - メモリ容量
   - NPU の有無

2. **実行コマンド**:
   - 使用したコマンドライン引数
   - モデル名

3. **エラーメッセージ**:
   - 完全なエラーメッセージ
   - スタックトレース

4. **システム診断結果**:
   - 上記診断ツールの出力結果

### よくある質問 (FAQ)

#### Q: どのモデルを選べばよいですか？
A: メモリ容量に応じて選択してください：
- 64GB+: matsuo-lab/weblab-10b
- 48GB+: rinna/youri-7b-chat
- 32GB+: cyberagent/open-calm-7b
- 27.8GB: rinna/youri-7b-chat + 積極的メモリ最適化

#### Q: NPUは必須ですか？
A: 必須ではありませんが、Windows 11 + NPU環境では大幅な性能向上が期待できます。

#### Q: 量子化による品質低下はどの程度ですか？
A: 設定により異なります：
- Safe: 2-5%の品質低下
- Balanced: 5-10%の品質低下
- Aggressive: 10-15%の品質低下

---

**問題が解決しない場合は、[GitHub Issues](https://github.com/kojima123/infer-os/issues)でサポートを受けてください！**

