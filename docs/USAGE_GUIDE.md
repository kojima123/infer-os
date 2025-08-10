
# 📖 Infer-OS 日本語重量級LLM 使用方法ガイド

このガイドでは、Infer-OS日本語重量級LLMデモの詳細な使用方法を説明します。

## 🎯 基本的な使用方法

### コマンドライン実行

#### 最もシンプルな実行
```bash
python infer_os_japanese_llm_demo.py --interactive
```

#### モデルを指定して実行
```bash
python infer_os_japanese_llm_demo.py --model rinna/youri-7b-chat --interactive
```

#### 全機能を有効にして実行
```bash
python infer_os_japanese_llm_demo.py \
  --model rinna/youri-7b-chat \
  --use-aggressive-memory \
  --enable-npu \
  --use-advanced-quant \
  --interactive
```

## 🤖 対応モデル

### 日本語重量級モデル一覧

#### matsuo-lab/weblab-10b (10Bパラメータ)
- **特徴**: 最重量級日本語特化モデル
- **専門分野**: 日本語理解・生成
- **推奨メモリ**: 64GB以上
- **使用例**:
```bash
python infer_os_japanese_llm_demo.py --model matsuo-lab/weblab-10b --use-aggressive-memory --interactive
```

#### rinna/youri-7b-chat (7Bパラメータ)
- **特徴**: 重量級日本語チャット特化
- **専門分野**: 対話・チャット
- **推奨メモリ**: 48GB以上
- **使用例**:
```bash
python infer_os_japanese_llm_demo.py --model rinna/youri-7b-chat --interactive
```

#### cyberagent/open-calm-7b (7Bパラメータ)
- **特徴**: 重量級バイリンガルモデル
- **専門分野**: 日英バイリンガル
- **推奨メモリ**: 42GB以上
- **使用例**:
```bash
python infer_os_japanese_llm_demo.py --model cyberagent/open-calm-7b --use-8bit --interactive
```

#### stabilityai/japanese-stablelm-instruct-alpha-7b (7Bパラメータ)
- **特徴**: 重量級指示追従特化
- **専門分野**: 指示理解・実行
- **推奨メモリ**: 42GB以上
- **使用例**:
```bash
python infer_os_japanese_llm_demo.py --model stabilityai/japanese-stablelm-instruct-alpha-7b --interactive
```

## ⚙️ コマンドライン引数詳細

### 基本設定

#### --model
使用するモデル名を指定
```bash
--model rinna/youri-7b-chat
```

#### --interactive
インタラクティブモードで実行
```bash
--interactive
```

#### --benchmark
ベンチマーク実行モード
```bash
--benchmark
```

#### --prompt
単発プロンプト実行
```bash
--prompt "日本の四季について説明してください"
```

### 量子化設定

#### --use-4bit
4bit量子化を使用（メモリ削減）
```bash
--use-4bit
```

#### --use-8bit
8bit量子化を使用（バランス型）
```bash
--use-8bit
```

#### --use-advanced-quant
高度な量子化最適化を使用
```bash
--use-advanced-quant
```

#### --quantization-profile
量子化プロファイルを指定
```bash
--quantization-profile safe      # 安全重視
--quantization-profile balanced  # バランス型（デフォルト）
--quantization-profile aggressive # 積極的最適化
```

### メモリ最適化

#### --use-aggressive-memory
積極的メモリ最適化（27.8GB環境対応）
```bash
--use-aggressive-memory
```

### NPU最適化

#### --enable-npu
Windows NPU最適化を有効化（デフォルト）
```bash
--enable-npu
```

#### --disable-npu
Windows NPU最適化を無効化
```bash
--disable-npu
```

### ONNX設定

#### --use-onnx-runtime
ONNX Runtimeを使用
```bash
--use-onnx-runtime
```

#### --onnx-optimization-level
ONNX最適化レベル（0-2）
```bash
--onnx-optimization-level 2  # 最高レベル
```

### その他

#### --list-models
利用可能モデル一覧を表示
```bash
--list-models
```

#### --samples
プロンプトサンプルを表示
```bash
--samples
```

#### --max-length
最大生成長を指定
```bash
--max-length 300
```

## 🎮 インタラクティブモード

### 基本操作

インタラクティブモードでは以下のコマンドが使用できます：

#### テキスト生成
任意のプロンプトを入力してテキスト生成
```
🤖 プロンプトを入力してください: 人工知能の未来について教えてください
```

#### ヘルプ表示
```
🤖 プロンプトを入力してください: help
```
または
```
🤖 プロンプトを入力してください: ヘルプ
```

#### サンプル表示
```
🤖 プロンプトを入力してください: samples
```
または
```
🤖 プロンプトを入力してください: サンプル
```

#### 終了
```
🤖 プロンプトを入力してください: exit
```
または
```
🤖 プロンプトを入力してください: quit
```

### インタラクティブモードの特徴

#### リアルタイム統計表示
- 生成時間
- トークン数
- スループット（tokens/sec）
- メモリ使用量
- CPU使用率

#### 日本語品質分析
- 品質レベル（優秀/良好/普通/要改善）
- 日本語比率
- 文字構成（ひらがな/カタカナ/漢字）

#### 最適化状態表示
- 量子化設定
- メモリ最適化状態
- NPU有効状態

## 📊 ベンチマーク機能

### 基本ベンチマーク実行
```bash
python infer_os_japanese_llm_demo.py --model rinna/youri-7b-chat --benchmark
```

### ベンチマーク内容
1. **人工知能の未来について説明してください**
2. **日本の四季の美しさについて詩を書いてください**
3. **量子コンピュータの仕組みを分かりやすく教えて**
4. **おすすめの日本料理レシピを教えてください**
5. **機械学習の基本概念について説明して**

### ベンチマーク結果
- 実行プロンプト数
- 総実行時間
- 平均生成時間
- 平均スループット
- 総生成トークン数

## 🎯 プロンプトサンプル

### 日常会話
```
今日の天気はどうですか？
おすすめの映画を教えてください
美味しい料理のレシピを教えて
```

### 技術・専門
```
人工知能の最新動向について説明してください
量子コンピュータの仕組みを教えて
機械学習のアルゴリズムについて
```

### 文化・歴史
```
日本の四季について説明してください
江戸時代の文化について教えて
日本の伝統芸能について
```

### 創作・文学
```
短い物語を書いてください
俳句を作ってください
詩を書いてください
```

### 教育・学習
```
数学の基本概念を説明して
歴史の重要な出来事について
科学の面白い現象について
```

## 🔧 環境別推奨設定

### 標準PC環境 (32GB+ メモリ)
```bash
python infer_os_japanese_llm_demo.py \
  --model rinna/youri-7b-chat \
  --use-8bit \
  --interactive
```

### 限定メモリ環境 (27.8GB)
```bash
python infer_os_japanese_llm_demo.py \
  --model rinna/youri-7b-chat \
  --use-aggressive-memory \
  --quantization-profile safe \
  --interactive
```

### NPU搭載PC (Windows 11)
```bash
python infer_os_japanese_llm_demo.py \
  --model rinna/youri-7b-chat \
  --enable-npu \
  --use-advanced-quant \
  --interactive
```

### 最適環境 (64GB+ メモリ + NPU)
```bash
python infer_os_japanese_llm_demo.py \
  --model matsuo-lab/weblab-10b \
  --use-aggressive-memory \
  --enable-npu \
  --use-advanced-quant \
  --quantization-profile aggressive \
  --interactive
```

### GPU環境 (NVIDIA CUDA)
```bash
python infer_os_japanese_llm_demo.py \
  --model rinna/youri-7b-chat \
  --use-8bit \
  --use-advanced-quant \
  --interactive
```

## 📈 性能最適化のコツ

### メモリ使用量を削減したい場合
1. `--use-aggressive-memory` を使用
2. `--quantization-profile safe` を選択
3. より軽量なモデルを選択
4. `--use-4bit` を使用

### 推論速度を向上させたい場合
1. `--enable-npu` を使用（Windows NPU環境）
2. `--use-advanced-quant` を使用
3. `--quantization-profile aggressive` を選択
4. `--use-onnx-runtime` を使用

### 生成品質を重視したい場合
1. `--quantization-profile safe` を使用
2. より大きなモデルを選択
3. `--max-length` を大きく設定

## 🚨 よくある問題と解決方法

### 生成結果が短い・「。」のみ
```bash
# より緩い設定で実行
python infer_os_japanese_llm_demo.py --quantization-profile safe --interactive
```

### メモリ不足エラー
```bash
# 積極的メモリ最適化を使用
python infer_os_japanese_llm_demo.py --use-aggressive-memory --interactive
```

### NPUが検出されない
```bash
# NPU無効で実行
python infer_os_japanese_llm_demo.py --disable-npu --interactive
```

### 推論が遅い
```bash
# 高度な最適化を有効化
python infer_os_japanese_llm_demo.py --use-advanced-quant --enable-npu --interactive
```

## 📊 出力情報の見方

### システム情報
```
📊 システム情報:
  Python: 3.10.0
  PyTorch: 2.0.0
  CPU: 16コア
  メモリ: 32.0GB
  使用中: 8.5GB (26.6%)
  利用可能: 23.5GB
```

### 最適化ライブラリ
```
🔧 最適化ライブラリ:
  Accelerate: ✅
  BitsAndBytes: ✅
  ONNX Runtime: ✅
  高度な量子化最適化: ✅
  Windows NPU最適化: ✅
```

### 生成結果統計
```
📊 日本語生成結果:
  生成時間: 2.34秒
  入力トークン: 15
  出力トークン: 42
  スループット: 17.9 tokens/sec

💾 リソース使用量:
  メモリ使用: 1.2GB
  総メモリ: 9.7GB
  CPU使用率: 45.2%

🇯🇵 日本語品質:
  品質レベル: 優秀
  日本語比率: 95.2%
  文字構成: ひらがな28, カタカナ5, 漢字9
```

---

**より詳細な情報は[性能最適化ガイド](PERFORMANCE_GUIDE.md)をご覧ください！**

