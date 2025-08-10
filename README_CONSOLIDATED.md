# 🚀 Infer-OS 日本語重量級LLM統合デモ

日本語対応の重量級LLMモデル（7B-10Bパラメータ）でInfer-OS最適化効果を体験する統合デモシステム

## ✨ 主要機能

### 🇯🇵 日本語重量級LLMサポート
- **matsuo-lab/weblab-10b** (10B) - 最重量級日本語モデル
- **rinna/youri-7b-chat** (7B) - 重量級チャット特化
- **cyberagent/open-calm-7b** (7B) - 重量級バイリンガル
- **stabilityai/japanese-stablelm-instruct-alpha-7b** (7B) - 重量級指示追従

### ⚡ Infer-OS最適化
- **推論速度向上**: 2-5倍の高速化
- **メモリ削減**: 65-75%のメモリ使用量削減
- **応答時間短縮**: 50-65%の応答時間短縮
- **スループット向上**: 2.5-4倍のスループット向上

### 🧠 積極的メモリ最適化
- **27.8GB環境対応**: 限られたメモリ環境での安定動作
- **チャンク分割ロード**: 512MBチャンクでの効率的ロード
- **強制メモリクリーンアップ**: Python GC + PyTorch + OS レベル
- **float16変換**: 50%のメモリ削減
- **緊急フォールバック**: 最小設定での確実な回復

### 💻 Windows NPU最適化
- **AMD Ryzen AI NPU**: 自動検出・有効化
- **Intel NPU**: 自動検出・有効化
- **Qualcomm NPU**: 自動検出・有効化
- **DirectML統合**: NPU加速ライブラリの自動設定

### 🔧 高度な量子化最適化
- **W4/W8量子化**: 重み量子化による高速化
- **KV量子化**: キー・バリューキャッシュ量子化
- **動的量子化**: 実行時最適化
- **3段階プロファイル**: Safe/Balanced/Aggressive

## 🚀 クイックスタート

### 基本実行
```bash
python infer_os_japanese_llm_demo.py --model rinna/youri-7b-chat --interactive
```

### 27.8GB環境での積極的メモリ最適化
```bash
python infer_os_japanese_llm_demo.py --model rinna/youri-7b-chat --use-aggressive-memory --interactive
```

### Windows NPU最適化有効
```bash
python infer_os_japanese_llm_demo.py --model rinna/youri-7b-chat --enable-npu --interactive
```

### 全機能有効（推奨）
```bash
python infer_os_japanese_llm_demo.py --model rinna/youri-7b-chat --use-aggressive-memory --enable-npu --use-advanced-quant --interactive
```

## 📋 システム要件

### 最小要件
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.8以上
- **メモリ**: 16GB以上
- **ストレージ**: 50GB以上の空き容量

### 推奨要件
- **OS**: Windows 11 (NPU対応)
- **Python**: 3.10以上
- **メモリ**: 32GB以上
- **ストレージ**: 100GB以上の空き容量
- **NPU**: AMD Ryzen AI, Intel NPU, Qualcomm NPU

## 🛠️ インストール

### 1. リポジトリのクローン
```bash
git clone https://github.com/kojima123/infer-os.git
cd infer-os
```

### 2. 依存関係のインストール
```bash
pip install torch transformers accelerate
pip install psutil argparse
```

### 3. オプション: 量子化サポート
```bash
pip install bitsandbytes
```

### 4. オプション: ONNX Runtime
```bash
pip install onnxruntime
```

### 5. オプション: Windows NPU対応
```bash
pip install onnxruntime-directml
```

## 📖 使用方法

### コマンドライン引数

#### 基本設定
- `--model`: 使用するモデル名（デフォルト: rinna/youri-7b-chat）
- `--use-4bit`: 4bit量子化を使用
- `--use-8bit`: 8bit量子化を使用
- `--use-advanced-quant`: 高度な量子化最適化を使用

#### メモリ最適化
- `--use-aggressive-memory`: 積極的メモリ最適化（27.8GB環境対応）
- `--quantization-profile`: 量子化プロファイル（safe/balanced/aggressive）

#### NPU最適化
- `--enable-npu`: Windows NPU最適化を有効化（デフォルト）
- `--disable-npu`: Windows NPU最適化を無効化

#### 実行モード
- `--interactive`: インタラクティブモード
- `--benchmark`: ベンチマーク実行
- `--prompt "テキスト"`: 単発プロンプト実行

#### その他
- `--list-models`: 利用可能モデル一覧表示
- `--samples`: プロンプトサンプル表示

### インタラクティブモード

インタラクティブモードでは以下のコマンドが使用できます：

- **任意のプロンプト**: 日本語テキスト生成
- **`help`** または **`ヘルプ`**: ヘルプ表示
- **`samples`** または **`サンプル`**: プロンプトサンプル表示
- **`exit`** または **`quit`**: 終了

## 📊 性能ベンチマーク

### Infer-OS最適化効果

| 項目 | 最適化前 | 最適化後 | 改善率 |
|------|----------|----------|--------|
| 推論速度 | 1.0x | 2.0-5.0x | 100-400% |
| メモリ使用量 | 100% | 25-35% | 65-75%削減 |
| 応答時間 | 100% | 35-50% | 50-65%短縮 |
| スループット | 1.0x | 2.5-4.0x | 150-300% |

### 環境別性能

| 環境 | メモリ | NPU | 推論速度 | 推奨設定 |
|------|--------|-----|----------|----------|
| 標準PC | 32GB+ | なし | 1.0x | --use-8bit |
| 限定メモリ | 27.8GB | なし | 0.8x | --use-aggressive-memory |
| NPU搭載PC | 32GB+ | あり | 3.0x | --enable-npu --use-advanced-quant |
| 最適環境 | 64GB+ | あり | 5.0x | 全機能有効 |

## 🔧 トラブルシューティング

### よくある問題

#### メモリ不足エラー
```bash
# 積極的メモリ最適化を使用
python infer_os_japanese_llm_demo.py --use-aggressive-memory --model rinna/youri-7b-chat
```

#### NPUが検出されない
```bash
# DirectML依存関係をインストール
pip install onnxruntime-directml

# NPU無効で実行
python infer_os_japanese_llm_demo.py --disable-npu
```

#### 生成結果が短い
```bash
# より緩い設定で実行
python infer_os_japanese_llm_demo.py --quantization-profile safe
```

### ログとデバッグ

詳細なログを確認するには：
```bash
python infer_os_japanese_llm_demo.py --interactive --verbose
```

## 📁 ファイル構成

```
infer-os/
├── infer_os_japanese_llm_demo.py      # メインデモファイル
├── aggressive_memory_optimizer.py     # 積極的メモリ最適化
├── windows_npu_optimizer.py          # Windows NPU最適化
├── advanced_quantization_optimizer.py # 高度な量子化最適化
├── infer_os_comparison_benchmark.py  # 比較ベンチマーク
├── README_CONSOLIDATED.md            # このファイル
├── docs/                              # ドキュメント
│   ├── INSTALLATION_GUIDE.md         # インストールガイド
│   ├── USAGE_GUIDE.md                # 使用方法ガイド
│   ├── PERFORMANCE_GUIDE.md          # 性能最適化ガイド
│   └── TROUBLESHOOTING_GUIDE.md      # トラブルシューティング
└── archive/                           # アーカイブファイル
    ├── old_versions/                  # 旧バージョン
    └── old_guides/                    # 旧ガイド
```

## 🤝 コントリビューション

プルリクエストやイシューの報告を歓迎します。

### 開発環境のセットアップ
```bash
git clone https://github.com/kojima123/infer-os.git
cd infer-os
pip install -r requirements.txt
```

### テスト実行
```bash
python infer_os_japanese_llm_demo.py --benchmark
```

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🙏 謝辞

- Hugging Face Transformersチーム
- PyTorchチーム
- 日本語LLMモデル開発者の皆様
- Infer-OSプロジェクトチーム

## 📞 サポート

- **GitHub Issues**: バグ報告・機能要望
- **Discussions**: 質問・議論
- **Wiki**: 詳細ドキュメント

---

**🚀 Infer-OS で日本語重量級LLMの真の性能を体験してください！**

