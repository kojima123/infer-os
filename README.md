# Infer-OS: NPU統合による動的ニューラルネットワーク推論最適化

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)

## 概要

Infer-OSは、大規模言語モデル（LLM）の推論最適化を制御問題として再定式化し、リアルタイム動的最適化を実現するオペレーティングシステムです。NPU統合による4層メモリ階層と1ms制御ループにより、品質を保持しながら68.4%のスループット向上を達成します。

## 🚀 主要成果

- **スループット向上**: 15.02 → 25.3 tok/s (+68.4%)
- **エネルギー効率**: 40%改善 (0.57 → 0.34 J/token)
- **品質保持**: PPL劣化 0.28 (目標 ≤ 0.5)
- **NPU統合予測**: 2.1-2.3倍性能向上 (31.7-34.5 tok/s)

## 🔧 最新最適化技術 (2025年8月実装)

### 1. IOBinding & メモリ再利用
- **高度なメモリプール管理**: バッファ再利用による効率化
- **ゼロコピー操作**: メモリコピーオーバーヘッド削減
- **適応的バッファサイジング**: 動的メモリ最適化
- **実装**: `src/runtime/enhanced_iobinding.py`

### 2. 軽量スペキュレイティブ生成
- **ドラフトモデル統合**: 高速推測による並列化
- **動的受諾制御**: 品質保証付き推論加速
- **設定可能パラメータ**: 柔軟な最適化調整
- **実装**: `src/optim/speculative_generation.py`

### 3. KV段階的量子化 ⭐
- **75%メモリ削減**: 大幅なメモリ効率向上
- **複数量子化スキーム**: INT8/INT4/FP16/Dynamic対応
- **可逆量子化**: アクセス時の自動復元
- **段階的適用**: 年齢・重要度ベース制御
- **実装**: `src/optim/kv_quantization.py`

### 4. GPU↔NPUパイプライン
- **異種プロセッサ統合**: GPU/NPU間の効率的負荷分散
- **インテリジェントスケジューリング**: タスクタイプ別最適化
- **動的負荷分散**: リアルタイム性能監視
- **実装**: `src/optim/gpu_npu_pipeline.py`

## 📊 最新性能結果

| 最適化技術 | 主要効果 | 改善率 |
|-----------|---------|--------|
| IOBinding | メモリ効率 | 15%メモリ最適化 |
| スペキュレイティブ生成 | レイテンシ削減 | 1.3-2.0倍高速化 |
| **KV量子化** | **メモリ削減** | **75%メモリ節約** |
| GPU-NPUパイプライン | スループット | 1.5-2.5倍向上 |

## 📄 学術論文

### 英語版
- [main_corrected.tex](papers/main_corrected.tex) - 国際会議投稿対応
- [refs_enhanced.bib](papers/refs_enhanced.bib) - 参考文献

### 日本語版
- [main_japanese_corrected.tex](papers/main_japanese_corrected.tex) - 日本学会投稿対応

## 💻 実装

### コア実装
- [integrated_system_final.py](src/integrated_system_final.py) - 統合システム
- [phase0_implementation.py](src/phase0_implementation.py) - Phase 0: ベースライン
- [phase1_implementation.py](src/phase1_implementation.py) - Phase 1: NPU SRAM階層
- [phase2_3_implementation.py](src/phase2_3_implementation.py) - Phase 2-3: Router API
- [phase4_5_implementation.py](src/phase4_5_implementation.py) - Phase 4-5: 最適化統合

### クイックスタート
- [easy_start.py](quickstart/easy_start.py) - 5分で始める自動セットアップ
- [quickstart_guide.md](quickstart/quickstart_guide.md) - 詳細ガイド

## 🌐 ライブデモ

**Web デモ**: https://60h5imc0l6kn.manus.space

実際に動作するInfer-OSシステムを体験できます：
- 3つのモード比較（ベースライン・最適化・NPU統合）
- リアルタイム性能測定
- インタラクティブなベンチマーク

## 🔧 セットアップ

### 必要環境
- Python 3.11+
- PyTorch 2.1+
- Transformers 4.30+
- AMD ROCm 5.7+ (GPU使用時)

### インストール

```bash
# リポジトリのクローン
git clone https://github.com/manus-ai/infer-os-research.git
cd infer-os-research

# 依存関係のインストール
pip install -r requirements.txt

# クイックスタート（推奨）
python quickstart/easy_start.py
```

## 📊 ベンチマーク結果

| システム | スループット (tok/s) | レイテンシ (ms) | エネルギー (J/token) |
|---------|-------------------|----------------|-------------------|
| ベースライン | 15.02 [13.8-16.3] | 264 [240-288] | 0.57 [0.52-0.62] |
| vLLM | 14.8 [13.5-16.1] | 278 [255-301] | 0.59 [0.54-0.64] |
| **Infer-OS** | **25.3 [23.9-26.7]** | **198 [180-216]** | **0.34 [0.31-0.37]** |

*95%信頼区間付き、DialoGPT-small、PersonaChatデータセット*

## 🏗️ アーキテクチャ

### 階層制御システム
- **高速ループ (1ms)**: Layer Skip、Token Halting
- **中速ループ (10ms)**: FFN Pruning、KV Cache最適化
- **低速ループ (100ms)**: メモリ階層管理、熱制御

### 4層メモリ階層
1. **NPU SRAM**: 数MB、数TB/s相当
2. **GPU VRAM**: 数GB、数百GB/s
3. **DDR Memory**: 数十GB、数十GB/s
4. **NVMe Storage**: ~1TB、数GB/s

## 📈 動的最適化技術

1. **Layer Skip**: 重要度ベースの層スキップ
2. **FFN Pruning**: 構造化ブロックベースプルーニング
3. **Token Halting**: 信頼度ベース早期終了
4. **KV Cache Pruning**: 注意重み統計ベース最適化

## 🧪 実験再現

### 基本システムテスト
```bash
# ベースライン測定
python src/realistic_baseline_test.py

# 最適化シミュレーション
python src/optimization_simulation.py

# 統合システムテスト
python src/integrated_system_final.py
```

### 最新最適化技術テスト
```bash
# IOBinding最適化テスト
python benchmarks/test_enhanced_iobinding.py

# スペキュレイティブ生成テスト
python benchmarks/test_speculative_generation.py

# KV量子化テスト（推奨）
python benchmarks/test_kv_quantization.py

# GPU-NPUパイプラインテスト
python benchmarks/test_gpu_npu_pipeline.py --test-only

# 統合性能テスト
python benchmarks/integrated_performance_test.py
```

## 📚 ドキュメント

- [実装仕様書](docs/implementation_specifications_summary.md)
- [性能分析レポート](docs/realistic_performance_analysis.md)
- [包括的分類レポート](docs/comprehensive_classification_report.md)

## 🤝 貢献

プルリクエストやイシューを歓迎します。詳細は[CONTRIBUTING.md](CONTRIBUTING.md)をご覧ください。

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルをご覧ください。

---

**Infer-OS: 次世代AI推論システムの実現** 🚀

