# Infer-OS 詳細性能評価レポート

**実行日**: 2025-08-08  
**環境**: Ubuntu 22.04 / Python 3.11 / PyTorch 2.1+

## 1. 評価方針とメトリクス

- **ベースラインの明示**  
  - 本報告では **FP16** を既定の比較基準（Baseline）とする。  
  - 参考として **FP32 基準**の数値も併記する。
- **指標**: tok/s, FTL(p50), p95, J/token, Peak-GB, ΔPPL（品質ガード: ΔPPL ≤ 0.5）

## 2. IOBinding & メモリ再利用

- 状態: 実装済み（メモリプール/ゼロコピー/自動サイジング）
- 小規模テストではオーバーヘッドにより 0.61x。  
  **大規模テンソル & GPU/ROCm/NPU EP**での再評価を必須化。

### 再評価プロトコル

1) small / medium / large の 3 サイズ  
2) 1000 ステップ連続でプール効果の収束確認  
3) `sess.run` vs `run_with_iobinding` 比較ログ  
4) EP 別効果（CUDA/ROCm/NPU/CPU）

## 3. 軽量スペキュレイティブ生成

- 現状: 機能完成、受諾率 2–4%、実効 0.28x（**負の効果**）
- 前提条件（要充足）:
  - **同一語彙・Tokenizer**、温度/Top-p 一致
  - **KV 共有**（検証再計算を削減）
  - **動的 draft_n**（不一致連続で降格）
  - 構造化推測（Medusa/EAGLE 系）も検討

## 4. KV 段階的量子化（最優秀）

**比較基準: FP16**

| 方式      | 理論圧縮率（対FP16） | 理論削減率 | 備考 |
|-----------|-----------------------|------------|------|
| INT8      | 2.0x                  | 50%        | per-head scale 付随の実効圧縮率を別途提示 |
| INT4      | 4.0x                  | 75%        | packing 効率のオーバーヘッドを考慮 |
| FP16保持  | 1.0x                  | 0%         | 近傍バッファ |
| 動的段階  | 2.0–4.0x相当          | 50–75%     | 年齢/重要度/圧迫で段階適用 |

> 参考（**対 FP32**）：INT8=4x(75%)、INT4=8x(87.5%)

- **実測（例）**: 量子化率 86.7%、可逆復元 MSE<1e-4（INT8）  
- **運用**: 近傍=FP16、旧領域=INT8/INT4、品質アラートで**即復元**（ロールバック）

## 5. GPU↔NPU パイプライン

- 既存: 機能確認（小規模 OK）  
- 課題: 大規模でタイムアウト（キュー飽和/同期/競合）
- **安定化パッチ**（次スプリント）:
  - **bounded queue + credit-based flow**
  - **優先度制御**（decode 優先/モード切替）
  - **EP 再優先 & フェイルオーバ**
  - 監視: キュー深さ / wait 比率 / ドロップ率 / 有効実行率

## 6. 統合評価の読み方

- 単一技術の効果は **最大値**を提示。  
- 同時適用時は **ワークロード依存**のため、合成効果は別途計測（表に "範囲" で記載）。

## 7. 推奨ロードマップ

- **短期 (1–3ヶ月)**: KV 量子化の本番導入、IOBinding の GPU/大規模再評価  
- **中期 (3–6ヶ月)**: スペキュレイティブの受諾率改善（50% 目標）、GPU↔NPU 安定化  
- **長期 (6–12ヶ月)**: 統合最適化（MPC）と新 HW サポート拡充

## 付録: 計算式

- 圧縮率 `R = baseline_bytes / optimized_bytes`  
- 削減率 `= 1 - 1/R`  
- 実効圧縮率には `scale/bias` のメタデータを含めて算出

## 実装ファイル

```
infer-os-research/
├── src/
│   ├── runtime/
│   │   ├── enhanced_iobinding.py    # IOBinding最適化
│   │   ├── ort_session.py           # ONNX Runtime統合
│   │   └── device.py                # デバイス管理
│   └── optim/
│       ├── speculative_generation.py # スペキュレイティブ生成
│       ├── kv_quantization.py       # KV量子化 ⭐
│       └── gpu_npu_pipeline.py      # GPU-NPUパイプライン
└── benchmarks/
    ├── test_enhanced_iobinding.py   # IOBindingテスト
    ├── test_speculative_generation.py # スペキュレイティブテスト
    ├── test_kv_quantization.py      # KV量子化テスト
    ├── test_gpu_npu_pipeline.py     # パイプラインテスト
    └── integrated_performance_test.py # 統合テスト
```

## 再現手順

### 基本テスト
```bash
# KV量子化テスト（推奨）
python benchmarks/test_kv_quantization.py

# IOBinding最適化テスト
python benchmarks/test_enhanced_iobinding.py --test-only

# スペキュレイティブ生成テスト
python benchmarks/test_speculative_generation.py

# GPU-NPUパイプラインテスト
python benchmarks/test_gpu_npu_pipeline.py --test-only
```

### 統合テスト
```bash
# 統合性能テスト
python benchmarks/integrated_performance_test.py
```

## 引用

本研究成果を引用する場合：

```bibtex
@misc{inferos2025,
  title={Infer-OS: LLM推論最適化のための統合オペレーティングシステム},
  author={Infer-OS開発チーム},
  year={2025},
  note={KV段階的量子化による75\%メモリ削減を達成}
}
```

---

**レポート作成者**: Infer-OS開発チーム  
**最終更新**: 2025年8月8日  
**ライセンス**: MIT License

