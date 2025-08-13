# Infer-OS × AMD GAIA 統合 実装仕様書 v1.0

## 📋 **ドキュメント情報**

- **文書名**: Infer-OS × AMD GAIA 統合 実装仕様書 v1.0
- **対象**: Windows 11 + GAIA（RAUX/CLI → Lemonade Server → ORT GenAI → NPU/iGPU）
- **想定読者**: ソフトウェア実装担当、MLOps、SRE
- **非目標**: NPU ドライバ実装や ORT コア改変（必要ならフェーズ3で検討）

## 🎯 **目的・スコープ**

### **目的**
Infer-OSの制御ループ・KV段階的量子化・IOBinding最適化を、GAIAスタックへ無停止で統合し、スループット向上・レイテンシ短縮・メモリ削減を実現。

### **スコープ**
- Lemonade Server（推論サーバ）とORT GenAI実行系への制御プレーン注入
- KV-cacheの**可逆段階的量子化（FP16/INT8/INT4/退避）**の組み込み
- iGPU（DML）優先のIOBindingとメモリ再利用
- Hybrid（NPU+iGPU）の運用最適化（Prefill→iGPU、Decode→NPU を基本、動的切替）

## 🔧 **前提・依存関係**

### **システム要件**
- **OS**: Windows 11（GAIA 要件）
- **ハード**: Ryzen AI（XDNA NPU）+ Radeon iGPU（RDNA、DML 経由）

### **実行基盤**
- **GAIA**: RAUX/gaia-cli → Lemonade Server → ONNX Runtime GenAI（EP: Hybrid / DML / CPU）
- **言語**: .NET / Python（Lemonade の実装言語に依存、標準は Python/TypeScript 混在）

### **開発環境**
- **言語/ビルド**: C++17/20、Rust 1.7x、Python 3.11、CMake、MSVC、Wheel/nuget（最小）
- **バージョン固定**: GAIA 推奨の NPU/iGPU ドライバ & ORT GenAI と同一

## 🏗️ **アーキテクチャ概要**

### **構成（レイヤ）**
```
RAUX / gaia-cli
   ↓
Lemonade Server（推論API）
   ↓        ↑
Infer-OS Control Agent（Windowsサービス／サイドカー）
   ↓        ↑
ONNX Runtime GenAI （EP: Hybrid / DML / CPU）
   ↓
NPU（Decode向け） / Radeon iGPU（Prefill/Attention向け）
```

### **主要コンポーネント**
- **Infer-OS Control Agent**: 1ms/10ms/100ms 制御ループ、KV 量子化ポリシ、IOBinding 管理、テレメトリ集約を担当
- **Lemonade Adapter**: Lemonade から前後処理（PreRun/PostRun）フック、KV 読み書きフック、設定反映を行う軽量プラグイン
- **ORT Adapter**: SessionOptions/RunOptions/IOBinding、QDQ（Quantize/Dequantize）カスタムOPの注入（フェーズ2）

### **フェーズ別導入**

#### **Phase 1: サイドカー（最短導入）**
- Lemonade 側は最小改変
- HTTP/CLI/環境変数で ORT/実行パラメータを動的変更
- KV は外部ラッパで管理

#### **Phase 2: プラグイン連携**
- Lemonade に KV-Manager と Pre/PostRun Hooks を実装
- QDQ カスタムOP を ORT グラフに挿入

#### **Phase 3: ハード連携**
- Prefill/Attention→iGPU、Decode→NPU の切替を制御ループで動的最適化
- KV の二層キャッシュ（VRAM 常駐＋必要時 NPU 転送）

## 🔧 **コンポーネント仕様**

### **4.1 Infer-OS Control Agent（Windows サービス）**

#### **役割**
制御ループ、方策決定、KV ポリシ管理、メトリクス、設定API

#### **実装**
Rust or C++ 常駐サービス + 内蔵 HTTP（127.0.0.1）

#### **公開 API（REST/JSON）**
- `POST /v1/policy`: 最適化方策を適用（例：kv.mode、kv.window、安全閾値、io.bind、scheduler.mode）
- `POST /v1/run-context`: 次ジョブの想定条件（L、batch、温度/上限、品質目標）を通知
- `GET /v1/metrics`: tokens/s、FTL、p95、VRAM/Host 使用量、ΔPPL 推定、NPU/iGPU 利用率
- `GET /v1/health`: 稼働確認

#### **制御ループ**
- **Fast（1ms）**: 負荷急変・熱・帯域観測、Token Halting/KV 直近窓の緊急保護
- **Mid（10ms）**: IOBinding/バッファ再利用、KV 量子化レベルの微調整
- **Slow（100ms）**: Prefill/Decode デバイス配分、量子化プロファイル切替、閾値再学習

#### **フェイルセーフ**
エラー時は **Baseline（最適化OFF）** へ自動フォールバック

### **4.2 Lemonade Adapter（プラグイン or ミドルウェア）**

#### **Hook ポイント**
- **PreRun**: モデル/入力条件/要求SLOを Infer-OS に通知、返却された方策を ORT に反映
- **KVWrite / KVRead**: KV 書込/読込時に 段階的量子化/復元を適用（Phase2）
- **PostRun**: 実績メトリクス送信（Infer-OS に学習データとして渡す）

#### **設定反映**
- ORT SessionOptions/RunOptions（スレッド、Graph Optimization Level、Arena、並列度）
- IOBinding（入力・出力バッファ、KV バッファの再利用）
- Hybrid EP の使用方針（Prefill→DML、Decode→NPU）

### **4.3 ORT Adapter**

#### **IOBinding**
DML デバイス上に事前確保したバッファプールを作り回す

#### **QDQ カスタムOP（Phase2）**
- KVQuantize(op=Q_kv)、KVDequantize(op=DQ_kv) を KV の境界に挿入
- ランタイムで精度レベル切替（FP16/INT8/INT4）

**注意**: NPU 直下での IOBinding / カスタム OP は制約がある可能性 → iGPU 側優先で実装し、NPU は Decode 偏重活用

## 🧮 **アルゴリズム仕様**

### **5.1 KV 段階的量子化（可逆）**

#### **レベル**
- **L0**: FP16（最新・重要）
- **L1**: INT8
- **L2**: INT4
- **L3**: 退避（圧縮/Drop）

#### **重要度スコア**
```
I = α·AttnWeight + β·AccessFreq + γ·SemanticRel – η·Age
```

#### **決定ロジック（擬似コード）**
```python
def decide_level(entry, mem_pressure, quality_budget, temp, bw):
    I = attn(entry)*α + freq(entry)*β + sem(entry)*γ - age(entry)*η
    if entry.is_recent(W=64): return L0  # 近傍安全窓は常にFP16
    if mem_pressure > 0.8 and I < 0.3:  return L3
    if mem_pressure > 0.6 and I < 0.5:  return L2
    if I < 0.7 or quality_budget < 0:   return L1
    return L0
```

#### **可逆性**
L1/L2 は復元メタ（scale/zero-point/shape/chunk id）を付与。品質劣化検知時は L0 へ昇格。

### **5.2 IOBinding ＆ バッファ再利用**
- 事前に iGPU(DML) 側で KV/入出力用の固定サイズリングバッファを確保
- 実行毎に OrtIoBinding.BindInput/BindOutput を使い ゼロコピーで回す
- 足りない場合のみ拡張（断片化を避ける）

### **5.3 Hybrid スケジューラ（Prefill/Decode）**

#### **方針**
Prefill/Attention は iGPU（DML）、Decode は NPU

#### **切替条件（Slow ループ）**
- temp>閾値 or NPU queue 深い → Decode を一時 iGPU へ
- 逆に iGPU サーマル高騰時は NPU へ回帰

#### **目的関数**
```
min{ p95 / TPS – λ·QualityReserve + μ·J/token }
```

## ⚙️ **設定（YAML）**

```yaml
infer_os:
  quality:
    max_delta_ppl: 0.5
    min_accept_rate: 0.5        # speculation を使う場合
  kv:
    recent_window: 64           # 常にFP16で保持
    level_thresholds:           # αβγη はモデルごとにチューニング
      L1_int8: 0.7
      L2_int4: 0.5
      L3_evict: 0.3
  io:
    enable_iobinding: true
    dml_pool_bytes: 2048MiB
    host_pool_bytes: 4096MiB
  scheduler:
    mode: hybrid                # hybrid|gpu_only|npu_only
    prefill_device: dml
    decode_device: npu
  loops:
    fast_ms: 1
    mid_ms: 10
    slow_ms: 100
  telemetry:
    push_interval_ms: 1000
```

## 🌐 **外部 API（Infer-OS Control Agent）**

### **POST /v1/policy**
- **入力**: `{ "kv":{ "mode":"dynamic", "recent_window":64 }, "io":{"enable_iobinding":true}, "scheduler":{"mode":"hybrid"} }`
- **出力**: `204 No Content`

### **POST /v1/run-context**
- **入力**: `{ "seq_len":1024, "batch":4, "target_ftl_ms":300, "quality_budget":0.5 }`

### **GET /v1/metrics**
- **出力例**:
```json
{
  "tps": 28.7, "ftl_ms": 210, "p95_ms": 320,
  "mem": { "vram_gb": 2.8, "host_gb": 6.3, "kv_levels": {"L0":22,"L1":58,"L2":17,"L3":3} },
  "util": { "npu": 0.41, "igpu": 0.63, "cpu": 0.22 },
  "quality": { "delta_ppl_est": 0.18, "spec_accept": 0.62 }
}
```

### **GET /v1/health**
- **出力**: `{status:"ok"}`

## 📊 **データモデル（KV メタ）**

```cpp
struct KvChunk {
  TensorId id;            // 層・ヘッド・時刻範囲
  Level level;            // L0/L1/L2/L3
  Shape shape;            // [H, T, D]
  ScaleZeroMeta qmeta;    // L1/L2 のみ
  uint64 last_access_ts;  // ns
  float  importance;      // I
  bool   pinned;          // recent_window 内は true
}
```

## 🎯 **性能目標（SLO）**

### **7B/8B**
- **TPS**: 1.3–1.6× 向上
- **メモリ**: −60–75% 削減
- **FTL**: −20–30% 短縮

### **20B**
- **TPS**: ≥1.8× 向上（Hybrid 前提）
- **メモリ**: −70–80% 削減

### **安定性**
24h 連続でリークなし、温度逸脱時の自動減速

## 📈 **テレメトリ・ロギング**

### **メトリクス収集**
1s 間隔で JSON Lines（logs/metrics_*.jsonl）へ書出し

### **例**
```
t, model, seq_len, batch, tps, ftl, p95, vram, host_mem, npu_util, igpu_util, kv_L0/L1/L2/L3, delta_ppl_est
```

### **重大イベント**
フェイルオーバ、量子化レベル大規模変化、EP 切替

## 🧪 **テスト計画**

### **単体**
KV Q/DQ の MSE、最大誤差、可逆性、recent_window 保護

### **結合**
Lemonade Adapter 経由で ORT IOBinding が効いているか（コピー回数、帯域）

### **負荷**
L={128,512,1024,2048}, batch={1,4,8}、連続 6h・24h

### **A/B**
Infer-OS ON/OFF、safe/balanced/aggressive プロファイル

### **回帰**
ドライバ更新/モデル更新で性能劣化が ±5% 内

### **合格基準（例）**
- KV 量子化時の ΔPPL 推定 ≤ 0.5（評価セットで検証）
- 7B/20B の SLO を 3/4 条件で満たす
- 異常時の Baseline フォールバックが 1 秒以内

## 🚀 **デプロイ・運用**

### **配布物**
- `inferos-agent.exe`（Windows サービス）
- `lemonade-inferos-adapter`（pip/nuget module）
- `ort-kv-qop.dll`（カスタムOP；Phase2 以降）

### **導入手順**
1. ドライバ/GAIA/ORT を推奨版へ固定
2. `inferos-agent.exe install`（サービス登録）
3. Lemonade の設定で Adapter を有効化（env/設定ファイル）
4. RAUX/CLI 経由でベースライン採取 → ON に切替

### **ロールバック**
Lemonade のフラグで Adapter 無効化、サービス停止

## 🔒 **セキュリティ・権限**

### **セキュリティ**
- Agent は localhost のみで待受、TLS は任意（社内閉域推奨）
- 最小権限（ネットワークなし、ファイル権限限定）
- 収集メトリクスにユーザ生成テキストを保存しない（ハッシュ/要約）

## ⚠️ **既知のリスクと対策**

### **NPU EP 制約**
IOBinding/カスタムOP不可な場合 → iGPU 側で最大効果を出す、NPU は Decode 偏重

### **更新追従**
GAIA/ORT 更新で API 変更 → サイドカー（Phase1）を維持し、プラグインはバージョンピン

### **温度・電力**
連続高負荷でクロック低下 → Slow ループで目標 TPS を再調整

## 📅 **スケジュール（目安）**

- **W1–2**: Phase1 実装（Agent・Adapter 最小、IOBinding、メトリクス）
- **W3–4**: A/B・長時間試験、プロファイル調整（safe/balanced/aggressive）
- **W5–7**: Phase2（KV フック、QDQ カスタムOP、コピー削減）
- **W8+**: Phase3（Hybrid 動的切替の高度化、NPU 常駐最適化）

## 📋 **受け入れ基準（Definition of Done）**

### **Phase1**
- 7B/8B モデルにて TPS ≥ 1.3×、メモリ −60%、FTL −20% を再現
- 24h 連続試験でエラー率 < 0.1%、メモリリークなし
- OFF→ON の切替が無停止（API で即時反映）
- ロールバック（Adapter 無効化）で元の GAIA 挙動に戻る

この仕様で着手すれば、まずはサイドカー導入で効果を見せ、次にプラグインでコピー削減・レイテンシ短縮を詰める流れに乗れます。

