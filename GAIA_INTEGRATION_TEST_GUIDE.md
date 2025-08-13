# GAIA × Infer-OS 統合テスト手順書

## 📋 概要

このドキュメントは、GAIA × Infer-OS統合システムのテスト手順を説明します。シミュレーションテストから実機テストまで、段階的なテスト方法を提供します。

## 🎯 テスト目標

- **統合動作確認**: 全コンポーネントの正常動作
- **パフォーマンス測定**: 最適化効果の定量評価
- **品質保証**: 推論品質の維持確認
- **運用耐性**: エラーハンドリング・フォールバック機能

## 📊 テスト結果サマリー（最新実行）

```
🎉 統合テスト成功! 成功率: 100.0%

📈 パフォーマンス向上: 71.0%
📉 メモリ削減: 49.1%
⚡ 実行時間: 1.6秒
```

## 🧪 Phase 1: シミュレーションテスト

### 1.1 前提条件

#### **必要な環境**
```bash
# Python 3.8以上
python --version

# 必要なライブラリ
pip install torch numpy fastapi uvicorn pydantic aiohttp pyyaml websockets
```

#### **ファイル構成確認**
```bash
# 必要なファイルが存在することを確認
ls -la gaia_*.py
# 出力例:
# gaia_integration_simulation_test.py
# gaia_sidecar_integration.py
# gaia_control_agent.py
# gaia_lemonade_adapter.py
# gaia_kv_quantization_engine.py
```

### 1.2 シミュレーションテスト実行

#### **基本実行**
```bash
# 統合シミュレーションテスト実行
python3 gaia_integration_simulation_test.py
```

#### **期待される出力**
```
🧪 GAIA × Infer-OS 統合シミュレーションテスト
============================================================
2025-08-13 08:37:08,002 - __main__ - INFO - 🔧 Phase 1: サイドカー統合テスト
2025-08-13 08:37:08,002 - __main__ - INFO - ✅ サイドカー統合テスト完了: 0.0ms
2025-08-13 08:37:08,002 - __main__ - INFO - 🔧 Phase 2: Control Agentテスト
2025-08-13 08:37:08,002 - __main__ - INFO - ✅ Control Agentテスト完了: 0.0ms
2025-08-13 08:37:08,002 - __main__ - INFO - 🔧 Phase 3: Lemonade Adapterテスト
2025-08-13 08:37:08,003 - __main__ - INFO - ✅ Lemonade Adapterテスト完了: 0.6ms
2025-08-13 08:37:08,003 - __main__ - INFO - 🔧 Phase 4: KV量子化エンジンテスト
2025-08-13 08:37:09,578 - __main__ - INFO - ✅ KV量子化エンジンテスト完了: 1574.6ms
2025-08-13 08:37:09,578 - __main__ - INFO - 🔧 Phase 5: 統合パフォーマンステスト
2025-08-13 08:37:09,578 - __main__ - INFO - 📊 baseline: 2.0 TPS, 2048.0MB
2025-08-13 08:37:09,578 - __main__ - INFO - 📊 kv_quantization_only: 2.7 TPS, 1433.6MB
2025-08-13 08:37:09,578 - __main__ - INFO - 📊 io_binding_only: 2.5 TPS, 1843.2MB
2025-08-13 08:37:09,578 - __main__ - INFO - 📊 full_optimization: 3.5 TPS, 1041.9MB
2025-08-13 08:37:09,578 - __main__ - INFO - ✅ 統合パフォーマンステスト完了: 0.1ms
2025-08-13 08:37:09,578 - __main__ - INFO - 📈 パフォーマンス向上: 71.0%
2025-08-13 08:37:09,578 - __main__ - INFO - 📉 メモリ削減: 49.1%
🎉 統合テスト成功! 成功率: 100.0%
```

### 1.3 テスト結果の解釈

#### **成功基準**
- **成功率**: 80%以上
- **パフォーマンス向上**: 50%以上
- **メモリ削減**: 30%以上
- **全コンポーネント**: エラーなし

#### **結果分析**
```json
{
  "summary": {
    "total_tests": 5,
    "successful_tests": 5,
    "success_rate_percent": 100.0
  },
  "performance_summary": {
    "performance_improvement_percent": 71.0,
    "memory_reduction_percent": 49.1,
    "target_tps_achieved": false
  }
}
```

## 🔧 Phase 2: 個別コンポーネントテスト

### 2.1 サイドカー統合テスト

```bash
# サイドカー統合の単体テスト
python3 gaia_sidecar_integration.py
```

**期待される結果**:
- 設定適用: 成功
- 統計情報取得: 正常
- 実行時間: 1ms未満

### 2.2 Control Agentテスト

```bash
# Control Agentの単体テスト
python3 gaia_control_agent.py
```

**期待される結果**:
- 認証トークン生成: 成功
- ポリシー更新: 成功
- メトリクス取得: 正常

### 2.3 Lemonade Adapterテスト

```bash
# Lemonade Adapterの単体テスト
python3 gaia_lemonade_adapter.py
```

**期待される結果**:
- 設定ファイル生成: 成功
- 動的最適化適用: 成功
- CLI引数生成: 正常

### 2.4 KV量子化エンジンテスト

```bash
# KV量子化エンジンの単体テスト
python3 gaia_kv_quantization_engine.py
```

**期待される結果**:
- 量子化実行: 成功
- 品質監視: 正常
- メモリ削減: 70%以上

## 🚀 Phase 3: 実機テスト（AMD GAIA環境）

### 3.1 前提条件

#### **ハードウェア要件**
- AMD Ryzen AI NPU搭載CPU
- Radeon iGPU
- 16GB以上のメモリ
- Windows 11 22H2以降

#### **ソフトウェア要件**
```bash
# GAIA CLI インストール確認
gaia-cli --version

# DirectML確認
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# 期待される出力: ['DmlExecutionProvider', 'CPUExecutionProvider']
```

### 3.2 実機テスト実行

#### **Step 1: 環境確認**
```bash
# NPU検出確認
python -c "
import subprocess
result = subprocess.run(['wmic', 'path', 'win32_processor', 'get', 'name'], 
                       capture_output=True, text=True)
print('NPU検出:', 'AI' in result.stdout)
"
```

#### **Step 2: GAIA統合テスト**
```bash
# 実機統合テスト実行
python3 gaia_integration_simulation_test.py --real-hardware
```

#### **Step 3: パフォーマンス測定**
```bash
# ベンチマーク実行
python3 -c "
from gaia_integration_simulation_test import GAIAIntegrationSimulationTest, SimulationConfig
import asyncio

config = SimulationConfig(
    test_duration_seconds=300,  # 5分間
    model_path='rinna/youri-7b-chat',
    enable_kv_quantization=True,
    enable_control_agent=True,
    enable_lemonade_adapter=True,
    performance_target_tps=10.0,  # 実機目標
    memory_target_mb=2048
)

async def run_real_test():
    test_runner = GAIAIntegrationSimulationTest(config)
    return await test_runner.run_full_simulation()

results = asyncio.run(run_real_test())
print('実機テスト結果:', results['performance_summary'])
"
```

### 3.3 実機テスト期待値

#### **パフォーマンス目標**
- **TPS向上**: 1.5-2.5倍
- **メモリ削減**: 60-75%
- **品質保持**: ΔPPL ≤ 0.5
- **レイテンシ**: 50%削減

#### **成功基準**
```
✅ NPU利用率: 70%以上
✅ iGPU利用率: 50%以上
✅ メモリ使用量: 目標値以下
✅ 推論品質: ベースライン±5%以内
✅ エラー率: 1%未満
```

## 🔍 Phase 4: トラブルシューティング

### 4.1 よくある問題と解決策

#### **問題1: NPU検出されない**
```bash
# 解決策
# 1. デバイスマネージャーでNPU確認
# 2. ドライバー更新
# 3. DirectML環境変数設定
set ONNXRUNTIME_PROVIDERS=DmlExecutionProvider,CPUExecutionProvider
```

#### **問題2: メモリ不足エラー**
```bash
# 解決策
# 1. 量子化レベル上げ
python3 -c "
from gaia_kv_quantization_engine import QuantizationLevel
config.level = QuantizationLevel.L2_INT4  # より積極的な量子化
"

# 2. バッチサイズ削減
# 3. モデルサイズ削減
```

#### **問題3: 品質劣化**
```bash
# 解決策
# 1. 量子化レベル下げ
# 2. 品質許容度調整
# 3. フォールバック機能確認
```

### 4.2 ログ分析

#### **重要なログパターン**
```bash
# 成功パターン
grep "✅" test_output.log

# エラーパターン
grep "❌" test_output.log

# パフォーマンス情報
grep "📊\|📈\|📉" test_output.log
```

## 📈 Phase 5: 継続的テスト

### 5.1 自動テストスケジュール

#### **日次テスト**
```bash
# crontab設定例
0 2 * * * cd /path/to/gaia && python3 gaia_integration_simulation_test.py >> daily_test.log 2>&1
```

#### **週次詳細テスト**
```bash
# 週次詳細テスト
0 3 * * 0 cd /path/to/gaia && python3 gaia_integration_simulation_test.py --extended >> weekly_test.log 2>&1
```

### 5.2 パフォーマンス監視

#### **メトリクス収集**
```python
# パフォーマンス監視スクリプト例
import json
import time
from datetime import datetime

def collect_metrics():
    # テスト実行
    results = run_simulation_test()
    
    # メトリクス記録
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "performance_improvement": results["performance_summary"]["performance_improvement_percent"],
        "memory_reduction": results["performance_summary"]["memory_reduction_percent"],
        "success_rate": results["summary"]["success_rate_percent"]
    }
    
    # ファイル保存
    with open("metrics_history.json", "a") as f:
        f.write(json.dumps(metrics) + "\n")

# 実行
collect_metrics()
```

## 🎯 成功基準まとめ

### シミュレーションテスト
- [x] **成功率**: 100% ✅
- [x] **パフォーマンス向上**: 71.0% ✅ (目標: 50%以上)
- [x] **メモリ削減**: 49.1% ✅ (目標: 30%以上)
- [x] **実行時間**: 1.6秒 ✅ (目標: 5秒以下)

### 実機テスト（目標値）
- [ ] **TPS向上**: 1.5-2.5倍
- [ ] **メモリ削減**: 60-75%
- [ ] **NPU利用率**: 70%以上
- [ ] **品質保持**: ΔPPL ≤ 0.5

## 🔧 次のステップ

1. **実機環境でのテスト実行**
2. **パフォーマンス最適化**
3. **本格運用への移行**
4. **継続的監視の設定**

---

## 📞 サポート

テスト実行中に問題が発生した場合:

1. **ログ確認**: エラーメッセージの詳細分析
2. **環境確認**: 前提条件の再チェック
3. **段階的実行**: 個別コンポーネントテストから開始
4. **設定調整**: パラメータの段階的調整

このガイドに従って、GAIA × Infer-OS統合システムの包括的なテストを実行してください。

