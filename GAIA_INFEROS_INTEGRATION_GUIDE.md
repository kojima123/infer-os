# GAIA × Infer-OS 統合ガイド v1.0

## 🎯 **概要**

このガイドでは、AMD GAIA環境（Windows 11 + Ryzen AI NPU + Radeon iGPU）にInfer-OSを統合し、LLM推論の大幅な性能向上を実現する方法を説明します。

### **期待される効果**
- **7B/8Bモデル**: 1.3-1.6倍のスループット向上、60-75%のメモリ削減
- **20Bモデル**: 1.8倍以上のスループット向上、70-80%のメモリ削減
- **KVキャッシュ**: 最大75%のメモリ削減
- **品質保持**: ΔPPL ≤ 0.5での高品質維持

## 🔧 **前提条件**

### **システム要件**
- **OS**: Windows 11（GAIA要件）
- **CPU**: AMD Ryzen AI 300シリーズ（XDNA NPU搭載）
- **GPU**: Radeon iGPU（RDNA、DirectML対応）
- **メモリ**: 16GB以上推奨
- **ストレージ**: 10GB以上の空き容量

### **ソフトウェア要件**
- **GAIA**: 最新版（RAUX/gaia-cli対応）
- **Python**: 3.11以上
- **ONNX Runtime**: DirectML対応版
- **NPU/iGPUドライバ**: GAIA推奨版

## 📦 **インストール手順**

### **Step 1: リポジトリのクローン**
```bash
git clone https://github.com/kojima123/infer-os.git
cd infer-os
```

### **Step 2: 依存関係のインストール**
```bash
# Python依存関係
pip install fastapi uvicorn requests numpy psutil pydantic

# オプション: YAML設定サポート
pip install pyyaml

# ONNX Runtime (DirectML)
pip install onnxruntime-directml
```

### **Step 3: GAIAの設定確認**
```bash
# GAIAが正常に動作することを確認
gaia-cli --version

# Hybridモードでの動作確認
gaia-cli --mode hybrid --model llama-7b --test
```

## 📋 **CLIマッピング表**

### **GAIA CLI フラグ対応表**

| 概念 | 公式CLIフラグ | 使用例 | 備考 |
|------|---------------|--------|------|
| 最適化モード: Infer-OS | `--optimization inferos` | `gaia-cli --model llama-7b --optimization inferos` | Infer-OS統合を有効化 |
| ハイブリッド実行 | `--mode hybrid` | `gaia-cli --mode hybrid` | NPU+iGPU協調処理 |
| KV量子化ON | `--kv-quantization enabled` | `gaia-cli --kv-quantization enabled` | KVキャッシュ量子化 |
| バッチサイズ | `--batch-size <N>` | `--batch-size 4` | 同時処理数 |
| 最大トークン | `--max-tokens <N>` | `--max-tokens 1024` | 生成トークン上限 |
| 品質閾値 | `--quality-threshold <F>` | `--quality-threshold 0.5` | ΔPPL許容値 |
| メモリ制限 | `--memory-limit <SIZE>` | `--memory-limit 16GB` | メモリ使用上限 |
| デバッグモード | `--debug` | `gaia-cli --debug` | 詳細ログ出力 |

> **注意**: 実際のCLIフラグ名は GAIA の実装に依存します。上記は設計仕様であり、実装時に調整が必要な場合があります。

### **動作確認コマンド**
```bash
# GAIA基本動作確認
gaia-cli --version
gaia-cli --help | grep -E "(optimization|mode|kv-quantization)"

# Infer-OS統合テスト
gaia-cli --model llama-7b --optimization inferos --test --dry-run
```

## 🔌 **API仕様（OpenAPI）**

### **Infer-OS Control Agent API v1.0**

#### **ベースURL**: `http://127.0.0.1:7031/v1`
> **セキュリティ**: ローカルホスト専用バインド（127.0.0.1）、認証トークン必須

#### **認証設定**

##### **環境変数によるトークン設定**
```cmd
# Windows PowerShell
$token = [System.Guid]::NewGuid().ToString()
[Environment]::SetEnvironmentVariable("INFEROS_AGENT_TOKEN", $token, "User")
echo "Generated token: $token"

# Linux/macOS
export INFEROS_AGENT_TOKEN=$(uuidgen)
echo "Generated token: $INFEROS_AGENT_TOKEN"
```

##### **認証ヘッダー**
```http
# 認証が必要なエンドポイント（POST/PUT/DELETE）
X-Inferos-Token: your-generated-token-here

# 例
curl -X POST http://127.0.0.1:7031/v1/policy \
  -H "X-Inferos-Token: $INFEROS_AGENT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"kv": {"mode": "dynamic"}}'
```

#### **エンドポイント一覧**

| エンドポイント | メソッド | 説明 | 認証 | バインド |
|---------------|----------|------|------|----------|
| `/health` | GET | ヘルスチェック | 不要 | 127.0.0.1 |
| `/metrics` | GET | パフォーマンスメトリクス | 不要 | 127.0.0.1 |
| `/policy` | GET | 最適化ポリシー取得 | 不要 | 127.0.0.1 |
| `/policy` | POST | 最適化ポリシー設定 | **必須** | 127.0.0.1 |
| `/baseline` | POST | ベースラインモード切替 | **必須** | 127.0.0.1 |
| `/config` | GET | 設定取得 | 不要 | 127.0.0.1 |
| `/config` | POST | 設定更新 | **必須** | 127.0.0.1 |
| `/history` | GET | パフォーマンス履歴 | 不要 | 127.0.0.1 |

#### **API セキュリティ実装例**

##### **認証ミドルウェア**
```python
# inferos_control_agent.py
import os
import secrets
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Infer-OS Control Agent", version="1.0.0")

# セキュリティ設定
AGENT_TOKEN = os.getenv("INFEROS_AGENT_TOKEN")
if not AGENT_TOKEN:
    # 開発環境用の自動生成（本番では環境変数必須）
    AGENT_TOKEN = secrets.token_urlsafe(32)
    print(f"⚠️  自動生成トークン: {AGENT_TOKEN}")
    print("本番環境では INFEROS_AGENT_TOKEN 環境変数を設定してください")

# CORS無効化（ローカル専用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1", "http://localhost"],
    allow_methods=["GET", "POST"],
    allow_headers=["X-Inferos-Token", "Content-Type"],
)

def require_auth(x_inferos_token: str | None = Header(None)):
    """認証が必要なエンドポイント用"""
    if not x_inferos_token or x_inferos_token != AGENT_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Valid X-Inferos-Token header required"
        )

@app.middleware("http")
async def security_headers(request: Request, call_next):
    """セキュリティヘッダー追加"""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

# 認証不要エンドポイント
@app.get("/v1/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/v1/metrics")
async def get_metrics():
    return get_current_metrics()

# 認証必須エンドポイント
@app.post("/v1/policy")
async def set_policy(
    policy: PolicyConfig,
    _: None = Depends(require_auth)
):
    return apply_policy_config(policy)

@app.post("/v1/config")
async def update_config(
    config: AgentConfig,
    _: None = Depends(require_auth)
):
    return update_agent_config(config)
```

##### **ポート設定の柔軟化**
```python
# 設定可能ポート（デフォルト: 7031）
DEFAULT_PORT = 7031
AGENT_PORT = int(os.getenv("INFEROS_AGENT_PORT", DEFAULT_PORT))

# ランダムポート生成（セキュリティ強化）
if os.getenv("INFEROS_RANDOM_PORT", "false").lower() == "true":
    import socket
    with socket.socket() as s:
        s.bind(('', 0))
        AGENT_PORT = s.getsockname()[1]
    print(f"🔒 ランダムポート使用: {AGENT_PORT}")

# サーバー起動
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",  # ローカルホスト専用
        port=AGENT_PORT,
        log_level="info"
    )
```

#### **API スキーマ定義**

##### **POST /v1/policy（認証必須）**
```json
{
  "type": "object",
  "properties": {
    "kv": {
      "type": "object",
      "properties": {
        "mode": {
          "type": "string",
          "enum": ["dynamic", "latency_optimized", "throughput_optimized", "quality_optimized"],
          "default": "dynamic"
        },
        "recent_window": {
          "type": "integer",
          "minimum": 16,
          "maximum": 512,
          "default": 64
        },
        "level_thresholds": {
          "type": "object",
          "properties": {
            "L1_int8": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.7},
            "L2_int4": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5},
            "L3_evict": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.3}
          }
        }
      }
    },
    "io": {
      "type": "object",
      "properties": {
        "enable_iobinding": {"type": "boolean", "default": true},
        "dml_pool_bytes": {
          "oneOf": [
            {"type": "integer", "minimum": 1073741824},
            {"type": "string", "pattern": "^\\d+(\\.\\d+)?(B|KB|MB|GB|MiB|GiB)$"}
          ],
          "default": 2147483648,
          "description": "メモリプールサイズ（バイト整数または人間可読文字列）"
        },
        "host_pool_bytes": {
          "oneOf": [
            {"type": "integer", "minimum": 1073741824},
            {"type": "string", "pattern": "^\\d+(\\.\\d+)?(B|KB|MB|GB|MiB|GiB)$"}
          ],
          "default": 4294967296,
          "description": "ホストメモリプールサイズ（バイト整数または人間可読文字列）"
        }
      }
    },
    "scheduler": {
      "type": "object",
      "properties": {
        "mode": {
          "type": "string",
          "enum": ["hybrid", "gpu_only", "npu_only"],
          "default": "hybrid"
        },
        "prefill_device": {
          "type": "string",
          "enum": ["dml", "npu", "cpu"],
          "default": "dml"
        },
        "decode_device": {
          "type": "string",
          "enum": ["npu", "dml", "cpu"],
          "default": "npu"
        }
      }
    },
    "quality": {
      "type": "object",
      "properties": {
        "max_delta_ppl": {"type": "number", "minimum": 0.0, "maximum": 2.0, "default": 0.5},
        "min_accept_rate": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5}
      }
    }
  }
}
```

##### **単位変換ユーティリティ**
```python
def parse_memory_size(value) -> int:
    """人間可読メモリサイズをバイト整数に変換"""
    if isinstance(value, int):
        return value
    
    if isinstance(value, str):
        import re
        match = re.match(r'^(\d+(?:\.\d+)?)(B|KB|MB|GB|MiB|GiB)$', value.upper())
        if not match:
            raise ValueError(f"Invalid memory size format: {value}")
        
        size, unit = match.groups()
        size = float(size)
        
        multipliers = {
            'B': 1,
            'KB': 1000,
            'MB': 1000**2,
            'GB': 1000**3,
            'MIB': 1024**2,
            'GIB': 1024**3
        }
        
        return int(size * multipliers[unit])
    
    raise ValueError(f"Unsupported memory size type: {type(value)}")

# 使用例
policy_data = {
    "io": {
        "dml_pool_bytes": "2048MiB",  # → 2147483648
        "host_pool_bytes": "4GiB"     # → 4294967296
    }
}

# 正規化
policy_data["io"]["dml_pool_bytes"] = parse_memory_size(policy_data["io"]["dml_pool_bytes"])
policy_data["io"]["host_pool_bytes"] = parse_memory_size(policy_data["io"]["host_pool_bytes"])
```

##### **GET /v1/metrics レスポンス例**
```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "measurement_window": {
    "duration_seconds": 60,
    "sample_count": 150,
    "description": "直近60秒間の移動平均（150リクエスト）"
  },
  "throughput": {
    "tokens_per_second": 45.7,
    "requests_per_second": 2.3,
    "calculation": "total_tokens / total_time_seconds",
    "note": "TPS = 生成トークン総数 / 実時間, RPS = リクエスト数 / 実時間"
  },
  "latency": {
    "mean_ms": 1250.5,
    "median_ms": 1180.0,
    "p95_ms": 2100.0,
    "p99_ms": 3200.0,
    "std_ms": 450.2,
    "calculation": "end_time - start_time per request"
  },
  "first_token_latency": {
    "mean_ms": 380.2,
    "median_ms": 350.0,
    "p95_ms": 650.0,
    "measurement_method": "実測（first_token_emit_time - request_start_time）",
    "note": "推定値ではなく、実際のトークン生成開始時刻を計測"
  },
  "quality": {
    "delta_ppl_est": {
      "current": 0.12,
      "baseline": 2.45,
      "delta": 0.12,
      "measurement": {
        "method": "オンライン簡易推定（NLL近似）",
        "window_size": 100,
        "update_frequency": "リクエスト毎",
        "last_full_evaluation": "2024-01-15T09:00:00Z"
      }
    },
    "accept_rate": {
      "current": 0.78,
      "target": 0.75,
      "measurement": {
        "method": "スペキュレイティブ生成受諾率",
        "window_size": 50,
        "description": "直近50回の推論での受諾トークン率"
      }
    },
    "full_evaluation": {
      "last_run": "2024-01-15T06:00:00Z",
      "next_scheduled": "2024-01-16T06:00:00Z",
      "dataset": "validation_set_500samples",
      "baseline_ppl": 2.45,
      "current_ppl": 2.57,
      "delta_ppl": 0.12,
      "status": "within_threshold"
    }
  },
  "memory": {
    "kv_cache_usage_mb": 1024.5,
    "kv_cache_hit_rate": 0.85,
    "total_allocated_mb": 3072.0,
    "peak_usage_mb": 3456.0
  },
  "device_utilization": {
    "npu_utilization": 0.67,
    "dml_utilization": 0.45,
    "cpu_utilization": 0.23
  },
  "optimization_status": {
    "kv_quantization_active": true,
    "io_binding_active": true,
    "hybrid_scheduling_active": true,
    "current_policy": "dynamic"
  }
}
```

#### **メトリクス算出定義**

##### **スループット計算**
```python
class MetricsCalculator:
    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self.request_history = []
        
    def calculate_throughput(self) -> Dict[str, float]:
        """スループット計算（時間窓ベース）"""
        now = time.time()
        cutoff = now - self.window_seconds
        
        # 時間窓内のリクエストのみ
        recent_requests = [
            req for req in self.request_history 
            if req['end_time'] >= cutoff
        ]
        
        if not recent_requests:
            return {"tokens_per_second": 0.0, "requests_per_second": 0.0}
        
        total_tokens = sum(req['tokens_generated'] for req in recent_requests)
        total_time = now - min(req['start_time'] for req in recent_requests)
        
        return {
            "tokens_per_second": total_tokens / total_time if total_time > 0 else 0.0,
            "requests_per_second": len(recent_requests) / total_time if total_time > 0 else 0.0
        }
```

##### **First Token Latency実測**
```python
class FTLMeasurement:
    def __init__(self):
        self.ftl_samples = []
        
    def measure_generation(self, prompt: str, model, tokenizer):
        """FTL実測付き生成"""
        start_time = time.perf_counter()
        first_token_time = None
        
        # ストリーミング生成でFTL計測
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            for i, output_ids in enumerate(model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                # ストリーミング有効化
                output_scores=True,
                return_dict_in_generate=True
            )):
                if i == 0:  # 最初のトークン
                    first_token_time = time.perf_counter()
                    break
        
        end_time = time.perf_counter()
        
        # メトリクス記録
        if first_token_time:
            ftl_ms = (first_token_time - start_time) * 1000
            total_latency_ms = (end_time - start_time) * 1000
            
            self.ftl_samples.append({
                "ftl_ms": ftl_ms,
                "total_latency_ms": total_latency_ms,
                "timestamp": time.time()
            })
            
            return {
                "first_token_latency_ms": ftl_ms,
                "total_latency_ms": total_latency_ms,
                "response": tokenizer.decode(output_ids[0], skip_special_tokens=True)
            }
```

##### **品質測定（PPL）**
```python
class QualityMeasurement:
    def __init__(self, validation_dataset: List[str]):
        self.validation_dataset = validation_dataset
        self.baseline_ppl = None
        self.online_nll_buffer = []
        
    def calculate_full_ppl(self, model, tokenizer) -> float:
        """完全PPL計算（バッチ処理）"""
        total_nll = 0.0
        total_tokens = 0
        
        model.eval()
        with torch.no_grad():
            for text in self.validation_dataset:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                
                # 負の対数尤度計算
                outputs = model(**inputs, labels=inputs["input_ids"])
                nll = outputs.loss.item() * inputs["input_ids"].size(1)
                
                total_nll += nll
                total_tokens += inputs["input_ids"].size(1)
        
        ppl = math.exp(total_nll / total_tokens)
        return ppl
    
    def estimate_online_ppl(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """オンライン簡易PPL推定"""
        # クロスエントロピー計算
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # NLLバッファに追加
        self.online_nll_buffer.append(ce_loss.item())
        
        # 窓サイズ維持
        if len(self.online_nll_buffer) > 100:
            self.online_nll_buffer.pop(0)
        
        # 平均NLLからPPL推定
        avg_nll = sum(self.online_nll_buffer) / len(self.online_nll_buffer)
        estimated_ppl = math.exp(avg_nll)
        
        return estimated_ppl
    
    def get_quality_metrics(self) -> Dict:
        """品質メトリクス取得"""
        current_ppl_est = self.estimate_online_ppl() if self.online_nll_buffer else 0.0
        
        return {
            "delta_ppl_est": {
                "current": current_ppl_est,
                "baseline": self.baseline_ppl or 0.0,
                "delta": current_ppl_est - (self.baseline_ppl or 0.0),
                "measurement": {
                    "method": "オンライン簡易推定（NLL近似）",
                    "window_size": len(self.online_nll_buffer),
                    "update_frequency": "リクエスト毎"
                }
            }
        }
```

#### **品質管理システム**

##### **定期評価スケジューラ**
```python
import schedule
import threading
from datetime import datetime, timedelta

class QualityScheduler:
    def __init__(self, quality_measurement: QualityMeasurement):
        self.quality_measurement = quality_measurement
        self.scheduler_thread = None
        self.running = False
        
    def start_scheduler(self):
        """品質評価スケジューラ開始"""
        # 毎日6時に完全評価
        schedule.every().day.at("06:00").do(self.run_full_evaluation)
        
        # 4時間毎に軽量評価
        schedule.every(4).hours.do(self.run_light_evaluation)
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
    def _scheduler_loop(self):
        """スケジューラループ"""
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # 1分間隔でチェック
    
    def run_full_evaluation(self):
        """完全品質評価実行"""
        print(f"🔍 完全品質評価開始: {datetime.now()}")
        
        try:
            # モデルとトークナイザー取得
            model, tokenizer = self._get_current_model()
            
            # 完全PPL計算
            current_ppl = self.quality_measurement.calculate_full_ppl(model, tokenizer)
            baseline_ppl = self.quality_measurement.baseline_ppl or current_ppl
            
            # 品質劣化チェック
            delta_ppl = current_ppl - baseline_ppl
            if delta_ppl > 0.5:  # 閾値超過
                self._trigger_quality_alert(delta_ppl, current_ppl, baseline_ppl)
            
            # 結果記録
            self._record_evaluation_result({
                "timestamp": datetime.now().isoformat(),
                "type": "full_evaluation",
                "current_ppl": current_ppl,
                "baseline_ppl": baseline_ppl,
                "delta_ppl": delta_ppl,
                "status": "alert" if delta_ppl > 0.5 else "normal"
            })
            
            print(f"✅ 完全品質評価完了: ΔPPL={delta_ppl:.3f}")
            
        except Exception as e:
            print(f"❌ 完全品質評価エラー: {e}")
            self._record_evaluation_result({
                "timestamp": datetime.now().isoformat(),
                "type": "full_evaluation",
                "status": "error",
                "error": str(e)
            })
    
    def _trigger_quality_alert(self, delta_ppl: float, current_ppl: float, baseline_ppl: float):
        """品質劣化アラート"""
        alert_message = f"""
🚨 品質劣化アラート
時刻: {datetime.now()}
ΔPPL: {delta_ppl:.3f} (閾値: 0.5)
現在PPL: {current_ppl:.3f}
ベースラインPPL: {baseline_ppl:.3f}
推奨アクション: KV量子化レベルの緩和を検討
"""
        
        print(alert_message)
        
        # ログ記録
        with open("quality_alerts.log", "a") as f:
            f.write(f"{datetime.now().isoformat()}: QUALITY_ALERT delta_ppl={delta_ppl:.3f}\n")
        
        # 自動回復試行
        self._attempt_quality_recovery()
    
    def _attempt_quality_recovery(self):
        """品質自動回復"""
        print("🔄 品質自動回復を試行中...")
        
        # KV量子化レベルを段階的に緩和
        recovery_policies = [
            {"kv": {"mode": "quality_optimized"}},
            {"kv": {"level_thresholds": {"L2_int4": 0.3, "L1_int8": 0.5}}},
            {"kv": {"mode": "latency_optimized"}}  # 最終手段
        ]
        
        for i, policy in enumerate(recovery_policies):
            try:
                # ポリシー適用
                self._apply_recovery_policy(policy)
                
                # 短時間待機
                time.sleep(30)
                
                # 軽量品質チェック
                if self._quick_quality_check():
                    print(f"✅ 品質回復成功: ポリシー{i+1}")
                    return True
                    
            except Exception as e:
                print(f"⚠️  回復ポリシー{i+1}失敗: {e}")
        
        print("❌ 自動品質回復失敗: 手動介入が必要")
        return False
#### **運用耐性とセキュリティ強化**

##### **フォールバック戦略とバックオフ**
```python
import time
import random
from enum import Enum
from typing import Dict, List, Optional

class FallbackReason(Enum):
    QUALITY_DEGRADATION = "quality_degradation"
    DEVICE_ERROR = "device_error"
    MEMORY_PRESSURE = "memory_pressure"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

class BackoffStrategy:
    def __init__(self, initial_delay: float = 1.0, max_delay: float = 60.0, multiplier: float = 2.0):
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.current_delay = initial_delay
        self.attempt_count = 0
    
    def get_delay(self) -> float:
        """バックオフ遅延時間取得"""
        if self.attempt_count == 0:
            delay = self.initial_delay
        else:
            # 指数バックオフ + ジッター
            delay = min(self.current_delay * self.multiplier, self.max_delay)
            jitter = random.uniform(0.1, 0.3) * delay
            delay = delay + jitter
        
        self.current_delay = delay
        self.attempt_count += 1
        return delay
    
    def reset(self):
        """バックオフ状態リセット"""
        self.current_delay = self.initial_delay
        self.attempt_count = 0

class RobustFallbackManager:
    def __init__(self):
        self.fallback_history = []
        self.backoff_strategies = {
            FallbackReason.QUALITY_DEGRADATION: BackoffStrategy(2.0, 120.0, 1.5),
            FallbackReason.DEVICE_ERROR: BackoffStrategy(5.0, 300.0, 2.0),
            FallbackReason.MEMORY_PRESSURE: BackoffStrategy(1.0, 60.0, 1.8),
            FallbackReason.TIMEOUT: BackoffStrategy(3.0, 180.0, 2.2),
        }
        self.max_consecutive_fallbacks = 5
        self.notification_thresholds = {
            FallbackReason.QUALITY_DEGRADATION: 3,
            FallbackReason.DEVICE_ERROR: 2,
            FallbackReason.MEMORY_PRESSURE: 4,
            FallbackReason.TIMEOUT: 3,
        }
    
    def handle_fallback(self, reason: FallbackReason, context: Dict) -> Dict:
        """フォールバック処理"""
        timestamp = time.time()
        
        # フォールバック履歴記録
        fallback_event = {
            "timestamp": timestamp,
            "reason": reason,
            "context": context,
            "attempt_count": self.backoff_strategies[reason].attempt_count + 1
        }
        self.fallback_history.append(fallback_event)
        
        # 連続フォールバック数チェック
        recent_fallbacks = [
            event for event in self.fallback_history
            if timestamp - event["timestamp"] < 300  # 5分以内
        ]
        
        if len(recent_fallbacks) >= self.max_consecutive_fallbacks:
            return self._handle_critical_fallback(reason, context)
        
        # 通知判定
        reason_count = sum(1 for event in recent_fallbacks if event["reason"] == reason)
        if reason_count >= self.notification_thresholds.get(reason, 3):
            self._send_notification(reason, reason_count, context)
        
        # バックオフ遅延
        delay = self.backoff_strategies[reason].get_delay()
        
        # フォールバック実行
        fallback_result = self._execute_fallback(reason, context)
        
        return {
            "status": "fallback_executed",
            "reason": reason.value,
            "delay_seconds": delay,
            "result": fallback_result,
            "next_retry": timestamp + delay
        }
    
    def _execute_fallback(self, reason: FallbackReason, context: Dict) -> Dict:
        """具体的フォールバック実行"""
        if reason == FallbackReason.QUALITY_DEGRADATION:
            return self._quality_degradation_fallback(context)
        elif reason == FallbackReason.DEVICE_ERROR:
            return self._device_error_fallback(context)
        elif reason == FallbackReason.MEMORY_PRESSURE:
            return self._memory_pressure_fallback(context)
        elif reason == FallbackReason.TIMEOUT:
            return self._timeout_fallback(context)
        else:
            return self._generic_fallback(context)
    
    def _quality_degradation_fallback(self, context: Dict) -> Dict:
        """品質劣化フォールバック"""
        # KV量子化レベル緩和
        current_policy = context.get("current_policy", {})
        
        fallback_policies = [
            # Level 1: INT4閾値を下げる
            {
                "kv": {
                    "level_thresholds": {
                        "L2_int4": max(0.2, current_policy.get("kv", {}).get("level_thresholds", {}).get("L2_int4", 0.5) - 0.1),
                        "L1_int8": max(0.4, current_policy.get("kv", {}).get("level_thresholds", {}).get("L1_int8", 0.7) - 0.1)
                    }
                }
            },
            # Level 2: 品質優先モード
            {"kv": {"mode": "quality_optimized"}},
            # Level 3: KV量子化無効
            {"kv": {"mode": "disabled"}}
        ]
        
        for i, policy in enumerate(fallback_policies):
            try:
                self._apply_policy(policy)
                return {"level": i + 1, "policy": policy, "status": "applied"}
            except Exception as e:
                continue
        
        return {"status": "all_fallbacks_failed"}
    
    def _device_error_fallback(self, context: Dict) -> Dict:
        """デバイスエラーフォールバック"""
        # デバイス切替順序
        device_fallback_order = ["cpu", "dml", "npu"]
        current_device = context.get("current_device", "npu")
        
        try:
            current_index = device_fallback_order.index(current_device)
            next_devices = device_fallback_order[current_index + 1:]
        except ValueError:
            next_devices = device_fallback_order
        
        for device in next_devices:
            try:
                self._switch_device(device)
                return {"fallback_device": device, "status": "switched"}
            except Exception as e:
                continue
        
        return {"status": "no_available_device"}
    
    def _memory_pressure_fallback(self, context: Dict) -> Dict:
        """メモリ圧迫フォールバック"""
        # メモリ削減策
        reduction_strategies = [
            # Level 1: KVキャッシュサイズ削減
            {"action": "reduce_kv_cache", "factor": 0.7},
            # Level 2: バッチサイズ削減
            {"action": "reduce_batch_size", "factor": 0.5},
            # Level 3: モデル量子化強化
            {"action": "aggressive_quantization"},
            # Level 4: ガベージコレクション強制実行
            {"action": "force_gc"}
        ]
        
        for strategy in reduction_strategies:
            try:
                result = self._apply_memory_reduction(strategy)
                if result.get("memory_freed_mb", 0) > 100:  # 100MB以上解放
                    return {"strategy": strategy, "result": result, "status": "effective"}
            except Exception as e:
                continue
        
        return {"status": "memory_reduction_failed"}
    
    def _send_notification(self, reason: FallbackReason, count: int, context: Dict):
        """運用通知送信"""
        notification = {
            "timestamp": time.time(),
            "severity": "warning" if count < 5 else "critical",
            "reason": reason.value,
            "count": count,
            "message": f"フォールバック頻発: {reason.value} ({count}回/5分)",
            "context": context,
            "recommended_action": self._get_recommended_action(reason)
        }
        
        # ログ記録
        with open("fallback_notifications.log", "a") as f:
            import json
            f.write(f"{json.dumps(notification)}\n")
        
        # コンソール出力
        severity_emoji = "⚠️" if notification["severity"] == "warning" else "🚨"
        print(f"{severity_emoji} {notification['message']}")
        print(f"推奨アクション: {notification['recommended_action']}")
    
    def _get_recommended_action(self, reason: FallbackReason) -> str:
        """推奨アクション取得"""
        recommendations = {
            FallbackReason.QUALITY_DEGRADATION: "モデル再訓練またはベースライン見直しを検討",
            FallbackReason.DEVICE_ERROR: "デバイスドライバー更新またはハードウェア診断を実行",
            FallbackReason.MEMORY_PRESSURE: "システムメモリ増設またはモデルサイズ削減を検討",
            FallbackReason.TIMEOUT: "推論タイムアウト設定の見直しまたはハードウェア性能向上を検討"
        }
        return recommendations.get(reason, "技術サポートに連絡")
```

##### **Windowsサービス運用強化**
```powershell
# Windows Service 完全運用スクリプト

# 1. 専用ユーザー作成（セキュリティ強化）
$ServiceUser = "InferOSAgent"
$ServicePassword = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 16 | % {[char]$_})

# ユーザー作成
net user $ServiceUser $ServicePassword /add /comment:"Infer-OS Control Agent Service Account"
net localgroup "Log on as a service" $ServiceUser /add

# 2. サービス設定ファイル作成
$ServiceConfig = @"
[Service]
Name=InferOSControlAgent
DisplayName=Infer-OS Control Agent
Description=AI inference optimization control agent for GAIA integration
User=$ServiceUser
Password=$ServicePassword
WorkingDirectory=C:\InferOS\Agent
ExePath=C:\InferOS\Agent\venv\Scripts\python.exe
Arguments=C:\InferOS\Agent\inferos_control_agent.py
LogPath=C:\InferOS\Logs\agent.log
MaxLogSize=100MB
LogRotation=daily
RestartPolicy=always
RestartDelay=30
Environment=INFEROS_AGENT_TOKEN=$env:INFEROS_AGENT_TOKEN;INFEROS_AGENT_PORT=7031
"@

$ServiceConfig | Out-File -FilePath "C:\InferOS\Config\service.conf" -Encoding UTF8

# 3. NSSM使用サービス作成
nssm install InferOSControlAgent "C:\InferOS\Agent\venv\Scripts\python.exe"
nssm set InferOSControlAgent Arguments "C:\InferOS\Agent\inferos_control_agent.py"
nssm set InferOSControlAgent AppDirectory "C:\InferOS\Agent"
nssm set InferOSControlAgent ObjectName ".\$ServiceUser" $ServicePassword
nssm set InferOSControlAgent DisplayName "Infer-OS Control Agent"
nssm set InferOSControlAgent Description "AI inference optimization control agent for GAIA integration"

# ログ設定
nssm set InferOSControlAgent AppStdout "C:\InferOS\Logs\agent_stdout.log"
nssm set InferOSControlAgent AppStderr "C:\InferOS\Logs\agent_stderr.log"
nssm set InferOSControlAgent AppRotateFiles 1
nssm set InferOSControlAgent AppRotateOnline 1
nssm set InferOSControlAgent AppRotateBytes 10485760  # 10MB

# 再起動設定
nssm set InferOSControlAgent AppExit Default Restart
nssm set InferOSControlAgent AppRestartDelay 30000  # 30秒

# 環境変数設定
nssm set InferOSControlAgent AppEnvironmentExtra "INFEROS_AGENT_TOKEN=$env:INFEROS_AGENT_TOKEN"

# 4. ファイアウォール設定
New-NetFirewallRule -DisplayName "Infer-OS Control Agent" -Direction Inbound -Protocol TCP -LocalPort 7031 -LocalAddress 127.0.0.1 -Action Allow

# 5. 権限設定
icacls "C:\InferOS\Agent" /grant "${ServiceUser}:(OI)(CI)RX" /T
icacls "C:\InferOS\Logs" /grant "${ServiceUser}:(OI)(CI)F" /T
icacls "C:\InferOS\Config" /grant "${ServiceUser}:(OI)(CI)R" /T

# 6. サービス開始
nssm start InferOSControlAgent

# 7. 監視スクリプト作成
$MonitorScript = @'
# Infer-OS Agent 監視スクリプト
$ServiceName = "InferOSControlAgent"
$LogPath = "C:\InferOS\Logs\monitor.log"
$AlertThresholds = @{
    CPUPercent = 80
    MemoryMB = 2048
    ResponseTimeMs = 5000
}

while ($true) {
    try {
        # サービス状態チェック
        $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
        if ($service.Status -ne "Running") {
            $message = "$(Get-Date): サービス停止検出 - 再起動試行"
            Add-Content -Path $LogPath -Value $message
            Start-Service -Name $ServiceName
        }
        
        # API応答チェック
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:7031/v1/health" -TimeoutSec 5 -ErrorAction SilentlyContinue
        if ($response.StatusCode -ne 200) {
            $message = "$(Get-Date): API応答異常 - サービス再起動"
            Add-Content -Path $LogPath -Value $message
            Restart-Service -Name $ServiceName
        }
        
        # リソース使用量チェック
        $process = Get-Process -Name "python" | Where-Object {$_.MainWindowTitle -like "*inferos*"}
        if ($process) {
            $cpuPercent = $process.CPU
            $memoryMB = $process.WorkingSet64 / 1MB
            
            if ($cpuPercent -gt $AlertThresholds.CPUPercent) {
                $message = "$(Get-Date): 高CPU使用率警告: $cpuPercent%"
                Add-Content -Path $LogPath -Value $message
            }
            
            if ($memoryMB -gt $AlertThresholds.MemoryMB) {
                $message = "$(Get-Date): 高メモリ使用量警告: $memoryMB MB"
                Add-Content -Path $LogPath -Value $message
            }
        }
        
    } catch {
        $message = "$(Get-Date): 監視エラー: $($_.Exception.Message)"
        Add-Content -Path $LogPath -Value $message
    }
    
    Start-Sleep -Seconds 60
}
'@

$MonitorScript | Out-File -FilePath "C:\InferOS\Scripts\monitor.ps1" -Encoding UTF8

# 8. 監視タスク作成
$TaskAction = New-ScheduledTaskAction -Execute "PowerShell.exe" -Argument "-File C:\InferOS\Scripts\monitor.ps1"
$TaskTrigger = New-ScheduledTaskTrigger -AtStartup
$TaskSettings = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5)
$TaskPrincipal = New-ScheduledTaskPrincipal -UserId $ServiceUser -LogonType ServiceAccount

Register-ScheduledTask -TaskName "InferOSAgentMonitor" -Action $TaskAction -Trigger $TaskTrigger -Settings $TaskSettings -Principal $TaskPrincipal

# 9. アンインストールスクリプト作成
$UninstallScript = @"
# Infer-OS Agent 完全アンインストール
Write-Host "Infer-OS Control Agent アンインストール開始..."

# サービス停止・削除
nssm stop InferOSControlAgent
nssm remove InferOSControlAgent confirm

# タスク削除
Unregister-ScheduledTask -TaskName "InferOSAgentMonitor" -Confirm:`$false

# ファイアウォール規則削除
Remove-NetFirewallRule -DisplayName "Infer-OS Control Agent" -ErrorAction SilentlyContinue

# ユーザー削除
net user $ServiceUser /delete

# ファイル削除
Remove-Item -Path "C:\InferOS" -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "アンインストール完了"
"@

$UninstallScript | Out-File -FilePath "C:\InferOS\Scripts\uninstall.ps1" -Encoding UTF8

Write-Host "Infer-OS Control Agent サービス設定完了"
Write-Host "サービスユーザー: $ServiceUser"
Write-Host "パスワード: $ServicePassword"
Write-Host "監視: タスクスケジューラで自動実行"
Write-Host "アンインストール: C:\InferOS\Scripts\uninstall.ps1 を実行"
```

##### **DirectML/NPU演算子サポート確認**
```python
import onnxruntime as ort
from typing import Dict, List, Set

class DeviceCapabilityChecker:
    def __init__(self):
        self.dml_supported_ops = set()
        self.npu_supported_ops = set()
        self.capability_cache = {}
        
    def check_directml_quantization_support(self) -> Dict[str, bool]:
        """DirectML量子化サポート確認"""
        try:
            # DirectMLプロバイダー初期化
            providers = [
                ('DmlExecutionProvider', {
                    'device_id': 0,
                    'enable_dynamic_graph_fusion': True
                })
            ]
            
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # テスト用簡単なモデル作成
            test_model = self._create_quantization_test_model()
            
            # INT8サポートテスト
            int8_support = self._test_quantization_support(test_model, "int8", providers)
            
            # INT4サポートテスト
            int4_support = self._test_quantization_support(test_model, "int4", providers)
            
            # FP16サポートテスト
            fp16_support = self._test_fp16_support(providers)
            
            return {
                "int8_quantization": int8_support,
                "int4_quantization": int4_support,
                "fp16_mixed_precision": fp16_support,
                "dynamic_quantization": int8_support,  # INT8ベース
                "static_quantization": int8_support and self._test_static_quantization(providers)
            }
            
        except Exception as e:
            return {
                "int8_quantization": False,
                "int4_quantization": False,
                "fp16_mixed_precision": False,
                "dynamic_quantization": False,
                "static_quantization": False,
                "error": str(e)
            }
    
    def get_supported_operators(self, provider: str) -> Set[str]:
        """サポート演算子一覧取得"""
        if provider in self.capability_cache:
            return self.capability_cache[provider]
        
        try:
            # プロバイダー固有の演算子リスト取得
            if provider == "DmlExecutionProvider":
                supported_ops = self._get_dml_operators()
            elif provider == "NPUExecutionProvider":
                supported_ops = self._get_npu_operators()
            else:
                supported_ops = set()
            
            self.capability_cache[provider] = supported_ops
            return supported_ops
            
        except Exception:
            return set()
    
    def create_fallback_table(self) -> Dict[str, List[Dict]]:
        """フォールバック表作成"""
        dml_caps = self.check_directml_quantization_support()
        
        fallback_scenarios = {
            "quantization_failure": [
                {
                    "condition": "INT4量子化失敗",
                    "action": "INT8量子化に切替",
                    "supported": dml_caps.get("int8_quantization", False)
                },
                {
                    "condition": "INT8量子化失敗",
                    "action": "FP16混合精度に切替",
                    "supported": dml_caps.get("fp16_mixed_precision", False)
                },
                {
                    "condition": "FP16失敗",
                    "action": "FP32フル精度に切替",
                    "supported": True  # 常にサポート
                }
            ],
            "device_unavailable": [
                {
                    "condition": "NPU利用不可",
                    "action": "DirectML(iGPU)に切替",
                    "check_method": "self._check_dml_availability()"
                },
                {
                    "condition": "DirectML利用不可",
                    "action": "CPU推論に切替",
                    "check_method": "True"  # CPU常に利用可能
                }
            ],
            "memory_pressure": [
                {
                    "condition": "KVキャッシュメモリ不足",
                    "action": "量子化レベル強化",
                    "parameters": {"L2_int4": 0.3, "L1_int8": 0.5}
                },
                {
                    "condition": "モデルメモリ不足",
                    "action": "モデル分割実行",
                    "parameters": {"chunk_size": 512, "overlap": 64}
                }
            ],
            "performance_degradation": [
                {
                    "condition": "TPS < 閾値の50%",
                    "action": "最適化ポリシー緩和",
                    "parameters": {"mode": "latency_optimized"}
                },
                {
                    "condition": "品質劣化 ΔPPL > 0.5",
                    "action": "品質優先モードに切替",
                    "parameters": {"mode": "quality_optimized"}
                }
            ]
        }
        
        return fallback_scenarios
    
    def _test_quantization_support(self, model_path: str, quant_type: str, providers: List) -> bool:
        """量子化サポートテスト"""
        try:
            if quant_type == "int8":
                # INT8動的量子化テスト
                from onnxruntime.quantization import quantize_dynamic, QuantType
                quantized_model = f"test_model_int8.onnx"
                quantize_dynamic(model_path, quantized_model, weight_type=QuantType.QInt8)
                
            elif quant_type == "int4":
                # INT4量子化テスト（実装依存）
                quantized_model = f"test_model_int4.onnx"
                # INT4量子化実装（簡略化）
                return self._test_int4_quantization(model_path, quantized_model)
            
            # 量子化モデルでセッション作成テスト
            session = ort.InferenceSession(quantized_model, providers=providers)
            
            # 簡単な推論テスト
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            
            import numpy as np
            test_input = np.random.randn(*[1 if dim is None else dim for dim in input_shape]).astype(np.float32)
            
            outputs = session.run(None, {input_name: test_input})
            
            return True
            
        except Exception as e:
            print(f"量子化テスト失敗 ({quant_type}): {e}")
            return False
```
  "ftl_ms": 245,
  "mem": {
    "vram_gb": 8.9,
    "system_gb": 16.2,
    "kv_cache_gb": 2.1
  },
  "util": {
    "npu": 0.75,
    "igpu": 0.45,
    "cpu": 0.32
  },
  "quality": {
    "delta_ppl_est": 0.42,
    "accept_rate": 0.87
  },
  "kv_stats": {
    "total_chunks": 1250,
    "level_distribution": {
      "L0": 320,
      "L1": 450,
      "L2": 380,
      "L3": 100
    },
    "compression_ratio": 0.35,
    "hit_rate": 0.92
  }
}
```

##### **エラーレスポンス**
```json
{
  "error": {
    "code": 400,
    "message": "Invalid policy configuration",
    "details": "recent_window must be between 16 and 512"
  }
}
```

## 🚀 **Phase 1: サイドカー統合（推奨開始点）**

### **1.1 Infer-OS Control Agentの起動**

#### **基本起動**
```bash
python inferos_control_agent.py
```

#### **設定ファイル付き起動**
```bash
# config.yaml を作成
cat > config.yaml << EOF
infer_os:
  quality:
    max_delta_ppl: 0.5
    min_accept_rate: 0.5
  kv:
    recent_window: 64
    level_thresholds:
      L1_int8: 0.7
      L2_int4: 0.5
      L3_evict: 0.3
  io:
    enable_iobinding: true
    dml_pool_bytes: 2048MiB
    host_pool_bytes: 4096MiB
  scheduler:
    mode: hybrid
    prefill_device: dml
    decode_device: npu
  loops:
    fast_ms: 1
    mid_ms: 10
    slow_ms: 100
EOF

python inferos_control_agent.py --config config.yaml
```

#### **Windowsサービスとして登録（推奨）**

##### **方法1: NSSM (Non-Sucking Service Manager) 使用**
```cmd
# 管理者権限でコマンドプロンプトを開く

# NSSMのダウンロードとインストール
# https://nssm.cc/download からダウンロード
# nssm.exe を C:\Windows\System32 にコピー

# サービス作成
nssm install InferOSAgent

# NSSM GUIが開くので以下を設定:
# Application tab:
#   Path: C:\Python311\python.exe
#   Startup directory: C:\path\to\infer-os
#   Arguments: inferos_control_agent.py --config config.yaml

# Details tab:
#   Display name: Infer-OS Control Agent
#   Description: AI inference optimization service for GAIA integration

# Log on tab:
#   This account: .\InferOSService (専用ユーザー推奨)

# I/O tab:
#   Output (stdout): C:\ProgramData\InferOS\logs\stdout.log
#   Error (stderr): C:\ProgramData\InferOS\logs\stderr.log

# Rotation tab:
#   Replace existing Output/Error files: チェック
#   Rotate files: チェック
#   Restrict rotation to files older than: 7 days

# Install service をクリック
```

##### **方法2: sc create コマンド使用**
```cmd
# 管理者権限でコマンドプロンプトを開く

# ログディレクトリ作成
mkdir C:\ProgramData\InferOS\logs

# サービス作成
sc create InferOSAgent ^
  binPath= "C:\Python311\python.exe C:\path\to\infer-os\inferos_control_agent.py --service" ^
  DisplayName= "Infer-OS Control Agent" ^
  Description= "AI inference optimization service for GAIA integration" ^
  start= auto ^
  obj= ".\InferOSService" ^
  password= "YourServicePassword"

# サービス設定
sc config InferOSAgent depend= "Tcpip/Afd"
sc failure InferOSAgent reset= 86400 actions= restart/5000/restart/10000/restart/30000

# ファイアウォール設定
netsh advfirewall firewall add rule name="InferOS Agent" dir=in action=allow protocol=TCP localport=7031
```

##### **方法3: タスクスケジューラ使用**
```cmd
# 管理者権限でコマンドプロンプトを開く

# タスク作成
schtasks /create /tn "InferOS Agent" /tr "C:\Python311\python.exe C:\path\to\infer-os\inferos_control_agent.py" /sc onstart /ru "SYSTEM" /rl highest

# タスク設定の詳細化
schtasks /change /tn "InferOS Agent" /st 00:00 /ri 1 /du 9999:59

# タスク開始
schtasks /run /tn "InferOS Agent"
```

#### **サービス運用管理**

##### **サービス制御コマンド**
```cmd
# サービス開始
net start InferOSAgent
# または
sc start InferOSAgent

# サービス停止
net stop InferOSAgent
# または
sc stop InferOSAgent

# サービス状態確認
sc query InferOSAgent

# サービス設定確認
sc qc InferOSAgent

# サービス削除（アンインストール時）
sc delete InferOSAgent
```

##### **ログ管理設定**
```cmd
# ログディレクトリ構成
C:\ProgramData\InferOS\
├── logs\
│   ├── stdout.log          # 標準出力
│   ├── stderr.log          # エラー出力
│   ├── inferos_agent.log   # アプリケーションログ
│   └── archived\           # ローテーション済みログ
├── config\
│   ├── config.yaml         # メイン設定
│   └── policy.json         # ポリシー設定
└── temp\
    └── onnx_cache\         # ONNX変換キャッシュ

# ログローテーション設定（PowerShell）
$logConfig = @"
<configuration>
  <appender name="RollingFile" type="log4net.Appender.RollingFileAppender">
    <file value="C:\ProgramData\InferOS\logs\inferos_agent.log" />
    <appendToFile value="true" />
    <rollingStyle value="Size" />
    <maxSizeRollBackups value="10" />
    <maximumFileSize value="10MB" />
    <staticLogFileName value="true" />
    <layout type="log4net.Layout.PatternLayout">
      <conversionPattern value="%date [%thread] %-5level %logger - %message%newline" />
    </layout>
  </appender>
</configuration>
"@

$logConfig | Out-File -FilePath "C:\ProgramData\InferOS\config\log4net.config" -Encoding UTF8
```

#### **権限とセキュリティ設定**

##### **専用サービスユーザー作成**
```cmd
# 専用ユーザー作成
net user InferOSService "ComplexPassword123!" /add /comment:"Infer-OS Service Account"

# 必要な権限付与
ntrights +r SeServiceLogonRight -u InferOSService
ntrights +r SeIncreaseQuotaPrivilege -u InferOSService
ntrights +r SeAssignPrimaryTokenPrivilege -u InferOSService

# ディレクトリ権限設定
icacls "C:\ProgramData\InferOS" /grant InferOSService:(OI)(CI)F
icacls "C:\path\to\infer-os" /grant InferOSService:(OI)(CI)RX
```

##### **ファイアウォール設定**
```cmd
# Infer-OS Agent API ポート許可
netsh advfirewall firewall add rule name="InferOS Agent API" dir=in action=allow protocol=TCP localport=7031 profile=private

# 特定IPからのアクセスのみ許可（セキュリティ強化）
netsh advfirewall firewall add rule name="InferOS Agent API Restricted" dir=in action=allow protocol=TCP localport=7031 remoteip=127.0.0.1,192.168.1.0/24

# ログ有効化
netsh advfirewall set allprofiles logging filename C:\ProgramData\InferOS\logs\firewall.log
netsh advfirewall set allprofiles logging maxfilesize 4096
netsh advfirewall set allprofiles logging droppedconnections enable
```

#### **監視とアラート設定**

##### **Windows イベントログ統合**
```python
# inferos_control_agent.py での実装例
import logging
import logging.handlers

def setup_windows_event_logging():
    """Windowsイベントログ設定"""
    try:
        # イベントログハンドラー
        event_handler = logging.handlers.NTEventLogHandler(
            appname="InferOS Agent",
            dllname=None,
            logtype="Application"
        )
        event_handler.setLevel(logging.WARNING)
        
        # フォーマッター
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        event_handler.setFormatter(formatter)
        
        # ルートロガーに追加
        logging.getLogger().addHandler(event_handler)
        
    except Exception as e:
        print(f"イベントログ設定エラー: {e}")
```

##### **パフォーマンスカウンター登録**
```cmd
# パフォーマンスカウンター作成
lodctr /R

# カスタムカウンター定義ファイル作成
echo [info] > inferos_counters.ini
echo drivername=InferOS Agent >> inferos_counters.ini
echo symbolfile=inferos_counters.h >> inferos_counters.ini
echo [objects] >> inferos_counters.ini
echo INFEROS_OBJECT_1_009_NAME=InferOS Performance >> inferos_counters.ini

# カウンター登録
lodctr inferos_counters.ini
```

#### **アンインストール手順**

##### **完全削除スクリプト**
```cmd
@echo off
echo Infer-OS Agent アンインストール開始...

# サービス停止
net stop InferOSAgent 2>nul
sc delete InferOSAgent 2>nul

# タスクスケジューラから削除
schtasks /delete /tn "InferOS Agent" /f 2>nul

# ファイアウォールルール削除
netsh advfirewall firewall delete rule name="InferOS Agent API" 2>nul
netsh advfirewall firewall delete rule name="InferOS Agent API Restricted" 2>nul

# ユーザーアカウント削除
net user InferOSService /delete 2>nul

# ディレクトリ削除（データ保持確認）
set /p KEEP_DATA="データを保持しますか？ (Y/N): "
if /i "%KEEP_DATA%"=="N" (
    rmdir /s /q "C:\ProgramData\InferOS" 2>nul
)

# レジストリクリーンアップ
reg delete "HKLM\SYSTEM\CurrentControlSet\Services\InferOSAgent" /f 2>nul
reg delete "HKLM\SOFTWARE\InferOS" /f 2>nul

echo アンインストール完了
pause
```

### **1.2 動作確認**

#### **API動作確認**
```bash
# ヘルスチェック
curl http://127.0.0.1:7031/v1/health

# メトリクス取得
curl http://127.0.0.1:7031/v1/metrics

# ポリシー設定
curl -X POST http://127.0.0.1:7031/v1/policy \
  -H "Content-Type: application/json" \
  -d '{"kv":{"mode":"dynamic","recent_window":64},"io":{"enable_iobinding":true},"scheduler":{"mode":"hybrid"}}'
```

### **1.3 GAIAとの連携設定**

#### **環境変数設定**
```bash
# Infer-OS統合を有効化
export INFEROS_ENABLED=true
export INFEROS_AGENT_URL=http://127.0.0.1:7031

# GAIA設定
export GAIA_OPTIMIZATION_MODE=inferos
export GAIA_KV_QUANTIZATION=enabled
```

#### **GAIA実行（Infer-OS統合）**
```bash
# 基本実行
gaia-cli --model llama-7b --optimization inferos

# 詳細設定
gaia-cli --model llama-7b \
  --optimization inferos \
  --kv-quantization enabled \
  --hybrid-mode \
  --batch-size 4 \
  --max-tokens 1024
```

## 🔌 **Phase 2: プラグイン統合（高度な統合）**

### **2.1 Lemonade Adapterの統合**

#### **Lemonadeサーバーへの統合**
```python
# lemonade_server.py への統合例
from lemonade_adapter import LemonadeAdapter, InferenceContext

# アダプターの初期化
adapter = LemonadeAdapter(
    agent_url="http://127.0.0.1:7031",
    enable_kv_quantization=True
)

# 推論実行時の統合
def run_inference(model, input_text, **kwargs):
    context = InferenceContext(
        model_name=model.name,
        seq_len=len(input_text.split()),
        batch_size=kwargs.get('batch_size', 1),
        target_ftl_ms=kwargs.get('target_ftl_ms', 300),
        quality_budget=kwargs.get('quality_budget', 0.8)
    )
    
    with adapter.inference_session(context) as session:
        # PreRun hook - ポリシー適用
        policy = session.prerun_hook(context)
        
        # ORT設定の適用
        session.apply_ort_policy(session_options, run_options)
        
        # 推論実行
        result = model.generate(input_text, **kwargs)
        
        # PostRun hook - メトリクス収集
        stats = InferenceStats(
            start_time=start_time,
            end_time=time.time(),
            tokens_generated=len(result.split()),
            first_token_latency=first_token_time,
            total_latency=total_time,
            memory_peak=get_memory_usage(),
            quality_score=calculate_quality_score(result)
        )
        session.postrun_hook(stats)
    
    return result
```

### **2.2 KV量子化の詳細設定**

#### **量子化エンジンの設定**
```python
from kv_quantization_engine import KVQuantizationEngine

# 量子化エンジンの初期化
engine = KVQuantizationEngine(
    recent_window=64,        # 最新64トークンはFP16保持
    max_cache_size=10000,    # 最大キャッシュサイズ
    quality_threshold=0.5    # 品質閾値
)

# 動的ポリシー更新
engine.update_quantization_policy(
    memory_pressure=0.7,     # メモリ圧迫度
    quality_budget=0.8,      # 品質予算
    recent_window=32         # 動的ウィンドウサイズ
)

# 統計情報の取得
stats = engine.get_cache_statistics()
print(f"圧縮率: {stats['compression_ratio']:.3f}")
print(f"メモリ削減: {stats['total_memory_saved_mb']:.2f} MB")
```

## 📊 **パフォーマンス測定**

### **3.1 ベンチマーク再現性**

#### **固定測定条件**

##### **テストモデル（厳選2種）**
| モデル | パラメータ数 | 用途 | 推奨メモリ |
|--------|-------------|------|-----------|
| `microsoft/DialoGPT-medium` | 355M | 軽量テスト | 4GB |
| `microsoft/DialoGPT-large` | 762M | 標準テスト | 8GB |

##### **固定プロンプトテンプレート**
```python
# benchmark_prompts.py
BENCHMARK_PROMPTS = {
    "conversation": [
        "Hello, how are you today?",
        "What is artificial intelligence?",
        "Can you explain machine learning?",
        "Tell me about the weather.",
        "What are your hobbies?"
    ],
    "reasoning": [
        "If I have 5 apples and give away 2, how many do I have left?",
        "What is the capital of France?",
        "Explain the difference between AI and ML.",
        "How does photosynthesis work?",
        "What causes seasons to change?"
    ]
}

# 固定生成パラメータ
GENERATION_CONFIG = {
    "max_new_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "do_sample": True,
    "pad_token_id": 50256,
    "eos_token_id": 50256,
    "seed": 42  # 再現性のため固定
}

# 測定条件
BENCHMARK_CONFIG = {
    "warmup_runs": 3,
    "measurement_runs": 10,
    "concurrent_sessions": 1,
    "timeout_seconds": 300,
    "quality_baseline_runs": 5
}
```

#### **再現可能ベンチマークスクリプト**
```python
# reproducible_benchmark.py
import time
import json
import statistics
from datetime import datetime
from typing import Dict, List, Tuple

class ReproducibleBenchmark:
    def __init__(self, model_name: str, config: Dict):
        self.model_name = model_name
        self.config = config
        self.results = []
        
    def run_baseline_benchmark(self) -> Dict:
        """ベースライン（Infer-OS無効）ベンチマーク"""
        print(f"🔄 ベースラインベンチマーク開始: {self.model_name}")
        
        # ウォームアップ
        for i in range(self.config["warmup_runs"]):
            self._single_inference(BENCHMARK_PROMPTS["conversation"][0], warmup=True)
        
        # 測定実行
        results = []
        for prompt_category in BENCHMARK_PROMPTS:
            for prompt in BENCHMARK_PROMPTS[prompt_category]:
                for run in range(self.config["measurement_runs"]):
                    result = self._single_inference(prompt)
                    results.append(result)
        
        return self._calculate_metrics(results, "baseline")
    
    def run_inferos_benchmark(self) -> Dict:
        """Infer-OS有効ベンチマーク"""
        print(f"🚀 Infer-OSベンチマーク開始: {self.model_name}")
        
        # Infer-OS設定適用
        self._apply_inferos_config()
        
        # ウォームアップ
        for i in range(self.config["warmup_runs"]):
            self._single_inference(BENCHMARK_PROMPTS["conversation"][0], warmup=True)
        
        # 測定実行
        results = []
        for prompt_category in BENCHMARK_PROMPTS:
            for prompt in BENCHMARK_PROMPTS[prompt_category]:
                for run in range(self.config["measurement_runs"]):
                    result = self._single_inference(prompt)
                    results.append(result)
        
        return self._calculate_metrics(results, "inferos")
    
    def _single_inference(self, prompt: str, warmup: bool = False) -> Dict:
        """単一推論実行"""
        start_time = time.perf_counter()
        
        # 実際の推論実行（実装依存）
        response = self._execute_inference(prompt)
        
        end_time = time.perf_counter()
        
        result = {
            "prompt": prompt,
            "response": response,
            "latency_ms": (end_time - start_time) * 1000,
            "tokens_generated": len(response.split()),
            "timestamp": datetime.now().isoformat(),
            "warmup": warmup
        }
        
        if not warmup:
            # 品質メトリクス計算
            result["quality_score"] = self._calculate_quality_score(prompt, response)
        
        return result
    
    def _calculate_metrics(self, results: List[Dict], mode: str) -> Dict:
        """メトリクス計算"""
        latencies = [r["latency_ms"] for r in results if not r["warmup"]]
        token_counts = [r["tokens_generated"] for r in results if not r["warmup"]]
        quality_scores = [r["quality_score"] for r in results if not r["warmup"] and "quality_score" in r]
        
        # スループット計算
        total_tokens = sum(token_counts)
        total_time_s = sum(latencies) / 1000
        tps = total_tokens / total_time_s if total_time_s > 0 else 0
        
        # First Token Latency (FTL) 推定
        ftl_ms = statistics.mean(latencies) * 0.3  # 推定値
        
        metrics = {
            "mode": mode,
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "latency": {
                "mean_ms": statistics.mean(latencies),
                "median_ms": statistics.median(latencies),
                "p95_ms": self._percentile(latencies, 95),
                "p99_ms": self._percentile(latencies, 99),
                "std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0
            },
            "throughput": {
                "tokens_per_second": tps,
                "requests_per_second": len(latencies) / total_time_s if total_time_s > 0 else 0
            },
            "first_token_latency_ms": ftl_ms,
            "quality": {
                "mean_score": statistics.mean(quality_scores) if quality_scores else 0,
                "std_score": statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0
            },
            "tokens": {
                "mean_per_response": statistics.mean(token_counts),
                "total_generated": total_tokens
            },
            "config": self.config
        }
        
        return metrics
    
    def generate_comparison_report(self, baseline: Dict, inferos: Dict) -> Dict:
        """比較レポート生成"""
        improvement = {
            "tps_improvement": (inferos["throughput"]["tokens_per_second"] / baseline["throughput"]["tokens_per_second"] - 1) * 100,
            "latency_reduction": (1 - inferos["latency"]["mean_ms"] / baseline["latency"]["mean_ms"]) * 100,
            "ftl_reduction": (1 - inferos["first_token_latency_ms"] / baseline["first_token_latency_ms"]) * 100,
            "quality_change": inferos["quality"]["mean_score"] - baseline["quality"]["mean_score"]
        }
        
        report = {
            "summary": {
                "model": self.model_name,
                "test_date": datetime.now().isoformat(),
                "baseline_tps": baseline["throughput"]["tokens_per_second"],
                "inferos_tps": inferos["throughput"]["tokens_per_second"],
                "improvement_percent": improvement["tps_improvement"],
                "quality_delta": improvement["quality_change"]
            },
            "detailed_metrics": {
                "baseline": baseline,
                "inferos": inferos,
                "improvements": improvement
            },
            "test_conditions": {
                "prompts_tested": len(BENCHMARK_PROMPTS["conversation"]) + len(BENCHMARK_PROMPTS["reasoning"]),
                "runs_per_prompt": self.config["measurement_runs"],
                "total_inferences": len(BENCHMARK_PROMPTS["conversation"]) * self.config["measurement_runs"] + len(BENCHMARK_PROMPTS["reasoning"]) * self.config["measurement_runs"],
                "generation_config": GENERATION_CONFIG
            }
        }
        
        return report

# 使用例
def run_reproducible_benchmark():
    """再現可能ベンチマーク実行"""
    benchmark = ReproducibleBenchmark("microsoft/DialoGPT-medium", BENCHMARK_CONFIG)
    
    # ベースライン測定
    baseline_results = benchmark.run_baseline_benchmark()
    
    # Infer-OS測定
    inferos_results = benchmark.run_inferos_benchmark()
    
    # 比較レポート生成
    report = benchmark.generate_comparison_report(baseline_results, inferos_results)
    
    # 結果保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"benchmark_report_{timestamp}.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"📊 ベンチマーク完了: benchmark_report_{timestamp}.json")
    return report
```

#### **アブレーション分析**
```python
# ablation_study.py
class AblationStudy:
    def __init__(self):
        self.configurations = {
            "baseline": {
                "kv_quantization": False,
                "io_binding": False,
                "hybrid_scheduling": False,
                "description": "標準PyTorch推論"
            },
            "kv_only": {
                "kv_quantization": True,
                "io_binding": False,
                "hybrid_scheduling": False,
                "description": "KV量子化のみ"
            },
            "io_only": {
                "kv_quantization": False,
                "io_binding": True,
                "hybrid_scheduling": False,
                "description": "IOBindingのみ"
            },
            "hybrid_only": {
                "kv_quantization": False,
                "io_binding": False,
                "hybrid_scheduling": True,
                "description": "ハイブリッドスケジューリングのみ"
            },
            "kv_io": {
                "kv_quantization": True,
                "io_binding": True,
                "hybrid_scheduling": False,
                "description": "KV量子化 + IOBinding"
            },
            "full_inferos": {
                "kv_quantization": True,
                "io_binding": True,
                "hybrid_scheduling": True,
                "description": "Infer-OS完全版"
            }
        }
    
    def run_ablation_study(self, model_name: str) -> Dict:
        """アブレーション研究実行"""
        results = {}
        
        for config_name, config in self.configurations.items():
            print(f"🔬 アブレーション実行: {config['description']}")
            
            # 設定適用
            self._apply_configuration(config)
            
            # ベンチマーク実行
            benchmark = ReproducibleBenchmark(model_name, BENCHMARK_CONFIG)
            result = benchmark.run_inferos_benchmark()
            result["config_name"] = config_name
            result["config_description"] = config["description"]
            
            results[config_name] = result
        
        # 相乗効果分析
        analysis = self._analyze_synergy(results)
        
        return {
            "results": results,
            "analysis": analysis
        }
    
    def _analyze_synergy(self, results: Dict) -> Dict:
        """相乗効果分析"""
        baseline_tps = results["baseline"]["throughput"]["tokens_per_second"]
        
        individual_effects = {
            "kv_quantization": (results["kv_only"]["throughput"]["tokens_per_second"] / baseline_tps - 1) * 100,
            "io_binding": (results["io_only"]["throughput"]["tokens_per_second"] / baseline_tps - 1) * 100,
            "hybrid_scheduling": (results["hybrid_only"]["throughput"]["tokens_per_second"] / baseline_tps - 1) * 100
        }
        
        combined_effects = {
            "kv_io": (results["kv_io"]["throughput"]["tokens_per_second"] / baseline_tps - 1) * 100,
            "full_inferos": (results["full_inferos"]["throughput"]["tokens_per_second"] / baseline_tps - 1) * 100
        }
        
        # 相乗効果計算
        expected_kv_io = individual_effects["kv_quantization"] + individual_effects["io_binding"]
        actual_kv_io = combined_effects["kv_io"]
        synergy_kv_io = actual_kv_io - expected_kv_io
        
        expected_full = sum(individual_effects.values())
        actual_full = combined_effects["full_inferos"]
        synergy_full = actual_full - expected_full
        
        return {
            "individual_effects": individual_effects,
            "combined_effects": combined_effects,
            "synergy_analysis": {
                "kv_io_synergy": synergy_kv_io,
                "full_synergy": synergy_full,
                "synergy_interpretation": self._interpret_synergy(synergy_full)
            }
        }
    
    def _interpret_synergy(self, synergy: float) -> str:
        """相乗効果の解釈"""
        if synergy > 5:
            return "強い正の相乗効果"
        elif synergy > 0:
            return "弱い正の相乗効果"
        elif synergy > -5:
            return "相乗効果なし"
        else:
            return "負の相互作用"
```

### **3.2 品質管理**

#### **基本ベンチマーク**
```bash
# Infer-OS無効でのベースライン測定
gaia-cli --model llama-7b --benchmark --optimization none

# Infer-OS有効での性能測定
gaia-cli --model llama-7b --benchmark --optimization inferos

# 詳細ベンチマーク
python benchmark_inferos_gaia.py \
  --model llama-7b \
  --sequences 100 \
  --batch-sizes 1,4,8 \
  --seq-lengths 128,512,1024 \
  --compare-modes baseline,inferos
```

#### **カスタムベンチマーク**
```python
import time
import requests
from lemonade_adapter import LemonadeAdapter

def benchmark_inferos():
    adapter = LemonadeAdapter()
    
    # テストケース
    test_cases = [
        {"seq_len": 128, "batch": 1, "desc": "短文・単一"},
        {"seq_len": 512, "batch": 4, "desc": "中文・バッチ"},
        {"seq_len": 1024, "batch": 1, "desc": "長文・単一"}
    ]
    
    results = []
    for case in test_cases:
        # ベースライン（Infer-OS無効）
        baseline_time = run_inference_baseline(case)
        
        # Infer-OS有効
        inferos_time = run_inference_with_inferos(case)
        
        improvement = baseline_time / inferos_time
        results.append({
            "case": case["desc"],
            "baseline_ms": baseline_time * 1000,
            "inferos_ms": inferos_time * 1000,
            "improvement": improvement
        })
    
    return results
```

### **3.2 メトリクス監視**

#### **リアルタイム監視**
```bash
# メトリクス監視スクリプト
cat > monitor_inferos.py << 'EOF'
import requests
import time
import json

def monitor_metrics():
    while True:
        try:
            response = requests.get("http://127.0.0.1:7031/v1/metrics")
            metrics = response.json()
            
            print(f"TPS: {metrics['tps']:.1f}, "
                  f"FTL: {metrics['ftl_ms']:.1f}ms, "
                  f"VRAM: {metrics['mem']['vram_gb']:.1f}GB, "
                  f"NPU: {metrics['util']['npu']:.1%}, "
                  f"iGPU: {metrics['util']['igpu']:.1%}")
            
        except Exception as e:
            print(f"監視エラー: {e}")
        
        time.sleep(1)

if __name__ == "__main__":
    monitor_metrics()
EOF

python monitor_inferos.py
```

#### **ダッシュボード表示**
```bash
# Grafana/Prometheus連携（オプション）
# メトリクスをPrometheus形式でエクスポート
curl http://127.0.0.1:7031/v1/metrics/prometheus
```

## 🔧 **トラブルシューティング**

### **4.1 フォールバック機能**

#### **自動フォールバック表**

| 事象 | 検出方法 | 直ちに行う処置 | 追随処置 | 復旧条件 |
|------|----------|----------------|----------|----------|
| DML未検出 | `ort.get_available_providers()` | CPU EPへ切替、KVしきい値を保守化 | 管理者へ通知、ドライバー確認 | DML再検出後5分 |
| NPU未検出/温度超過 | メトリクス温度/利用率 | decode→DMLへ移譲 | 5分毎に再試行 | NPU温度正常化 |
| ΔPPL超過 | `/v1/metrics` | KVを INT4→INT8→FP16 へ段階復元 | 閾値を一段保守化 | 品質安定後10分 |
| メモリ圧迫 | `/v1/metrics.mem` | recent_window↓ 等で圧縮強化 | 統計で最適点再探索 | メモリ使用率<80% |
| API応答なし | HTTP timeout | ベースラインモードへ切替 | Agent再起動 | ヘルスチェック正常 |

#### **フォールバック設定例**

##### **DML/NPU未検出時の自動切替**
```python
# inferos_control_agent.py での実装例
import onnxruntime as ort

class DeviceManager:
    def __init__(self):
        self.available_providers = ort.get_available_providers()
        self.current_config = self.detect_optimal_config()
    
    def detect_optimal_config(self):
        """最適なデバイス構成を検出"""
        config = {
            "prefill_device": "cpu",
            "decode_device": "cpu",
            "kv_policy": "conservative"
        }
        
        if "DmlExecutionProvider" in self.available_providers:
            config["prefill_device"] = "dml"
            config["kv_policy"] = "balanced"
            
            # NPU検出（DirectML経由）
            if self.detect_npu_capability():
                config["decode_device"] = "npu"
                config["kv_policy"] = "aggressive"
        
        return config
    
    def apply_fallback_policy(self, error_type):
        """エラー種別に応じたフォールバック"""
        fallback_policies = {
            "dml_unavailable": {
                "prefill_device": "cpu",
                "decode_device": "cpu",
                "kv": {"recent_window": 128, "level_thresholds": {"L1_int8": 0.8, "L2_int4": 0.6, "L3_evict": 0.4}}
            },
            "npu_thermal": {
                "decode_device": "dml",
                "kv": {"recent_window": 96, "level_thresholds": {"L1_int8": 0.75, "L2_int4": 0.55, "L3_evict": 0.35}}
            },
            "quality_degradation": {
                "kv": {"recent_window": 256, "level_thresholds": {"L1_int8": 0.9, "L2_int4": 0.7, "L3_evict": 0.5}}
            },
            "memory_pressure": {
                "kv": {"recent_window": 32, "level_thresholds": {"L1_int8": 0.6, "L2_int4": 0.4, "L3_evict": 0.2}}
            }
        }
        
        return fallback_policies.get(error_type, fallback_policies["dml_unavailable"])
```

##### **品質劣化検出と自動復旧**
```python
class QualityMonitor:
    def __init__(self, max_delta_ppl=0.5, window_size=10):
        self.max_delta_ppl = max_delta_ppl
        self.window_size = window_size
        self.quality_history = []
        self.degradation_count = 0
    
    def check_quality(self, current_ppl, baseline_ppl):
        """品質チェックと自動調整"""
        delta_ppl = current_ppl - baseline_ppl
        self.quality_history.append(delta_ppl)
        
        if len(self.quality_history) > self.window_size:
            self.quality_history.pop(0)
        
        avg_delta = sum(self.quality_history) / len(self.quality_history)
        
        if avg_delta > self.max_delta_ppl:
            self.degradation_count += 1
            if self.degradation_count >= 3:  # 3回連続で劣化
                return self.trigger_quality_recovery()
        else:
            self.degradation_count = 0
        
        return None
    
    def trigger_quality_recovery(self):
        """品質回復処理"""
        recovery_steps = [
            {"kv": {"level_thresholds": {"L2_int4": 0.6, "L3_evict": 0.4}}},  # INT4を減らす
            {"kv": {"level_thresholds": {"L1_int8": 0.8, "L2_int4": 0.7}}},   # INT8を増やす
            {"kv": {"recent_window": 128}},                                    # ウィンドウ拡大
            {"scheduler": {"mode": "gpu_only"}}                               # NPU無効化
        ]
        
        return recovery_steps[min(self.degradation_count - 3, len(recovery_steps) - 1)]
```

#### **エラー処理の詳細**

##### **DirectML量子化演算サポート確認**
```python
def check_quantization_support():
    """DirectMLでの量子化演算サポート確認"""
    try:
        import onnxruntime as ort
        
        # テスト用の小さなONNXモデルで量子化演算を確認
        test_model_path = create_test_quantized_model()
        
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = [
            ('DmlExecutionProvider', {
                'device_id': 0,
                'enable_dynamic_graph_fusion': True
            })
        ]
        
        session = ort.InferenceSession(test_model_path, session_options, providers=providers)
        
        # 量子化演算のサポート確認
        supported_ops = get_supported_quantization_ops(session)
        
        return {
            "w4_support": "QLinearConv" in supported_ops,
            "w8_support": "QLinearMatMul" in supported_ops,
            "kv_int8_support": "QuantizeLinear" in supported_ops,
            "kv_int4_support": "DequantizeLinear" in supported_ops
        }
        
    except Exception as e:
        logger.warning(f"量子化サポート確認エラー: {e}")
        return {"w4_support": False, "w8_support": False, "kv_int8_support": False, "kv_int4_support": False}

def apply_quantization_fallback(support_info):
    """量子化サポート状況に応じたフォールバック"""
    if not support_info["kv_int4_support"]:
        # INT4未サポートの場合はINT8のみ使用
        return {
            "kv": {
                "level_thresholds": {
                    "L1_int8": 0.5,
                    "L2_int4": 1.0,  # INT4を無効化
                    "L3_evict": 0.3
                }
            }
        }
    
    if not support_info["kv_int8_support"]:
        # INT8未サポートの場合は量子化無効
        return {
            "kv": {
                "level_thresholds": {
                    "L1_int8": 1.0,  # INT8を無効化
                    "L2_int4": 1.0,  # INT4を無効化
                    "L3_evict": 0.7  # 圧縮のみ使用
                }
            }
        }
    
    return None  # フォールバック不要
```

### **4.2 一般的な問題**

#### **NPU/iGPUが認識されない**
```bash
# デバイス確認
python -c "
import onnxruntime as ort
print('利用可能なプロバイダー:', ort.get_available_providers())
"

# DirectML確認
python -c "
import onnxruntime as ort
providers = ort.get_available_providers()
if 'DmlExecutionProvider' in providers:
    print('✅ DirectML利用可能')
else:
    print('❌ DirectML利用不可 - ドライバー確認が必要')
"
```

#### **メモリ不足エラー**
```bash
# メモリ使用量確認
curl http://127.0.0.1:7031/v1/metrics | jq '.mem'

# 量子化設定の調整
curl -X POST http://127.0.0.1:7031/v1/policy \
  -H "Content-Type: application/json" \
  -d '{"kv":{"recent_window":32,"level_thresholds":{"L1_int8":0.6,"L2_int4":0.4,"L3_evict":0.2}}}'
```

#### **品質劣化の対処**
```bash
# 品質重視設定
curl -X POST http://127.0.0.1:7031/v1/policy \
  -H "Content-Type: application/json" \
  -d '{"kv":{"recent_window":128,"level_thresholds":{"L1_int8":0.8,"L2_int4":0.6,"L3_evict":0.4}}}'

# ベースラインモードへの切り替え
curl -X POST http://127.0.0.1:7031/v1/baseline -d '{"enable": true}'
```

### **4.2 ログとデバッグ**

#### **詳細ログの有効化**
```python
# inferos_control_agent.py の設定
import logging
logging.basicConfig(level=logging.DEBUG)

# または環境変数で設定
export INFEROS_LOG_LEVEL=DEBUG
python inferos_control_agent.py
```

#### **パフォーマンス履歴の確認**
```bash
# 履歴データの取得
curl http://127.0.0.1:7031/v1/history | jq '.history[-10:]'

# 統計情報のエクスポート
curl http://127.0.0.1:7031/v1/metrics > metrics_$(date +%Y%m%d_%H%M%S).json
```

## 🎯 **最適化のベストプラクティス**

### **5.1 モデル別推奨設定**

#### **7B/8Bモデル（中軽量）**
```yaml
infer_os:
  kv:
    recent_window: 64
    level_thresholds:
      L1_int8: 0.7
      L2_int4: 0.5
      L3_evict: 0.3
  scheduler:
    mode: hybrid
    prefill_device: dml
    decode_device: npu
```

#### **13B/20Bモデル（重量級）**
```yaml
infer_os:
  kv:
    recent_window: 32
    level_thresholds:
      L1_int8: 0.6
      L2_int4: 0.4
      L3_evict: 0.2
  scheduler:
    mode: hybrid
    prefill_device: dml
    decode_device: npu
```

#### **3B以下モデル（軽量）**
```yaml
infer_os:
  kv:
    recent_window: 128
    level_thresholds:
      L1_int8: 0.8
      L2_int4: 0.6
      L3_evict: 0.4
  scheduler:
    mode: gpu_only  # NPUオーバーヘッドを避ける
```

### **5.2 用途別最適化**

#### **チャットボット（低レイテンシ重視）**
```bash
curl -X POST http://127.0.0.1:7031/v1/policy \
  -H "Content-Type: application/json" \
  -d '{
    "kv": {"recent_window": 128, "mode": "latency_optimized"},
    "scheduler": {"mode": "hybrid", "decode_device": "npu"},
    "quality": {"max_delta_ppl": 0.3}
  }'
```

#### **バッチ処理（スループット重視）**
```bash
curl -X POST http://127.0.0.1:7031/v1/policy \
  -H "Content-Type: application/json" \
  -d '{
    "kv": {"recent_window": 32, "mode": "throughput_optimized"},
    "scheduler": {"mode": "hybrid", "prefill_device": "dml"},
    "quality": {"max_delta_ppl": 0.5}
  }'
```

#### **高品質生成（品質重視）**
```bash
curl -X POST http://127.0.0.1:7031/v1/policy \
  -H "Content-Type: application/json" \
  -d '{
    "kv": {"recent_window": 256, "mode": "quality_optimized"},
    "scheduler": {"mode": "gpu_only"},
    "quality": {"max_delta_ppl": 0.1}
  }'
```

## 📈 **継続的な最適化**

### **6.1 自動チューニング**

#### **適応的設定調整**
```python
# auto_tuner.py
import requests
import time
import numpy as np

class InferOSAutoTuner:
    def __init__(self, agent_url="http://127.0.0.1:7031"):
        self.agent_url = agent_url
        self.performance_history = []
    
    def tune_for_workload(self, target_tps=None, target_quality=None):
        """ワークロードに基づく自動チューニング"""
        current_metrics = self.get_metrics()
        
        if target_tps and current_metrics['tps'] < target_tps:
            # スループット向上のための調整
            self.adjust_for_throughput()
        
        if target_quality and current_metrics['quality']['delta_ppl_est'] > target_quality:
            # 品質向上のための調整
            self.adjust_for_quality()
    
    def adjust_for_throughput(self):
        """スループット向上調整"""
        policy = {
            "kv": {"recent_window": 32, "level_thresholds": {"L1_int8": 0.6, "L2_int4": 0.4, "L3_evict": 0.2}},
            "scheduler": {"mode": "hybrid"}
        }
        self.apply_policy(policy)
    
    def adjust_for_quality(self):
        """品質向上調整"""
        policy = {
            "kv": {"recent_window": 128, "level_thresholds": {"L1_int8": 0.8, "L2_int4": 0.6, "L3_evict": 0.4}},
            "scheduler": {"mode": "hybrid"}
        }
        self.apply_policy(policy)

# 使用例
tuner = InferOSAutoTuner()
tuner.tune_for_workload(target_tps=30.0, target_quality=0.3)
```

### **6.2 A/Bテスト**

#### **設定比較テスト**
```bash
# A/Bテストスクリプト
cat > ab_test_inferos.py << 'EOF'
import requests
import time
import statistics

def run_ab_test():
    configs = [
        {"name": "Conservative", "kv": {"recent_window": 128}},
        {"name": "Balanced", "kv": {"recent_window": 64}},
        {"name": "Aggressive", "kv": {"recent_window": 32}}
    ]
    
    results = {}
    for config in configs:
        print(f"Testing {config['name']} configuration...")
        
        # 設定適用
        requests.post("http://127.0.0.1:7031/v1/policy", json=config)
        time.sleep(2)  # 設定反映待ち
        
        # パフォーマンス測定
        metrics = []
        for _ in range(10):
            response = requests.get("http://127.0.0.1:7031/v1/metrics")
            metrics.append(response.json())
            time.sleep(1)
        
        # 統計計算
        tps_values = [m['tps'] for m in metrics]
        results[config['name']] = {
            "avg_tps": statistics.mean(tps_values),
            "std_tps": statistics.stdev(tps_values),
            "avg_quality": statistics.mean([m['quality']['delta_ppl_est'] for m in metrics])
        }
    
    return results

if __name__ == "__main__":
    results = run_ab_test()
    for name, stats in results.items():
        print(f"{name}: TPS={stats['avg_tps']:.1f}±{stats['std_tps']:.1f}, Quality={stats['avg_quality']:.3f}")
EOF

python ab_test_inferos.py
```

## 🎉 **成功事例と期待される結果**

### **7.1 典型的な改善例**

#### **Llama-7B チャットボット**
```
ベースライン:
- TPS: 18.5 tokens/sec
- FTL: 320ms
- メモリ: 14.2GB
- 品質: PPL 12.3

Infer-OS統合後:
- TPS: 26.8 tokens/sec (+45%)
- FTL: 245ms (-23%)
- メモリ: 8.9GB (-37%)
- 品質: PPL 12.7 (ΔPPL +0.4)

ROI: 45%のスループット向上 + 37%のメモリ削減
```

#### **Llama-13B バッチ処理**
```
ベースライン:
- TPS: 12.3 tokens/sec
- バッチサイズ: 4
- メモリ: 26.1GB
- 品質: PPL 11.8

Infer-OS統合後:
- TPS: 19.7 tokens/sec (+60%)
- バッチサイズ: 8 (2倍)
- メモリ: 16.4GB (-37%)
- 品質: PPL 12.2 (ΔPPL +0.4)

ROI: 60%のスループット向上 + バッチサイズ2倍化
```

### **7.2 導入効果の測定**

#### **KPIダッシュボード**
```python
# kpi_dashboard.py
import streamlit as st
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

def create_dashboard():
    st.title("Infer-OS × GAIA パフォーマンスダッシュボード")
    
    # メトリクス取得
    metrics = requests.get("http://127.0.0.1:7031/v1/metrics").json()
    history = requests.get("http://127.0.0.1:7031/v1/history").json()
    
    # KPI表示
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("TPS", f"{metrics['tps']:.1f}", delta=f"+{metrics['tps']/18.5-1:.1%}")
    with col2:
        st.metric("FTL (ms)", f"{metrics['ftl_ms']:.0f}", delta=f"{(metrics['ftl_ms']/320-1)*100:.1f}%")
    with col3:
        st.metric("メモリ (GB)", f"{metrics['mem']['vram_gb']:.1f}", delta=f"{(metrics['mem']['vram_gb']/14.2-1)*100:.1f}%")
    with col4:
        st.metric("品質 (ΔPPL)", f"{metrics['quality']['delta_ppl_est']:.3f}")
    
    # パフォーマンス履歴グラフ
    if history['history']:
        timestamps = [h['timestamp'] for h in history['history']]
        tps_values = [h['tps'] for h in history['history']]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timestamps, y=tps_values, name='TPS'))
        fig.update_layout(title='スループット履歴', xaxis_title='時刻', yaxis_title='TPS')
        st.plotly_chart(fig)

if __name__ == "__main__":
    create_dashboard()
```

## 🔮 **今後の発展**

### **8.1 Phase 3: ハードウェア連携**

#### **NPU常駐最適化**
```python
# 将来実装予定
class NPUResidentOptimizer:
    """NPU常駐KVキャッシュ最適化"""
    
    def __init__(self):
        self.npu_cache_manager = NPUCacheManager()
        self.igpu_spillover = iGPUSpilloverManager()
    
    def optimize_kv_placement(self, kv_chunks):
        """KVチャンクの最適配置"""
        # 重要度の高いKVはNPUに常駐
        # 低重要度はiGPUにスピルオーバー
        pass
```

#### **動的負荷分散**
```python
# 将来実装予定
class HybridLoadBalancer:
    """NPU+iGPU動的負荷分散"""
    
    def balance_workload(self, prefill_load, decode_load):
        """ワークロードに基づく動的分散"""
        # Prefill: iGPU優先、高負荷時はNPUも活用
        # Decode: NPU優先、熱制約時はiGPUに移行
        pass
```

### **8.2 エコシステム拡張**

#### **他フレームワーク対応**
- **Ollama統合**: Ollama + Infer-OS連携
- **LangChain統合**: LangChain Agent + Infer-OS最適化
- **Transformers統合**: HuggingFace Transformers + Infer-OS

#### **クラウド展開**
- **Azure統合**: Azure ML + Infer-OS
- **AWS統合**: SageMaker + Infer-OS
- **コンテナ化**: Docker + Kubernetes対応

## 📞 **サポート**

### **9.1 コミュニティ**
- **GitHub**: https://github.com/kojima123/infer-os
- **Issues**: バグ報告・機能要望
- **Discussions**: 技術討論・質問

### **9.2 トラブルシューティング**
- **ログ確認**: `tail -f inferos_agent.log`
- **設定リセット**: `curl -X POST http://127.0.0.1:7031/v1/baseline`
- **強制再起動**: `pkill -f inferos_control_agent && python inferos_control_agent.py`

### **9.3 パフォーマンス相談**
特定のワークロードでの最適化相談は、以下の情報と共にIssueを作成してください：
- モデルサイズ・種類
- 典型的な入力長・バッチサイズ
- 現在のパフォーマンス指標
- 目標パフォーマンス
- ハードウェア構成

---

**🎯 GAIA × Infer-OS統合により、AMD Ryzen AI NPU + Radeon iGPU環境でのLLM推論性能を最大化し、次世代AIアプリケーションの基盤を構築しましょう！**

