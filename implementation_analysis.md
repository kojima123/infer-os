# 現在の実装と公式サンプルの比較分析

## 現在の実装の問題点

### 1. 公式推奨アプローチとの乖離

#### ❌ 現在の実装
```python
# カスタムPyTorchモデル → ONNX変換
class SimpleNPUModel(nn.Module):
    def __init__(self):
        self.linear1 = nn.Linear(512, 1024)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(1024, 1000)
```

#### ✅ 公式推奨アプローチ
```python
# OnnxRuntime GenAI (OGA)フレームワーク使用
# 事前量子化済み4-bit LLMモデル使用
# AMD hybrid collectionからの公式モデル
```

### 2. LLM実装の根本的な問題

#### ❌ 現在の実装
- **ダミーモデル**: 実際のLLMではない単純なLinear層
- **テキスト生成なし**: NPU推論テストのみ
- **量子化なし**: FP32モデルのまま
- **公式モデル未使用**: カスタムモデル作成

#### ✅ 公式推奨
- **実際のLLM**: Llama2、Llama3、Phi-3等
- **4-bit量子化**: NPU最適化済み
- **OGA統合**: OnnxRuntime GenAI使用
- **公式モデル**: AMD hybrid collectionから

### 3. VitisAI ExecutionProvider設定の問題

#### ❌ 現在の実装
```python
vitisai_options = {
    'config_file': 'vaip_config.json',
    'timeout': self.timeout_seconds,
    'device_timeout': self.timeout_seconds,  # 非公式オプション
    'polling_timeout': 1,                    # 非公式オプション
    'exec_timeout': self.timeout_seconds     # 非公式オプション
}
```

#### ✅ 公式推奨
```python
provider_options = [{
    "cache_dir": "/tmp/my_cache",
    "cache_key": "model_md5",
    "log_level": "info"
}]
```

### 4. infer-OS統合の問題

#### ❌ 現在の実装
- **独自実装**: カスタムinfer-OS設定
- **環境変数**: 非公式の環境変数使用
- **設定ファイル**: 独自のJSON設定

#### ✅ 公式推奨
- **Lemonade SDK**: 公式高レベルPython SDK
- **OGA統合**: OnnxRuntime GenAI経由
- **標準API**: 公式インターフェース使用

## 余分な処理の特定

### 1. 不要なXRT環境設定
```python
# ❌ 余分な処理
xrt_env_vars = {
    'XRT_TIMEOUT': str(timeout_seconds * 1000),
    'XRT_DEVICE_TIMEOUT': str(timeout_seconds * 1000),
    'VITIS_AI_TIMEOUT': str(timeout_seconds),
    'FLEXML_TIMEOUT': str(timeout_seconds),
    'XRT_POLLING_TIMEOUT': '1000',
    'XRT_EXEC_TIMEOUT': str(timeout_seconds * 1000),
    'VAIML_TIMEOUT': str(timeout_seconds)
}
```

### 2. 不要なONNXモデル作成
```python
# ❌ 余分な処理
torch.onnx.export(model, dummy_input, model_path, ...)
onnx_model = onnx.load(model_path)
onnx_model.ir_version = 10
```

### 3. 不要なタイムアウトハンドラー
```python
# ❌ 余分な処理
class XRTTimeoutHandler:
    def timeout_handler(self, signum, frame):
        # 複雑なタイムアウト処理
```

## 誤った実装の特定

### 1. LLMアーキテクチャの誤解
- **現在**: Linear層のみのダミーモデル
- **正解**: 実際のTransformerベースLLM

### 2. NPU最適化の誤解
- **現在**: カスタムモデルのONNX変換
- **正解**: 事前量子化済み公式モデル使用

### 3. infer-OS統合の誤解
- **現在**: 独自の環境変数とフラグ
- **正解**: Lemonade SDK経由の標準統合

### 4. ベンチマーク対象の誤解
- **現在**: NPU推論テストのみ
- **正解**: 実際のLLMテキスト生成性能

## 推奨される修正方針

### 1. 公式OGAフレームワーク採用
- OnnxRuntime GenAI使用
- AMD hybrid collectionモデル使用
- Lemonade SDK統合

### 2. 実際のLLM実装
- 事前量子化済みモデル
- 実際のテキスト生成
- 性能測定

### 3. 設定の簡素化
- 公式プロバイダーオプションのみ
- 不要な環境変数削除
- 標準的なセッション設定

### 4. infer-OSベンチマーク対応
- ON/OFF比較機能
- 実際の生成性能測定
- 標準的なメトリクス

