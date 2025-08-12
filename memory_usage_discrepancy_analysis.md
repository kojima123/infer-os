# インタラクティブモードと単発実行モードのメモリ使用率乖離問題分析

## 📊 発見された重大な問題

### メモリ使用率の大幅な乖離

| 実行モード | メモリ使用率 | 差異 |
|------------|--------------|------|
| **インタラクティブモード** | **39.7%** | 基準 |
| **単発実行モード** | **88.5%** | **+48.8%** |

**乖離率**: 約123%の増加（2倍以上の差）

## 🔍 原因分析

### 1. **実行フローの違い**

#### **インタラクティブモード**
```python
def run_interactive_mode(self):
    # NPU監視開始
    self.start_npu_monitoring()
    
    # 初期状態でメモリ使用率測定 → 39.7%
    
    try:
        while True:
            # ユーザー入力待機（メモリ解放される）
            prompt = input("💬 プロンプトを入力してください: ")
            
            # テキスト生成（必要時のみメモリ使用）
            result = self.generate_text_with_ollama_fixed(prompt, max_tokens=100)
    finally:
        self.stop_npu_monitoring()
```

#### **単発実行モード**
```python
def main():
    # システム初期化
    system = OllamaInferOSFixedController()
    system.initialize_system()
    
    # 単発生成
    system.start_npu_monitoring()
    result = system.generate_text_with_ollama_fixed(args.prompt, args.tokens, args.template)
    
    # メモリ使用率測定 → 88.5%（生成直後の高負荷状態）
    system.stop_npu_monitoring()
    system.show_npu_stats()  # ここでメモリ使用率測定
```

### 2. **メモリ測定タイミングの違い**

#### **インタラクティブモード**
- **測定タイミング**: 初期化直後、ユーザー入力待機中
- **状態**: アイドル状態、メモリが解放されている
- **Ollamaモデル**: 必要時のみロード

#### **単発実行モード**
- **測定タイミング**: テキスト生成直後
- **状態**: 高負荷状態、メモリが大量使用中
- **Ollamaモデル**: フルロード状態

### 3. **Ollamaモデルのメモリ管理**

#### **gpt-oss:20b (12.8GB)のメモリ使用パターン**

##### **インタラクティブモード**
```
初期化時: モデル部分ロード
待機中: メモリ解放・最適化
生成時: 必要部分のみロード
待機中: 再度メモリ解放
```

##### **単発実行モード**
```
初期化時: モデル部分ロード
生成時: フルロード（12.8GB）
測定時: フルロード状態維持 ← 高メモリ使用率
終了時: メモリ解放
```

### 4. **infer-OS最適化設定の影響**

#### **メモリ最適化設定**
```python
if self.infer_os_config["memory_optimization"]:
    os.environ['OLLAMA_MAX_LOADED_MODELS'] = '1'  # 1つのモデルのみ
    os.environ['OLLAMA_NUM_PARALLEL'] = '1'  # 並列処理制限
    os.environ['OLLAMA_LOAD_TIMEOUT'] = '60'  # タイムアウト短縮
```

**問題**: 単発実行モードでは、生成直後にメモリ解放される前に測定が行われる

### 5. **NPU監視とメモリ測定の関係**

#### **show_npu_stats()関数の処理**
```python
def show_npu_stats(self):
    # システム情報取得
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()  # ここでメモリ使用率測定
    print(f"💾 メモリ使用率: {memory.percent:.1f}%")
```

**問題**: `psutil.virtual_memory()`は瞬間的なメモリ使用率を取得するため、測定タイミングで大きく変わる

## 📈 詳細な実行フロー比較

### インタラクティブモード実行フロー

```
1. システム初期化
   └─ メモリ使用率: 低い（基本システムのみ）

2. show_system_status()呼び出し
   └─ メモリ使用率測定: 39.7% ← ここで測定

3. インタラクティブループ開始
   └─ ユーザー入力待機（アイドル状態）

4. テキスト生成（必要時のみ）
   └─ 一時的にメモリ使用率上昇

5. 生成完了後、再度待機状態
   └─ メモリ使用率低下
```

### 単発実行モード実行フロー

```
1. システム初期化
   └─ メモリ使用率: 低い

2. NPU監視開始
   └─ バックグラウンド処理開始

3. テキスト生成実行
   └─ Ollamaモデルフルロード
   └─ メモリ使用率急上昇: 88.5%

4. 生成完了

5. show_npu_stats()呼び出し
   └─ メモリ使用率測定: 88.5% ← ここで測定（高負荷状態）

6. システム終了
   └─ メモリ解放
```

## 🚨 問題の本質

### 1. **測定タイミングの不整合**
- インタラクティブモード: アイドル状態で測定
- 単発実行モード: 高負荷状態で測定

### 2. **Ollamaモデルのメモリ管理**
- 12.8GBの大型モデルの動的ロード/アンロード
- 生成時の一時的な大量メモリ使用

### 3. **infer-OS最適化の効果測定困難**
- 測定タイミングの違いにより、真の最適化効果が見えない
- メモリ使用率の比較が無意味になる

## 💡 修正案

### 1. **統一された測定タイミング**

#### **修正前**
```python
# インタラクティブモード
def run_interactive_mode(self):
    self.show_system_status()  # 初期化直後に測定
    
# 単発実行モード  
def main():
    result = system.generate_text_with_ollama_fixed(...)
    system.show_npu_stats()  # 生成直後に測定
```

#### **修正後**
```python
# 両モード共通
def measure_memory_consistently(self):
    # 生成前後の両方で測定
    memory_before = psutil.virtual_memory().percent
    
    # テキスト生成
    result = self.generate_text_with_ollama_fixed(...)
    
    # 生成直後
    memory_after = psutil.virtual_memory().percent
    
    # 5秒待機してメモリ安定化
    time.sleep(5)
    memory_stable = psutil.virtual_memory().percent
    
    return {
        "before": memory_before,
        "peak": memory_after,
        "stable": memory_stable
    }
```

### 2. **メモリ使用率の段階的測定**

```python
def comprehensive_memory_analysis(self):
    """包括的メモリ分析"""
    measurements = {
        "initialization": psutil.virtual_memory().percent,
        "pre_generation": None,
        "during_generation": None,
        "post_generation": None,
        "stabilized": None
    }
    
    # 生成前
    measurements["pre_generation"] = psutil.virtual_memory().percent
    
    # 生成中（バックグラウンドで監視）
    def monitor_during_generation():
        max_usage = 0
        while self.generating:
            current = psutil.virtual_memory().percent
            max_usage = max(max_usage, current)
            time.sleep(0.5)
        measurements["during_generation"] = max_usage
    
    # 生成実行
    monitor_thread = threading.Thread(target=monitor_during_generation)
    self.generating = True
    monitor_thread.start()
    
    result = self.generate_text_with_ollama_fixed(...)
    
    self.generating = False
    monitor_thread.join()
    
    # 生成直後
    measurements["post_generation"] = psutil.virtual_memory().percent
    
    # 5秒後（安定化）
    time.sleep(5)
    measurements["stabilized"] = psutil.virtual_memory().percent
    
    return measurements, result
```

### 3. **infer-OS最適化効果の正確な測定**

```python
def accurate_optimization_comparison(self, prompt, tokens):
    """正確な最適化効果比較"""
    
    # 最適化有効で測定
    self.infer_os_enabled = True
    self.apply_infer_os_optimizations()
    memory_on, result_on = self.comprehensive_memory_analysis()
    
    # メモリクリア
    self.clear_memory_cache()
    time.sleep(10)
    
    # 最適化無効で測定
    self.infer_os_enabled = False
    memory_off, result_off = self.comprehensive_memory_analysis()
    
    return {
        "optimization_on": memory_on,
        "optimization_off": memory_off,
        "results": {"on": result_on, "off": result_off}
    }
```

### 4. **メモリクリア機能の追加**

```python
def clear_memory_cache(self):
    """メモリキャッシュクリア"""
    try:
        # Ollamaモデルアンロード
        requests.post(f"{self.ollama_api}/unload", 
                     json={"model": self.current_model["name"]})
        
        # Python ガベージコレクション
        import gc
        gc.collect()
        
        # システムメモリクリア（Linux）
        if os.name == 'posix':
            os.system('sync && echo 3 > /proc/sys/vm/drop_caches')
        
        print("🧹 メモリキャッシュクリア完了")
        
    except Exception as e:
        print(f"⚠️ メモリクリアエラー: {e}")
```

## 🎯 期待される修正効果

### 1. **正確な比較測定**
- 同一条件でのメモリ使用率測定
- infer-OS最適化の真の効果確認
- 実行モード間の一貫性

### 2. **詳細なメモリプロファイル**
```
📊 詳細メモリ使用率分析:
  🔧 初期化時: 35.2%
  📝 生成前: 38.1%
  🔥 生成中最大: 89.3%
  📊 生成直後: 85.7%
  ✅ 安定化後: 42.6%
  
  💡 infer-OS最適化効果:
    🔥 ピーク使用率: 89.3% → 76.4% (12.9%削減)
    ✅ 安定化使用率: 42.6% → 38.9% (3.7%削減)
```

### 3. **実用的な性能指標**
- ピーク時メモリ使用率
- 安定化後メモリ使用率
- メモリ解放速度
- 最適化効果の定量化

## 📋 修正優先順位

### 高優先度（即座に修正）
1. **統一された測定タイミング**: 両モードで同じタイミングで測定
2. **メモリ安定化待機**: 生成後5秒待機してから測定
3. **詳細なメモリプロファイル**: 段階的測定の実装

### 中優先度（次回修正）
1. **メモリクリア機能**: 正確な比較のためのクリア機能
2. **動的メモリ監視**: 生成中のリアルタイム監視
3. **最適化効果の定量化**: 数値による効果測定

### 低優先度（将来的改善）
1. **メモリ使用率予測**: 事前の使用率予測
2. **自動メモリ最適化**: 使用率に応じた自動調整
3. **メモリリーク検出**: 長期実行時のリーク検出

## 🔧 推奨修正アプローチ

### 1. **即座に実装可能な修正**
```python
def show_memory_consistently(self, context=""):
    """一貫したメモリ使用率表示"""
    # 5秒待機してメモリ安定化
    time.sleep(5)
    memory = psutil.virtual_memory()
    print(f"💾 メモリ使用率 ({context}): {memory.percent:.1f}%")
    return memory.percent
```

### 2. **両モードでの統一測定**
```python
# インタラクティブモード
memory_usage = self.show_memory_consistently("待機中")

# 単発実行モード  
memory_usage = self.show_memory_consistently("生成後安定化")
```

## 📊 修正後の期待結果

### 修正前（現状）
```
インタラクティブモード: 39.7% (アイドル状態)
単発実行モード: 88.5% (高負荷状態)
比較: 意味のない比較
```

### 修正後（期待）
```
インタラクティブモード: 45.3% (安定化後)
単発実行モード: 47.1% (安定化後)
比較: 有意義な比較が可能

infer-OS最適化効果:
  ON: 45.3% → OFF: 52.8% (7.5%削減効果)
```

この修正により、真のinfer-OS最適化効果を正確に測定できるようになります。

