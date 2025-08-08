# 🚀 5分で始める言語モデル クイックスタートガイド

**所要時間**: 約5分  
**対象**: プログラミング初心者〜中級者  
**環境**: Windows/Mac/Linux（Ryzen AI推奨）

---

## 📋 必要なもの

- Python 3.8以上がインストールされたPC
- インターネット接続
- 8GB以上のメモリ（16GB推奨）
- 10GB以上の空きディスク容量

---

## ⚡ 超高速セットアップ（3ステップ）

### ステップ1️⃣: 環境準備（1分）

```bash
# 仮想環境を作成
python -m venv ai_env

# 仮想環境を有効化
# Windows:
ai_env\Scripts\activate
# Mac/Linux:
source ai_env/bin/activate

# 必要なライブラリをインストール
pip install torch transformers accelerate
```

### ステップ2️⃣: モデルをダウンロード（2分）

```python
# quick_setup.py として保存
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("🤖 言語モデルをダウンロード中...")

# 高速・軽量モデル（Phi-3 Mini）を使用
model_name = "microsoft/Phi-3-mini-4k-instruct"

# トークナイザーとモデルをダウンロード
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

print("✅ ダウンロード完了！")

# 簡単なテスト
def chat(message):
    inputs = tokenizer(f"User: {message}\nAssistant:", return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Assistant:")[-1].strip()

# テスト実行
print("\n🎉 セットアップ完了！テストしてみましょう：")
print("応答:", chat("こんにちは！"))
```

### ステップ3️⃣: 実行（30秒）

```bash
python quick_setup.py
```

---

## 🎯 用途別おすすめモデル

### 💬 チャット・対話
```python
model_name = "microsoft/Phi-3-mini-4k-instruct"  # 高速
# または
model_name = "meta-llama/Llama-2-7b-chat-hf"    # 高品質
```

### 💻 コード生成
```python
model_name = "codellama/CodeLlama-7b-Instruct-hf"
```

### 🇯🇵 日本語特化
```python
model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
```

---

## 🔧 完全版セットアップスクリプト

以下のスクリプトをコピー＆ペーストして `easy_start.py` として保存：

```python
#!/usr/bin/env python3
"""
🚀 5分で始める言語モデル - 完全版
"""

import os
import sys

def install_requirements():
    """必要なライブラリを自動インストール"""
    print("📦 必要なライブラリをインストール中...")
    
    packages = [
        "torch",
        "transformers>=4.36.0", 
        "accelerate"
    ]
    
    for package in packages:
        os.system(f"{sys.executable} -m pip install {package}")
    
    print("✅ インストール完了！")

def setup_model():
    """モデルのセットアップ"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    print("🤖 言語モデルを準備中...")
    
    # 軽量・高速モデルを使用
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        print("✅ モデル準備完了！")
        return tokenizer, model
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        print("💡 インターネット接続を確認してください")
        return None, None

def chat_interface(tokenizer, model):
    """チャットインターフェース"""
    import torch
    
    print("\n🎉 言語モデルが使用可能になりました！")
    print("💬 何でも話しかけてください（終了: 'quit'）")
    print("-" * 50)
    
    while True:
        user_input = input("\nあなた: ")
        
        if user_input.lower() in ['quit', 'exit', '終了']:
            print("👋 お疲れさまでした！")
            break
        
        # AI応答の生成
        inputs = tokenizer(f"User: {user_input}\nAssistant:", return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        ai_response = response.split("Assistant:")[-1].strip()
        
        print(f"🤖 AI: {ai_response}")

def main():
    """メイン処理"""
    print("🚀 5分で始める言語モデル")
    print("=" * 40)
    
    # ステップ1: ライブラリインストール
    try:
        import torch
        import transformers
        print("✅ 必要なライブラリは既にインストール済み")
    except ImportError:
        install_requirements()
    
    # ステップ2: モデルセットアップ
    tokenizer, model = setup_model()
    
    if tokenizer and model:
        # ステップ3: チャット開始
        chat_interface(tokenizer, model)
    else:
        print("❌ セットアップに失敗しました")

if __name__ == "__main__":
    main()
```

---

## 🎮 使い方

### 基本的な使い方
```bash
python easy_start.py
```

### カスタマイズ例

#### 1. 異なるモデルを使用
```python
# スクリプト内の model_name を変更
model_name = "meta-llama/Llama-2-7b-chat-hf"  # より高品質
```

#### 2. 応答の調整
```python
# 温度を下げる（より一貫した応答）
temperature=0.3

# 温度を上げる（より創造的な応答）
temperature=1.0

# 応答長を調整
max_new_tokens=50   # 短い応答
max_new_tokens=300  # 長い応答
```

---

## 🔍 トラブルシューティング

### よくある問題と解決方法

#### ❌ メモリ不足エラー
```python
# 解決方法: より小さなモデルを使用
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

#### ❌ ダウンロードが遅い
```python
# 解決方法: キャッシュディレクトリを指定
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir="./model_cache"
)
```

#### ❌ 日本語が文字化け
```python
# 解決方法: 日本語対応モデルを使用
model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
```

---

## 📊 性能比較表

| モデル | サイズ | 速度 | メモリ | 用途 |
|--------|--------|------|--------|------|
| Phi-3 Mini | 3.8B | ⭐⭐⭐⭐⭐ | 2.4GB | 高速チャット |
| Llama 2 7B | 7B | ⭐⭐⭐ | 4.5GB | 高品質対話 |
| CodeLlama 7B | 7B | ⭐⭐⭐ | 4.8GB | コード生成 |
| TinyLlama | 1.1B | ⭐⭐⭐⭐⭐ | 1.2GB | 軽量・高速 |

---

## 🎯 次のステップ

### レベルアップしたい場合

1. **より高性能なモデル**を試す
   - Llama 2 13B（高品質）
   - Mixtral 8x7B（最新技術）

2. **専門用途向けモデル**を探す
   - 医療: BioGPT
   - 法律: Legal-BERT
   - 科学: SciBERT

3. **ファインチューニング**を学ぶ
   - 独自データでの学習
   - 特定タスクへの最適化

### 参考リンク

- [Hugging Face Model Hub](https://huggingface.co/models)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

---

## 🎉 完了！

これで言語モデルが使えるようになりました！

**何か質問があれば、AIに直接聞いてみてください：**
```
あなた: Pythonでリストをソートする方法を教えて
🤖 AI: Pythonでリストをソートするには...
```

**楽しいAIライフを！** 🚀✨

