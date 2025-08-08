#!/usr/bin/env python3
"""
🚀 5分で始める言語モデル - 完全版
初心者でも簡単に言語モデルを使い始められるスクリプト

使用方法:
    python easy_start.py

作成者: Manus AI
バージョン: 1.0
"""

import os
import sys
import subprocess
import time

def print_banner():
    """開始バナーの表示"""
    print("🚀" + "=" * 50 + "🚀")
    print("    5分で始める言語モデル - クイックスタート")
    print("🚀" + "=" * 50 + "🚀")
    print()

def check_python_version():
    """Python バージョンの確認"""
    print("🔍 Python バージョンを確認中...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8以上が必要です")
        print(f"   現在のバージョン: {sys.version}")
        print("   https://python.org から最新版をダウンロードしてください")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} - OK")
    return True

def install_requirements():
    """必要なライブラリを自動インストール"""
    print("\n📦 必要なライブラリをインストール中...")
    print("   これには数分かかる場合があります...")
    
    packages = [
        "torch",
        "transformers>=4.36.0", 
        "accelerate"
    ]
    
    for i, package in enumerate(packages, 1):
        print(f"   [{i}/{len(packages)}] {package} をインストール中...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package, "--quiet"
            ], check=True, capture_output=True, text=True)
            
            print(f"   ✅ {package} インストール完了")
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ {package} のインストールに失敗: {e}")
            return False
    
    print("✅ 全てのライブラリのインストール完了！")
    return True

def check_dependencies():
    """依存関係の確認"""
    print("\n🔍 依存関係を確認中...")
    
    required_packages = ["torch", "transformers", "accelerate"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (未インストール)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 {len(missing_packages)}個のパッケージをインストールする必要があります")
        response = input("   今すぐインストールしますか？ (y/n): ").lower()
        
        if response == 'y':
            return install_requirements()
        else:
            print("   インストールをキャンセルしました")
            return False
    
    print("✅ 全ての依存関係が満たされています")
    return True

def select_model():
    """モデル選択"""
    print("\n🤖 使用するモデルを選択してください:")
    print()
    
    models = {
        "1": {
            "name": "microsoft/Phi-3-mini-4k-instruct",
            "description": "Phi-3 Mini (推奨)",
            "features": "高速・軽量・日本語対応",
            "size": "3.8B",
            "memory": "~2.4GB"
        },
        "2": {
            "name": "meta-llama/Llama-2-7b-chat-hf", 
            "description": "Llama 2 7B Chat",
            "features": "高品質・汎用性",
            "size": "7B",
            "memory": "~4.5GB"
        },
        "3": {
            "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "description": "TinyLlama (超軽量)",
            "features": "超高速・省メモリ",
            "size": "1.1B", 
            "memory": "~1.2GB"
        }
    }
    
    for key, model in models.items():
        print(f"   {key}. {model['description']}")
        print(f"      特徴: {model['features']}")
        print(f"      サイズ: {model['size']} | メモリ: {model['memory']}")
        print()
    
    while True:
        choice = input("選択してください (1-3, デフォルト: 1): ").strip()
        
        if choice == "" or choice == "1":
            return models["1"]["name"]
        elif choice in models:
            return models[choice]["name"]
        else:
            print("❌ 無効な選択です。1-3の数字を入力してください。")

def setup_model(model_name):
    """モデルのセットアップ"""
    print(f"\n🤖 言語モデルを準備中: {model_name}")
    print("   初回実行時はモデルのダウンロードに時間がかかります...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print("   📥 トークナイザーをダウンロード中...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("   📥 モデルをダウンロード中...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        print("✅ モデル準備完了！")
        return tokenizer, model
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        print("💡 解決方法:")
        print("   - インターネット接続を確認してください")
        print("   - より軽量なモデル (TinyLlama) を試してください")
        print("   - メモリ不足の場合は他のアプリケーションを終了してください")
        return None, None

def run_quick_test(tokenizer, model):
    """クイックテスト"""
    print("\n🧪 クイックテストを実行中...")
    
    try:
        import torch
        
        test_prompt = "こんにちは！"
        inputs = tokenizer(f"User: {test_prompt}\nAssistant:", return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        ai_response = response.split("Assistant:")[-1].strip()
        
        print(f"   入力: {test_prompt}")
        print(f"   出力: {ai_response}")
        print("✅ テスト成功！モデルが正常に動作しています")
        return True
        
    except Exception as e:
        print(f"❌ テストに失敗: {e}")
        return False

def chat_interface(tokenizer, model):
    """チャットインターフェース"""
    import torch
    
    print("\n" + "🎉" * 20)
    print("   言語モデルが使用可能になりました！")
    print("🎉" * 20)
    print()
    print("💬 何でも話しかけてください")
    print("📝 使用例:")
    print("   - 「Pythonでリストをソートする方法は？」")
    print("   - 「今日の予定を整理して」")
    print("   - 「面白い話をして」")
    print()
    print("🚪 終了するには 'quit' または 'exit' と入力してください")
    print("-" * 60)
    
    conversation_count = 0
    
    while True:
        try:
            user_input = input("\n😊 あなた: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '終了', 'q']:
                print("\n👋 お疲れさまでした！")
                print(f"   今回の会話数: {conversation_count}回")
                print("   また使ってくださいね！")
                break
            
            if not user_input:
                print("💡 何か入力してください")
                continue
            
            print("🤖 AI: 考え中...", end="", flush=True)
            
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
            
            print(f"\r🤖 AI: {ai_response}")
            conversation_count += 1
            
        except KeyboardInterrupt:
            print("\n\n👋 Ctrl+C で終了しました")
            break
        except Exception as e:
            print(f"\n❌ エラーが発生しました: {e}")
            print("💡 もう一度試してください")

def show_tips():
    """使用のコツを表示"""
    print("\n💡 使用のコツ:")
    print("   📝 具体的な質問をすると良い回答が得られます")
    print("   🔄 満足いく回答が得られない場合は、質問を変えて再試行")
    print("   ⚡ 応答が遅い場合は、より軽量なモデルを選択")
    print("   🌐 日本語と英語の両方で質問できます")

def main():
    """メイン処理"""
    print_banner()
    
    # ステップ1: Python バージョン確認
    if not check_python_version():
        return
    
    # ステップ2: 依存関係確認・インストール
    if not check_dependencies():
        return
    
    # ステップ3: モデル選択
    model_name = select_model()
    
    # ステップ4: モデルセットアップ
    tokenizer, model = setup_model(model_name)
    
    if not tokenizer or not model:
        print("\n❌ セットアップに失敗しました")
        print("💡 問題が解決しない場合は、より軽量なモデルを試してください")
        return
    
    # ステップ5: クイックテスト
    if not run_quick_test(tokenizer, model):
        print("⚠️  テストに失敗しましたが、チャットを試すことはできます")
    
    # ステップ6: 使用のコツ表示
    show_tips()
    
    # ステップ7: チャット開始
    chat_interface(tokenizer, model)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 プログラムを終了しました")
    except Exception as e:
        print(f"\n❌ 予期しないエラーが発生しました: {e}")
        print("💡 問題が続く場合は、Pythonとライブラリを最新版に更新してください")

