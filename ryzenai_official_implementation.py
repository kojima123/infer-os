#!/usr/bin/env python3
"""
AMD公式RyzenAI実装
Lemonade SDK + OGA-Hybrid使用
公式ドキュメント: https://ryzenai.docs.amd.com/en/latest/llm/high_level_python.html
"""

import os
import sys
import time
import subprocess
from typing import Optional, List, Dict, Any
import argparse

# Lemonade SDK インポート（公式API）
try:
    from lemonade.api import from_pretrained
    LEMONADE_AVAILABLE = True
    print("✅ Lemonade SDK インポート成功")
except ImportError as e:
    LEMONADE_AVAILABLE = False
    print(f"⚠️ Lemonade SDK インポートエラー: {e}")
    print("💡 以下のコマンドでインストールしてください:")
    print("  conda create -n ryzenai-llm python=3.10")
    print("  conda activate ryzenai-llm")
    print("  pip install lemonade-sdk[llm-oga-hybrid]")
    print("  lemonade-install --ryzenai hybrid")

class RyzenAIOfficialLLM:
    """AMD公式RyzenAI LLM実装"""
    
    def __init__(self, model_name: str = "amd/Llama-3.2-1B-Instruct-awq-g128-int4-asym-fp16-onnx-hybrid"):
        """
        AMD公式RyzenAI LLM初期化
        
        Args:
            model_name: 使用するモデル名（AMD公式モデル）
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        
        print("🇯🇵 AMD公式RyzenAI LLM初期化開始")
        print(f"📦 モデル: {model_name}")
        
        if LEMONADE_AVAILABLE:
            self._initialize_model()
        else:
            print("❌ Lemonade SDK が利用できません")
    
    def _initialize_model(self):
        """モデル初期化"""
        try:
            print("🔧 Lemonade SDK モデル初期化中...")
            print("⚡ NPU + CPU ハイブリッドモードで初期化...")
            
            # 公式API使用
            self.model, self.tokenizer = from_pretrained(
                self.model_name,
                recipe="oga-hybrid"  # NPU + CPU ハイブリッドモード
            )
            
            print("✅ モデル初期化完了")
            print(f"  📱 モデル: {self.model_name}")
            print(f"  🔧 レシピ: oga-hybrid (NPU + CPU)")
            
            self.is_initialized = True
            
            # 初期化テスト
            self._initialization_test()
            
        except Exception as e:
            print(f"❌ モデル初期化エラー: {e}")
            print("💡 以下を確認してください:")
            print("  1. RyzenAI 1.5.1がインストールされているか")
            print("  2. NPUドライバーが正しくインストールされているか")
            print("  3. Conda環境が正しく設定されているか")
            self.is_initialized = False
    
    def _initialization_test(self):
        """初期化テスト"""
        try:
            print("🧪 初期化テスト実行中...")
            
            test_prompt = "Hello"
            input_ids = self.tokenizer(test_prompt, return_tensors="pt").input_ids
            
            # 短いテスト生成
            response = self.model.generate(input_ids, max_new_tokens=5)
            test_output = self.tokenizer.decode(response[0])
            
            print(f"✅ 初期化テスト成功")
            print(f"  📝 テスト入力: {test_prompt}")
            print(f"  📝 テスト出力: {test_output}")
            
        except Exception as e:
            print(f"⚠️ 初期化テスト警告: {e}")
    
    def generate_text(self, prompt: str, max_new_tokens: int = 64, **kwargs) -> str:
        """
        テキスト生成
        
        Args:
            prompt: 入力プロンプト
            max_new_tokens: 最大生成トークン数
            **kwargs: 追加の生成パラメータ
            
        Returns:
            str: 生成されたテキスト
        """
        if not self.is_initialized:
            print("❌ モデルが初期化されていません")
            return prompt
        
        try:
            print("⚡ RyzenAI NPU推論実行中...")
            print(f"📝 プロンプト: \"{prompt}\"")
            print(f"🔢 最大生成トークン数: {max_new_tokens}")
            
            start_time = time.time()
            
            # トークン化
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            print(f"🔤 入力トークン数: {input_ids.shape[1]}")
            
            # NPU推論実行
            response = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # デコード
            generated_text = self.tokenizer.decode(response[0])
            
            # 統計情報
            total_tokens = response.shape[1]
            new_tokens = total_tokens - input_ids.shape[1]
            tokens_per_second = new_tokens / generation_time if generation_time > 0 else 0
            
            print(f"✅ RyzenAI NPU推論完了")
            print(f"  ⏱️ 生成時間: {generation_time:.2f}秒")
            print(f"  🔤 生成トークン数: {new_tokens}")
            print(f"  🚀 生成速度: {tokens_per_second:.1f}トークン/秒")
            
            return generated_text
            
        except Exception as e:
            print(f"❌ テキスト生成エラー: {e}")
            return prompt
    
    def generate_japanese(self, prompt: str, max_new_tokens: int = 100) -> str:
        """
        日本語テキスト生成（最適化設定）
        
        Args:
            prompt: 日本語プロンプト
            max_new_tokens: 最大生成トークン数
            
        Returns:
            str: 生成された日本語テキスト
        """
        print("🇯🇵 日本語生成モード")
        
        # 日本語生成に最適化されたパラメータ
        japanese_params = {
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.1,
            'do_sample': True,
        }
        
        return self.generate_text(prompt, max_new_tokens, **japanese_params)
    
    def interactive_mode(self):
        """インタラクティブモード"""
        if not self.is_initialized:
            print("❌ モデルが初期化されていません")
            return
        
        print("\n🇯🇵 AMD公式RyzenAI インタラクティブモード")
        print("=" * 60)
        print("⚡ NPU + CPU ハイブリッドモード")
        print(f"📦 モデル: {self.model_name}")
        print("\n💡 'exit'または'quit'で終了")
        print("=" * 60)
        
        while True:
            try:
                prompt = input("\n🤖 プロンプトを入力してください: ").strip()
                
                if prompt.lower() in ['exit', 'quit', '終了']:
                    print("👋 AMD公式RyzenAI LLMを終了します")
                    break
                
                if not prompt:
                    print("⚠️ プロンプトを入力してください")
                    continue
                
                print("\n🔄 生成中...")
                
                # 日本語判定（簡易）
                if any('\u3040' <= char <= '\u309F' or '\u30A0' <= char <= '\u30FF' or '\u4E00' <= char <= '\u9FAF' for char in prompt):
                    result = self.generate_japanese(prompt)
                else:
                    result = self.generate_text(prompt)
                
                print(f"\n📝 生成結果:")
                print("-" * 40)
                print(result)
                print("-" * 40)
                
            except KeyboardInterrupt:
                print("\n\n👋 AMD公式RyzenAI LLMを終了します")
                break
            except Exception as e:
                print(f"\n❌ エラーが発生しました: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """システム情報取得"""
        info = {
            'lemonade_available': LEMONADE_AVAILABLE,
            'model_initialized': self.is_initialized,
            'model_name': self.model_name,
            'recipe': 'oga-hybrid',
        }
        
        # NPU情報取得（可能な場合）
        try:
            # タスクマネージャーでNPU確認を促す
            info['npu_check'] = "タスクマネージャー -> パフォーマンス -> NPU0 で使用率を確認してください"
        except:
            pass
        
        return info

def check_environment():
    """環境確認"""
    print("🔍 RyzenAI環境確認")
    print("=" * 40)
    
    # Conda環境確認
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'None')
    print(f"📦 Conda環境: {conda_env}")
    
    # Python バージョン確認
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"🐍 Python バージョン: {python_version}")
    
    # Lemonade SDK確認
    print(f"🍋 Lemonade SDK: {'✅ 利用可能' if LEMONADE_AVAILABLE else '❌ 未インストール'}")
    
    # 環境変数確認
    ryzenai_path = os.environ.get('RYZEN_AI_INSTALLATION_PATH', 'Not set')
    print(f"📁 RYZEN_AI_INSTALLATION_PATH: {ryzenai_path}")
    
    print("=" * 40)
    
    if not LEMONADE_AVAILABLE:
        print("\n💡 インストール手順:")
        print("1. conda create -n ryzenai-llm python=3.10")
        print("2. conda activate ryzenai-llm")
        print("3. pip install lemonade-sdk[llm-oga-hybrid]")
        print("4. lemonade-install --ryzenai hybrid")

def run_validation_command():
    """公式検証コマンド実行"""
    if not LEMONADE_AVAILABLE:
        print("❌ Lemonade SDK が利用できません")
        return
    
    print("🧪 公式検証コマンド実行")
    
    # 公式検証コマンド
    cmd = [
        "lemonade",
        "-i", "amd/Llama-3.2-1B-Instruct-awq-g128-int4-asym-fp16-onnx-hybrid",
        "oga-load",
        "--device", "hybrid",
        "--dtype", "int4",
        "llm-prompt",
        "--max-new-tokens", "64",
        "-p", "Hello, how are you?"
    ]
    
    try:
        print(f"📝 実行コマンド: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ 検証成功")
            print("📝 出力:")
            print(result.stdout)
        else:
            print("❌ 検証失敗")
            print("📝 エラー:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ 検証タイムアウト（120秒）")
    except Exception as e:
        print(f"❌ 検証エラー: {e}")

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="AMD公式RyzenAI LLMデモ")
    parser.add_argument("--model", default="amd/Llama-3.2-1B-Instruct-awq-g128-int4-asym-fp16-onnx-hybrid", help="使用するモデル名")
    parser.add_argument("--interactive", action="store_true", help="インタラクティブモード")
    parser.add_argument("--prompt", type=str, help="単発プロンプト実行")
    parser.add_argument("--check-env", action="store_true", help="環境確認")
    parser.add_argument("--validate", action="store_true", help="公式検証コマンド実行")
    
    args = parser.parse_args()
    
    print("🎯 AMD公式RyzenAI LLMデモ")
    print("=" * 50)
    
    if args.check_env:
        check_environment()
        return
    
    if args.validate:
        run_validation_command()
        return
    
    if not LEMONADE_AVAILABLE:
        print("⚠️ Lemonade SDK が利用できません")
        check_environment()
        return
    
    # RyzenAI LLM初期化
    llm = RyzenAIOfficialLLM(model_name=args.model)
    
    if not llm.is_initialized:
        print("❌ モデル初期化に失敗しました")
        return
    
    # システム情報表示
    info = llm.get_system_info()
    print("\n📊 システム情報:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    if args.interactive:
        # インタラクティブモード
        llm.interactive_mode()
    elif args.prompt:
        # 単発プロンプト実行
        print(f"\n📝 プロンプト: {args.prompt}")
        print("🔄 生成中...")
        
        result = llm.generate_text(args.prompt)
        
        print(f"\n📝 生成結果:")
        print("-" * 40)
        print(result)
        print("-" * 40)
    else:
        # デフォルトテスト
        test_prompts = [
            "Hello, how are you?",
            "人参について教えてください。"
        ]
        
        for prompt in test_prompts:
            print(f"\n🧪 テスト実行: {prompt}")
            result = llm.generate_text(prompt, max_new_tokens=50)
            
            print(f"📝 生成結果:")
            print("-" * 40)
            print(result)
            print("-" * 40)

if __name__ == "__main__":
    main()

