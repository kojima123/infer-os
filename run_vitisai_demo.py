"""
VitisAI NPUデモ実行スクリプト
真のNPU処理を実現するVitisAI ExecutionProvider専用デモ

使用方法:
    python run_vitisai_demo.py --interactive
    python run_vitisai_demo.py --prompt "人参について教えてください。"
"""

import sys
import os
import argparse
import time
import traceback
from typing import Dict, Any, Optional

# 必要なライブラリのインポート確認
try:
    import torch
    import transformers
    import onnx
    import onnxruntime as ort
    import numpy as np
    DEPENDENCIES_OK = True
except ImportError as e:
    print(f"❌ 必要なライブラリが不足しています: {e}")
    DEPENDENCIES_OK = False

# VitisAI NPUエンジンのインポート
try:
    from vitisai_npu_engine import VitisAINPUEngine
    VITISAI_NPU_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ VitisAI NPUエンジンが利用できません: {e}")
    VITISAI_NPU_AVAILABLE = False

class VitisAINPUDemo:
    """VitisAI NPUデモ"""
    
    def __init__(self, model_name: str = "rinna/youri-7b-chat"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.vitisai_engine = None
        
        print(f"🚀 VitisAI NPUデモ初期化")
        print(f"📱 モデル: {model_name}")
        print(f"🎯 真のNPU処理実現版")
        
        # 依存関係チェック
        if not DEPENDENCIES_OK:
            print("❌ 必要なライブラリをインストールしてください:")
            print("   pip install torch transformers onnx onnxruntime-vitisai")
            sys.exit(1)
    
    def setup(self) -> bool:
        """セットアップ"""
        try:
            print("\n🔧 セットアップ開始...")
            
            # 環境確認
            self._check_environment()
            
            # トークナイザーロード
            print("📝 トークナイザーをロード中...")
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✅ トークナイザーロード完了")
            
            # モデルロード
            print("🤖 モデルをロード中...")
            from transformers import AutoModelForCausalLM
            
            load_start = time.time()
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            load_time = time.time() - load_start
            
            print(f"✅ モデルロード完了 ({load_time:.1f}秒)")
            
            # VitisAI NPUエンジンセットアップ
            if VITISAI_NPU_AVAILABLE:
                print("🚀 VitisAI NPUエンジンセットアップ中...")
                self.vitisai_engine = VitisAINPUEngine(self.model, self.tokenizer)
                
                if self.vitisai_engine.setup_vitisai_npu():
                    print("✅ VitisAI NPUエンジンセットアップ完了")
                    print("🎉 真のNPU処理が利用可能です！")
                else:
                    print("⚠️ VitisAI NPUエンジンセットアップ失敗、CPU推論を使用")
                    self.vitisai_engine = None
            else:
                print("⚠️ VitisAI NPUエンジンが利用できません、CPU推論を使用")
            
            print("✅ セットアップ完了")
            return True
            
        except Exception as e:
            print(f"❌ セットアップエラー: {e}")
            traceback.print_exc()
            return False
    
    def _check_environment(self):
        """環境確認"""
        print("🔍 VitisAI環境確認中...")
        
        # VitisAI ExecutionProvider確認
        available_providers = ort.get_available_providers()
        if 'VitisAIExecutionProvider' in available_providers:
            print("✅ VitisAI ExecutionProvider利用可能")
        else:
            print("❌ VitisAI ExecutionProvider利用不可")
            print("💡 onnxruntime-vitisaiをインストールしてください")
        
        # 環境変数確認
        ryzen_ai_path = os.environ.get('RYZEN_AI_INSTALLATION_PATH')
        if ryzen_ai_path:
            print(f"✅ RYZEN_AI_INSTALLATION_PATH: {ryzen_ai_path}")
        else:
            print("⚠️ RYZEN_AI_INSTALLATION_PATH未設定")
        
        xlnx_target = os.environ.get('XLNX_TARGET_NAME')
        if xlnx_target:
            print(f"✅ XLNX_TARGET_NAME: {xlnx_target}")
        else:
            print("⚠️ XLNX_TARGET_NAME未設定")
    
    def generate_text(self, prompt: str, max_new_tokens: int = 50, 
                     temperature: float = 0.7) -> Dict[str, Any]:
        """テキスト生成"""
        try:
            # VitisAI NPU推論を優先
            if self.vitisai_engine and self.vitisai_engine.is_vitisai_ready:
                print("⚡ VitisAI NPU推論を使用中...")
                result = self.vitisai_engine.generate_with_vitisai_npu(
                    prompt, max_new_tokens, temperature
                )
                
                if "error" not in result:
                    return result
                else:
                    print(f"⚠️ VitisAI NPU推論エラー: {result['error']}")
                    print("🔄 CPU推論にフォールバック")
            
            # CPU推論（フォールバック）
            print("🖥️ CPU推論を使用中...")
            return self._generate_with_cpu(prompt, max_new_tokens, temperature)
            
        except Exception as e:
            print(f"❌ テキスト生成エラー: {e}")
            traceback.print_exc()
            return {"error": f"生成エラー: {e}"}
    
    def _generate_with_cpu(self, prompt: str, max_new_tokens: int, 
                          temperature: float) -> Dict[str, Any]:
        """CPU推論"""
        try:
            start_time = time.time()
            
            # 入力トークン化
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # デバイス移動
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 生成設定
            from transformers import GenerationConfig
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
            
            # テキスト生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            # デコード
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            generation_time = time.time() - start_time
            input_tokens = len(inputs['input_ids'][0])
            output_tokens = len(outputs[0]) - input_tokens
            tokens_per_sec = output_tokens / generation_time if generation_time > 0 else 0
            
            return {
                "generated_text": generated_text,
                "generation_time": generation_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tokens_per_sec": tokens_per_sec,
                "inference_method": "CPU"
            }
            
        except Exception as e:
            print(f"❌ CPU推論エラー: {e}")
            traceback.print_exc()
            return {"error": f"CPU推論エラー: {e}"}
    
    def run_interactive(self):
        """インタラクティブモード"""
        print("\n🇯🇵 VitisAI NPUデモ - インタラクティブモード")
        print("🎯 真のNPU処理実現版")
        print("💡 'exit'または'quit'で終了、'stats'でVitisAI統計表示")
        print("=" * 60)
        
        while True:
            try:
                prompt = input("\n🤖 プロンプトを入力してください: ").strip()
                
                if not prompt:
                    print("⚠️ プロンプトを入力してください")
                    continue
                
                if prompt.lower() in ['exit', 'quit', '終了']:
                    print("👋 インタラクティブモードを終了します")
                    break
                
                if prompt.lower() == 'stats':
                    self._show_vitisai_stats()
                    continue
                
                if prompt.lower() == 'help':
                    self._show_help()
                    continue
                
                if prompt.lower() == 'env':
                    self._check_environment()
                    continue
                
                # テキスト生成実行
                print("\n🔄 生成中...")
                result = self.generate_text(prompt)
                
                if "error" in result:
                    print(f"❌ エラー: {result['error']}")
                    continue
                
                # 結果表示
                print(f"\n✅ 生成完了:")
                print(f"📝 生成テキスト: {result['generated_text']}")
                print(f"⏱️ 生成時間: {result['generation_time']:.2f}秒")
                print(f"📊 入力トークン数: {result['input_tokens']}")
                print(f"📊 出力トークン数: {result['output_tokens']}")
                print(f"🚀 生成速度: {result['tokens_per_sec']:.1f} トークン/秒")
                print(f"🔧 推論方法: {result['inference_method']}")
                
                # VitisAI NPU統計情報表示
                if 'npu_provider' in result:
                    print(f"🎯 NPUプロバイダー: {result['npu_provider']}")
                if 'npu_inference_count' in result:
                    print(f"⚡ NPU推論回数: {result['npu_inference_count']}")
                    if 'avg_npu_time' in result:
                        print(f"⚡ 平均NPU時間: {result['avg_npu_time']:.3f}秒")
                
            except KeyboardInterrupt:
                print("\n👋 インタラクティブモードを終了します")
                break
            except Exception as e:
                print(f"❌ 予期しないエラー: {e}")
                traceback.print_exc()
    
    def _show_vitisai_stats(self):
        """VitisAI統計情報表示"""
        print("\n📊 VitisAI NPU統計情報:")
        
        if self.vitisai_engine:
            stats = self.vitisai_engine.get_vitisai_stats()
            print(f"  🎯 VitisAI準備状態: {'✅ 準備完了' if stats['is_vitisai_ready'] else '❌ 未準備'}")
            print(f"  ⚡ NPU推論回数: {stats['npu_inference_count']}")
            print(f"  ⏱️ 総NPU時間: {stats['total_npu_time']:.3f}秒")
            print(f"  📊 平均NPU時間: {stats['avg_npu_time']:.3f}秒")
            print(f"  📁 設定ファイル: {stats['config_file']}")
            print(f"  🎯 NPUオーバレイ: {stats['npu_overlay']}")
            print(f"  📂 Ryzen AIパス: {stats['ryzen_ai_path']}")
        else:
            print("  ⚠️ VitisAI NPUエンジンが利用できません")
    
    def _show_help(self):
        """ヘルプ表示"""
        print("\n📖 VitisAI NPUデモヘルプ:")
        print("  - 日本語でプロンプトを入力してください")
        print("  - 'exit'または'quit'で終了")
        print("  - 'stats'でVitisAI統計情報を表示")
        print("  - 'env'で環境情報を表示")
        print("  - 'help'でこのヘルプを表示")
        print("\n🎯 VitisAI NPUの特徴:")
        print("  - 真のNPU処理（Neural Processing Unit）")
        print("  - VitisAI ExecutionProvider使用")
        print("  - INT8量子化最適化")
        print("  - 高速・低電力処理")
        print("\n📊 NPU動作確認:")
        print("  - タスクマネージャーでNPU使用率を確認")
        print("  - 推論方法が'VitisAI NPU'と表示")
        print("  - quicktestと同様のVitisAI EPログ出力")
    
    def cleanup(self):
        """クリーンアップ"""
        if self.vitisai_engine:
            self.vitisai_engine.cleanup()

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="VitisAI NPUデモ")
    parser.add_argument("--model", type=str, default="rinna/youri-7b-chat",
                       help="使用するモデル名")
    parser.add_argument("--interactive", action="store_true",
                       help="インタラクティブモードで実行")
    parser.add_argument("--prompt", type=str,
                       help="単発実行用プロンプト")
    parser.add_argument("--max-tokens", type=int, default=50,
                       help="最大生成トークン数")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="生成温度")
    
    args = parser.parse_args()
    
    print("🚀 VitisAI NPUデモ開始")
    print("🎯 真のNPU処理実現版")
    print("=" * 60)
    
    # デモ初期化
    demo = VitisAINPUDemo(args.model)
    
    try:
        # セットアップ
        if not demo.setup():
            print("❌ セットアップに失敗しました")
            sys.exit(1)
        
        # 実行モード選択
        if args.interactive:
            demo.run_interactive()
        elif args.prompt:
            print(f"\n🔄 単発実行: {args.prompt}")
            result = demo.generate_text(args.prompt, args.max_tokens, args.temperature)
            
            if "error" in result:
                print(f"❌ エラー: {result['error']}")
            else:
                print(f"✅ 生成結果: {result['generated_text']}")
                print(f"⏱️ 生成時間: {result['generation_time']:.2f}秒")
                print(f"🚀 生成速度: {result['tokens_per_sec']:.1f} トークン/秒")
                print(f"🔧 推論方法: {result['inference_method']}")
        else:
            print("💡 --interactive または --prompt を指定してください")
            print("例: python run_vitisai_demo.py --interactive")
            print("例: python run_vitisai_demo.py --prompt \"人参について教えてください。\"")
    
    finally:
        # クリーンアップ
        demo.cleanup()

if __name__ == "__main__":
    main()

