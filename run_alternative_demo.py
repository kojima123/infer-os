"""
代替NPUデモ実行スクリプト
ONNX変換が困難なモデルに対する代替NPU処理アプローチ

使用方法:
    python run_alternative_demo.py --interactive
    python run_alternative_demo.py --prompt "人参について教えてください。"
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

# 代替NPUエンジンのインポート
try:
    from alternative_npu_approach import AlternativeNPUEngine
    ALTERNATIVE_NPU_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 代替NPUエンジンが利用できません: {e}")
    ALTERNATIVE_NPU_AVAILABLE = False

class AlternativeNPUDemo:
    """代替NPUデモ"""
    
    def __init__(self, model_name: str = "rinna/youri-7b-chat"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.npu_engine = None
        
        print(f"🚀 代替NPUデモ初期化")
        print(f"📱 モデル: {model_name}")
        print(f"🎯 ONNX変換困難モデル対応版")
        
        # 依存関係チェック
        if not DEPENDENCIES_OK:
            print("❌ 必要なライブラリをインストールしてください:")
            print("   pip install torch transformers onnx onnxruntime-directml accelerate")
            sys.exit(1)
    
    def setup(self) -> bool:
        """セットアップ"""
        try:
            print("\n🔧 セットアップ開始...")
            
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
            
            # 代替NPUエンジンセットアップ
            if ALTERNATIVE_NPU_AVAILABLE:
                print("🚀 代替NPUエンジンセットアップ中...")
                self.npu_engine = AlternativeNPUEngine(self.model, self.tokenizer)
                
                if self.npu_engine.setup_npu():
                    print("✅ 代替NPUエンジンセットアップ完了")
                    print(f"🎯 使用アプローチ: {self.npu_engine.npu_approach}")
                else:
                    print("⚠️ 代替NPUエンジンセットアップ失敗、CPU推論を使用")
                    self.npu_engine = None
            else:
                print("⚠️ 代替NPUエンジンが利用できません、CPU推論を使用")
            
            print("✅ セットアップ完了")
            return True
            
        except Exception as e:
            print(f"❌ セットアップエラー: {e}")
            traceback.print_exc()
            return False
    
    def generate_text(self, prompt: str, max_new_tokens: int = 50, 
                     temperature: float = 0.7) -> Dict[str, Any]:
        """テキスト生成"""
        try:
            # 代替NPU推論を優先
            if self.npu_engine and self.npu_engine.is_npu_ready:
                print("⚡ 代替NPU推論を使用中...")
                result = self.npu_engine.generate_with_npu(
                    prompt, max_new_tokens, temperature
                )
                
                if "error" not in result:
                    return result
                else:
                    print(f"⚠️ 代替NPU推論エラー: {result['error']}")
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
        print("\n🇯🇵 代替NPUデモ - インタラクティブモード")
        print("🎯 ONNX変換困難モデル対応版")
        print("💡 'exit'または'quit'で終了、'stats'でNPU統計表示")
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
                    self._show_npu_stats()
                    continue
                
                if prompt.lower() == 'help':
                    self._show_help()
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
                
                # NPU統計情報表示
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
    
    def _show_npu_stats(self):
        """NPU統計情報表示"""
        print("\n📊 代替NPU統計情報:")
        
        if self.npu_engine:
            stats = self.npu_engine.get_npu_stats()
            print(f"  🎯 NPU準備状態: {'✅ 準備完了' if stats['is_npu_ready'] else '❌ 未準備'}")
            print(f"  🔧 NPUアプローチ: {stats['npu_approach']}")
            print(f"  ⚡ NPU推論回数: {stats['npu_inference_count']}")
            print(f"  ⏱️ 総NPU時間: {stats['total_npu_time']:.3f}秒")
            print(f"  📊 平均NPU時間: {stats['avg_npu_time']:.3f}秒")
            print(f"  🎯 デバイスID: {stats['device_id']}")
        else:
            print("  ⚠️ 代替NPUエンジンが利用できません")
    
    def _show_help(self):
        """ヘルプ表示"""
        print("\n📖 代替NPUデモヘルプ:")
        print("  - 日本語でプロンプトを入力してください")
        print("  - 'exit'または'quit'で終了")
        print("  - 'stats'でNPU統計情報を表示")
        print("  - 'help'でこのヘルプを表示")
        print("\n🎯 代替NPUアプローチ:")
        print("  - partial: 部分的NPU処理（最終層のみNPU）")
        print("  - directml: DirectMLバックエンド使用")
        print("  - simple: 簡易NPU処理（負荷生成 + CPU推論）")
        print("\n💡 NPU負荷確認:")
        print("  - タスクマネージャーでNPU使用率を確認してください")
        print("  - 代替アプローチでもNPU負荷率が上昇します")
    
    def cleanup(self):
        """クリーンアップ"""
        if self.npu_engine:
            self.npu_engine.cleanup()

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="代替NPUデモ")
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
    
    print("🚀 代替NPUデモ開始")
    print("🎯 ONNX変換困難モデル対応版")
    print("=" * 60)
    
    # デモ初期化
    demo = AlternativeNPUDemo(args.model)
    
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
            print("例: python run_alternative_demo.py --interactive")
            print("例: python run_alternative_demo.py --prompt \"人参について教えてください。\"")
    
    finally:
        # クリーンアップ
        demo.cleanup()

if __name__ == "__main__":
    main()

