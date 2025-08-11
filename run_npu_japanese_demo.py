#!/usr/bin/env python3
"""
NPU最適化日本語モデル統合デモ
真のNPU活用を実現する包括的なデモスクリプト
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path


class NPUJapaneseDemoRunner:
    """NPU日本語デモ実行クラス"""
    
    def __init__(self):
        self.available_models = {
            "llama3-8b-amd-npu": {
                "size": "8B",
                "type": "NPU最適化済み",
                "japanese": "多言語対応",
                "npu_ready": True,
                "recommended": True,
                "description": "最も安定したNPU対応日本語モデル"
            },
            "ALMA-Ja-V3-amd-npu": {
                "size": "7B", 
                "type": "翻訳特化NPU",
                "japanese": "日本語翻訳特化",
                "npu_ready": True,
                "recommended": True,
                "description": "日本語翻訳に最適化されたNPUモデル"
            },
            "cyberagent/Llama-3.1-70B-Japanese-Instruct-2407": {
                "size": "70B",
                "type": "大規模日本語",
                "japanese": "日本語特化",
                "npu_ready": False,
                "recommended": False,
                "description": "最重量級日本語モデル（ONNX変換チャレンジ）"
            }
        }
        
        self.test_prompts = {
            "general": [
                "人工知能の未来について教えてください。",
                "日本の四季の美しさについて説明してください。",
                "量子コンピューターの仕組みを簡単に説明してください。"
            ],
            "translation": [
                "次の英語を日本語に翻訳してください: 'The future of artificial intelligence is bright and full of possibilities.'",
                "次の日本語を英語に翻訳してください: '桜の花が咲く春は日本で最も美しい季節です。'",
                "次の文章を自然な日本語に翻訳してください: 'Machine learning algorithms are revolutionizing various industries.'"
            ],
            "technical": [
                "NPU（Neural Processing Unit）とGPUの違いについて詳しく説明してください。",
                "深層学習における量子化の重要性について教えてください。",
                "エッジAIの利点と課題について論じてください。"
            ]
        }
    
    def show_welcome(self):
        """ウェルカムメッセージ表示"""
        print("🚀 NPU最適化日本語モデル統合デモ")
        print("🎯 真のNPU活用実現版")
        print("=" * 70)
        print("💡 このデモでは以下を実現します:")
        print("  ✅ NPU最適化済み日本語モデルの動作確認")
        print("  ✅ 真のNPU処理によるハードウェア負荷率向上")
        print("  ✅ 高品質な日本語テキスト生成")
        print("  ✅ 複数モデルの性能比較")
        print("=" * 70)
    
    def show_model_selection(self):
        """モデル選択画面表示"""
        print("\n📱 利用可能なモデル:")
        print("-" * 50)
        
        for i, (model_key, info) in enumerate(self.available_models.items(), 1):
            status = "✅ 推奨" if info["recommended"] else "🔄 実験的"
            npu_status = "⚡ NPU対応" if info["npu_ready"] else "🔧 ONNX変換必要"
            
            print(f"{i}. {model_key}")
            print(f"   📊 サイズ: {info['size']}")
            print(f"   🔧 タイプ: {info['type']}")
            print(f"   🇯🇵 日本語: {info['japanese']}")
            print(f"   {npu_status} | {status}")
            print(f"   📝 説明: {info['description']}")
            print()
    
    def check_model_availability(self, model_name: str) -> bool:
        """モデルの利用可能性チェック"""
        print(f"🔍 {model_name} の利用可能性チェック中...")
        
        # ローカルディレクトリ確認
        if os.path.exists(model_name):
            print(f"✅ ローカルモデル発見: {model_name}")
            
            # NPU最適化ファイル確認
            npu_files = [
                "pytorch_llama3_8b_w_bit_4_awq_amd.pt",
                "alma_w_bit_4_awq_fa_amd.pt"
            ]
            
            for npu_file in npu_files:
                npu_path = Path(model_name) / npu_file
                if npu_path.exists():
                    print(f"⚡ NPU最適化ファイル確認: {npu_file}")
                    return True
            
            # 通常のモデルファイル確認
            config_path = Path(model_name) / "config.json"
            if config_path.exists():
                print(f"📋 設定ファイル確認: config.json")
                return True
        
        print(f"❌ {model_name} が見つかりません")
        return False
    
    def download_model_if_needed(self, model_name: str) -> bool:
        """必要に応じてモデルダウンロード"""
        if self.check_model_availability(model_name):
            return True
        
        print(f"📥 {model_name} をダウンロードしますか？")
        response = input("y/n: ").lower().strip()
        
        if response in ['y', 'yes', 'はい']:
            print(f"📥 {model_name} ダウンロード開始...")
            
            try:
                cmd = ["python", "download_npu_models.py", "--download", model_name]
                result = subprocess.run(cmd, check=True, text=True)
                
                print(f"✅ {model_name} ダウンロード完了")
                return True
                
            except subprocess.CalledProcessError as e:
                print(f"❌ ダウンロードエラー: {e}")
                return False
            except Exception as e:
                print(f"❌ 予期しないエラー: {e}")
                return False
        else:
            print("❌ ダウンロードをスキップしました")
            return False
    
    def run_model_demo(self, model_name: str, prompt_type: str = "general"):
        """モデルデモ実行"""
        print(f"\n🚀 {model_name} デモ実行開始")
        print(f"📝 プロンプトタイプ: {prompt_type}")
        print("-" * 50)
        
        # プロンプト選択
        prompts = self.test_prompts.get(prompt_type, self.test_prompts["general"])
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n🤖 テスト {i}/{len(prompts)}")
            print(f"📝 プロンプト: {prompt}")
            print("🔄 生成中...")
            
            # モデル実行
            start_time = time.time()
            
            try:
                cmd = [
                    "python", "npu_optimized_japanese_models.py",
                    "--model", model_name,
                    "--prompt", prompt,
                    "--max-tokens", "150"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                execution_time = time.time() - start_time
                
                if result.returncode == 0:
                    print(f"✅ 生成完了 ({execution_time:.1f}秒)")
                    
                    # 出力から応答部分を抽出
                    output_lines = result.stdout.split('\n')
                    response_found = False
                    
                    for line in output_lines:
                        if "📝 応答:" in line:
                            response = line.replace("📝 応答:", "").strip()
                            print(f"💬 応答: {response}")
                            response_found = True
                            break
                    
                    if not response_found:
                        print("⚠️ 応答の抽出に失敗しました")
                        print("📄 生出力:")
                        print(result.stdout[-500:])  # 最後の500文字
                else:
                    print(f"❌ 実行エラー (終了コード: {result.returncode})")
                    print("📄 エラー出力:")
                    print(result.stderr[-500:])
                    
            except subprocess.TimeoutExpired:
                print("⏰ タイムアウト（5分）")
            except Exception as e:
                print(f"❌ 予期しないエラー: {e}")
            
            print("-" * 30)
    
    def run_performance_comparison(self):
        """性能比較実行"""
        print("\n🏁 NPU最適化モデル性能比較")
        print("=" * 60)
        
        # NPU対応モデルのみ比較
        npu_models = [key for key, info in self.available_models.items() if info["npu_ready"]]
        
        test_prompt = "人工知能の未来について簡潔に説明してください。"
        
        results = {}
        
        for model_name in npu_models:
            print(f"\n📊 {model_name} 性能測定中...")
            
            if not self.check_model_availability(model_name):
                print(f"❌ {model_name} が利用できません")
                continue
            
            start_time = time.time()
            
            try:
                cmd = [
                    "python", "npu_optimized_japanese_models.py",
                    "--model", model_name,
                    "--prompt", test_prompt,
                    "--max-tokens", "100"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                
                execution_time = time.time() - start_time
                
                if result.returncode == 0:
                    results[model_name] = {
                        "time": execution_time,
                        "success": True,
                        "size": self.available_models[model_name]["size"]
                    }
                    print(f"✅ 完了: {execution_time:.1f}秒")
                else:
                    results[model_name] = {
                        "time": execution_time,
                        "success": False,
                        "size": self.available_models[model_name]["size"]
                    }
                    print(f"❌ 失敗: {execution_time:.1f}秒")
                    
            except subprocess.TimeoutExpired:
                results[model_name] = {
                    "time": 180,
                    "success": False,
                    "size": self.available_models[model_name]["size"]
                }
                print("⏰ タイムアウト")
            except Exception as e:
                print(f"❌ エラー: {e}")
        
        # 結果表示
        print("\n📊 性能比較結果")
        print("=" * 60)
        
        for model_name, result in results.items():
            status = "✅ 成功" if result["success"] else "❌ 失敗"
            print(f"📱 {model_name}")
            print(f"   📊 サイズ: {result['size']}")
            print(f"   ⏱️ 実行時間: {result['time']:.1f}秒")
            print(f"   🎯 結果: {status}")
            print()
    
    def run_interactive_mode(self, model_name: str):
        """インタラクティブモード実行"""
        print(f"\n🎮 {model_name} インタラクティブモード")
        print("💡 'exit'で終了")
        print("-" * 50)
        
        try:
            cmd = [
                "python", "npu_optimized_japanese_models.py",
                "--model", model_name,
                "--interactive"
            ]
            
            subprocess.run(cmd, check=True)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ インタラクティブモードエラー: {e}")
        except KeyboardInterrupt:
            print("\n👋 インタラクティブモードを終了しました")
        except Exception as e:
            print(f"❌ 予期しないエラー: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="NPU最適化日本語モデル統合デモ")
    parser.add_argument("--model", help="使用するモデル名")
    parser.add_argument("--prompt-type", default="general", 
                       choices=["general", "translation", "technical"],
                       help="テストプロンプトタイプ")
    parser.add_argument("--interactive", action="store_true", help="インタラクティブモード")
    parser.add_argument("--compare", action="store_true", help="性能比較実行")
    parser.add_argument("--download-all", action="store_true", help="NPU対応モデル一括ダウンロード")
    
    args = parser.parse_args()
    
    demo = NPUJapaneseDemoRunner()
    demo.show_welcome()
    
    if args.download_all:
        print("\n📥 NPU対応モデル一括ダウンロード開始...")
        try:
            cmd = ["python", "download_npu_models.py", "--download-all-npu"]
            subprocess.run(cmd, check=True)
            print("✅ 一括ダウンロード完了")
        except Exception as e:
            print(f"❌ 一括ダウンロードエラー: {e}")
        return
    
    if args.compare:
        demo.run_performance_comparison()
        return
    
    if args.model:
        # 指定モデルでデモ実行
        if args.model not in demo.available_models:
            print(f"❌ 未知のモデル: {args.model}")
            demo.show_model_selection()
            return
        
        if not demo.download_model_if_needed(args.model):
            print("❌ モデルの準備に失敗しました")
            return
        
        if args.interactive:
            demo.run_interactive_mode(args.model)
        else:
            demo.run_model_demo(args.model, args.prompt_type)
    else:
        # インタラクティブなモデル選択
        demo.show_model_selection()
        
        try:
            choice = input("\n📱 使用するモデルを選択してください (1-3): ").strip()
            
            model_list = list(demo.available_models.keys())
            
            if choice.isdigit() and 1 <= int(choice) <= len(model_list):
                selected_model = model_list[int(choice) - 1]
                
                if not demo.download_model_if_needed(selected_model):
                    print("❌ モデルの準備に失敗しました")
                    return
                
                # デモタイプ選択
                print("\n🎯 実行タイプを選択してください:")
                print("1. 基本デモ（一般的な質問）")
                print("2. 翻訳デモ（翻訳タスク）")
                print("3. 技術デモ（技術的な質問）")
                print("4. インタラクティブモード")
                print("5. 性能比較")
                
                demo_choice = input("選択 (1-5): ").strip()
                
                if demo_choice == "1":
                    demo.run_model_demo(selected_model, "general")
                elif demo_choice == "2":
                    demo.run_model_demo(selected_model, "translation")
                elif demo_choice == "3":
                    demo.run_model_demo(selected_model, "technical")
                elif demo_choice == "4":
                    demo.run_interactive_mode(selected_model)
                elif demo_choice == "5":
                    demo.run_performance_comparison()
                else:
                    print("❌ 無効な選択です")
            else:
                print("❌ 無効な選択です")
                
        except KeyboardInterrupt:
            print("\n👋 デモを終了しました")
        except Exception as e:
            print(f"❌ 予期しないエラー: {e}")


if __name__ == "__main__":
    main()

