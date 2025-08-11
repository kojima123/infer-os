#!/usr/bin/env python3
"""
統合NPU + Infer-OS最適化デモ実行スクリプト
包括的な最適化システムの統合デモ
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path


class IntegratedOptimizationDemo:
    """統合最適化デモクラス"""
    
    def __init__(self):
        self.available_models = {
            "llama3-8b-amd-npu": {
                "size": "8B",
                "type": "NPU最適化済み",
                "infer_os_compatible": True,
                "npu_ready": True,
                "recommended": True,
                "description": "NPU + Infer-OS統合最適化対応"
            },
            "ALMA-Ja-V3-amd-npu": {
                "size": "7B",
                "type": "翻訳特化NPU",
                "infer_os_compatible": True,
                "npu_ready": True,
                "recommended": True,
                "description": "翻訳特化 + Infer-OS統合最適化"
            },
            "cyberagent/Llama-3.1-70B-Japanese-Instruct-2407": {
                "size": "70B",
                "type": "大規模日本語",
                "infer_os_compatible": True,
                "npu_ready": False,
                "recommended": False,
                "description": "最重量級 + Infer-OS統合最適化"
            },
            "rinna/youri-7b-chat": {
                "size": "7B",
                "type": "日本語チャット",
                "infer_os_compatible": True,
                "npu_ready": False,
                "recommended": True,
                "description": "Infer-OS最適化対応（NPU変換可能）"
            }
        }
        
        self.optimization_modes = {
            "full": {
                "name": "完全統合最適化",
                "npu": True,
                "infer_os": True,
                "aggressive_memory": True,
                "advanced_quant": True,
                "windows_npu": True,
                "description": "全ての最適化機能を有効化"
            },
            "npu_only": {
                "name": "NPU最適化のみ",
                "npu": True,
                "infer_os": False,
                "aggressive_memory": False,
                "advanced_quant": False,
                "windows_npu": False,
                "description": "NPU最適化のみ有効"
            },
            "infer_os_only": {
                "name": "Infer-OS最適化のみ",
                "npu": False,
                "infer_os": True,
                "aggressive_memory": True,
                "advanced_quant": True,
                "windows_npu": True,
                "description": "Infer-OS最適化のみ有効"
            },
            "balanced": {
                "name": "バランス最適化",
                "npu": True,
                "infer_os": True,
                "aggressive_memory": True,
                "advanced_quant": False,
                "windows_npu": False,
                "description": "安定性重視の最適化"
            }
        }
        
        self.test_scenarios = {
            "basic": {
                "name": "基本性能テスト",
                "prompts": [
                    "人工知能について簡潔に説明してください。",
                    "日本の文化について教えてください。",
                    "量子コンピューターとは何ですか？"
                ],
                "max_tokens": 100
            },
            "advanced": {
                "name": "高度な生成テスト",
                "prompts": [
                    "人工知能の未来について詳しく論じてください。技術的な進歩、社会への影響、倫理的な課題について包括的に説明してください。",
                    "日本の四季の美しさについて、文学的な表現を用いて詩的に描写してください。",
                    "量子コンピューターの仕組みを、専門知識のない人にも分かりやすく、具体例を交えて説明してください。"
                ],
                "max_tokens": 300
            },
            "translation": {
                "name": "翻訳性能テスト",
                "prompts": [
                    "次の英語を自然な日本語に翻訳してください: 'The future of artificial intelligence is bright and full of possibilities.'",
                    "次の日本語を自然な英語に翻訳してください: '桜の花が咲く春は日本で最も美しい季節です。'",
                    "次の技術文書を日本語に翻訳してください: 'Machine learning algorithms are revolutionizing various industries by enabling automated decision-making processes.'"
                ],
                "max_tokens": 150
            }
        }
    
    def show_welcome(self):
        """ウェルカムメッセージ表示"""
        print("🚀 統合NPU + Infer-OS最適化デモ")
        print("🎯 真の包括的最適化システム体験")
        print("=" * 80)
        print("💡 このデモでは以下の統合最適化を体験できます:")
        print("  ⚡ NPU最適化 (VitisAI ExecutionProvider)")
        print("  🧠 Infer-OS最適化 (積極的メモリ、高度量子化)")
        print("  🪟 Windows NPU最適化 (AMD/Intel/Qualcomm)")
        print("  📊 包括的性能監視")
        print("  🎮 インタラクティブ対話")
        print("=" * 80)
    
    def show_model_selection(self):
        """モデル選択画面表示"""
        print("\n📱 統合最適化対応モデル:")
        print("-" * 60)
        
        for i, (model_key, info) in enumerate(self.available_models.items(), 1):
            status = "✅ 推奨" if info["recommended"] else "🔄 実験的"
            npu_status = "⚡ NPU対応" if info["npu_ready"] else "🔧 NPU変換可能"
            infer_os_status = "🧠 Infer-OS対応" if info["infer_os_compatible"] else "❌ 非対応"
            
            print(f"{i}. {model_key}")
            print(f"   📊 サイズ: {info['size']}")
            print(f"   🔧 タイプ: {info['type']}")
            print(f"   {npu_status} | {infer_os_status} | {status}")
            print(f"   📝 説明: {info['description']}")
            print()
    
    def show_optimization_modes(self):
        """最適化モード選択画面表示"""
        print("\n🔧 最適化モード:")
        print("-" * 50)
        
        for i, (mode_key, info) in enumerate(self.optimization_modes.items(), 1):
            print(f"{i}. {info['name']}")
            print(f"   📝 説明: {info['description']}")
            print(f"   ⚡ NPU: {'✅' if info['npu'] else '❌'}")
            print(f"   🧠 Infer-OS: {'✅' if info['infer_os'] else '❌'}")
            print(f"   💾 積極的メモリ: {'✅' if info['aggressive_memory'] else '❌'}")
            print(f"   📊 高度量子化: {'✅' if info['advanced_quant'] else '❌'}")
            print()
    
    def show_test_scenarios(self):
        """テストシナリオ選択画面表示"""
        print("\n🎯 テストシナリオ:")
        print("-" * 40)
        
        for i, (scenario_key, info) in enumerate(self.test_scenarios.items(), 1):
            print(f"{i}. {info['name']}")
            print(f"   📝 説明: プロンプト{len(info['prompts'])}個、最大{info['max_tokens']}トークン")
            print()
    
    def check_dependencies(self) -> bool:
        """依存関係チェック"""
        print("🔍 統合最適化システム依存関係チェック中...")
        
        required_files = [
            "integrated_npu_infer_os.py",
            "npu_optimized_japanese_models.py",
            "download_npu_models.py"
        ]
        
        missing_files = []
        for file_name in required_files:
            if not os.path.exists(file_name):
                missing_files.append(file_name)
        
        if missing_files:
            print("❌ 必要なファイルが見つかりません:")
            for file_name in missing_files:
                print(f"  - {file_name}")
            return False
        
        print("✅ 統合最適化システムファイル確認完了")
        return True
    
    def run_integrated_demo(self, model_name: str, optimization_mode: str, test_scenario: str):
        """統合デモ実行"""
        print(f"\n🚀 統合最適化デモ実行開始")
        print(f"📱 モデル: {model_name}")
        print(f"🔧 最適化モード: {optimization_mode}")
        print(f"🎯 テストシナリオ: {test_scenario}")
        print("-" * 60)
        
        # 最適化設定取得
        opt_config = self.optimization_modes[optimization_mode]
        scenario_config = self.test_scenarios[test_scenario]
        
        # コマンド構築
        cmd = ["python", "integrated_npu_infer_os.py", "--model", model_name]
        
        if not opt_config["npu"]:
            cmd.append("--disable-npu")
        if not opt_config["infer_os"]:
            cmd.append("--disable-infer-os")
        
        # 各プロンプトでテスト実行
        for i, prompt in enumerate(scenario_config["prompts"], 1):
            print(f"\n🤖 テスト {i}/{len(scenario_config['prompts'])}")
            print(f"📝 プロンプト: {prompt}")
            print("🔄 統合最適化生成中...")
            
            test_cmd = cmd + [
                "--prompt", prompt,
                "--max-tokens", str(scenario_config["max_tokens"])
            ]
            
            start_time = time.time()
            
            try:
                result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=300)
                execution_time = time.time() - start_time
                
                if result.returncode == 0:
                    print(f"✅ 生成完了 ({execution_time:.1f}秒)")
                    
                    # 応答抽出
                    output_lines = result.stdout.split('\n')
                    for line in output_lines:
                        if "📝 応答:" in line:
                            response = line.replace("📝 応答:", "").strip()
                            print(f"💬 応答: {response}")
                            break
                else:
                    print(f"❌ 実行エラー (終了コード: {result.returncode})")
                    print("📄 エラー出力:")
                    print(result.stderr[-300:])
                    
            except subprocess.TimeoutExpired:
                print("⏰ タイムアウト（5分）")
            except Exception as e:
                print(f"❌ 予期しないエラー: {e}")
            
            print("-" * 40)
    
    def run_performance_comparison(self):
        """性能比較実行"""
        print("\n🏁 統合最適化性能比較")
        print("=" * 70)
        
        # 比較対象モデル
        comparison_models = ["llama3-8b-amd-npu", "rinna/youri-7b-chat"]
        comparison_modes = ["full", "npu_only", "infer_os_only"]
        
        test_prompt = "人工知能の未来について簡潔に説明してください。"
        
        results = {}
        
        for model_name in comparison_models:
            if model_name not in self.available_models:
                continue
            
            results[model_name] = {}
            
            for mode in comparison_modes:
                print(f"\n📊 {model_name} - {self.optimization_modes[mode]['name']}")
                
                opt_config = self.optimization_modes[mode]
                cmd = ["python", "integrated_npu_infer_os.py", "--model", model_name, "--prompt", test_prompt, "--max-tokens", "100"]
                
                if not opt_config["npu"]:
                    cmd.append("--disable-npu")
                if not opt_config["infer_os"]:
                    cmd.append("--disable-infer-os")
                
                start_time = time.time()
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                    execution_time = time.time() - start_time
                    
                    if result.returncode == 0:
                        results[model_name][mode] = {
                            "time": execution_time,
                            "success": True
                        }
                        print(f"✅ 完了: {execution_time:.1f}秒")
                    else:
                        results[model_name][mode] = {
                            "time": execution_time,
                            "success": False
                        }
                        print(f"❌ 失敗: {execution_time:.1f}秒")
                        
                except subprocess.TimeoutExpired:
                    results[model_name][mode] = {
                        "time": 180,
                        "success": False
                    }
                    print("⏰ タイムアウト")
                except Exception as e:
                    print(f"❌ エラー: {e}")
        
        # 結果表示
        print("\n📊 性能比較結果")
        print("=" * 70)
        
        for model_name, model_results in results.items():
            print(f"\n📱 {model_name}")
            for mode, result in model_results.items():
                status = "✅ 成功" if result["success"] else "❌ 失敗"
                mode_name = self.optimization_modes[mode]["name"]
                print(f"   🔧 {mode_name}: {result['time']:.1f}秒 {status}")
    
    def run_interactive_mode(self, model_name: str, optimization_mode: str):
        """インタラクティブモード実行"""
        print(f"\n🎮 統合最適化インタラクティブモード")
        print(f"📱 モデル: {model_name}")
        print(f"🔧 最適化: {self.optimization_modes[optimization_mode]['name']}")
        print("💡 'exit'で終了")
        print("-" * 60)
        
        opt_config = self.optimization_modes[optimization_mode]
        cmd = ["python", "integrated_npu_infer_os.py", "--model", model_name, "--interactive"]
        
        if not opt_config["npu"]:
            cmd.append("--disable-npu")
        if not opt_config["infer_os"]:
            cmd.append("--disable-infer-os")
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ インタラクティブモードエラー: {e}")
        except KeyboardInterrupt:
            print("\n👋 インタラクティブモードを終了しました")
        except Exception as e:
            print(f"❌ 予期しないエラー: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="統合NPU + Infer-OS最適化デモ")
    parser.add_argument("--model", help="使用するモデル名")
    parser.add_argument("--optimization-mode", default="full", 
                       choices=["full", "npu_only", "infer_os_only", "balanced"],
                       help="最適化モード")
    parser.add_argument("--test-scenario", default="basic",
                       choices=["basic", "advanced", "translation"],
                       help="テストシナリオ")
    parser.add_argument("--interactive", action="store_true", help="インタラクティブモード")
    parser.add_argument("--compare", action="store_true", help="性能比較実行")
    parser.add_argument("--check-deps", action="store_true", help="依存関係チェック")
    
    args = parser.parse_args()
    
    demo = IntegratedOptimizationDemo()
    demo.show_welcome()
    
    if args.check_deps:
        if demo.check_dependencies():
            print("✅ 全ての依存関係が満たされています")
        else:
            print("❌ 依存関係に問題があります")
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
        
        if args.interactive:
            demo.run_interactive_mode(args.model, args.optimization_mode)
        else:
            demo.run_integrated_demo(args.model, args.optimization_mode, args.test_scenario)
    else:
        # インタラクティブな選択
        if not demo.check_dependencies():
            print("❌ 依存関係を先に解決してください")
            return
        
        demo.show_model_selection()
        
        try:
            # モデル選択
            choice = input("\n📱 使用するモデルを選択してください (1-4): ").strip()
            model_list = list(demo.available_models.keys())
            
            if not (choice.isdigit() and 1 <= int(choice) <= len(model_list)):
                print("❌ 無効な選択です")
                return
            
            selected_model = model_list[int(choice) - 1]
            
            # 最適化モード選択
            demo.show_optimization_modes()
            mode_choice = input("\n🔧 最適化モードを選択してください (1-4): ").strip()
            mode_list = list(demo.optimization_modes.keys())
            
            if not (mode_choice.isdigit() and 1 <= int(mode_choice) <= len(mode_list)):
                print("❌ 無効な選択です")
                return
            
            selected_mode = mode_list[int(mode_choice) - 1]
            
            # 実行タイプ選択
            print("\n🎯 実行タイプを選択してください:")
            print("1. 基本テスト")
            print("2. 高度テスト")
            print("3. 翻訳テスト")
            print("4. インタラクティブモード")
            print("5. 性能比較")
            
            exec_choice = input("選択 (1-5): ").strip()
            
            if exec_choice == "1":
                demo.run_integrated_demo(selected_model, selected_mode, "basic")
            elif exec_choice == "2":
                demo.run_integrated_demo(selected_model, selected_mode, "advanced")
            elif exec_choice == "3":
                demo.run_integrated_demo(selected_model, selected_mode, "translation")
            elif exec_choice == "4":
                demo.run_interactive_mode(selected_model, selected_mode)
            elif exec_choice == "5":
                demo.run_performance_comparison()
            else:
                print("❌ 無効な選択です")
                
        except KeyboardInterrupt:
            print("\n👋 デモを終了しました")
        except Exception as e:
            print(f"❌ 予期しないエラー: {e}")


if __name__ == "__main__":
    main()

