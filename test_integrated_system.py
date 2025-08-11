#!/usr/bin/env python3
"""
統合NPU + Infer-OS最適化システム検証テスト
包括的な性能測定と機能検証
"""

import os
import sys
import time
import json
import traceback
import subprocess
from typing import Dict, List, Any, Optional
from pathlib import Path


class IntegratedSystemValidator:
    """統合最適化システム検証クラス"""
    
    def __init__(self):
        self.test_results = {
            "system_check": {},
            "dependency_check": {},
            "model_tests": {},
            "optimization_tests": {},
            "performance_benchmarks": {},
            "integration_tests": {}
        }
        
        self.test_models = [
            "llama3-8b-amd-npu",
            "ALMA-Ja-V3-amd-npu",
            "rinna/youri-7b-chat"
        ]
        
        self.optimization_modes = [
            "full",
            "npu_only", 
            "infer_os_only",
            "balanced"
        ]
        
        self.test_prompts = [
            "人工知能について説明してください。",
            "日本の文化について教えてください。",
            "量子コンピューターとは何ですか？"
        ]
    
    def run_full_validation(self) -> Dict[str, Any]:
        """完全検証実行"""
        print("🚀 統合NPU + Infer-OS最適化システム完全検証開始")
        print("=" * 80)
        
        # Phase 1: システムチェック
        print("\n📋 Phase 1: システム環境チェック")
        self._check_system_environment()
        
        # Phase 2: 依存関係チェック
        print("\n📦 Phase 2: 依存関係チェック")
        self._check_dependencies()
        
        # Phase 3: モデルテスト
        print("\n🤖 Phase 3: モデル機能テスト")
        self._test_models()
        
        # Phase 4: 最適化テスト
        print("\n⚡ Phase 4: 最適化機能テスト")
        self._test_optimizations()
        
        # Phase 5: 性能ベンチマーク
        print("\n📊 Phase 5: 性能ベンチマーク")
        self._run_performance_benchmarks()
        
        # Phase 6: 統合テスト
        print("\n🔗 Phase 6: 統合機能テスト")
        self._test_integration()
        
        # 結果サマリー
        print("\n📋 検証結果サマリー")
        self._display_validation_summary()
        
        return self.test_results
    
    def _check_system_environment(self):
        """システム環境チェック"""
        print("🔍 システム環境確認中...")
        
        # Python環境
        python_version = sys.version
        self.test_results["system_check"]["python_version"] = python_version
        print(f"🐍 Python: {python_version.split()[0]}")
        
        # OS情報
        import platform
        os_info = platform.platform()
        self.test_results["system_check"]["os_info"] = os_info
        print(f"💻 OS: {os_info}")
        
        # メモリ情報
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            memory_gb = memory_info.total / (1024**3)
            self.test_results["system_check"]["memory_gb"] = memory_gb
            print(f"💾 メモリ: {memory_gb:.1f}GB")
        except ImportError:
            print("⚠️ psutilが利用できません")
        
        # GPU/NPU情報
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                self.test_results["system_check"]["gpu_available"] = True
                self.test_results["system_check"]["gpu_count"] = gpu_count
                self.test_results["system_check"]["gpu_name"] = gpu_name
                print(f"🎮 GPU: {gpu_name} ({gpu_count}個)")
            else:
                self.test_results["system_check"]["gpu_available"] = False
                print("🎮 GPU: 利用不可")
        except ImportError:
            print("⚠️ PyTorchが利用できません")
        
        # NPU環境変数
        npu_env_vars = [
            "RYZEN_AI_INSTALLATION_PATH",
            "XLNX_VART_FIRMWARE", 
            "XLNX_TARGET_NAME"
        ]
        
        npu_env_status = {}
        for var in npu_env_vars:
            value = os.environ.get(var)
            npu_env_status[var] = value is not None
            status = "✅" if value else "❌"
            print(f"🔧 {var}: {status}")
        
        self.test_results["system_check"]["npu_env_vars"] = npu_env_status
    
    def _check_dependencies(self):
        """依存関係チェック"""
        print("📦 依存関係確認中...")
        
        # 必須ライブラリ
        required_libraries = [
            "torch",
            "transformers", 
            "onnx",
            "onnxruntime",
            "psutil"
        ]
        
        library_status = {}
        for lib in required_libraries:
            try:
                __import__(lib)
                library_status[lib] = True
                print(f"✅ {lib}: 利用可能")
            except ImportError:
                library_status[lib] = False
                print(f"❌ {lib}: 利用不可")
        
        self.test_results["dependency_check"]["libraries"] = library_status
        
        # ONNX Runtime プロバイダー
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            self.test_results["dependency_check"]["onnx_providers"] = providers
            
            print("🔧 ONNX Runtime プロバイダー:")
            for provider in providers:
                print(f"  - {provider}")
                
            # VitisAI ExecutionProvider確認
            vitisai_available = 'VitisAIExecutionProvider' in providers
            self.test_results["dependency_check"]["vitisai_available"] = vitisai_available
            status = "✅" if vitisai_available else "❌"
            print(f"⚡ VitisAI ExecutionProvider: {status}")
            
        except ImportError:
            print("❌ ONNX Runtimeが利用できません")
        
        # 統合システムファイル
        required_files = [
            "integrated_npu_infer_os.py",
            "run_integrated_demo.py",
            "npu_optimized_japanese_models.py"
        ]
        
        file_status = {}
        for file_name in required_files:
            exists = os.path.exists(file_name)
            file_status[file_name] = exists
            status = "✅" if exists else "❌"
            print(f"📄 {file_name}: {status}")
        
        self.test_results["dependency_check"]["required_files"] = file_status
    
    def _test_models(self):
        """モデル機能テスト"""
        print("🤖 モデル機能テスト実行中...")
        
        for model_name in self.test_models:
            print(f"\n📱 テスト対象: {model_name}")
            
            model_test_result = {
                "load_test": False,
                "generation_test": False,
                "error_messages": []
            }
            
            # モデルロードテスト
            try:
                print("🔄 モデルロードテスト...")
                cmd = [
                    "python", "integrated_npu_infer_os.py",
                    "--model", model_name,
                    "--prompt", "テスト",
                    "--max-tokens", "10",
                    "--disable-npu",
                    "--disable-infer-os"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    model_test_result["load_test"] = True
                    print("✅ モデルロード成功")
                    
                    # 生成テスト
                    if "📝 応答:" in result.stdout:
                        model_test_result["generation_test"] = True
                        print("✅ テキスト生成成功")
                    else:
                        print("⚠️ テキスト生成応答なし")
                else:
                    error_msg = result.stderr[-200:] if result.stderr else "Unknown error"
                    model_test_result["error_messages"].append(error_msg)
                    print(f"❌ モデルロード失敗: {error_msg}")
                    
            except subprocess.TimeoutExpired:
                model_test_result["error_messages"].append("Timeout")
                print("⏰ モデルロードタイムアウト")
            except Exception as e:
                model_test_result["error_messages"].append(str(e))
                print(f"❌ モデルテストエラー: {e}")
            
            self.test_results["model_tests"][model_name] = model_test_result
    
    def _test_optimizations(self):
        """最適化機能テスト"""
        print("⚡ 最適化機能テスト実行中...")
        
        test_model = "rinna/youri-7b-chat"  # 安定したテスト用モデル
        test_prompt = "人工知能について簡潔に説明してください。"
        
        for mode in self.optimization_modes:
            print(f"\n🔧 最適化モード: {mode}")
            
            optimization_test_result = {
                "execution_success": False,
                "execution_time": 0,
                "optimization_applied": False,
                "error_messages": []
            }
            
            try:
                cmd = [
                    "python", "integrated_npu_infer_os.py",
                    "--model", test_model,
                    "--prompt", test_prompt,
                    "--max-tokens", "50"
                ]
                
                # 最適化モード設定
                if mode == "npu_only":
                    cmd.append("--disable-infer-os")
                elif mode == "infer_os_only":
                    cmd.append("--disable-npu")
                elif mode == "balanced":
                    cmd.extend(["--quantization-profile", "balanced"])
                
                start_time = time.time()
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                execution_time = time.time() - start_time
                
                optimization_test_result["execution_time"] = execution_time
                
                if result.returncode == 0:
                    optimization_test_result["execution_success"] = True
                    print(f"✅ 実行成功 ({execution_time:.1f}秒)")
                    
                    # 最適化適用確認
                    if "最適化" in result.stdout or "NPU" in result.stdout:
                        optimization_test_result["optimization_applied"] = True
                        print("✅ 最適化適用確認")
                    else:
                        print("⚠️ 最適化適用未確認")
                else:
                    error_msg = result.stderr[-200:] if result.stderr else "Unknown error"
                    optimization_test_result["error_messages"].append(error_msg)
                    print(f"❌ 実行失敗: {error_msg}")
                    
            except subprocess.TimeoutExpired:
                optimization_test_result["error_messages"].append("Timeout")
                print("⏰ 最適化テストタイムアウト")
            except Exception as e:
                optimization_test_result["error_messages"].append(str(e))
                print(f"❌ 最適化テストエラー: {e}")
            
            self.test_results["optimization_tests"][mode] = optimization_test_result
    
    def _run_performance_benchmarks(self):
        """性能ベンチマーク実行"""
        print("📊 性能ベンチマーク実行中...")
        
        benchmark_model = "rinna/youri-7b-chat"
        benchmark_prompts = self.test_prompts
        
        benchmark_results = {}
        
        for mode in ["infer_os_only", "full"]:  # 安定したモードでベンチマーク
            print(f"\n⚡ ベンチマークモード: {mode}")
            
            mode_results = {
                "total_time": 0,
                "average_time": 0,
                "successful_runs": 0,
                "failed_runs": 0,
                "prompt_results": []
            }
            
            total_time = 0
            successful_runs = 0
            
            for i, prompt in enumerate(benchmark_prompts, 1):
                print(f"🔄 ベンチマーク {i}/{len(benchmark_prompts)}")
                
                try:
                    cmd = [
                        "python", "integrated_npu_infer_os.py",
                        "--model", benchmark_model,
                        "--prompt", prompt,
                        "--max-tokens", "100"
                    ]
                    
                    if mode == "infer_os_only":
                        cmd.append("--disable-npu")
                    
                    start_time = time.time()
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                    execution_time = time.time() - start_time
                    
                    prompt_result = {
                        "prompt": prompt[:50] + "...",
                        "execution_time": execution_time,
                        "success": result.returncode == 0
                    }
                    
                    if result.returncode == 0:
                        successful_runs += 1
                        total_time += execution_time
                        print(f"✅ 成功 ({execution_time:.1f}秒)")
                    else:
                        print(f"❌ 失敗 ({execution_time:.1f}秒)")
                    
                    mode_results["prompt_results"].append(prompt_result)
                    
                except subprocess.TimeoutExpired:
                    print("⏰ タイムアウト")
                    mode_results["prompt_results"].append({
                        "prompt": prompt[:50] + "...",
                        "execution_time": 120,
                        "success": False
                    })
                except Exception as e:
                    print(f"❌ エラー: {e}")
                    mode_results["prompt_results"].append({
                        "prompt": prompt[:50] + "...",
                        "execution_time": 0,
                        "success": False
                    })
            
            mode_results["total_time"] = total_time
            mode_results["successful_runs"] = successful_runs
            mode_results["failed_runs"] = len(benchmark_prompts) - successful_runs
            mode_results["average_time"] = total_time / successful_runs if successful_runs > 0 else 0
            
            benchmark_results[mode] = mode_results
            
            print(f"📊 {mode} 結果:")
            print(f"   成功: {successful_runs}/{len(benchmark_prompts)}")
            print(f"   平均時間: {mode_results['average_time']:.1f}秒")
        
        self.test_results["performance_benchmarks"] = benchmark_results
    
    def _test_integration(self):
        """統合機能テスト"""
        print("🔗 統合機能テスト実行中...")
        
        integration_tests = {
            "demo_script_test": False,
            "interactive_mode_test": False,
            "comparison_test": False,
            "dependency_integration": False
        }
        
        # デモスクリプトテスト
        try:
            print("🎮 デモスクリプトテスト...")
            result = subprocess.run(
                ["python", "run_integrated_demo.py", "--check-deps"],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0 and "依存関係" in result.stdout:
                integration_tests["demo_script_test"] = True
                print("✅ デモスクリプト動作確認")
            else:
                print("❌ デモスクリプト動作失敗")
                
        except Exception as e:
            print(f"❌ デモスクリプトテストエラー: {e}")
        
        # 依存関係統合テスト
        try:
            print("📦 依存関係統合テスト...")
            
            # 統合システムインポートテスト
            sys.path.append(os.getcwd())
            
            try:
                from integrated_npu_infer_os import IntegratedNPUInferOS
                integration_tests["dependency_integration"] = True
                print("✅ 統合システムインポート成功")
            except ImportError as e:
                print(f"❌ 統合システムインポート失敗: {e}")
                
        except Exception as e:
            print(f"❌ 依存関係統合テストエラー: {e}")
        
        self.test_results["integration_tests"] = integration_tests
    
    def _display_validation_summary(self):
        """検証結果サマリー表示"""
        print("=" * 80)
        print("📋 統合NPU + Infer-OS最適化システム検証結果サマリー")
        print("=" * 80)
        
        # システム環境
        print("\n💻 システム環境:")
        system_check = self.test_results["system_check"]
        print(f"   Python: {system_check.get('python_version', 'Unknown').split()[0]}")
        print(f"   OS: {system_check.get('os_info', 'Unknown')}")
        print(f"   メモリ: {system_check.get('memory_gb', 0):.1f}GB")
        print(f"   GPU: {'✅' if system_check.get('gpu_available', False) else '❌'}")
        
        # 依存関係
        print("\n📦 依存関係:")
        dep_check = self.test_results["dependency_check"]
        libraries = dep_check.get("libraries", {})
        for lib, status in libraries.items():
            print(f"   {lib}: {'✅' if status else '❌'}")
        
        vitisai_status = dep_check.get("vitisai_available", False)
        print(f"   VitisAI EP: {'✅' if vitisai_status else '❌'}")
        
        # モデルテスト
        print("\n🤖 モデルテスト:")
        model_tests = self.test_results["model_tests"]
        for model, result in model_tests.items():
            load_status = "✅" if result["load_test"] else "❌"
            gen_status = "✅" if result["generation_test"] else "❌"
            print(f"   {model}: ロード{load_status} 生成{gen_status}")
        
        # 最適化テスト
        print("\n⚡ 最適化テスト:")
        opt_tests = self.test_results["optimization_tests"]
        for mode, result in opt_tests.items():
            success_status = "✅" if result["execution_success"] else "❌"
            time_info = f"({result['execution_time']:.1f}秒)" if result["execution_time"] > 0 else ""
            print(f"   {mode}: {success_status} {time_info}")
        
        # 性能ベンチマーク
        print("\n📊 性能ベンチマーク:")
        benchmarks = self.test_results["performance_benchmarks"]
        for mode, result in benchmarks.items():
            success_rate = f"{result['successful_runs']}/{result['successful_runs'] + result['failed_runs']}"
            avg_time = f"{result['average_time']:.1f}秒" if result['average_time'] > 0 else "N/A"
            print(f"   {mode}: 成功率{success_rate} 平均{avg_time}")
        
        # 統合テスト
        print("\n🔗 統合テスト:")
        integration = self.test_results["integration_tests"]
        for test, status in integration.items():
            print(f"   {test}: {'✅' if status else '❌'}")
        
        # 総合評価
        print("\n🏆 総合評価:")
        
        # 成功率計算
        total_tests = 0
        passed_tests = 0
        
        # 依存関係成功率
        lib_total = len(libraries)
        lib_passed = sum(libraries.values())
        total_tests += lib_total
        passed_tests += lib_passed
        
        # モデルテスト成功率
        model_total = len(model_tests) * 2  # ロード + 生成
        model_passed = sum(result["load_test"] + result["generation_test"] for result in model_tests.values())
        total_tests += model_total
        passed_tests += model_passed
        
        # 最適化テスト成功率
        opt_total = len(opt_tests)
        opt_passed = sum(result["execution_success"] for result in opt_tests.values())
        total_tests += opt_total
        passed_tests += opt_passed
        
        # 統合テスト成功率
        int_total = len(integration)
        int_passed = sum(integration.values())
        total_tests += int_total
        passed_tests += int_passed
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"   総合成功率: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            print("   評価: 🎉 優秀 - 統合システムは正常に動作しています")
        elif success_rate >= 60:
            print("   評価: ✅ 良好 - 一部の機能に問題がありますが使用可能です")
        elif success_rate >= 40:
            print("   評価: ⚠️ 注意 - 複数の問題があります。修正が必要です")
        else:
            print("   評価: ❌ 不良 - 重大な問題があります。システムの見直しが必要です")
        
        print("=" * 80)
    
    def save_results(self, filename: str = "validation_results.json"):
        """検証結果保存"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, ensure_ascii=False, indent=2)
            print(f"📄 検証結果を保存しました: {filename}")
        except Exception as e:
            print(f"❌ 結果保存エラー: {e}")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="統合NPU + Infer-OS最適化システム検証")
    parser.add_argument("--save-results", action="store_true", help="結果をJSONファイルに保存")
    parser.add_argument("--output-file", default="validation_results.json", help="出力ファイル名")
    
    args = parser.parse_args()
    
    validator = IntegratedSystemValidator()
    
    try:
        results = validator.run_full_validation()
        
        if args.save_results:
            validator.save_results(args.output_file)
        
        print("\n🏁 統合システム検証完了")
        
    except KeyboardInterrupt:
        print("\n👋 検証を中断しました")
    except Exception as e:
        print(f"\n❌ 検証エラー: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

