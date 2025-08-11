"""
NPU環境チェッカー
真のNPU動作に必要な環境要件を詳細にチェック

使用方法:
    python npu_environment_checker.py
"""

import sys
import os
import subprocess
import platform
import traceback
from typing import Dict, List, Tuple, Any

class NPUEnvironmentChecker:
    """NPU環境チェッカー"""
    
    def __init__(self):
        self.results = {}
        self.warnings = []
        self.errors = []
        
        print("🔍 NPU環境チェッカー開始")
        print("🎯 真のNPU動作に必要な環境要件を詳細チェック")
        print("=" * 60)
    
    def check_all(self) -> Dict[str, Any]:
        """全項目チェック"""
        try:
            # システム情報チェック
            self.check_system_info()
            
            # ハードウェアチェック
            self.check_hardware()
            
            # ソフトウェアチェック
            self.check_software()
            
            # ONNXRuntimeチェック
            self.check_onnxruntime()
            
            # NPUドライバーチェック
            self.check_npu_drivers()
            
            # Ryzen AI SDKチェック
            self.check_ryzen_ai_sdk()
            
            # 総合評価
            self.evaluate_npu_readiness()
            
            return self.results
            
        except Exception as e:
            print(f"❌ 環境チェックエラー: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    def check_system_info(self):
        """システム情報チェック"""
        print("\n🖥️ システム情報チェック:")
        print("-" * 40)
        
        try:
            # OS情報
            os_info = platform.platform()
            print(f"📋 OS: {os_info}")
            
            # Windows 11チェック
            if "Windows" in os_info:
                if "Windows-11" in os_info or "Windows-10" in os_info:
                    windows_version = self._get_windows_version()
                    print(f"🪟 Windowsバージョン: {windows_version}")
                    
                    if "Windows-11" in os_info:
                        self.results["windows_11"] = True
                        print("✅ Windows 11検出")
                    else:
                        self.results["windows_11"] = False
                        self.warnings.append("Windows 11推奨（現在: Windows 10）")
                        print("⚠️ Windows 10検出（Windows 11推奨）")
                else:
                    self.results["windows_11"] = False
                    self.errors.append("Windows 11が必要")
                    print("❌ 未対応OS（Windows 11が必要）")
            else:
                self.results["windows_11"] = False
                self.errors.append("Windows OSが必要")
                print("❌ 非Windows OS（Windows 11が必要）")
            
            # アーキテクチャ
            arch = platform.architecture()
            print(f"🏗️ アーキテクチャ: {arch}")
            
            # Python情報
            python_version = sys.version
            print(f"🐍 Python: {python_version}")
            
            # Python 3.8-3.11チェック
            python_major = sys.version_info.major
            python_minor = sys.version_info.minor
            
            if python_major == 3 and 8 <= python_minor <= 11:
                self.results["python_compatible"] = True
                print("✅ Python互換バージョン")
            else:
                self.results["python_compatible"] = False
                self.errors.append(f"Python 3.8-3.11が必要（現在: {python_major}.{python_minor}）")
                print(f"❌ Python非互換バージョン（3.8-3.11が必要）")
            
        except Exception as e:
            print(f"❌ システム情報チェックエラー: {e}")
            self.errors.append(f"システム情報取得エラー: {e}")
    
    def _get_windows_version(self) -> str:
        """Windowsバージョン詳細取得"""
        try:
            result = subprocess.run(
                ['ver'], 
                capture_output=True, 
                text=True, 
                shell=True
            )
            return result.stdout.strip()
        except:
            return "不明"
    
    def check_hardware(self):
        """ハードウェアチェック"""
        print("\n🔧 ハードウェアチェック:")
        print("-" * 40)
        
        try:
            # CPU情報
            cpu_info = self._get_cpu_info()
            print(f"🖥️ CPU: {cpu_info}")
            
            # AMD Ryzen AIチェック
            if "AMD" in cpu_info and "Ryzen" in cpu_info:
                if any(series in cpu_info for series in ["7040", "8040", "8045", "AI"]):
                    self.results["ryzen_ai_cpu"] = True
                    print("✅ AMD Ryzen AI CPU検出")
                else:
                    self.results["ryzen_ai_cpu"] = False
                    self.warnings.append("Ryzen AI CPU未確認（7040/8040シリーズ推奨）")
                    print("⚠️ Ryzen AI CPU未確認")
            else:
                self.results["ryzen_ai_cpu"] = False
                self.errors.append("AMD Ryzen AI CPUが必要")
                print("❌ 非AMD CPU（Ryzen AI CPU必要）")
            
            # メモリ情報
            memory_info = self._get_memory_info()
            print(f"💾 メモリ: {memory_info}")
            
            # 16GB以上チェック
            memory_gb = self._parse_memory_size(memory_info)
            if memory_gb >= 16:
                self.results["sufficient_memory"] = True
                print("✅ 十分なメモリ容量")
            else:
                self.results["sufficient_memory"] = False
                self.warnings.append(f"メモリ不足（現在: {memory_gb}GB、推奨: 16GB以上）")
                print(f"⚠️ メモリ不足（{memory_gb}GB < 16GB）")
            
        except Exception as e:
            print(f"❌ ハードウェアチェックエラー: {e}")
            self.errors.append(f"ハードウェア情報取得エラー: {e}")
    
    def _get_cpu_info(self) -> str:
        """CPU情報取得"""
        try:
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'name'], 
                capture_output=True, 
                text=True
            )
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip() and 'Name' not in line:
                    return line.strip()
            return "不明"
        except:
            return platform.processor()
    
    def _get_memory_info(self) -> str:
        """メモリ情報取得"""
        try:
            result = subprocess.run(
                ['wmic', 'computersystem', 'get', 'TotalPhysicalMemory'], 
                capture_output=True, 
                text=True
            )
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip() and 'TotalPhysicalMemory' not in line:
                    bytes_memory = int(line.strip())
                    gb_memory = bytes_memory / (1024**3)
                    return f"{gb_memory:.1f}GB"
            return "不明"
        except:
            return "不明"
    
    def _parse_memory_size(self, memory_info: str) -> float:
        """メモリサイズ解析"""
        try:
            if "GB" in memory_info:
                return float(memory_info.replace("GB", ""))
            return 0.0
        except:
            return 0.0
    
    def check_software(self):
        """ソフトウェアチェック"""
        print("\n📦 ソフトウェアチェック:")
        print("-" * 40)
        
        # 必要パッケージリスト
        required_packages = [
            "torch",
            "transformers", 
            "onnx",
            "onnxruntime",
            "numpy"
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                version = self._get_package_version(package)
                print(f"✅ {package}: {version}")
                self.results[f"{package}_installed"] = True
            except ImportError:
                print(f"❌ {package}: 未インストール")
                self.results[f"{package}_installed"] = False
                self.errors.append(f"{package}が未インストール")
    
    def _get_package_version(self, package_name: str) -> str:
        """パッケージバージョン取得"""
        try:
            module = __import__(package_name)
            return getattr(module, '__version__', '不明')
        except:
            return "不明"
    
    def check_onnxruntime(self):
        """ONNXRuntimeチェック"""
        print("\n🔍 ONNXRuntimeチェック:")
        print("-" * 40)
        
        try:
            import onnxruntime as ort
            
            # バージョン
            version = ort.__version__
            print(f"📦 ONNXRuntimeバージョン: {version}")
            
            # プロバイダー
            providers = ort.get_available_providers()
            print(f"📋 利用可能プロバイダー: {len(providers)}個")
            
            # 重要プロバイダーチェック
            important_providers = {
                'VitisAIExecutionProvider': 'NPU専用（最重要）',
                'DmlExecutionProvider': 'DirectML（GPU）',
                'CUDAExecutionProvider': 'NVIDIA GPU',
                'CPUExecutionProvider': 'CPU（フォールバック）'
            }
            
            for provider, description in important_providers.items():
                if provider in providers:
                    print(f"  ✅ {provider}: {description}")
                    self.results[f"{provider.lower()}_available"] = True
                    
                    if provider == 'VitisAIExecutionProvider':
                        print("    🎯 真のNPU処理が可能！")
                else:
                    print(f"  ❌ {provider}: {description}")
                    self.results[f"{provider.lower()}_available"] = False
                    
                    if provider == 'VitisAIExecutionProvider':
                        self.errors.append("VitisAI ExecutionProvider未インストール（NPU処理に必須）")
            
        except ImportError:
            print("❌ ONNXRuntime未インストール")
            self.errors.append("ONNXRuntime未インストール")
        except Exception as e:
            print(f"❌ ONNXRuntimeチェックエラー: {e}")
            self.errors.append(f"ONNXRuntimeエラー: {e}")
    
    def check_npu_drivers(self):
        """NPUドライバーチェック"""
        print("\n🚗 NPUドライバーチェック:")
        print("-" * 40)
        
        try:
            # デバイスマネージャー情報取得
            result = subprocess.run(
                ['wmic', 'path', 'win32_pnpentity', 'get', 'name'], 
                capture_output=True, 
                text=True
            )
            
            devices = result.stdout.lower()
            
            # NPU関連デバイス検索
            npu_keywords = ['npu', 'neural processing', 'ai accelerator', 'ryzen ai']
            npu_found = False
            
            for keyword in npu_keywords:
                if keyword in devices:
                    npu_found = True
                    print(f"✅ NPU関連デバイス検出: {keyword}")
                    break
            
            if npu_found:
                self.results["npu_driver_installed"] = True
                print("✅ NPUドライバーが存在する可能性があります")
            else:
                self.results["npu_driver_installed"] = False
                self.warnings.append("NPU関連デバイスが見つかりません")
                print("⚠️ NPU関連デバイス未検出")
            
            # AMD関連ドライバー
            if 'amd' in devices:
                print("✅ AMDドライバー検出")
                self.results["amd_driver_installed"] = True
            else:
                print("⚠️ AMDドライバー未検出")
                self.results["amd_driver_installed"] = False
                self.warnings.append("AMDドライバー未検出")
            
        except Exception as e:
            print(f"❌ ドライバーチェックエラー: {e}")
            self.errors.append(f"ドライバーチェックエラー: {e}")
    
    def check_ryzen_ai_sdk(self):
        """Ryzen AI SDKチェック"""
        print("\n🛠️ Ryzen AI SDKチェック:")
        print("-" * 40)
        
        # 一般的なインストールパス
        sdk_paths = [
            "C:\\AMD\\RyzenAI",
            "C:\\Program Files\\AMD\\RyzenAI",
            "C:\\Program Files (x86)\\AMD\\RyzenAI"
        ]
        
        sdk_found = False
        for path in sdk_paths:
            if os.path.exists(path):
                print(f"✅ Ryzen AI SDK検出: {path}")
                self.results["ryzen_ai_sdk_installed"] = True
                sdk_found = True
                
                # バージョン確認
                version_file = os.path.join(path, "version.txt")
                if os.path.exists(version_file):
                    try:
                        with open(version_file, 'r') as f:
                            version = f.read().strip()
                        print(f"📦 SDKバージョン: {version}")
                    except:
                        print("📦 SDKバージョン: 不明")
                break
        
        if not sdk_found:
            print("❌ Ryzen AI SDK未検出")
            self.results["ryzen_ai_sdk_installed"] = False
            self.errors.append("Ryzen AI SDK未インストール")
        
        # 環境変数チェック
        ryzen_ai_path = os.environ.get('RYZEN_AI_INSTALLATION_PATH')
        if ryzen_ai_path:
            print(f"✅ RYZEN_AI_INSTALLATION_PATH: {ryzen_ai_path}")
            self.results["ryzen_ai_env_var"] = True
        else:
            print("⚠️ RYZEN_AI_INSTALLATION_PATH未設定")
            self.results["ryzen_ai_env_var"] = False
            self.warnings.append("RYZEN_AI_INSTALLATION_PATH環境変数未設定")
    
    def evaluate_npu_readiness(self):
        """NPU準備状況評価"""
        print("\n📊 NPU準備状況評価:")
        print("=" * 60)
        
        # 必須要件
        critical_requirements = [
            ("ryzen_ai_cpu", "AMD Ryzen AI CPU"),
            ("vitisaiexecutionprovider_available", "VitisAI ExecutionProvider"),
            ("ryzen_ai_sdk_installed", "Ryzen AI SDK")
        ]
        
        # 推奨要件
        recommended_requirements = [
            ("windows_11", "Windows 11"),
            ("sufficient_memory", "16GB以上メモリ"),
            ("npu_driver_installed", "NPUドライバー")
        ]
        
        # 必須要件チェック
        critical_passed = 0
        print("🔴 必須要件:")
        for key, name in critical_requirements:
            status = self.results.get(key, False)
            if status:
                print(f"  ✅ {name}")
                critical_passed += 1
            else:
                print(f"  ❌ {name}")
        
        # 推奨要件チェック
        recommended_passed = 0
        print("\n🟡 推奨要件:")
        for key, name in recommended_requirements:
            status = self.results.get(key, False)
            if status:
                print(f"  ✅ {name}")
                recommended_passed += 1
            else:
                print(f"  ⚠️ {name}")
        
        # 総合評価
        critical_total = len(critical_requirements)
        recommended_total = len(recommended_requirements)
        
        print(f"\n📈 評価結果:")
        print(f"  🔴 必須要件: {critical_passed}/{critical_total}")
        print(f"  🟡 推奨要件: {recommended_passed}/{recommended_total}")
        
        if critical_passed == critical_total:
            if recommended_passed == recommended_total:
                print("  🎯 評価: ✅ NPU完全対応")
                self.results["npu_readiness"] = "完全対応"
            else:
                print("  🎯 評価: ✅ NPU基本対応")
                self.results["npu_readiness"] = "基本対応"
        else:
            print("  🎯 評価: ❌ NPU未対応")
            self.results["npu_readiness"] = "未対応"
        
        # 警告・エラー表示
        if self.warnings:
            print(f"\n⚠️ 警告 ({len(self.warnings)}件):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if self.errors:
            print(f"\n❌ エラー ({len(self.errors)}件):")
            for error in self.errors:
                print(f"  - {error}")
        
        # 推奨アクション
        self._show_recommended_actions()
    
    def _show_recommended_actions(self):
        """推奨アクション表示"""
        print(f"\n💡 推奨アクション:")
        
        if not self.results.get("vitisaiexecutionprovider_available", False):
            print("  1. VitisAI ExecutionProvider インストール:")
            print("     pip install onnxruntime-vitisai")
        
        if not self.results.get("ryzen_ai_sdk_installed", False):
            print("  2. Ryzen AI SDK インストール:")
            print("     https://www.amd.com/en/products/software/ryzen-ai.html")
        
        if not self.results.get("ryzen_ai_cpu", False):
            print("  3. AMD Ryzen AI CPU搭載システムが必要")
            print("     （7040/8040シリーズ以降）")
        
        if not self.results.get("npu_driver_installed", False):
            print("  4. NPUドライバー更新:")
            print("     AMD公式サイトから最新ドライバーダウンロード")
        
        print(f"\n📖 詳細ガイド: TRUE_NPU_SETUP_GUIDE.md を参照")

def main():
    """メイン関数"""
    checker = NPUEnvironmentChecker()
    results = checker.check_all()
    
    print(f"\n🏁 NPU環境チェック完了")
    print(f"📊 総合評価: {results.get('npu_readiness', '不明')}")

if __name__ == "__main__":
    main()

