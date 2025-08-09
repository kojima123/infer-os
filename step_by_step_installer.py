#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 段階的インストール自動化スクリプト

setuptools/pipエラーを回避してInfer-OS NPU環境を構築

機能:
- setuptools/pipエラー自動修復
- 段階的パッケージインストール
- バイナリ版優先インストール
- 包括的動作確認テスト

使用方法:
    python step_by_step_installer.py
"""

import sys
import subprocess
import time
import traceback
from datetime import datetime
import json

class StepByStepInstaller:
    """段階的インストールツール"""
    
    def __init__(self):
        self.log_messages = []
        self.installed_packages = {}
        self.failed_packages = []
        
    def log(self, message: str, level: str = "INFO"):
        """ログ出力"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        colors = {
            "INFO": "\033[37m",     # 白
            "SUCCESS": "\033[92m",  # 緑
            "WARNING": "\033[93m",  # 黄
            "ERROR": "\033[91m",    # 赤
            "RESET": "\033[0m"      # リセット
        }
        
        color = colors.get(level, colors["INFO"])
        reset = colors["RESET"]
        
        log_entry = f"[{timestamp}] {message}"
        self.log_messages.append(log_entry)
        
        print(f"{color}{log_entry}{reset}")
    
    def check_python_environment(self):
        """Python環境確認"""
        self.log("Python環境を確認中...")
        
        version_info = sys.version_info
        python_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
        
        self.log(f"Python バージョン: {python_version}")
        self.log(f"Python 実行ファイル: {sys.executable}")
        
        # 仮想環境確認
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            self.log("✅ 仮想環境で実行中", "SUCCESS")
        else:
            self.log("⚠️ システムPython環境で実行中", "WARNING")
        
        return True
    
    def upgrade_basic_tools(self):
        """基本ツールアップグレード"""
        self.log("基本ツール（pip, setuptools, wheel）をアップグレード中...")
        
        basic_tools = [
            ("pip", "pip"),
            ("setuptools", "setuptools"),
            ("wheel", "wheel")
        ]
        
        success_count = 0
        
        for tool_name, package_name in basic_tools:
            try:
                self.log(f"  {tool_name} をアップグレード中...")
                
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--upgrade", package_name],
                    capture_output=True, text=True, timeout=180
                )
                
                if result.returncode == 0:
                    self.log(f"  ✅ {tool_name} アップグレード成功", "SUCCESS")
                    success_count += 1
                else:
                    self.log(f"  ❌ {tool_name} アップグレードエラー: {result.stderr}", "ERROR")
                    
            except subprocess.TimeoutExpired:
                self.log(f"  ❌ {tool_name} アップグレードタイムアウト", "ERROR")
            except Exception as e:
                self.log(f"  ❌ {tool_name} アップグレードエラー: {e}", "ERROR")
        
        self.log(f"基本ツールアップグレード: {success_count}/{len(basic_tools)} 成功")
        return success_count >= 2  # pip, setuptoolsが成功すればOK
    
    def install_package_safe(self, package_spec: str, binary_only: bool = True, timeout: int = 300):
        """安全なパッケージインストール"""
        try:
            self.log(f"  {package_spec} をインストール中...")
            
            cmd = [sys.executable, "-m", "pip", "install"]
            
            if binary_only:
                cmd.extend(["--only-binary=:all:", package_spec])
            else:
                cmd.append(package_spec)
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            
            if result.returncode == 0:
                self.log(f"  ✅ {package_spec} インストール成功", "SUCCESS")
                self.installed_packages[package_spec] = "success"
                return True
            else:
                self.log(f"  ❌ {package_spec} インストールエラー: {result.stderr}", "ERROR")
                self.failed_packages.append(package_spec)
                return False
                
        except subprocess.TimeoutExpired:
            self.log(f"  ❌ {package_spec} インストールタイムアウト", "ERROR")
            self.failed_packages.append(package_spec)
            return False
        except Exception as e:
            self.log(f"  ❌ {package_spec} インストールエラー: {e}", "ERROR")
            self.failed_packages.append(package_spec)
            return False
    
    def install_numpy_safe(self):
        """NumPy安全インストール"""
        self.log("NumPyを安全にインストール中...")
        
        # 複数のバージョンを試行
        numpy_versions = [
            "numpy==1.24.3",
            "numpy==1.23.5", 
            "numpy==1.24.4",
            "numpy<2.0"
        ]
        
        for numpy_spec in numpy_versions:
            self.log(f"  {numpy_spec} を試行中...")
            
            if self.install_package_safe(numpy_spec, binary_only=True, timeout=300):
                # インストール成功後、動作確認
                try:
                    import numpy as np
                    version = np.__version__
                    self.log(f"  ✅ NumPy {version} 動作確認成功", "SUCCESS")
                    return True
                except Exception as e:
                    self.log(f"  ❌ NumPy動作確認エラー: {e}", "ERROR")
                    continue
        
        self.log("❌ NumPyインストールに失敗しました", "ERROR")
        return False
    
    def install_scientific_packages(self):
        """科学計算パッケージインストール"""
        self.log("科学計算パッケージをインストール中...")
        
        packages = [
            ("scipy", "scipy"),
            ("pandas", "pandas"),
            ("matplotlib", "matplotlib")
        ]
        
        success_count = 0
        
        for package_name, package_spec in packages:
            if self.install_package_safe(package_spec, binary_only=True):
                success_count += 1
            else:
                # バイナリ版で失敗した場合、通常インストール試行
                self.log(f"  {package_name} バイナリ版失敗、通常インストール試行...")
                if self.install_package_safe(package_spec, binary_only=False):
                    success_count += 1
        
        self.log(f"科学計算パッケージ: {success_count}/{len(packages)} 成功")
        return success_count >= 2  # 2つ以上成功すればOK
    
    def install_ml_packages(self):
        """機械学習パッケージインストール"""
        self.log("機械学習パッケージをインストール中...")
        
        packages = [
            ("PyTorch", "torch"),
            ("ONNX Runtime", "onnxruntime")
        ]
        
        success_count = 0
        
        for package_name, package_spec in packages:
            if self.install_package_safe(package_spec, binary_only=False, timeout=600):
                success_count += 1
        
        self.log(f"機械学習パッケージ: {success_count}/{len(packages)} 成功")
        return success_count >= 1  # 1つ以上成功すればOK
    
    def install_utility_packages(self):
        """ユーティリティパッケージインストール"""
        self.log("ユーティリティパッケージをインストール中...")
        
        packages = [
            ("requests", "requests"),
            ("psutil", "psutil")
        ]
        
        success_count = 0
        
        for package_name, package_spec in packages:
            if self.install_package_safe(package_spec, binary_only=False):
                success_count += 1
        
        self.log(f"ユーティリティパッケージ: {success_count}/{len(packages)} 成功")
        return success_count >= 1
    
    def test_package_imports(self):
        """パッケージインポートテスト"""
        self.log("パッケージインポートをテスト中...")
        
        test_packages = [
            ("NumPy", "import numpy as np; np.__version__"),
            ("SciPy", "import scipy; scipy.__version__"),
            ("Pandas", "import pandas as pd; pd.__version__"),
            ("Matplotlib", "import matplotlib; matplotlib.__version__"),
            ("PyTorch", "import torch; torch.__version__"),
            ("ONNX Runtime", "import onnxruntime as ort; ort.__version__"),
            ("Requests", "import requests; requests.__version__"),
            ("PSUtil", "import psutil; psutil.__version__")
        ]
        
        success_count = 0
        package_versions = {}
        
        for package_name, test_code in test_packages:
            try:
                result = eval(test_code)
                self.log(f"  ✅ {package_name}: {result}", "SUCCESS")
                package_versions[package_name] = result
                success_count += 1
            except Exception as e:
                self.log(f"  ❌ {package_name}: {e}", "ERROR")
                package_versions[package_name] = f"Error: {e}"
        
        self.log(f"パッケージインポートテスト: {success_count}/{len(test_packages)} 成功")
        
        # バージョン情報保存
        self.package_versions = package_versions
        
        return success_count >= 6  # 6つ以上成功すればOK
    
    def test_numpy_functionality(self):
        """NumPy機能テスト"""
        self.log("NumPy機能をテスト中...")
        
        try:
            import numpy as np
            
            # 基本演算テスト
            arr1 = np.array([1, 2, 3, 4, 5])
            arr2 = np.array([2, 3, 4, 5, 6])
            
            result_add = arr1 + arr2
            result_mul = arr1 * arr2
            result_dot = np.dot(arr1, arr2)
            
            self.log(f"  配列演算: 加算={result_add.sum()}, 乗算={result_mul.sum()}, 内積={result_dot}")
            
            # 行列演算テスト
            matrix1 = np.random.rand(5, 5)
            matrix2 = np.random.rand(5, 5)
            matrix_result = np.matmul(matrix1, matrix2)
            
            self.log(f"  行列演算: 結果形状={matrix_result.shape}")
            
            # 統計関数テスト
            data = np.random.normal(0, 1, 100)
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            self.log(f"  統計関数: 平均={mean_val:.3f}, 標準偏差={std_val:.3f}")
            
            self.log("✅ NumPy機能テスト成功", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"❌ NumPy機能テストエラー: {e}", "ERROR")
            return False
    
    def save_installation_report(self):
        """インストールレポート保存"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # JSONレポート
            report_data = {
                "timestamp": datetime.now().isoformat(),
                "python_version": sys.version,
                "python_executable": sys.executable,
                "installed_packages": self.installed_packages,
                "failed_packages": self.failed_packages,
                "package_versions": getattr(self, 'package_versions', {}),
                "log_messages": self.log_messages
            }
            
            json_path = f'installation_report_{timestamp}.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            # テキストレポート
            txt_path = f'installation_log_{timestamp}.txt'
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("段階的インストールレポート\n")
                f.write("=" * 50 + "\n")
                f.write(f"実行日時: {datetime.now().isoformat()}\n")
                f.write(f"Python: {sys.version}\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("インストール成功パッケージ:\n")
                for pkg, status in self.installed_packages.items():
                    f.write(f"  ✅ {pkg}: {status}\n")
                
                f.write(f"\n失敗パッケージ:\n")
                for pkg in self.failed_packages:
                    f.write(f"  ❌ {pkg}\n")
                
                f.write(f"\nパッケージバージョン:\n")
                for pkg, version in getattr(self, 'package_versions', {}).items():
                    f.write(f"  {pkg}: {version}\n")
                
                f.write("\n" + "=" * 50 + "\n")
                f.write("詳細ログ:\n")
                for log_entry in self.log_messages:
                    f.write(log_entry + "\n")
            
            self.log(f"💾 インストールレポート保存: {json_path}, {txt_path}")
            
        except Exception as e:
            self.log(f"レポート保存エラー: {e}", "WARNING")
    
    def run_installation(self):
        """インストール実行"""
        self.log("🔧 段階的インストールを開始します")
        self.log("=" * 60)
        
        try:
            # Python環境確認
            self.check_python_environment()
            
            # 基本ツールアップグレード
            if not self.upgrade_basic_tools():
                self.log("⚠️ 基本ツールアップグレードに問題がありますが続行します", "WARNING")
            
            # NumPy安全インストール
            if not self.install_numpy_safe():
                self.log("❌ NumPyインストールに失敗しました", "ERROR")
                return False
            
            # 科学計算パッケージインストール
            self.install_scientific_packages()
            
            # 機械学習パッケージインストール
            self.install_ml_packages()
            
            # ユーティリティパッケージインストール
            self.install_utility_packages()
            
            # パッケージインポートテスト
            if not self.test_package_imports():
                self.log("⚠️ 一部パッケージのインポートに失敗しましたが続行します", "WARNING")
            
            # NumPy機能テスト
            self.test_numpy_functionality()
            
            # レポート保存
            self.save_installation_report()
            
            # 成功メッセージ
            self.log("=" * 60)
            self.log("🎉 段階的インストール完了！", "SUCCESS")
            self.log("=" * 60)
            
            self.log("次のステップ:")
            self.log("1. python infer_os_npu_test.py --mode basic  # 基本テスト")
            self.log("2. python infer_os_npu_test.py --mode comprehensive  # 包括的テスト")
            
            return True
            
        except Exception as e:
            self.log(f"❌ インストールエラー: {e}", "ERROR")
            self.log(traceback.format_exc(), "ERROR")
            return False

def main():
    """メイン実行関数"""
    print("""
============================================================================
🔧 段階的インストール自動化スクリプト
============================================================================

このスクリプトはsetuptools/pipエラーを回避してInfer-OS NPU環境を構築します。

インストール内容:
- 基本ツール (pip, setuptools, wheel) アップグレード
- NumPy (1.24.3) 安全インストール
- 科学計算パッケージ (scipy, pandas, matplotlib)
- 機械学習パッケージ (torch, onnxruntime)
- ユーティリティ (requests, psutil)

特徴:
- バイナリ版優先インストール（コンパイルエラー回避）
- 段階的インストール（エラー時の影響最小化）
- 包括的動作確認テスト
- 詳細レポート生成

実行時間: 約10-20分
インターネット接続が必要です

============================================================================
""")
    
    try:
        installer = StepByStepInstaller()
        success = installer.run_installation()
        
        if success:
            print("\n🎉 段階的インストール成功！Infer-OS NPUテストが実行可能になりました。")
        else:
            print("\n❌ インストールに問題が発生しました。レポートを確認してください。")
            
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーにより中断されました")
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()

