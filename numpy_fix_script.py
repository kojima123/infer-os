#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 NumPy修復自動化スクリプト

Infer-OS NPUテスト用のNumPyエラーを自動修復

機能:
- NumPyエラー自動検出
- 安全な修復手順実行
- 依存関係整合性確認
- 動作確認テスト

使用方法:
    python numpy_fix_script.py
"""

import sys
import subprocess
import time
import traceback
from datetime import datetime

class NumPyFixer:
    """NumPy修復ツール"""
    
    def __init__(self):
        self.log_messages = []
        
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
    
    def check_numpy_error(self):
        """NumPyエラー確認"""
        self.log("NumPyエラーを確認中...")
        
        try:
            import numpy as np
            self.log(f"✅ NumPy正常: v{np.__version__}", "SUCCESS")
            return False
        except ImportError as e:
            error_msg = str(e)
            if "numpy._core._multiarray_umath" in error_msg:
                self.log("❌ NumPy C拡張エラー検出", "ERROR")
                return True
            elif "source directory" in error_msg:
                self.log("❌ NumPyソースディレクトリエラー検出", "ERROR")
                return True
            else:
                self.log(f"❌ NumPy未知エラー: {error_msg}", "ERROR")
                return True
        except Exception as e:
            self.log(f"❌ NumPy確認エラー: {e}", "ERROR")
            return True
    
    def check_python_version(self):
        """Python バージョン確認"""
        self.log("Python環境を確認中...")
        
        version_info = sys.version_info
        python_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
        
        self.log(f"Python バージョン: {python_version}")
        self.log(f"Python 実行ファイル: {sys.executable}")
        
        if version_info.major == 3 and 8 <= version_info.minor <= 12:
            self.log("✅ Python バージョン: 対応", "SUCCESS")
            return True
        else:
            self.log(f"⚠️ Python バージョン: {python_version} (推奨: 3.8-3.12)", "WARNING")
            return False
    
    def uninstall_numpy(self):
        """NumPy アンインストール"""
        self.log("破損したNumPyをアンインストール中...")
        
        try:
            # 通常のアンインストール
            result = subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "numpy", "-y"],
                capture_output=True, text=True, timeout=120
            )
            
            if result.returncode == 0:
                self.log("✅ NumPy アンインストール成功", "SUCCESS")
            else:
                self.log(f"⚠️ NumPy アンインストール警告: {result.stderr}", "WARNING")
            
            # 強制アンインストール（念のため）
            try:
                result2 = subprocess.run(
                    [sys.executable, "-m", "pip", "uninstall", "numpy", "-y", "--break-system-packages"],
                    capture_output=True, text=True, timeout=60
                )
            except:
                pass
            
            return True
            
        except subprocess.TimeoutExpired:
            self.log("❌ NumPy アンインストールタイムアウト", "ERROR")
            return False
        except Exception as e:
            self.log(f"❌ NumPy アンインストールエラー: {e}", "ERROR")
            return False
    
    def clear_pip_cache(self):
        """pipキャッシュクリア"""
        self.log("pipキャッシュをクリア中...")
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "cache", "purge"],
                capture_output=True, text=True, timeout=60
            )
            
            if result.returncode == 0:
                self.log("✅ pipキャッシュクリア成功", "SUCCESS")
            else:
                self.log("⚠️ pipキャッシュクリア警告", "WARNING")
            
            return True
            
        except Exception as e:
            self.log(f"⚠️ pipキャッシュクリアエラー: {e}", "WARNING")
            return False
    
    def install_stable_numpy(self):
        """安定版NumPyインストール"""
        self.log("安定版NumPyをインストール中...")
        
        # 推奨バージョン（Python バージョンに応じて）
        version_info = sys.version_info
        if version_info.minor >= 12:
            numpy_version = "1.24.3"  # Python 3.12対応
        elif version_info.minor >= 11:
            numpy_version = "1.24.3"  # Python 3.11対応
        else:
            numpy_version = "1.21.6"  # 古いPython対応
        
        try:
            self.log(f"NumPy {numpy_version} をインストール中...")
            
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", f"numpy=={numpy_version}"],
                capture_output=True, text=True, timeout=300
            )
            
            if result.returncode == 0:
                self.log(f"✅ NumPy {numpy_version} インストール成功", "SUCCESS")
                return True
            else:
                self.log(f"❌ NumPy インストールエラー: {result.stderr}", "ERROR")
                
                # フォールバック: バージョン指定なし
                self.log("フォールバック: 最新安定版をインストール中...")
                result2 = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "numpy<2.0"],
                    capture_output=True, text=True, timeout=300
                )
                
                if result2.returncode == 0:
                    self.log("✅ NumPy フォールバックインストール成功", "SUCCESS")
                    return True
                else:
                    return False
                
        except subprocess.TimeoutExpired:
            self.log("❌ NumPy インストールタイムアウト", "ERROR")
            return False
        except Exception as e:
            self.log(f"❌ NumPy インストールエラー: {e}", "ERROR")
            return False
    
    def install_dependencies(self):
        """依存関係インストール"""
        self.log("依存関係をインストール中...")
        
        dependencies = [
            "scipy<1.11",
            "pandas<2.1", 
            "matplotlib<3.8",
            "requests",
            "psutil"
        ]
        
        success_count = 0
        
        for dep in dependencies:
            try:
                self.log(f"  {dep} をインストール中...")
                
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", dep],
                    capture_output=True, text=True, timeout=180
                )
                
                if result.returncode == 0:
                    self.log(f"  ✅ {dep} インストール成功", "SUCCESS")
                    success_count += 1
                else:
                    self.log(f"  ⚠️ {dep} インストール警告", "WARNING")
                    
            except Exception as e:
                self.log(f"  ❌ {dep} インストールエラー: {e}", "ERROR")
        
        self.log(f"依存関係インストール: {success_count}/{len(dependencies)} 成功")
        return success_count >= len(dependencies) * 0.8  # 80%以上成功すればOK
    
    def test_numpy_functionality(self):
        """NumPy機能テスト"""
        self.log("NumPy機能をテスト中...")
        
        try:
            import numpy as np
            
            # バージョン確認
            version = np.__version__
            self.log(f"NumPy バージョン: {version}")
            
            # 基本機能テスト
            arr1 = np.array([1, 2, 3, 4, 5])
            arr2 = np.array([2, 3, 4, 5, 6])
            
            # 演算テスト
            result_add = arr1 + arr2
            result_mul = arr1 * arr2
            result_dot = np.dot(arr1, arr2)
            
            self.log(f"配列演算テスト: 加算={result_add.sum()}, 乗算={result_mul.sum()}, 内積={result_dot}")
            
            # 行列演算テスト
            matrix1 = np.random.rand(10, 10)
            matrix2 = np.random.rand(10, 10)
            matrix_result = np.matmul(matrix1, matrix2)
            
            self.log(f"行列演算テスト: 結果形状={matrix_result.shape}")
            
            # 統計関数テスト
            data = np.random.normal(0, 1, 1000)
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            self.log(f"統計関数テスト: 平均={mean_val:.3f}, 標準偏差={std_val:.3f}")
            
            self.log("✅ NumPy機能テスト成功", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"❌ NumPy機能テストエラー: {e}", "ERROR")
            return False
    
    def test_dependencies(self):
        """依存関係テスト"""
        self.log("依存関係をテスト中...")
        
        test_packages = [
            ("scipy", "import scipy; scipy.__version__"),
            ("pandas", "import pandas as pd; pd.__version__"),
            ("matplotlib", "import matplotlib; matplotlib.__version__"),
            ("requests", "import requests; requests.__version__"),
            ("psutil", "import psutil; psutil.__version__")
        ]
        
        success_count = 0
        
        for package_name, test_code in test_packages:
            try:
                exec(test_code)
                self.log(f"  ✅ {package_name} テスト成功", "SUCCESS")
                success_count += 1
            except Exception as e:
                self.log(f"  ❌ {package_name} テストエラー: {e}", "ERROR")
        
        self.log(f"依存関係テスト: {success_count}/{len(test_packages)} 成功")
        return success_count >= len(test_packages) * 0.8
    
    def save_fix_log(self):
        """修復ログ保存"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_path = f'numpy_fix_log_{timestamp}.txt'
            
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write("NumPy修復ログ\n")
                f.write("=" * 50 + "\n")
                f.write(f"実行日時: {datetime.now().isoformat()}\n")
                f.write(f"Python: {sys.version}\n")
                f.write("=" * 50 + "\n\n")
                
                for log_entry in self.log_messages:
                    f.write(log_entry + "\n")
            
            self.log(f"💾 修復ログ保存: {log_path}")
            
        except Exception as e:
            self.log(f"ログ保存エラー: {e}", "WARNING")
    
    def run_fix(self):
        """修復実行"""
        self.log("🔧 NumPy修復を開始します")
        self.log("=" * 60)
        
        try:
            # Python環境確認
            if not self.check_python_version():
                self.log("⚠️ Python環境に問題がありますが続行します", "WARNING")
            
            # NumPyエラー確認
            if not self.check_numpy_error():
                self.log("✅ NumPyエラーは検出されませんでした", "SUCCESS")
                return True
            
            # NumPyアンインストール
            if not self.uninstall_numpy():
                self.log("❌ NumPyアンインストールに失敗しました", "ERROR")
                return False
            
            # pipキャッシュクリア
            self.clear_pip_cache()
            
            # 安定版NumPyインストール
            if not self.install_stable_numpy():
                self.log("❌ NumPyインストールに失敗しました", "ERROR")
                return False
            
            # 依存関係インストール
            self.install_dependencies()
            
            # NumPy機能テスト
            if not self.test_numpy_functionality():
                self.log("❌ NumPy機能テストに失敗しました", "ERROR")
                return False
            
            # 依存関係テスト
            self.test_dependencies()
            
            # ログ保存
            self.save_fix_log()
            
            # 成功メッセージ
            self.log("=" * 60)
            self.log("🎉 NumPy修復完了！", "SUCCESS")
            self.log("=" * 60)
            
            self.log("次のステップ:")
            self.log("1. python infer_os_npu_test.py --mode basic  # 基本テスト")
            self.log("2. python infer_os_npu_test.py --mode comprehensive  # 包括的テスト")
            
            return True
            
        except Exception as e:
            self.log(f"❌ 修復エラー: {e}", "ERROR")
            self.log(traceback.format_exc(), "ERROR")
            return False

def main():
    """メイン実行関数"""
    print("""
============================================================================
🔧 NumPy修復自動化スクリプト
============================================================================

このスクリプトはInfer-OS NPUテスト用のNumPyエラーを自動修復します。

修復内容:
- 破損したNumPyの完全削除
- pipキャッシュクリア
- 安定版NumPy (1.24.3) インストール
- 依存関係 (scipy, pandas等) インストール
- 機能テスト・動作確認

実行時間: 約5-10分
インターネット接続が必要です

============================================================================
""")
    
    try:
        fixer = NumPyFixer()
        success = fixer.run_fix()
        
        if success:
            print("\n🎉 NumPy修復成功！Infer-OS NPUテストが実行可能になりました。")
        else:
            print("\n❌ NumPy修復に問題が発生しました。ログを確認してください。")
            
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーにより中断されました")
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()

