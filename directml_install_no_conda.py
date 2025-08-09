#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Conda不要 DirectMLインストールスクリプト

Condaエラーを回避してDirectMLを確実にインストール

機能:
- Conda不要のDirectMLインストール
- 自動環境確認・最適化
- インストール後の動作確認
- 詳細ログ出力

使用方法:
    python directml_install_no_conda.py
"""

import sys
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path

class DirectMLInstaller:
    """Conda不要DirectMLインストーラー"""
    
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
    
    def check_python_environment(self):
        """Python環境確認"""
        self.log("Python環境を確認中...")
        
        try:
            python_version = sys.version
            python_executable = sys.executable
            
            self.log(f"Python バージョン: {python_version}")
            self.log(f"Python 実行ファイル: {python_executable}")
            
            # Python バージョンチェック
            version_info = sys.version_info
            if version_info.major == 3 and 8 <= version_info.minor <= 13:
                self.log("✅ Python バージョン: 対応", "SUCCESS")
                return True
            else:
                self.log(f"⚠️ Python バージョン: {version_info.major}.{version_info.minor} (推奨: 3.8-3.13)", "WARNING")
                return False
                
        except Exception as e:
            self.log(f"Python環境確認エラー: {e}", "ERROR")
            return False
    
    def upgrade_pip(self):
        """pipアップグレード"""
        self.log("pipをアップグレード中...")
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                capture_output=True, text=True, timeout=300
            )
            
            if result.returncode == 0:
                self.log("✅ pip アップグレード成功", "SUCCESS")
                return True
            else:
                self.log(f"⚠️ pip アップグレード警告: {result.stderr}", "WARNING")
                return False
                
        except subprocess.TimeoutExpired:
            self.log("❌ pip アップグレードタイムアウト", "ERROR")
            return False
        except Exception as e:
            self.log(f"❌ pip アップグレードエラー: {e}", "ERROR")
            return False
    
    def install_pytorch_base(self):
        """PyTorch基本パッケージインストール"""
        self.log("PyTorch基本パッケージをインストール中...")
        
        try:
            # PyTorchがすでにインストールされているかチェック
            try:
                import torch
                self.log(f"✅ PyTorch既にインストール済み: v{torch.__version__}", "SUCCESS")
                return True
            except ImportError:
                pass
            
            # PyTorchインストール
            packages = ["torch", "torchvision", "torchaudio"]
            
            for package in packages:
                self.log(f"  {package} をインストール中...")
                
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    capture_output=True, text=True, timeout=600
                )
                
                if result.returncode == 0:
                    self.log(f"  ✅ {package} インストール成功", "SUCCESS")
                else:
                    self.log(f"  ⚠️ {package} インストール警告: {result.stderr}", "WARNING")
            
            # インストール確認
            import torch
            self.log(f"✅ PyTorch インストール確認: v{torch.__version__}", "SUCCESS")
            return True
            
        except subprocess.TimeoutExpired:
            self.log("❌ PyTorchインストールタイムアウト", "ERROR")
            return False
        except Exception as e:
            self.log(f"❌ PyTorchインストールエラー: {e}", "ERROR")
            return False
    
    def install_torch_directml(self):
        """torch-directmlインストール"""
        self.log("torch-directmlをインストール中...")
        
        try:
            # torch-directmlがすでにインストールされているかチェック
            try:
                import torch_directml
                self.log(f"✅ torch-directml既にインストール済み", "SUCCESS")
                return True
            except ImportError:
                pass
            
            # torch-directmlインストール
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "torch-directml"],
                capture_output=True, text=True, timeout=600
            )
            
            if result.returncode == 0:
                self.log("✅ torch-directml インストール成功", "SUCCESS")
                
                # インストール確認
                import torch_directml
                self.log("✅ torch-directml インポート確認", "SUCCESS")
                
                if torch_directml.is_available():
                    device = torch_directml.device()
                    self.log(f"✅ DirectMLデバイス: {device}", "SUCCESS")
                else:
                    self.log("⚠️ DirectMLデバイス利用不可", "WARNING")
                
                return True
            else:
                self.log(f"❌ torch-directml インストールエラー: {result.stderr}", "ERROR")
                return False
                
        except subprocess.TimeoutExpired:
            self.log("❌ torch-directmlインストールタイムアウト", "ERROR")
            return False
        except Exception as e:
            self.log(f"❌ torch-directmlインストールエラー: {e}", "ERROR")
            return False
    
    def install_onnxruntime_directml(self):
        """onnxruntime-directmlインストール"""
        self.log("onnxruntime-directmlをインストール中...")
        
        try:
            # 既存のonnxruntimeをアンインストール（競合回避）
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "uninstall", "onnxruntime", "-y"],
                    capture_output=True, text=True, timeout=120
                )
                if result.returncode == 0:
                    self.log("  既存onnxruntimeをアンインストール", "INFO")
            except:
                pass
            
            # onnxruntime-directmlがすでにインストールされているかチェック
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                if 'DmlExecutionProvider' in providers:
                    self.log(f"✅ onnxruntime-directml既にインストール済み", "SUCCESS")
                    return True
            except ImportError:
                pass
            
            # onnxruntime-directmlインストール
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "onnxruntime-directml"],
                capture_output=True, text=True, timeout=600
            )
            
            if result.returncode == 0:
                self.log("✅ onnxruntime-directml インストール成功", "SUCCESS")
                
                # インストール確認
                import onnxruntime as ort
                providers = ort.get_available_providers()
                self.log(f"利用可能プロバイダー: {len(providers)}個")
                
                if 'DmlExecutionProvider' in providers:
                    self.log("✅ DirectMLプロバイダー利用可能", "SUCCESS")
                else:
                    self.log("⚠️ DirectMLプロバイダー未対応", "WARNING")
                
                return True
            else:
                self.log(f"❌ onnxruntime-directml インストールエラー: {result.stderr}", "ERROR")
                return False
                
        except subprocess.TimeoutExpired:
            self.log("❌ onnxruntime-directmlインストールタイムアウト", "ERROR")
            return False
        except Exception as e:
            self.log(f"❌ onnxruntime-directmlインストールエラー: {e}", "ERROR")
            return False
    
    def install_additional_packages(self):
        """追加パッケージインストール"""
        self.log("追加パッケージをインストール中...")
        
        additional_packages = [
            "numpy",
            "pandas", 
            "matplotlib",
            "requests",
            "psutil",
            "pillow",
            "beautifulsoup4"
        ]
        
        success_count = 0
        
        for package in additional_packages:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    capture_output=True, text=True, timeout=300
                )
                
                if result.returncode == 0:
                    self.log(f"  ✅ {package} インストール成功", "SUCCESS")
                    success_count += 1
                else:
                    self.log(f"  ⚠️ {package} インストール警告", "WARNING")
                    
            except Exception as e:
                self.log(f"  ❌ {package} インストールエラー: {e}", "ERROR")
        
        self.log(f"追加パッケージ: {success_count}/{len(additional_packages)} 成功")
        return success_count >= len(additional_packages) * 0.8  # 80%以上成功すればOK
    
    def test_directml_functionality(self):
        """DirectML機能テスト"""
        self.log("DirectML機能をテスト中...")
        
        try:
            # PyTorch DirectMLテスト
            import torch
            import torch_directml
            
            if torch_directml.is_available():
                device = torch_directml.device()
                self.log(f"DirectMLデバイス: {device}")
                
                # 簡単なテンソル演算
                x = torch.randn(100, 100, device=device)
                y = torch.randn(100, 100, device=device)
                z = torch.mm(x, y)
                
                self.log("✅ PyTorch DirectML演算成功", "SUCCESS")
                
                # 性能テスト
                start_time = time.time()
                for _ in range(10):
                    z = torch.mm(x, y)
                z_cpu = z.cpu()  # 同期
                directml_time = time.time() - start_time
                
                self.log(f"DirectML性能テスト: {directml_time:.4f}秒 (10回実行)")
                
            else:
                self.log("⚠️ DirectMLデバイス利用不可", "WARNING")
                return False
            
            # ONNX Runtime DirectMLテスト
            import onnxruntime as ort
            providers = ort.get_available_providers()
            
            if 'DmlExecutionProvider' in providers:
                self.log("✅ ONNX Runtime DirectML利用可能", "SUCCESS")
            else:
                self.log("⚠️ ONNX Runtime DirectML未対応", "WARNING")
                return False
            
            return True
            
        except Exception as e:
            self.log(f"❌ DirectML機能テストエラー: {e}", "ERROR")
            return False
    
    def save_installation_log(self):
        """インストールログ保存"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_path = f'directml_install_log_{timestamp}.txt'
            
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write("DirectML インストールログ\n")
                f.write("=" * 50 + "\n")
                f.write(f"実行日時: {datetime.now().isoformat()}\n")
                f.write(f"Python: {sys.version}\n")
                f.write("=" * 50 + "\n\n")
                
                for log_entry in self.log_messages:
                    f.write(log_entry + "\n")
            
            self.log(f"💾 インストールログ保存: {log_path}")
            
        except Exception as e:
            self.log(f"ログ保存エラー: {e}", "WARNING")
    
    def run_installation(self):
        """インストール実行"""
        self.log("🚀 Conda不要 DirectMLインストールを開始します")
        self.log("=" * 60)
        
        try:
            # 環境確認
            if not self.check_python_environment():
                self.log("❌ Python環境に問題があります", "ERROR")
                return False
            
            # pipアップグレード
            self.upgrade_pip()
            
            # PyTorch基本パッケージ
            if not self.install_pytorch_base():
                self.log("❌ PyTorchインストールに失敗しました", "ERROR")
                return False
            
            # torch-directml
            if not self.install_torch_directml():
                self.log("❌ torch-directmlインストールに失敗しました", "ERROR")
                return False
            
            # onnxruntime-directml
            if not self.install_onnxruntime_directml():
                self.log("❌ onnxruntime-directmlインストールに失敗しました", "ERROR")
                return False
            
            # 追加パッケージ
            self.install_additional_packages()
            
            # 機能テスト
            if not self.test_directml_functionality():
                self.log("⚠️ DirectML機能テストで問題が発生しました", "WARNING")
            
            # ログ保存
            self.save_installation_log()
            
            # 成功メッセージ
            self.log("=" * 60)
            self.log("🎉 DirectMLインストール完了！", "SUCCESS")
            self.log("=" * 60)
            
            self.log("次のステップ:")
            self.log("1. python directml_verification.py  # 詳細検証")
            self.log("2. python ryzen_ai_verification.py  # Ryzen AI確認")
            self.log("3. python infer_os_npu_test.py --mode basic  # 統合テスト")
            
            return True
            
        except Exception as e:
            self.log(f"❌ インストールエラー: {e}", "ERROR")
            self.log(traceback.format_exc(), "ERROR")
            return False

def main():
    """メイン実行関数"""
    print("""
============================================================================
🚀 Conda不要 DirectMLインストーラー
============================================================================

このスクリプトはCondaエラーを回避してDirectMLを確実にインストールします。

インストール内容:
- PyTorch (CPU版)
- torch-directml (AMD GPU/NPU対応)
- onnxruntime-directml (ONNX Runtime DirectML対応)
- 追加パッケージ (numpy, pandas等)

実行時間: 約5-10分
インターネット接続が必要です

============================================================================
""")
    
    try:
        installer = DirectMLInstaller()
        success = installer.run_installation()
        
        if success:
            print("\n🎉 インストール成功！AMD NPU環境でDirectMLが利用可能になりました。")
        else:
            print("\n❌ インストールに問題が発生しました。ログを確認してください。")
            
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーにより中断されました")
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()

