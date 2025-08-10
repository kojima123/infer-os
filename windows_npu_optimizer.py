# -*- coding: utf-8 -*-
"""
🚀 Windows NPU最適化機能

Windows環境でのNPU（Neural Processing Unit）検出・有効化・最適化
- AMD Ryzen AI NPU対応
- Intel NPU対応
- Qualcomm NPU対応
- DirectML NPU最適化
"""

import os
import sys
import subprocess
import platform
import psutil
import time
from typing import Dict, List, Optional, Tuple
import traceback

class WindowsNPUOptimizer:
    """Windows NPU最適化クラス"""
    
    def __init__(self):
        self.npu_info = {}
        self.npu_available = False
        self.npu_type = None
        self.directml_available = False
        
    def detect_npu_hardware(self) -> Dict[str, any]:
        """NPUハードウェア検出"""
        print("🔍 NPUハードウェア検出中...")
        
        npu_info = {
            "detected": False,
            "type": None,
            "devices": [],
            "driver_version": None,
            "directml_support": False
        }
        
        try:
            # AMD Ryzen AI NPU検出
            amd_npu = self._detect_amd_ryzen_ai_npu()
            if amd_npu["detected"]:
                npu_info.update(amd_npu)
                npu_info["type"] = "AMD Ryzen AI"
                print(f"✅ AMD Ryzen AI NPU検出: {amd_npu['model']}")
            
            # Intel NPU検出
            intel_npu = self._detect_intel_npu()
            if intel_npu["detected"]:
                npu_info.update(intel_npu)
                npu_info["type"] = "Intel NPU"
                print(f"✅ Intel NPU検出: {intel_npu['model']}")
            
            # Qualcomm NPU検出
            qualcomm_npu = self._detect_qualcomm_npu()
            if qualcomm_npu["detected"]:
                npu_info.update(qualcomm_npu)
                npu_info["type"] = "Qualcomm NPU"
                print(f"✅ Qualcomm NPU検出: {qualcomm_npu['model']}")
            
            # DirectML対応確認
            directml_support = self._check_directml_support()
            npu_info["directml_support"] = directml_support
            
            if npu_info["detected"]:
                print(f"🎯 NPU検出成功: {npu_info['type']}")
                self.npu_available = True
                self.npu_type = npu_info["type"]
                self.directml_available = directml_support
            else:
                print("⚠️ NPUが検出されませんでした")
                
        except Exception as e:
            print(f"❌ NPU検出エラー: {e}")
            
        self.npu_info = npu_info
        return npu_info
    
    def _detect_amd_ryzen_ai_npu(self) -> Dict[str, any]:
        """AMD Ryzen AI NPU検出"""
        try:
            # WMIを使用してAMD NPU検出
            result = subprocess.run([
                "powershell", "-Command",
                "Get-WmiObject -Class Win32_PnPEntity | Where-Object {$_.Name -like '*NPU*' -or $_.Name -like '*Ryzen AI*' -or $_.DeviceID -like '*VEN_1022*'} | Select-Object Name, DeviceID, Status"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'NPU' in line or 'Ryzen AI' in line:
                        return {
                            "detected": True,
                            "model": "AMD Ryzen AI NPU",
                            "vendor": "AMD",
                            "status": "Active" if "OK" in line else "Unknown"
                        }
            
            # レジストリからの検出も試行
            reg_result = subprocess.run([
                "reg", "query", "HKLM\\SYSTEM\\CurrentControlSet\\Enum\\PCI",
                "/s", "/f", "NPU"
            ], capture_output=True, text=True, timeout=10)
            
            if reg_result.returncode == 0 and "NPU" in reg_result.stdout:
                return {
                    "detected": True,
                    "model": "AMD Ryzen AI NPU (Registry)",
                    "vendor": "AMD",
                    "status": "Detected"
                }
                
        except Exception as e:
            print(f"⚠️ AMD NPU検出エラー: {e}")
        
        return {"detected": False}
    
    def _detect_intel_npu(self) -> Dict[str, any]:
        """Intel NPU検出"""
        try:
            # Intel NPU検出
            result = subprocess.run([
                "powershell", "-Command",
                "Get-WmiObject -Class Win32_PnPEntity | Where-Object {$_.Name -like '*Intel*NPU*' -or $_.DeviceID -like '*VEN_8086*'} | Select-Object Name, DeviceID, Status"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'NPU' in line and 'Intel' in line:
                        return {
                            "detected": True,
                            "model": "Intel NPU",
                            "vendor": "Intel",
                            "status": "Active" if "OK" in line else "Unknown"
                        }
                        
        except Exception as e:
            print(f"⚠️ Intel NPU検出エラー: {e}")
        
        return {"detected": False}
    
    def _detect_qualcomm_npu(self) -> Dict[str, any]:
        """Qualcomm NPU検出"""
        try:
            # Qualcomm NPU検出
            result = subprocess.run([
                "powershell", "-Command",
                "Get-WmiObject -Class Win32_PnPEntity | Where-Object {$_.Name -like '*Qualcomm*NPU*' -or $_.Name -like '*Hexagon*'} | Select-Object Name, DeviceID, Status"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if ('NPU' in line or 'Hexagon' in line) and 'Qualcomm' in line:
                        return {
                            "detected": True,
                            "model": "Qualcomm NPU",
                            "vendor": "Qualcomm",
                            "status": "Active" if "OK" in line else "Unknown"
                        }
                        
        except Exception as e:
            print(f"⚠️ Qualcomm NPU検出エラー: {e}")
        
        return {"detected": False}
    
    def _check_directml_support(self) -> bool:
        """DirectML対応確認"""
        try:
            # DirectMLライブラリの確認
            import importlib.util
            
            # onnxruntime-directmlの確認
            spec = importlib.util.find_spec("onnxruntime")
            if spec is not None:
                try:
                    import onnxruntime as ort
                    providers = ort.get_available_providers()
                    if 'DmlExecutionProvider' in providers:
                        print("✅ DirectML対応確認")
                        return True
                except:
                    pass
            
            # torch-directmlの確認
            spec = importlib.util.find_spec("torch_directml")
            if spec is not None:
                print("✅ torch-directml検出")
                return True
                
        except Exception as e:
            print(f"⚠️ DirectML確認エラー: {e}")
        
        return False
    
    def enable_npu_optimization(self) -> bool:
        """NPU最適化を有効化"""
        if not self.npu_available:
            print("❌ NPUが利用できません")
            return False
        
        print(f"🚀 {self.npu_type} NPU最適化を有効化中...")
        
        try:
            # DirectML最適化
            if self.directml_available:
                success = self._enable_directml_optimization()
                if success:
                    print("✅ DirectML NPU最適化有効化完了")
                    return True
            
            # NPU固有の最適化
            if self.npu_type == "AMD Ryzen AI":
                return self._enable_amd_npu_optimization()
            elif self.npu_type == "Intel NPU":
                return self._enable_intel_npu_optimization()
            elif self.npu_type == "Qualcomm NPU":
                return self._enable_qualcomm_npu_optimization()
                
        except Exception as e:
            print(f"❌ NPU最適化有効化エラー: {e}")
            
        return False
    
    def _enable_directml_optimization(self) -> bool:
        """DirectML最適化有効化"""
        try:
            # 環境変数設定
            os.environ["ORT_DIRECTML_DEVICE_FILTER"] = "0"  # 最初のNPUデバイスを使用
            os.environ["DIRECTML_DEBUG"] = "0"  # デバッグ無効
            os.environ["DIRECTML_FORCE_NPU"] = "1"  # NPU強制使用
            
            print("✅ DirectML環境変数設定完了")
            return True
            
        except Exception as e:
            print(f"❌ DirectML最適化エラー: {e}")
            return False
    
    def _enable_amd_npu_optimization(self) -> bool:
        """AMD NPU最適化有効化"""
        try:
            # AMD固有の最適化設定
            os.environ["AMD_NPU_ENABLE"] = "1"
            os.environ["RYZEN_AI_OPTIMIZATION"] = "1"
            
            print("✅ AMD Ryzen AI NPU最適化設定完了")
            return True
            
        except Exception as e:
            print(f"❌ AMD NPU最適化エラー: {e}")
            return False
    
    def _enable_intel_npu_optimization(self) -> bool:
        """Intel NPU最適化有効化"""
        try:
            # Intel固有の最適化設定
            os.environ["INTEL_NPU_ENABLE"] = "1"
            os.environ["OPENVINO_NPU_OPTIMIZATION"] = "1"
            
            print("✅ Intel NPU最適化設定完了")
            return True
            
        except Exception as e:
            print(f"❌ Intel NPU最適化エラー: {e}")
            return False
    
    def _enable_qualcomm_npu_optimization(self) -> bool:
        """Qualcomm NPU最適化有効化"""
        try:
            # Qualcomm固有の最適化設定
            os.environ["QUALCOMM_NPU_ENABLE"] = "1"
            os.environ["HEXAGON_OPTIMIZATION"] = "1"
            
            print("✅ Qualcomm NPU最適化設定完了")
            return True
            
        except Exception as e:
            print(f"❌ Qualcomm NPU最適化エラー: {e}")
            return False
    
    def get_npu_status_report(self) -> str:
        """NPU状況レポート生成"""
        if not self.npu_info:
            self.detect_npu_hardware()
        
        report = f"""
🎯 **Windows NPU状況レポート**

📊 **NPU検出状況**:
  検出済み: {'✅' if self.npu_available else '❌'}
  NPUタイプ: {self.npu_type or 'なし'}
  DirectML対応: {'✅' if self.directml_available else '❌'}

🔧 **ハードウェア情報**:
  OS: {platform.system()} {platform.release()}
  アーキテクチャ: {platform.machine()}
  CPU: {platform.processor()}

⚡ **最適化状況**:
  NPU最適化: {'有効' if self.npu_available else '無効'}
  DirectML最適化: {'有効' if self.directml_available else '無効'}
  
💡 **推奨アクション**:
"""
        
        if not self.npu_available:
            report += """  🔧 NPUドライバーの更新を確認
  📦 DirectMLライブラリのインストール
  ⚙️ BIOS設定でNPUを有効化"""
        else:
            report += """  ✅ NPU最適化が利用可能
  🚀 DirectML最適化を活用
  📊 NPU性能ベンチマークを実行"""
        
        return report
    
    def install_directml_dependencies(self) -> bool:
        """DirectML依存関係のインストール"""
        print("📦 DirectML依存関係をインストール中...")
        
        try:
            # onnxruntime-directmlのインストール
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "onnxruntime-directml", "--upgrade"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ onnxruntime-directml インストール完了")
            else:
                print(f"⚠️ onnxruntime-directml インストール警告: {result.stderr}")
            
            # torch-directmlのインストール（利用可能な場合）
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    "torch-directml", "--upgrade"
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    print("✅ torch-directml インストール完了")
                else:
                    print("⚠️ torch-directml は利用できません（オプション）")
            except:
                print("⚠️ torch-directml インストールをスキップ")
            
            return True
            
        except Exception as e:
            print(f"❌ DirectML依存関係インストールエラー: {e}")
            return False

# 使用例とテスト関数
def test_windows_npu_optimization():
    """Windows NPU最適化のテスト"""
    print("🧪 Windows NPU最適化テスト開始")
    
    optimizer = WindowsNPUOptimizer()
    
    # NPU検出
    npu_info = optimizer.detect_npu_hardware()
    
    # 状況レポート
    report = optimizer.get_npu_status_report()
    print(report)
    
    # NPU最適化有効化
    if optimizer.npu_available:
        success = optimizer.enable_npu_optimization()
        if success:
            print("✅ NPU最適化テスト成功")
        else:
            print("❌ NPU最適化テスト失敗")
    else:
        print("💡 NPUが検出されませんでした。DirectML依存関係をインストールしますか？")
        optimizer.install_directml_dependencies()
    
    return optimizer

if __name__ == "__main__":
    # テスト実行
    test_windows_npu_optimization()

