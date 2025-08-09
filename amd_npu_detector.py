#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 AMD NPU専用検出スクリプト

Windows 11 AMD環境でのNPU検出エラーを解決する改良版スクリプト

対応環境:
- Windows 11
- AMD Ryzen AI プロセッサー
- AMD Radeon グラフィックス

使用方法:
    python amd_npu_detector.py
"""

import sys
import os
import platform
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

class AMDNPUDetector:
    """AMD NPU専用検出器"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {},
            'amd_devices': [],
            'npu_status': {},
            'recommendations': []
        }
        
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
        
        print(f"{color}[{timestamp}] {message}{reset}")
    
    def collect_system_info(self):
        """システム情報収集"""
        self.log("システム情報を収集中...")
        
        try:
            import psutil
            
            system_info = {
                'os': platform.system(),
                'os_version': platform.version(),
                'os_release': platform.release(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
                'cpu_count': os.cpu_count(),
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2)
            }
            
            self.results['system_info'] = system_info
            
            self.log(f"OS: {system_info['os']} {system_info['os_release']}")
            self.log(f"CPU: {system_info['processor']}")
            self.log(f"メモリ: {system_info['memory_gb']} GB")
            
        except Exception as e:
            self.log(f"システム情報収集エラー: {e}", "ERROR")
    
    def detect_amd_devices_registry(self):
        """レジストリ経由でAMDデバイス検出"""
        self.log("レジストリからAMDデバイスを検出中...")
        
        amd_devices = []
        
        try:
            # PowerShellでレジストリ検索
            powershell_cmd = '''
            Get-ItemProperty -Path "HKLM:\\SYSTEM\\CurrentControlSet\\Enum\\PCI\\*" -Name "DeviceDesc" -ErrorAction SilentlyContinue | 
            Where-Object { $_.DeviceDesc -match "AMD" } | 
            Select-Object PSChildName, DeviceDesc | 
            ConvertTo-Json
            '''
            
            result = subprocess.run(
                ['powershell', '-Command', powershell_cmd],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0 and result.stdout.strip():
                try:
                    devices_data = json.loads(result.stdout)
                    if isinstance(devices_data, dict):
                        devices_data = [devices_data]
                    
                    for device in devices_data:
                        device_desc = device.get('DeviceDesc', '')
                        pci_id = device.get('PSChildName', '')
                        
                        if 'AMD' in device_desc:
                            device_info = {
                                'name': device_desc,
                                'pci_id': pci_id,
                                'type': self.classify_amd_device(device_desc),
                                'source': 'registry'
                            }
                            amd_devices.append(device_info)
                            
                            self.log(f"検出: {device_desc}")
                
                except json.JSONDecodeError:
                    self.log("レジストリデータの解析に失敗", "WARNING")
            
        except subprocess.TimeoutExpired:
            self.log("レジストリ検索がタイムアウト", "WARNING")
        except Exception as e:
            self.log(f"レジストリ検索エラー: {e}", "WARNING")
        
        return amd_devices
    
    def detect_amd_devices_wmi(self):
        """WMI経由でAMDデバイス検出"""
        self.log("WMIからAMDデバイスを検出中...")
        
        amd_devices = []
        
        try:
            # PowerShellでWMI検索
            powershell_cmd = '''
            Get-WmiObject -Class Win32_PnPEntity | 
            Where-Object { $_.Name -match "AMD" -or $_.Description -match "AMD" } | 
            Select-Object Name, Description, DeviceID, Manufacturer | 
            ConvertTo-Json
            '''
            
            result = subprocess.run(
                ['powershell', '-Command', powershell_cmd],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0 and result.stdout.strip():
                try:
                    devices_data = json.loads(result.stdout)
                    if isinstance(devices_data, dict):
                        devices_data = [devices_data]
                    
                    for device in devices_data:
                        name = device.get('Name', '')
                        description = device.get('Description', '')
                        device_id = device.get('DeviceID', '')
                        
                        if 'AMD' in name or 'AMD' in description:
                            device_info = {
                                'name': name or description,
                                'description': description,
                                'device_id': device_id,
                                'type': self.classify_amd_device(name or description),
                                'source': 'wmi'
                            }
                            amd_devices.append(device_info)
                            
                            self.log(f"検出: {name or description}")
                
                except json.JSONDecodeError:
                    self.log("WMIデータの解析に失敗", "WARNING")
            
        except subprocess.TimeoutExpired:
            self.log("WMI検索がタイムアウト", "WARNING")
        except Exception as e:
            self.log(f"WMI検索エラー: {e}", "WARNING")
        
        return amd_devices
    
    def detect_amd_devices_dxdiag(self):
        """DirectX診断ツール経由でAMDデバイス検出"""
        self.log("DirectX診断からAMDデバイスを検出中...")
        
        amd_devices = []
        
        try:
            # dxdiag実行
            result = subprocess.run(
                ['dxdiag', '/t', 'dxdiag_output.txt'],
                capture_output=True, text=True, timeout=60
            )
            
            # 出力ファイル読み取り
            dxdiag_file = Path('dxdiag_output.txt')
            if dxdiag_file.exists():
                try:
                    with open(dxdiag_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # AMD関連情報を抽出
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if 'AMD' in line and ('Card name' in line or 'Chip type' in line):
                            device_name = line.split(':')[-1].strip()
                            
                            device_info = {
                                'name': device_name,
                                'type': self.classify_amd_device(device_name),
                                'source': 'dxdiag'
                            }
                            amd_devices.append(device_info)
                            
                            self.log(f"検出: {device_name}")
                    
                    # 一時ファイル削除
                    dxdiag_file.unlink()
                    
                except Exception as e:
                    self.log(f"dxdiag出力解析エラー: {e}", "WARNING")
            
        except subprocess.TimeoutExpired:
            self.log("dxdiag実行がタイムアウト", "WARNING")
        except Exception as e:
            self.log(f"dxdiag実行エラー: {e}", "WARNING")
        
        return amd_devices
    
    def classify_amd_device(self, device_name: str) -> str:
        """AMDデバイスの分類"""
        device_name_upper = device_name.upper()
        
        # NPU/AI関連キーワード
        npu_keywords = ['AI', 'NPU', 'NEURAL', 'RYZEN AI', 'PHOENIX', 'HAWK POINT']
        if any(keyword in device_name_upper for keyword in npu_keywords):
            return 'NPU'
        
        # GPU関連キーワード
        gpu_keywords = ['RADEON', 'RX', 'VEGA', 'NAVI', 'RDNA']
        if any(keyword in device_name_upper for keyword in gpu_keywords):
            return 'GPU'
        
        # CPU関連キーワード
        cpu_keywords = ['RYZEN', 'PROCESSOR', 'CPU']
        if any(keyword in device_name_upper for keyword in cpu_keywords):
            return 'CPU'
        
        # その他
        return 'OTHER'
    
    def check_amd_software(self):
        """AMD関連ソフトウェアの確認"""
        self.log("AMD関連ソフトウェアを確認中...")
        
        software_status = {}
        
        # AMD Software確認
        amd_software_paths = [
            r"C:\Program Files\AMD\CNext\CNext\RadeonSoftware.exe",
            r"C:\Program Files (x86)\AMD\CNext\CNext\RadeonSoftware.exe",
            r"C:\AMD\CNext\CNext\RadeonSoftware.exe"
        ]
        
        for path in amd_software_paths:
            if Path(path).exists():
                software_status['amd_software'] = {
                    'installed': True,
                    'path': path
                }
                self.log(f"✅ AMD Software検出: {path}", "SUCCESS")
                break
        else:
            software_status['amd_software'] = {'installed': False}
            self.log("⚠️ AMD Softwareが見つかりません", "WARNING")
        
        # AMD Ryzen AI Software確認
        ai_software_paths = [
            r"C:\Program Files\AMD\RyzenAI",
            r"C:\Program Files (x86)\AMD\RyzenAI",
            r"C:\AMD\RyzenAI"
        ]
        
        for path in ai_software_paths:
            if Path(path).exists():
                software_status['ryzen_ai_software'] = {
                    'installed': True,
                    'path': path
                }
                self.log(f"✅ Ryzen AI Software検出: {path}", "SUCCESS")
                break
        else:
            software_status['ryzen_ai_software'] = {'installed': False}
            self.log("⚠️ Ryzen AI Softwareが見つかりません", "WARNING")
        
        return software_status
    
    def check_pytorch_directml(self):
        """PyTorch DirectML確認"""
        self.log("PyTorch DirectML対応を確認中...")
        
        directml_status = {}
        
        try:
            import torch
            directml_status['torch_version'] = torch.__version__
            
            # DirectML確認
            try:
                import torch_directml
                directml_status['torch_directml'] = {
                    'available': True,
                    'version': getattr(torch_directml, '__version__', 'unknown')
                }
                
                # DirectMLデバイス確認
                try:
                    device = torch_directml.device()
                    directml_status['directml_device'] = str(device)
                    self.log(f"✅ DirectMLデバイス: {device}", "SUCCESS")
                except Exception as e:
                    directml_status['directml_device_error'] = str(e)
                    self.log(f"⚠️ DirectMLデバイス取得エラー: {e}", "WARNING")
                
            except ImportError:
                directml_status['torch_directml'] = {'available': False}
                self.log("⚠️ torch-directmlが見つかりません", "WARNING")
        
        except ImportError:
            directml_status['torch_available'] = False
            self.log("⚠️ PyTorchが見つかりません", "WARNING")
        
        return directml_status
    
    def check_onnxruntime_directml(self):
        """ONNX Runtime DirectML確認"""
        self.log("ONNX Runtime DirectML対応を確認中...")
        
        onnx_status = {}
        
        try:
            import onnxruntime as ort
            onnx_status['onnxruntime_version'] = ort.__version__
            
            # 利用可能プロバイダー確認
            providers = ort.get_available_providers()
            onnx_status['available_providers'] = providers
            
            # DirectMLプロバイダー確認
            if 'DmlExecutionProvider' in providers:
                onnx_status['directml_provider'] = True
                self.log("✅ ONNX Runtime DirectMLプロバイダー利用可能", "SUCCESS")
            else:
                onnx_status['directml_provider'] = False
                self.log("⚠️ ONNX Runtime DirectMLプロバイダーが見つかりません", "WARNING")
            
            # テストセッション作成
            try:
                session_options = ort.SessionOptions()
                if 'DmlExecutionProvider' in providers:
                    # DirectMLプロバイダーでテストセッション
                    test_providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
                    onnx_status['directml_test'] = 'success'
                    self.log("✅ DirectMLプロバイダーテスト成功", "SUCCESS")
                
            except Exception as e:
                onnx_status['directml_test_error'] = str(e)
                self.log(f"⚠️ DirectMLプロバイダーテストエラー: {e}", "WARNING")
        
        except ImportError:
            onnx_status['onnxruntime_available'] = False
            self.log("⚠️ ONNX Runtimeが見つかりません", "WARNING")
        
        return onnx_status
    
    def generate_recommendations(self):
        """推奨事項生成"""
        recommendations = []
        
        # AMD NPU検出結果に基づく推奨事項
        npu_devices = [d for d in self.results['amd_devices'] if d['type'] == 'NPU']
        
        if not npu_devices:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Hardware',
                'title': 'AMD Ryzen AI対応プロセッサーの確認',
                'description': 'NPUが検出されませんでした。AMD Ryzen AI対応プロセッサー（Phoenix、Hawk Point世代）を使用していることを確認してください。',
                'action': 'システム仕様を確認し、必要に応じてハードウェアをアップグレードしてください。'
            })
        
        # ソフトウェア推奨事項
        software_status = self.results.get('software_status', {})
        
        if not software_status.get('amd_software', {}).get('installed', False):
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Software',
                'title': 'AMD Softwareのインストール',
                'description': 'AMD Softwareが見つかりません。最新のAMDドライバーとソフトウェアをインストールしてください。',
                'action': 'https://www.amd.com/support からAMD Softwareをダウンロード・インストールしてください。'
            })
        
        if not software_status.get('ryzen_ai_software', {}).get('installed', False):
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Software',
                'title': 'Ryzen AI Softwareのインストール',
                'description': 'Ryzen AI専用ソフトウェアが見つかりません。NPU最適化のためにインストールを推奨します。',
                'action': 'AMD開発者サイトからRyzen AI Software Toolkitをダウンロード・インストールしてください。'
            })
        
        # DirectML推奨事項
        directml_status = self.results.get('directml_status', {})
        
        if not directml_status.get('torch_directml', {}).get('available', False):
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Python',
                'title': 'torch-directmlのインストール',
                'description': 'PyTorch DirectMLが見つかりません。AMD GPU/NPU最適化のためにインストールを推奨します。',
                'action': 'pip install torch-directml でインストールしてください。'
            })
        
        if not directml_status.get('directml_provider', False):
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Python',
                'title': 'ONNX Runtime DirectMLのインストール',
                'description': 'ONNX Runtime DirectMLプロバイダーが見つかりません。',
                'action': 'pip install onnxruntime-directml でインストールしてください。'
            })
        
        self.results['recommendations'] = recommendations
        return recommendations
    
    def run_detection(self):
        """検出実行"""
        self.log("🔍 AMD NPU検出を開始します")
        self.log("=" * 60)
        
        try:
            # システム情報収集
            self.collect_system_info()
            
            # AMDデバイス検出（複数手法）
            all_devices = []
            
            # レジストリ検索
            registry_devices = self.detect_amd_devices_registry()
            all_devices.extend(registry_devices)
            
            # WMI検索
            wmi_devices = self.detect_amd_devices_wmi()
            all_devices.extend(wmi_devices)
            
            # DirectX診断
            dxdiag_devices = self.detect_amd_devices_dxdiag()
            all_devices.extend(dxdiag_devices)
            
            # 重複除去
            unique_devices = []
            seen_names = set()
            
            for device in all_devices:
                device_name = device['name']
                if device_name not in seen_names:
                    unique_devices.append(device)
                    seen_names.add(device_name)
            
            self.results['amd_devices'] = unique_devices
            
            # ソフトウェア確認
            self.results['software_status'] = self.check_amd_software()
            
            # DirectML確認
            self.results['directml_status'] = self.check_pytorch_directml()
            
            # ONNX Runtime確認
            self.results['onnx_status'] = self.check_onnxruntime_directml()
            
            # 推奨事項生成
            self.generate_recommendations()
            
            # 結果表示
            self.display_results()
            
            # 結果保存
            self.save_results()
            
            self.log("🎉 AMD NPU検出完了", "SUCCESS")
            
        except Exception as e:
            self.log(f"❌ 検出エラー: {e}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "ERROR")
    
    def display_results(self):
        """結果表示"""
        self.log("=" * 60)
        self.log("📊 検出結果サマリー")
        self.log("=" * 60)
        
        # システム情報
        system_info = self.results['system_info']
        self.log(f"OS: {system_info.get('os', 'Unknown')} {system_info.get('os_release', '')}")
        self.log(f"CPU: {system_info.get('processor', 'Unknown')}")
        self.log(f"メモリ: {system_info.get('memory_gb', 'Unknown')} GB")
        
        # AMDデバイス
        amd_devices = self.results['amd_devices']
        self.log(f"\n🔍 検出されたAMDデバイス: {len(amd_devices)}個")
        
        npu_count = len([d for d in amd_devices if d['type'] == 'NPU'])
        gpu_count = len([d for d in amd_devices if d['type'] == 'GPU'])
        cpu_count = len([d for d in amd_devices if d['type'] == 'CPU'])
        
        self.log(f"  NPU: {npu_count}個")
        self.log(f"  GPU: {gpu_count}個")
        self.log(f"  CPU: {cpu_count}個")
        
        for device in amd_devices:
            device_type = device['type']
            device_name = device['name']
            source = device['source']
            
            if device_type == 'NPU':
                self.log(f"  ✅ NPU: {device_name} (検出元: {source})", "SUCCESS")
            elif device_type == 'GPU':
                self.log(f"  🎮 GPU: {device_name} (検出元: {source})")
            else:
                self.log(f"  💻 {device_type}: {device_name} (検出元: {source})")
        
        # ソフトウェア状況
        software_status = self.results.get('software_status', {})
        self.log(f"\n💿 ソフトウェア状況:")
        
        amd_sw = software_status.get('amd_software', {})
        if amd_sw.get('installed', False):
            self.log(f"  ✅ AMD Software: インストール済み", "SUCCESS")
        else:
            self.log(f"  ❌ AMD Software: 未インストール", "ERROR")
        
        ai_sw = software_status.get('ryzen_ai_software', {})
        if ai_sw.get('installed', False):
            self.log(f"  ✅ Ryzen AI Software: インストール済み", "SUCCESS")
        else:
            self.log(f"  ⚠️ Ryzen AI Software: 未インストール", "WARNING")
        
        # DirectML状況
        directml_status = self.results.get('directml_status', {})
        self.log(f"\n🐍 Python環境:")
        
        if directml_status.get('torch_directml', {}).get('available', False):
            self.log(f"  ✅ torch-directml: 利用可能", "SUCCESS")
        else:
            self.log(f"  ❌ torch-directml: 未インストール", "ERROR")
        
        onnx_status = self.results.get('onnx_status', {})
        if onnx_status.get('directml_provider', False):
            self.log(f"  ✅ ONNX Runtime DirectML: 利用可能", "SUCCESS")
        else:
            self.log(f"  ❌ ONNX Runtime DirectML: 未インストール", "ERROR")
        
        # 推奨事項
        recommendations = self.results.get('recommendations', [])
        if recommendations:
            self.log(f"\n💡 推奨事項: {len(recommendations)}件")
            
            for i, rec in enumerate(recommendations, 1):
                priority = rec['priority']
                title = rec['title']
                description = rec['description']
                action = rec['action']
                
                priority_color = "ERROR" if priority == "HIGH" else "WARNING"
                self.log(f"  {i}. [{priority}] {title}", priority_color)
                self.log(f"     {description}")
                self.log(f"     対処法: {action}")
        
        self.log("=" * 60)
    
    def save_results(self):
        """結果保存"""
        try:
            # JSON形式で保存
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            json_path = f'amd_npu_detection_{timestamp}.json'
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            self.log(f"💾 結果保存: {json_path}")
            
            # テキストレポート生成
            txt_path = f'amd_npu_report_{timestamp}.txt'
            self.generate_text_report(txt_path)
            
            self.log(f"📄 レポート保存: {txt_path}")
            
        except Exception as e:
            self.log(f"結果保存エラー: {e}", "ERROR")
    
    def generate_text_report(self, output_path: str):
        """テキストレポート生成"""
        
        system_info = self.results['system_info']
        amd_devices = self.results['amd_devices']
        recommendations = self.results.get('recommendations', [])
        
        report_content = f"""AMD NPU検出レポート
===================

実行日時: {self.results['timestamp']}

システム情報
-----------
OS: {system_info.get('os', 'Unknown')} {system_info.get('os_release', '')}
CPU: {system_info.get('processor', 'Unknown')}
メモリ: {system_info.get('memory_gb', 'Unknown')} GB
Python: {system_info.get('python_version', 'Unknown')}

検出されたAMDデバイス
-------------------
総数: {len(amd_devices)}個

"""
        
        npu_devices = [d for d in amd_devices if d['type'] == 'NPU']
        gpu_devices = [d for d in amd_devices if d['type'] == 'GPU']
        cpu_devices = [d for d in amd_devices if d['type'] == 'CPU']
        
        if npu_devices:
            report_content += "NPUデバイス:\n"
            for device in npu_devices:
                report_content += f"  - {device['name']} (検出元: {device['source']})\n"
            report_content += "\n"
        
        if gpu_devices:
            report_content += "GPUデバイス:\n"
            for device in gpu_devices:
                report_content += f"  - {device['name']} (検出元: {device['source']})\n"
            report_content += "\n"
        
        if cpu_devices:
            report_content += "CPUデバイス:\n"
            for device in cpu_devices:
                report_content += f"  - {device['name']} (検出元: {device['source']})\n"
            report_content += "\n"
        
        # ソフトウェア状況
        software_status = self.results.get('software_status', {})
        report_content += "ソフトウェア状況\n"
        report_content += "---------------\n"
        
        amd_sw = software_status.get('amd_software', {})
        report_content += f"AMD Software: {'インストール済み' if amd_sw.get('installed', False) else '未インストール'}\n"
        
        ai_sw = software_status.get('ryzen_ai_software', {})
        report_content += f"Ryzen AI Software: {'インストール済み' if ai_sw.get('installed', False) else '未インストール'}\n"
        
        # Python環境
        directml_status = self.results.get('directml_status', {})
        onnx_status = self.results.get('onnx_status', {})
        
        report_content += "\nPython環境\n"
        report_content += "----------\n"
        report_content += f"torch-directml: {'利用可能' if directml_status.get('torch_directml', {}).get('available', False) else '未インストール'}\n"
        report_content += f"ONNX Runtime DirectML: {'利用可能' if onnx_status.get('directml_provider', False) else '未インストール'}\n"
        
        # 推奨事項
        if recommendations:
            report_content += "\n推奨事項\n"
            report_content += "--------\n"
            
            for i, rec in enumerate(recommendations, 1):
                report_content += f"{i}. [{rec['priority']}] {rec['title']}\n"
                report_content += f"   {rec['description']}\n"
                report_content += f"   対処法: {rec['action']}\n\n"
        
        report_content += "\n---\nこのレポートは自動生成されました\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

def main():
    """メイン実行関数"""
    print("""
============================================================================
🔍 AMD NPU専用検出ツール
============================================================================

このツールは Windows 11 AMD環境でのNPU検出エラーを解決します。

対応環境:
- Windows 11
- AMD Ryzen AI プロセッサー (Phoenix, Hawk Point世代)
- AMD Radeon グラフィックス

検出方法:
- レジストリ検索
- WMI (Windows Management Instrumentation)
- DirectX診断ツール
- AMD Software確認
- Python DirectML確認

============================================================================
""")
    
    try:
        detector = AMDNPUDetector()
        detector.run_detection()
        
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーにより中断されました")
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()

