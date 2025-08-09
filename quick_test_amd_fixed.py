#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Infer-OS AMD NPU対応クイックテスト (修正版)

Windows 11 AMD環境でのNPU検出エラーを修正したクイックテストスクリプト

修正内容:
- AMD NPU専用検出ロジック追加
- PowerShell/WMI/レジストリ検索の実装
- DirectML対応確認
- エラーハンドリング強化

使用方法:
    python quick_test_amd_fixed.py
"""

import sys
import platform
import time
import traceback
import subprocess
import json
from datetime import datetime
from pathlib import Path

def print_header():
    """ヘッダー表示"""
    print("=" * 70)
    print("🚀 Infer-OS AMD NPU対応クイックテスト (修正版)")
    print("=" * 70)
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {platform.python_version()}")
    print(f"OS: {platform.system()} {platform.release()}")
    print("=" * 70)

def test_basic_environment():
    """基本環境テスト"""
    print("\n📋 1. 基本環境テスト")
    print("-" * 30)
    
    tests = []
    
    # Python バージョンチェック
    python_version = platform.python_version()
    if python_version.startswith('3.'):
        major, minor = map(int, python_version.split('.')[:2])
        if major == 3 and 9 <= minor <= 11:
            print(f"✅ Python バージョン: {python_version} (対応)")
            tests.append(True)
        else:
            print(f"⚠️ Python バージョン: {python_version} (推奨: 3.9-3.11)")
            tests.append(False)
    else:
        print(f"❌ Python バージョン: {python_version} (非対応)")
        tests.append(False)
    
    # OS チェック
    if platform.system() == 'Windows':
        print(f"✅ OS: Windows {platform.release()}")
        tests.append(True)
    else:
        print(f"⚠️ OS: {platform.system()} (Windows推奨)")
        tests.append(False)
    
    # メモリチェック
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb >= 16:
            print(f"✅ メモリ: {memory_gb:.1f}GB (十分)")
        elif memory_gb >= 8:
            print(f"⚠️ メモリ: {memory_gb:.1f}GB (最低限)")
        else:
            print(f"❌ メモリ: {memory_gb:.1f}GB (不足)")
        tests.append(memory_gb >= 8)
    except ImportError:
        print("⚠️ メモリ: 確認できません (psutil未インストール)")
        tests.append(False)
    
    return all(tests)

def test_required_packages():
    """必須パッケージテスト"""
    print("\n📦 2. 必須パッケージテスト")
    print("-" * 30)
    
    required_packages = [
        ('numpy', 'NumPy'),
        ('torch', 'PyTorch'),
        ('onnxruntime', 'ONNX Runtime'),
        ('flask', 'Flask'),
        ('requests', 'Requests'),
        ('psutil', 'psutil')
    ]
    
    results = []
    
    for package_name, display_name in required_packages:
        try:
            module = __import__(package_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name}: v{version}")
            results.append(True)
        except ImportError:
            print(f"❌ {display_name}: 未インストール")
            results.append(False)
    
    return all(results)

def detect_amd_devices_powershell():
    """PowerShell経由でAMDデバイス検出"""
    print("\n🔍 3. AMD NPU/GPU検出テスト (PowerShell)")
    print("-" * 30)
    
    detected_devices = []
    
    try:
        # PowerShellでWMI検索
        powershell_cmd = '''
        Get-WmiObject -Class Win32_PnPEntity | 
        Where-Object { $_.Name -match "AMD" -or $_.Description -match "AMD" } | 
        Select-Object Name, Description, DeviceID | 
        ConvertTo-Json
        '''
        
        print("PowerShellでAMDデバイスを検索中...")
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
                            'type': classify_amd_device(name or description)
                        }
                        detected_devices.append(device_info)
                        
                        # デバイスタイプに応じた表示
                        device_type = device_info['type']
                        if device_type == 'NPU':
                            print(f"✅ NPU検出: {name or description}")
                        elif device_type == 'GPU':
                            print(f"🎮 GPU検出: {name or description}")
                        else:
                            print(f"💻 AMD デバイス: {name or description}")
            
            except json.JSONDecodeError:
                print("⚠️ PowerShellデータの解析に失敗")
        
        else:
            print("⚠️ PowerShell検索でエラーが発生")
            if result.stderr:
                print(f"エラー詳細: {result.stderr}")
    
    except subprocess.TimeoutExpired:
        print("⚠️ PowerShell検索がタイムアウト")
    except FileNotFoundError:
        print("❌ PowerShellが見つかりません")
    except Exception as e:
        print(f"❌ PowerShell検索エラー: {e}")
    
    return detected_devices

def detect_amd_devices_registry():
    """レジストリ経由でAMDデバイス検出"""
    print("\n🔍 3-2. AMD NPU/GPU検出テスト (レジストリ)")
    print("-" * 30)
    
    detected_devices = []
    
    try:
        # PowerShellでレジストリ検索
        powershell_cmd = '''
        Get-ItemProperty -Path "HKLM:\\SYSTEM\\CurrentControlSet\\Enum\\PCI\\*" -Name "DeviceDesc" -ErrorAction SilentlyContinue | 
        Where-Object { $_.DeviceDesc -match "AMD" } | 
        Select-Object PSChildName, DeviceDesc | 
        ConvertTo-Json
        '''
        
        print("レジストリからAMDデバイスを検索中...")
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
                            'type': classify_amd_device(device_desc),
                            'source': 'registry'
                        }
                        detected_devices.append(device_info)
                        
                        # デバイスタイプに応じた表示
                        device_type = device_info['type']
                        if device_type == 'NPU':
                            print(f"✅ NPU検出: {device_desc}")
                        elif device_type == 'GPU':
                            print(f"🎮 GPU検出: {device_desc}")
                        else:
                            print(f"💻 AMD デバイス: {device_desc}")
            
            except json.JSONDecodeError:
                print("⚠️ レジストリデータの解析に失敗")
        
        else:
            print("⚠️ レジストリ検索でエラーが発生")
    
    except subprocess.TimeoutExpired:
        print("⚠️ レジストリ検索がタイムアウト")
    except Exception as e:
        print(f"❌ レジストリ検索エラー: {e}")
    
    return detected_devices

def classify_amd_device(device_name: str) -> str:
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

def test_directml_support():
    """DirectML対応テスト"""
    print("\n🐍 4. DirectML対応テスト")
    print("-" * 30)
    
    directml_available = False
    
    # PyTorch DirectML確認
    try:
        import torch
        print(f"✅ PyTorch: v{torch.__version__}")
        
        try:
            import torch_directml
            print(f"✅ torch-directml: 利用可能")
            
            # DirectMLデバイス確認
            try:
                device = torch_directml.device()
                print(f"✅ DirectMLデバイス: {device}")
                directml_available = True
            except Exception as e:
                print(f"⚠️ DirectMLデバイス取得エラー: {e}")
        
        except ImportError:
            print("❌ torch-directml: 未インストール")
            print("   インストール: pip install torch-directml")
    
    except ImportError:
        print("❌ PyTorch: 未インストール")
    
    # ONNX Runtime DirectML確認
    try:
        import onnxruntime as ort
        print(f"✅ ONNX Runtime: v{ort.__version__}")
        
        providers = ort.get_available_providers()
        print(f"利用可能プロバイダー: {len(providers)}個")
        
        if 'DmlExecutionProvider' in providers:
            print("✅ DirectMLプロバイダー: 利用可能")
            directml_available = True
        else:
            print("❌ DirectMLプロバイダー: 未対応")
            print("   インストール: pip install onnxruntime-directml")
        
    except ImportError:
        print("❌ ONNX Runtime: 未インストール")
    
    return directml_available

def test_basic_computation():
    """基本計算テスト"""
    print("\n🧮 5. 基本計算テスト")
    print("-" * 30)
    
    try:
        import numpy as np
        import time
        
        # NumPy行列演算テスト
        print("NumPy行列演算テスト実行中...")
        size = 500
        
        start_time = time.time()
        a = np.random.randn(size, size)
        b = np.random.randn(size, size)
        c = np.dot(a, b)
        numpy_time = time.time() - start_time
        
        print(f"✅ NumPy ({size}x{size}): {numpy_time:.3f}秒")
        
        # PyTorch テンソル演算テスト
        try:
            import torch
            
            print("PyTorch テンソル演算テスト実行中...")
            
            start_time = time.time()
            x = torch.randn(size, size)
            y = torch.randn(size, size)
            z = torch.mm(x, y)
            torch_time = time.time() - start_time
            
            print(f"✅ PyTorch CPU ({size}x{size}): {torch_time:.3f}秒")
            
            # DirectML利用可能な場合
            try:
                import torch_directml
                device = torch_directml.device()
                
                start_time = time.time()
                x_dml = torch.randn(size, size, device=device)
                y_dml = torch.randn(size, size, device=device)
                z_dml = torch.mm(x_dml, y_dml)
                # DirectMLの同期
                z_dml_cpu = z_dml.cpu()
                directml_time = time.time() - start_time
                
                speedup = torch_time / directml_time
                print(f"✅ PyTorch DirectML ({size}x{size}): {directml_time:.3f}秒 (高速化: {speedup:.1f}x)")
                
            except ImportError:
                print("ℹ️ torch-directml: 未インストール")
            except Exception as e:
                print(f"⚠️ DirectMLテストエラー: {e}")
            
        except ImportError:
            print("⚠️ PyTorch: 未インストール")
        
        return True
        
    except ImportError:
        print("❌ NumPy: 未インストール")
        return False
    except Exception as e:
        print(f"❌ 計算テストエラー: {e}")
        return False

def test_infer_os_basic():
    """Infer-OS基本テスト"""
    print("\n🔧 6. Infer-OS基本テスト")
    print("-" * 30)
    
    # srcディレクトリをパスに追加
    import os
    src_path = os.path.join(os.getcwd(), 'src')
    if os.path.exists(src_path):
        sys.path.insert(0, src_path)
        print(f"✅ srcディレクトリ検出: {src_path}")
    else:
        print(f"⚠️ srcディレクトリ未検出: {src_path}")
        return False
    
    # 各コンポーネントの基本インポートテスト
    components = [
        ('runtime.enhanced_iobinding', 'EnhancedIOBinding', 'IOBinding最適化'),
        ('optim.kv_quantization', 'KVQuantizationManager', 'KV量子化'),
        ('optim.speculative_generation', 'SpeculativeGenerationEngine', 'スペキュレイティブ生成'),
        ('optim.gpu_npu_pipeline', 'GPUNPUPipeline', 'GPU-NPUパイプライン')
    ]
    
    results = []
    
    for module_name, class_name, display_name in components:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            
            # 基本初期化テスト
            instance = cls()
            print(f"✅ {display_name}: 初期化成功")
            results.append(True)
            
        except ImportError as e:
            print(f"❌ {display_name}: インポートエラー ({e})")
            results.append(False)
        except Exception as e:
            print(f"⚠️ {display_name}: 初期化エラー ({e})")
            results.append(False)
    
    return any(results)  # 少なくとも1つ成功すればOK

def generate_summary(test_results, amd_devices):
    """結果サマリー生成"""
    print("\n" + "=" * 70)
    print("📊 テスト結果サマリー")
    print("=" * 70)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    print(f"総テスト数: {total_tests}")
    print(f"成功: {passed_tests}")
    print(f"失敗: {total_tests - passed_tests}")
    print(f"成功率: {passed_tests/total_tests*100:.1f}%")
    
    print("\n📋 詳細結果:")
    test_names = [
        "基本環境",
        "必須パッケージ", 
        "AMD NPU/GPU検出",
        "DirectML対応",
        "基本計算",
        "Infer-OS基本"
    ]
    
    for i, (test_name, result) in enumerate(zip(test_names, test_results.values())):
        status = "✅ 成功" if result else "❌ 失敗"
        print(f"{i+1}. {test_name}: {status}")
    
    # AMD デバイス詳細
    print(f"\n🔍 検出されたAMDデバイス: {len(amd_devices)}個")
    npu_devices = [d for d in amd_devices if d['type'] == 'NPU']
    gpu_devices = [d for d in amd_devices if d['type'] == 'GPU']
    
    if npu_devices:
        print("NPUデバイス:")
        for device in npu_devices:
            print(f"  ✅ {device['name']}")
    
    if gpu_devices:
        print("GPUデバイス:")
        for device in gpu_devices:
            print(f"  🎮 {device['name']}")
    
    print("\n🎯 推奨事項:")
    
    if not test_results['basic_env']:
        print("- Python 3.9-3.11の使用を推奨します")
        print("- 16GB以上のメモリを推奨します")
    
    if not test_results['packages']:
        print("- 不足パッケージをインストールしてください:")
        print("  pip install torch onnxruntime numpy flask requests psutil")
    
    if not test_results['amd_detection']:
        print("- AMD NPU/GPUドライバーを最新版に更新してください")
        print("- AMD Software: https://www.amd.com/support")
    
    if not test_results['directml']:
        print("- DirectML対応パッケージをインストールしてください:")
        print("  pip install torch-directml onnxruntime-directml")
    
    if not test_results['infer_os']:
        print("- Infer-OSプロジェクトが正しくクローンされているか確認してください")
        print("- 依存関係を再インストールしてください")
    
    if passed_tests == total_tests:
        print("🎉 全てのテストが成功しました！")
        print("次のステップ:")
        print("1. python amd_npu_detector.py  # 詳細AMD NPU検出")
        print("2. python infer_os_npu_test.py  # 統合テスト")
        print("3. python infer-os-demo/src/main.py  # Webデモ起動")
    elif passed_tests >= total_tests * 0.7:
        print("⚠️ 基本的な動作は可能ですが、一部問題があります")
        print("詳細テストで問題を特定してください:")
        print("python amd_npu_detector.py")
    else:
        print("❌ 重要な問題があります")
        print("AMD NPU専用検出ツールを実行してください:")
        print("python amd_npu_detector.py")
    
    print("=" * 70)

def main():
    """メイン実行関数"""
    print_header()
    
    # 各テスト実行
    test_results = {}
    all_amd_devices = []
    
    try:
        test_results['basic_env'] = test_basic_environment()
        test_results['packages'] = test_required_packages()
        
        # AMD デバイス検出（複数手法）
        powershell_devices = detect_amd_devices_powershell()
        registry_devices = detect_amd_devices_registry()
        
        # 重複除去
        all_amd_devices = powershell_devices + registry_devices
        unique_devices = []
        seen_names = set()
        
        for device in all_amd_devices:
            device_name = device['name']
            if device_name not in seen_names:
                unique_devices.append(device)
                seen_names.add(device_name)
        
        all_amd_devices = unique_devices
        test_results['amd_detection'] = len(all_amd_devices) > 0
        
        test_results['directml'] = test_directml_support()
        test_results['computation'] = test_basic_computation()
        test_results['infer_os'] = test_infer_os_basic()
        
    except KeyboardInterrupt:
        print("\n⚠️ テストが中断されました")
        return
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        print(traceback.format_exc())
        return
    
    # 結果サマリー
    generate_summary(test_results, all_amd_devices)
    
    # 結果保存
    try:
        import json
        result_data = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'python_version': platform.python_version(),
                'os': f"{platform.system()} {platform.release()}",
                'machine': platform.machine()
            },
            'test_results': test_results,
            'amd_devices': all_amd_devices
        }
        
        with open('quick_test_amd_results.json', 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"\n💾 結果保存: quick_test_amd_results.json")
        
    except Exception as e:
        print(f"\n⚠️ 結果保存エラー: {e}")

if __name__ == "__main__":
    main()

