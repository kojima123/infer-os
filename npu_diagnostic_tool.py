#!/usr/bin/env python3
"""
NPU状態診断ツール
再起動後のNPU問題を診断し、復旧方法を提案する
"""

import sys
import subprocess
import json
import time
from typing import Dict, List, Any

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class NPUDiagnosticTool:
    def __init__(self):
        self.results = {}
        
    def run_full_diagnosis(self) -> Dict[str, Any]:
        """完全なNPU診断を実行"""
        print("🔍 NPU状態診断ツール開始")
        print("=" * 60)
        
        # 1. システムレベル診断
        print("\n📋 1. システムレベル診断")
        self.results['system'] = self.diagnose_system_level()
        
        # 2. DirectML診断
        print("\n🔧 2. DirectML診断")
        self.results['directml'] = self.diagnose_directml()
        
        # 3. ONNX Runtime診断
        print("\n⚙️ 3. ONNX Runtime診断")
        self.results['onnxruntime'] = self.diagnose_onnxruntime()
        
        # 4. デバイス一覧診断
        print("\n💻 4. デバイス一覧診断")
        self.results['devices'] = self.diagnose_devices()
        
        # 5. 復旧提案
        print("\n🔧 5. 復旧提案")
        self.results['recovery'] = self.suggest_recovery()
        
        # 6. 診断結果サマリー
        print("\n📊 6. 診断結果サマリー")
        self.print_summary()
        
        return self.results
    
    def diagnose_system_level(self) -> Dict[str, Any]:
        """システムレベルのNPU診断"""
        result = {
            'npu_devices': [],
            'device_manager_status': 'unknown',
            'driver_status': 'unknown'
        }
        
        try:
            # デバイスマネージャー情報取得（PowerShell経由）
            print("  🔍 デバイスマネージャー確認中...")
            
            # NPUデバイス検索
            powershell_cmd = '''
            Get-PnpDevice | Where-Object {
                $_.FriendlyName -like "*NPU*" -or 
                $_.FriendlyName -like "*Neural*" -or
                $_.FriendlyName -like "*AI*" -or
                $_.HardwareID -like "*NPU*"
            } | Select-Object FriendlyName, Status, InstanceId | ConvertTo-Json
            '''
            
            try:
                proc = subprocess.run(
                    ['powershell', '-Command', powershell_cmd],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if proc.returncode == 0 and proc.stdout.strip():
                    devices = json.loads(proc.stdout)
                    if isinstance(devices, dict):
                        devices = [devices]
                    
                    result['npu_devices'] = devices
                    result['device_manager_status'] = 'success'
                    
                    print(f"    ✅ NPUデバイス発見: {len(devices)}個")
                    for device in devices:
                        print(f"      📱 {device.get('FriendlyName', 'Unknown')}: {device.get('Status', 'Unknown')}")
                else:
                    result['device_manager_status'] = 'no_devices'
                    print("    ⚠️ NPUデバイスが見つかりません")
                    
            except subprocess.TimeoutExpired:
                result['device_manager_status'] = 'timeout'
                print("    ❌ デバイスマネージャー確認タイムアウト")
            except json.JSONDecodeError:
                result['device_manager_status'] = 'parse_error'
                print("    ❌ デバイス情報の解析に失敗")
            except Exception as e:
                result['device_manager_status'] = f'error: {e}'
                print(f"    ❌ デバイスマネージャー確認エラー: {e}")
                
        except Exception as e:
            result['device_manager_status'] = f'system_error: {e}'
            print(f"    ❌ システムレベル診断エラー: {e}")
        
        return result
    
    def diagnose_directml(self) -> Dict[str, Any]:
        """DirectML診断"""
        result = {
            'available': False,
            'version': 'unknown',
            'devices': [],
            'error': None
        }
        
        try:
            print("  🔍 DirectML状態確認中...")
            
            # DirectMLの基本確認
            try:
                import onnxruntime as ort
                
                # DirectMLプロバイダーの利用可能性確認
                available_providers = ort.get_available_providers()
                
                if 'DmlExecutionProvider' in available_providers:
                    result['available'] = True
                    print("    ✅ DirectMLプロバイダー利用可能")
                    
                    # バージョン情報
                    result['version'] = ort.__version__
                    print(f"    📋 ONNX Runtime バージョン: {ort.__version__}")
                    
                    # デバイス情報取得試行
                    try:
                        # 簡単なセッション作成でデバイス確認
                        providers = [('DmlExecutionProvider', {'device_id': 0})]
                        
                        # ダミーモデルでテスト
                        import numpy as np
                        from onnx import helper, TensorProto
                        
                        # 最小限のONNXモデル作成
                        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1])
                        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1])
                        
                        identity_node = helper.make_node('Identity', inputs=['input'], outputs=['output'])
                        graph = helper.make_graph([identity_node], 'test', [input_tensor], [output_tensor])
                        model = helper.make_model(graph)
                        model.ir_version = 6
                        model.opset_import[0].version = 9
                        
                        session = ort.InferenceSession(
                            model.SerializeToString(),
                            providers=providers
                        )
                        
                        active_providers = session.get_providers()
                        result['devices'] = active_providers
                        
                        print(f"    📱 アクティブプロバイダー: {active_providers}")
                        
                        # テスト実行
                        test_input = np.array([[1.0]], dtype=np.float32)
                        test_output = session.run(['output'], {'input': test_input})
                        
                        print("    ✅ DirectMLテスト実行成功")
                        
                    except Exception as device_error:
                        result['error'] = f'device_test_failed: {device_error}'
                        print(f"    ⚠️ DirectMLデバイステスト失敗: {device_error}")
                        
                else:
                    result['available'] = False
                    print("    ❌ DirectMLプロバイダー利用不可")
                    print(f"    📋 利用可能プロバイダー: {available_providers}")
                    
            except ImportError:
                result['error'] = 'onnxruntime_not_available'
                print("    ❌ ONNX Runtime がインストールされていません")
            except Exception as e:
                result['error'] = f'directml_error: {e}'
                print(f"    ❌ DirectML確認エラー: {e}")
                
        except Exception as e:
            result['error'] = f'diagnosis_error: {e}'
            print(f"    ❌ DirectML診断エラー: {e}")
        
        return result
    
    def diagnose_onnxruntime(self) -> Dict[str, Any]:
        """ONNX Runtime診断"""
        result = {
            'installed': ONNXRUNTIME_AVAILABLE,
            'version': 'unknown',
            'providers': [],
            'directml_support': False
        }
        
        if ONNXRUNTIME_AVAILABLE:
            try:
                import onnxruntime as ort
                
                result['version'] = ort.__version__
                result['providers'] = ort.get_available_providers()
                result['directml_support'] = 'DmlExecutionProvider' in result['providers']
                
                print(f"    ✅ ONNX Runtime: {ort.__version__}")
                print(f"    📋 利用可能プロバイダー: {len(result['providers'])}個")
                for provider in result['providers']:
                    status = "✅" if provider == 'DmlExecutionProvider' else "📋"
                    print(f"      {status} {provider}")
                    
            except Exception as e:
                result['error'] = str(e)
                print(f"    ❌ ONNX Runtime診断エラー: {e}")
        else:
            print("    ❌ ONNX Runtime がインストールされていません")
        
        return result
    
    def diagnose_devices(self) -> Dict[str, Any]:
        """デバイス一覧診断"""
        result = {
            'gpu_devices': [],
            'npu_devices': [],
            'total_devices': 0
        }
        
        try:
            print("  🔍 計算デバイス一覧確認中...")
            
            # GPU情報取得
            if TORCH_AVAILABLE:
                try:
                    import torch
                    
                    if torch.cuda.is_available():
                        gpu_count = torch.cuda.device_count()
                        result['gpu_devices'] = []
                        
                        for i in range(gpu_count):
                            gpu_info = {
                                'id': i,
                                'name': torch.cuda.get_device_name(i),
                                'memory': torch.cuda.get_device_properties(i).total_memory
                            }
                            result['gpu_devices'].append(gpu_info)
                            
                        print(f"    🎮 GPU発見: {gpu_count}個")
                        for gpu in result['gpu_devices']:
                            memory_gb = gpu['memory'] / (1024**3)
                            print(f"      📱 GPU {gpu['id']}: {gpu['name']} ({memory_gb:.1f}GB)")
                    else:
                        print("    ⚠️ CUDA対応GPUが見つかりません")
                        
                except Exception as e:
                    print(f"    ❌ GPU情報取得エラー: {e}")
            
            # NPU情報は前のシステム診断結果を参照
            if 'system' in self.results:
                result['npu_devices'] = self.results['system'].get('npu_devices', [])
            
            result['total_devices'] = len(result['gpu_devices']) + len(result['npu_devices'])
            
        except Exception as e:
            print(f"    ❌ デバイス一覧診断エラー: {e}")
        
        return result
    
    def suggest_recovery(self) -> Dict[str, Any]:
        """復旧方法の提案"""
        suggestions = []
        priority = 'low'
        
        # DirectML問題の場合
        if not self.results.get('directml', {}).get('available', False):
            suggestions.append({
                'issue': 'DirectML利用不可',
                'solution': 'ONNX Runtime (DirectML版) の再インストール',
                'command': 'pip uninstall onnxruntime && pip install onnxruntime-directml',
                'priority': 'high'
            })
            priority = 'high'
        
        # NPUデバイス認識問題の場合
        npu_devices = self.results.get('system', {}).get('npu_devices', [])
        if not npu_devices:
            suggestions.append({
                'issue': 'NPUデバイス未認識',
                'solution': 'NPUドライバーの再インストール',
                'command': 'デバイスマネージャーでNPUドライバーを更新',
                'priority': 'high'
            })
            priority = 'high'
        
        # デバイス状態問題の場合
        for device in npu_devices:
            if device.get('Status') != 'OK':
                suggestions.append({
                    'issue': f'NPUデバイス状態異常: {device.get("Status")}',
                    'solution': 'デバイスの無効化→有効化',
                    'command': 'デバイスマネージャーでデバイスを無効化後、再有効化',
                    'priority': 'medium'
                })
                if priority == 'low':
                    priority = 'medium'
        
        # 一般的な復旧方法
        suggestions.append({
            'issue': '一般的な復旧',
            'solution': 'システム再起動',
            'command': 'shutdown /r /t 0',
            'priority': 'low'
        })
        
        # 提案の表示
        print(f"  🔧 復旧提案 (優先度: {priority})")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"    {i}. {suggestion['issue']}")
            print(f"       💡 解決方法: {suggestion['solution']}")
            print(f"       📝 コマンド: {suggestion['command']}")
            print(f"       ⚡ 優先度: {suggestion['priority']}")
            print()
        
        return {
            'suggestions': suggestions,
            'priority': priority,
            'count': len(suggestions)
        }
    
    def print_summary(self):
        """診断結果サマリーの表示"""
        print("📊 診断結果サマリー")
        print("-" * 40)
        
        # システム状態
        system = self.results.get('system', {})
        npu_count = len(system.get('npu_devices', []))
        print(f"🖥️  システム: NPUデバイス {npu_count}個")
        
        # DirectML状態
        directml = self.results.get('directml', {})
        directml_status = "✅ 利用可能" if directml.get('available') else "❌ 利用不可"
        print(f"🔧 DirectML: {directml_status}")
        
        # ONNX Runtime状態
        onnxrt = self.results.get('onnxruntime', {})
        onnxrt_status = "✅ インストール済み" if onnxrt.get('installed') else "❌ 未インストール"
        print(f"⚙️  ONNX Runtime: {onnxrt_status}")
        
        # 復旧提案
        recovery = self.results.get('recovery', {})
        suggestion_count = recovery.get('count', 0)
        priority = recovery.get('priority', 'unknown')
        print(f"🔧 復旧提案: {suggestion_count}個 (優先度: {priority})")
        
        # 総合判定
        if directml.get('available') and npu_count > 0:
            print("\n🎯 総合判定: ✅ NPU使用可能")
        elif directml.get('available'):
            print("\n🎯 総合判定: ⚠️ DirectML利用可能、NPU要確認")
        else:
            print("\n🎯 総合判定: ❌ NPU使用不可、復旧が必要")

def main():
    """メイン実行関数"""
    diagnostic = NPUDiagnosticTool()
    results = diagnostic.run_full_diagnosis()
    
    # 結果をJSONファイルに保存
    try:
        with open('npu_diagnostic_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 診断結果を npu_diagnostic_results.json に保存しました")
    except Exception as e:
        print(f"\n❌ 診断結果保存エラー: {e}")

if __name__ == "__main__":
    main()

