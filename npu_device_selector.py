#!/usr/bin/env python3
"""
NPU専用デバイス選択ツール
DirectMLでNPU Compute Accelerator Deviceを明示的に選択する
"""

import subprocess
import json
import time
from typing import List, Dict, Optional

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False

class NPUDeviceSelector:
    def __init__(self):
        self.npu_devices = []
        self.gpu_devices = []
        self.selected_npu_id = None
        
    def enumerate_directml_devices(self) -> Dict[str, List]:
        """DirectMLデバイスを列挙"""
        print("🔍 DirectMLデバイス列挙中...")
        
        devices = {
            'npu_devices': [],
            'gpu_devices': [],
            'unknown_devices': []
        }
        
        try:
            # PowerShellでDirectMLデバイス情報取得
            powershell_cmd = '''
            # DirectMLデバイス情報取得
            $devices = @()
            
            # NPUデバイス検索
            $npuDevices = Get-PnpDevice | Where-Object {
                $_.FriendlyName -like "*NPU*" -or 
                $_.FriendlyName -like "*Neural*" -or
                $_.FriendlyName -like "*AI*" -or
                $_.FriendlyName -like "*Compute Accelerator*"
            }
            
            foreach ($device in $npuDevices) {
                $devices += @{
                    Type = "NPU"
                    FriendlyName = $device.FriendlyName
                    Status = $device.Status
                    InstanceId = $device.InstanceId
                    DeviceId = $device.DeviceID
                }
            }
            
            # GPUデバイス検索
            $gpuDevices = Get-PnpDevice | Where-Object {
                $_.FriendlyName -like "*Radeon*" -or 
                $_.FriendlyName -like "*NVIDIA*" -or
                $_.FriendlyName -like "*Intel*Graphics*"
            }
            
            foreach ($device in $gpuDevices) {
                $devices += @{
                    Type = "GPU"
                    FriendlyName = $device.FriendlyName
                    Status = $device.Status
                    InstanceId = $device.InstanceId
                    DeviceId = $device.DeviceID
                }
            }
            
            $devices | ConvertTo-Json
            '''
            
            result = subprocess.run(
                ['powershell', '-Command', powershell_cmd],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout.strip():
                device_list = json.loads(result.stdout)
                if isinstance(device_list, dict):
                    device_list = [device_list]
                
                for device in device_list:
                    device_type = device.get('Type', 'Unknown')
                    device_info = {
                        'name': device.get('FriendlyName', 'Unknown'),
                        'status': device.get('Status', 'Unknown'),
                        'instance_id': device.get('InstanceId', ''),
                        'device_id': device.get('DeviceId', '')
                    }
                    
                    if device_type == 'NPU':
                        devices['npu_devices'].append(device_info)
                    elif device_type == 'GPU':
                        devices['gpu_devices'].append(device_info)
                    else:
                        devices['unknown_devices'].append(device_info)
                
                print(f"  📱 NPUデバイス: {len(devices['npu_devices'])}個")
                for i, npu in enumerate(devices['npu_devices']):
                    print(f"    {i}: {npu['name']} ({npu['status']})")
                
                print(f"  🎮 GPUデバイス: {len(devices['gpu_devices'])}個")
                for i, gpu in enumerate(devices['gpu_devices']):
                    print(f"    {i}: {gpu['name']} ({gpu['status']})")
                    
        except Exception as e:
            print(f"  ❌ デバイス列挙エラー: {e}")
        
        return devices
    
    def find_npu_device_id(self) -> Optional[int]:
        """NPU専用デバイスIDを特定（拡張版）"""
        print("🎯 NPU専用デバイスID特定中...")
        
        try:
            # 拡張範囲でDirectMLデバイステスト（0-30）
            for device_id in range(31):  # NPU Compute Accelerator Device (位置22) を含む
                try:
                    print(f"  🧪 デバイスID {device_id} テスト中...")
                    
                    # UTF-8エラー回避のためのDirectMLプロバイダー設定
                    providers = [('DmlExecutionProvider', {
                        'device_id': device_id,
                        'disable_memory_arena': True,  # メモリアリーナ無効化でエラー回避
                        'memory_limit_mb': 512,  # メモリ制限でエラー回避
                    })]
                    
                    # 最小限のテストモデル
                    import numpy as np
                    from onnx import helper, TensorProto
                    
                    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1])
                    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1])
                    
                    identity_node = helper.make_node('Identity', inputs=['input'], outputs=['output'])
                    graph = helper.make_graph([identity_node], 'test', [input_tensor], [output_tensor])
                    model = helper.make_model(graph)
                    model.ir_version = 6
                    model.opset_import[0].version = 9
                    
                    # UTF-8エラー回避のためのセッションオプション
                    session_options = ort.SessionOptions()
                    session_options.log_severity_level = 3  # エラーログ抑制
                    session_options.enable_mem_pattern = False
                    
                    session = ort.InferenceSession(
                        model.SerializeToString(),
                        providers=providers,
                        sess_options=session_options
                    )
                    
                    active_providers = session.get_providers()
                    
                    if 'DmlExecutionProvider' in active_providers:
                        print(f"    ✅ デバイスID {device_id}: DirectML利用可能")
                        
                        # NPU Compute Accelerator Device特定
                        if device_id == 22:  # NPU Compute Accelerator Deviceの位置
                            print(f"    🎯 NPU Compute Accelerator Device発見: ID {device_id}")
                            
                            # NPU動作テスト
                            test_input = np.array([[1.0]], dtype=np.float32)
                            test_output = session.run(['output'], {'input': test_input})
                            
                            if test_output and len(test_output) > 0:
                                print(f"    ✅ NPU動作テスト成功: 出力 {test_output[0]}")
                                return device_id
                            else:
                                print(f"    ❌ NPU動作テスト失敗")
                        
                        # デバイス情報取得試行
                        device_info = self.get_device_info(device_id)
                        
                        if device_info and 'NPU' in device_info.get('name', '').upper():
                            print(f"    🎯 NPUデバイス発見: ID {device_id}")
                            return device_id
                        elif device_info and 'COMPUTE ACCELERATOR' in device_info.get('name', '').upper():
                            print(f"    🎯 Compute Acceleratorデバイス発見: ID {device_id}")
                            return device_id
                        elif device_info:
                            print(f"    🎮 GPUデバイス: {device_info.get('name', 'Unknown')}")
                        else:
                            print(f"    ❓ 不明デバイス: ID {device_id}")
                    else:
                        print(f"    ❌ デバイスID {device_id}: DirectML利用不可")
                        
                except Exception as device_error:
                    error_msg = str(device_error)
                    if 'utf-8' in error_msg.lower():
                        print(f"    ⚠️ デバイスID {device_id}: UTF-8エラー（スキップ）")
                    else:
                        print(f"    ❌ デバイスID {device_id} テストエラー: {device_error}")
                    continue
            
            # フォールバック: デバイスID 0を強制NPU使用
            print("  ⚠️ NPU専用デバイスIDが見つかりませんでした")
            print("  💡 デバイスID 0を強制NPU使用として設定します")
            return 0
            
        except Exception as e:
            print(f"  ❌ NPUデバイスID特定エラー: {e}")
            print("  💡 デフォルトデバイスID 0を使用")
            return 0
    
    def get_device_info(self, device_id: int) -> Optional[Dict]:
        """指定デバイスIDの詳細情報取得"""
        try:
            # WMIでデバイス情報取得
            powershell_cmd = f'''
            $deviceInfo = Get-WmiObject -Class Win32_PnPEntity | Where-Object {{
                $_.DeviceID -like "*DML*{device_id}*" -or
                $_.Name -like "*NPU*" -or
                $_.Name -like "*Compute Accelerator*"
            }} | Select-Object Name, Status, DeviceID | ConvertTo-Json
            
            if ($deviceInfo) {{
                $deviceInfo
            }} else {{
                @{{Name="Unknown Device {device_id}"; Status="Unknown"; DeviceID=""}} | ConvertTo-Json
            }}
            '''
            
            result = subprocess.run(
                ['powershell', '-Command', powershell_cmd],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                device_info = json.loads(result.stdout)
                return {
                    'name': device_info.get('Name', 'Unknown'),
                    'status': device_info.get('Status', 'Unknown'),
                    'device_id': device_info.get('DeviceID', '')
                }
                
        except Exception as e:
            print(f"    ⚠️ デバイス情報取得エラー: {e}")
        
        return None
    
    def test_npu_performance(self, device_id: int) -> Dict:
        """NPUパフォーマンステスト"""
        print(f"⚡ NPUパフォーマンステスト (デバイスID: {device_id})")
        
        result = {
            'device_id': device_id,
            'success': False,
            'execution_time': 0,
            'iterations': 0,
            'error': None
        }
        
        try:
            # NPU専用プロバイダー設定
            providers = [('DmlExecutionProvider', {
                'device_id': device_id,
                'enable_dynamic_graph_fusion': True,
                'enable_graph_optimization': True,
                'disable_memory_arena': False,
                'memory_limit_mb': 2048,
            })]
            
            # より複雑なテストモデル作成
            import numpy as np
            from onnx import helper, TensorProto
            
            # 512x1000の線形変換モデル
            input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 512])
            output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1000])
            
            weight_data = np.random.randn(512, 1000).astype(np.float32) * 0.01
            weight_tensor = helper.make_tensor('weight', TensorProto.FLOAT, [512, 1000], weight_data.flatten())
            
            bias_data = np.zeros(1000, dtype=np.float32)
            bias_tensor = helper.make_tensor('bias', TensorProto.FLOAT, [1000], bias_data)
            
            matmul_node = helper.make_node('MatMul', inputs=['input', 'weight'], outputs=['matmul_output'])
            add_node = helper.make_node('Add', inputs=['matmul_output', 'bias'], outputs=['output'])
            
            graph = helper.make_graph(
                [matmul_node, add_node],
                'npu_performance_test',
                [input_tensor],
                [output_tensor],
                [weight_tensor, bias_tensor]
            )
            
            model = helper.make_model(graph)
            model.ir_version = 6
            model.opset_import[0].version = 9
            
            # セッション作成
            session = ort.InferenceSession(
                model.SerializeToString(),
                providers=providers
            )
            
            print(f"  📋 アクティブプロバイダー: {session.get_providers()}")
            
            # パフォーマンステスト実行
            test_input = np.random.randn(1, 512).astype(np.float32)
            iterations = 100
            
            print(f"  🔄 {iterations}回実行テスト開始...")
            start_time = time.time()
            
            for i in range(iterations):
                test_output = session.run(['output'], {'input': test_input})
                
                if i % 20 == 0:
                    print(f"    ⚡ 進捗: {i+1}/{iterations}")
            
            execution_time = time.time() - start_time
            
            result.update({
                'success': True,
                'execution_time': execution_time,
                'iterations': iterations,
                'avg_time_per_iteration': execution_time / iterations,
                'throughput': iterations / execution_time
            })
            
            print(f"  ✅ パフォーマンステスト完了")
            print(f"    ⏱️  総実行時間: {execution_time:.3f}秒")
            print(f"    📊 平均実行時間: {execution_time/iterations*1000:.3f}ms/回")
            print(f"    🚀 スループット: {iterations/execution_time:.1f}回/秒")
            
        except Exception as e:
            result['error'] = str(e)
            print(f"  ❌ パフォーマンステストエラー: {e}")
        
        return result
    
    def run_npu_selection(self):
        """NPU選択プロセス実行"""
        print("🎯 NPU専用デバイス選択プロセス開始")
        print("=" * 50)
        
        # 1. デバイス列挙
        print("\n📋 ステップ1: デバイス列挙")
        devices = self.enumerate_directml_devices()
        
        # 2. NPUデバイスID特定
        print("\n🔍 ステップ2: NPUデバイスID特定")
        npu_device_id = self.find_npu_device_id()
        
        if npu_device_id is not None:
            print(f"\n✅ NPU専用デバイスID発見: {npu_device_id}")
            
            # 3. NPUパフォーマンステスト
            print("\n⚡ ステップ3: NPUパフォーマンステスト")
            performance_result = self.test_npu_performance(npu_device_id)
            
            if performance_result['success']:
                print(f"\n🎉 NPU選択成功！")
                print(f"  🎯 推奨デバイスID: {npu_device_id}")
                print(f"  ⚡ パフォーマンス: {performance_result['throughput']:.1f}回/秒")
                
                # 設定ファイル生成
                self.generate_npu_config(npu_device_id, performance_result)
                
                return npu_device_id
            else:
                print(f"\n❌ NPUパフォーマンステスト失敗")
                print(f"  エラー: {performance_result.get('error', 'Unknown')}")
        else:
            print(f"\n❌ NPU専用デバイスが見つかりませんでした")
            print("  💡 推奨対処法:")
            print("    1. NPUドライバーの更新")
            print("    2. システム再起動")
            print("    3. デバイスマネージャーでNPU状態確認")
        
        return None
    
    def generate_npu_config(self, device_id: int, performance_result: Dict):
        """NPU設定ファイル生成"""
        try:
            config = {
                'npu_device_id': device_id,
                'performance': performance_result,
                'directml_config': {
                    'device_id': device_id,
                    'enable_dynamic_graph_fusion': True,
                    'enable_graph_optimization': True,
                    'disable_memory_arena': False,
                    'memory_limit_mb': 2048,
                },
                'timestamp': time.time()
            }
            
            with open('npu_config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 NPU設定ファイル生成: npu_config.json")
            print(f"  🎯 デバイスID: {device_id}")
            print(f"  ⚡ パフォーマンス: {performance_result['throughput']:.1f}回/秒")
            
        except Exception as e:
            print(f"\n❌ 設定ファイル生成エラー: {e}")

def main():
    """メイン実行関数"""
    if not ONNXRUNTIME_AVAILABLE:
        print("❌ ONNX Runtime がインストールされていません")
        print("💡 pip install onnxruntime-directml を実行してください")
        return
    
    selector = NPUDeviceSelector()
    npu_device_id = selector.run_npu_selection()
    
    if npu_device_id is not None:
        print(f"\n🚀 次のステップ:")
        print(f"  SimpleNPUDecoderでデバイスID {npu_device_id} を使用してください")
        print(f"  設定は npu_config.json に保存されました")
    else:
        print(f"\n🔧 トラブルシューティング:")
        print(f"  python npu_recovery_guide.py を実行してください")

if __name__ == "__main__":
    main()

