#!/usr/bin/env python3
"""
NPU復旧ガイドツール
再起動後のNPU問題に対する具体的な復旧手順を提供
"""

import subprocess
import time
import sys
from typing import List, Dict

class NPURecoveryGuide:
    def __init__(self):
        self.recovery_steps = []
        
    def run_recovery_wizard(self):
        """NPU復旧ウィザードを実行"""
        print("🔧 NPU復旧ウィザード開始")
        print("=" * 50)
        
        print("\n📋 再起動後のNPU問題に対する復旧手順を実行します")
        print("⚠️  管理者権限が必要な場合があります")
        
        # ステップ1: NPUドライバー状態確認
        print("\n🔍 ステップ1: NPUドライバー状態確認")
        self.check_npu_driver_status()
        
        # ステップ2: DirectML再初期化
        print("\n🔄 ステップ2: DirectML再初期化")
        self.reinitialize_directml()
        
        # ステップ3: NPUデバイス再認識
        print("\n🔌 ステップ3: NPUデバイス再認識")
        self.refresh_npu_devices()
        
        # ステップ4: ONNX Runtime再設定
        print("\n⚙️ ステップ4: ONNX Runtime再設定")
        self.reconfigure_onnxruntime()
        
        # ステップ5: 復旧確認テスト
        print("\n✅ ステップ5: 復旧確認テスト")
        self.verify_recovery()
        
        # 復旧結果サマリー
        print("\n📊 復旧結果サマリー")
        self.print_recovery_summary()
    
    def check_npu_driver_status(self):
        """NPUドライバー状態確認"""
        try:
            print("  🔍 NPUドライバー状態を確認中...")
            
            # PowerShellでNPUデバイス状態確認
            powershell_cmd = '''
            Get-PnpDevice | Where-Object {
                $_.FriendlyName -like "*NPU*" -or 
                $_.FriendlyName -like "*Neural*" -or
                $_.FriendlyName -like "*AI*"
            } | Select-Object FriendlyName, Status, ProblemCode
            '''
            
            result = subprocess.run(
                ['powershell', '-Command', powershell_cmd],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                output = result.stdout.strip()
                if output:
                    print("    📱 NPUデバイス発見:")
                    print(f"    {output}")
                    
                    # 問題のあるデバイスがある場合の対処提案
                    if "Error" in output or "Problem" in output:
                        print("    ⚠️ 問題のあるNPUデバイスが検出されました")
                        self.suggest_driver_fix()
                    else:
                        print("    ✅ NPUドライバー状態正常")
                else:
                    print("    ❌ NPUデバイスが見つかりません")
                    self.suggest_driver_install()
            else:
                print(f"    ❌ ドライバー確認エラー: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("    ❌ ドライバー確認タイムアウト")
        except Exception as e:
            print(f"    ❌ ドライバー確認エラー: {e}")
    
    def suggest_driver_fix(self):
        """ドライバー修復提案"""
        print("    💡 ドライバー修復手順:")
        print("      1. デバイスマネージャーを開く")
        print("      2. 問題のあるNPUデバイスを右クリック")
        print("      3. 「デバイスのアンインストール」を選択")
        print("      4. 「このデバイスのドライバーソフトウェアを削除する」にチェック")
        print("      5. システム再起動")
        print("      6. Windows Updateでドライバー自動インストール")
    
    def suggest_driver_install(self):
        """ドライバーインストール提案"""
        print("    💡 NPUドライバーインストール手順:")
        print("      1. メーカーサイトから最新NPUドライバーをダウンロード")
        print("      2. 管理者権限でインストール実行")
        print("      3. システム再起動")
        print("      4. デバイスマネージャーでNPU認識確認")
    
    def reinitialize_directml(self):
        """DirectML再初期化"""
        try:
            print("  🔄 DirectML再初期化中...")
            
            # ONNX Runtime DirectMLの再インストール提案
            print("    💡 DirectML再初期化手順:")
            print("      1. 現在のONNX Runtimeをアンインストール")
            print("         pip uninstall onnxruntime onnxruntime-directml")
            print("      2. 最新版DirectML対応ONNX Runtimeをインストール")
            print("         pip install onnxruntime-directml")
            print("      3. Pythonプロセス再起動")
            
            # 自動実行オプション
            user_input = input("    ❓ 自動でONNX Runtime再インストールを実行しますか？ (y/N): ")
            
            if user_input.lower() == 'y':
                print("    🔄 ONNX Runtime再インストール実行中...")
                
                # アンインストール
                subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'onnxruntime', 'onnxruntime-directml'], 
                             capture_output=True)
                
                # 再インストール
                result = subprocess.run([sys.executable, '-m', 'pip', 'install', 'onnxruntime-directml'], 
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("    ✅ ONNX Runtime DirectML再インストール完了")
                else:
                    print(f"    ❌ 再インストールエラー: {result.stderr}")
            else:
                print("    📝 手動での再インストールを推奨します")
                
        except Exception as e:
            print(f"    ❌ DirectML再初期化エラー: {e}")
    
    def refresh_npu_devices(self):
        """NPUデバイス再認識"""
        try:
            print("  🔌 NPUデバイス再認識中...")
            
            # デバイス再スキャン
            print("    🔍 ハードウェア変更のスキャン実行中...")
            
            powershell_cmd = '''
            $devcon = Get-Command devcon.exe -ErrorAction SilentlyContinue
            if ($devcon) {
                devcon rescan
                Write-Output "デバイス再スキャン完了"
            } else {
                # devcon.exeが無い場合の代替方法
                Get-PnpDevice | Where-Object {$_.Status -eq "Error"} | Enable-PnpDevice -Confirm:$false
                Write-Output "問題デバイス再有効化完了"
            }
            '''
            
            result = subprocess.run(
                ['powershell', '-Command', powershell_cmd],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print(f"    ✅ {result.stdout.strip()}")
            else:
                print("    💡 手動でのデバイス再認識手順:")
                print("      1. デバイスマネージャーを開く")
                print("      2. メニューから「操作」→「ハードウェア変更のスキャン」")
                print("      3. NPUデバイスの状態確認")
                
        except subprocess.TimeoutExpired:
            print("    ❌ デバイス再認識タイムアウト")
        except Exception as e:
            print(f"    ❌ デバイス再認識エラー: {e}")
    
    def reconfigure_onnxruntime(self):
        """ONNX Runtime再設定"""
        try:
            print("  ⚙️ ONNX Runtime設定確認中...")
            
            # ONNX Runtime設定確認
            try:
                import onnxruntime as ort
                
                providers = ort.get_available_providers()
                print(f"    📋 利用可能プロバイダー: {providers}")
                
                if 'DmlExecutionProvider' in providers:
                    print("    ✅ DirectMLプロバイダー利用可能")
                    
                    # 簡単なテスト実行
                    self.test_directml_basic()
                    
                else:
                    print("    ❌ DirectMLプロバイダー利用不可")
                    print("    💡 ONNX Runtime DirectMLの再インストールが必要です")
                    
            except ImportError:
                print("    ❌ ONNX Runtime がインストールされていません")
                print("    💡 pip install onnxruntime-directml を実行してください")
                
        except Exception as e:
            print(f"    ❌ ONNX Runtime設定確認エラー: {e}")
    
    def test_directml_basic(self):
        """DirectML基本テスト"""
        try:
            print("    🧪 DirectML基本テスト実行中...")
            
            import onnxruntime as ort
            import numpy as np
            from onnx import helper, TensorProto
            
            # 最小限のテストモデル作成
            input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1])
            output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1])
            
            identity_node = helper.make_node('Identity', inputs=['input'], outputs=['output'])
            graph = helper.make_graph([identity_node], 'test', [input_tensor], [output_tensor])
            model = helper.make_model(graph)
            model.ir_version = 6
            model.opset_import[0].version = 9
            
            # DirectMLプロバイダーでセッション作成
            providers = [('DmlExecutionProvider', {'device_id': 0})]
            session = ort.InferenceSession(
                model.SerializeToString(),
                providers=providers
            )
            
            # テスト実行
            test_input = np.array([[1.0]], dtype=np.float32)
            test_output = session.run(['output'], {'input': test_input})
            
            active_providers = session.get_providers()
            
            if 'DmlExecutionProvider' in active_providers:
                print("    ✅ DirectML基本テスト成功")
                print(f"    📱 アクティブプロバイダー: {active_providers}")
            else:
                print("    ⚠️ DirectMLプロバイダーが非アクティブ")
                print(f"    📱 アクティブプロバイダー: {active_providers}")
                
        except Exception as e:
            print(f"    ❌ DirectML基本テストエラー: {e}")
    
    def verify_recovery(self):
        """復旧確認テスト"""
        try:
            print("  ✅ NPU復旧確認テスト実行中...")
            
            # シンプルNPUデコーダーのテスト
            try:
                from simple_npu_decode import SimpleNPUDecoder
                
                # ダミーのモデルとトークナイザーでテスト
                class DummyModel:
                    pass
                
                class DummyTokenizer:
                    def encode(self, text):
                        return [1, 2, 3]
                    
                    def decode(self, tokens):
                        return "テスト出力"
                
                decoder = SimpleNPUDecoder(DummyModel(), DummyTokenizer())
                
                if decoder.npu_session is not None:
                    print("    ✅ NPUデコーダー初期化成功")
                    
                    # 簡単なテスト実行
                    test_result = decoder.decode_with_npu("テスト", max_tokens=1)
                    print("    ✅ NPUデコードテスト成功")
                    
                    return True
                else:
                    print("    ❌ NPUデコーダー初期化失敗")
                    return False
                    
            except ImportError:
                print("    ⚠️ SimpleNPUDecoderが見つかりません")
                return False
            except Exception as e:
                print(f"    ❌ NPUデコーダーテストエラー: {e}")
                return False
                
        except Exception as e:
            print(f"    ❌ 復旧確認テストエラー: {e}")
            return False
    
    def print_recovery_summary(self):
        """復旧結果サマリー"""
        print("📊 復旧結果サマリー")
        print("-" * 30)
        
        # 最終確認テスト実行
        recovery_success = self.verify_recovery()
        
        if recovery_success:
            print("🎉 NPU復旧成功！")
            print("✅ NPUデコーダーが正常に動作しています")
            print("\n🚀 次のステップ:")
            print("  python infer_os_japanese_llm_demo.py --enable-npu --interactive")
        else:
            print("❌ NPU復旧未完了")
            print("\n🔧 追加の復旧手順:")
            print("  1. システム再起動")
            print("  2. NPUドライバーの手動更新")
            print("  3. Windows Updateの実行")
            print("  4. メーカーサポートへの問い合わせ")

def main():
    """メイン実行関数"""
    recovery = NPURecoveryGuide()
    recovery.run_recovery_wizard()

if __name__ == "__main__":
    main()

