"""
NPUデバッグ情報表示スクリプト
プロバイダー情報とNPUアプローチの詳細確認用

使用方法:
    python debug_npu_info.py
"""

import sys
import traceback

def check_onnxruntime_providers():
    """ONNXRuntimeプロバイダー確認"""
    try:
        import onnxruntime as ort
        
        print("🔍 ONNXRuntime プロバイダー情報:")
        print("=" * 50)
        
        # 利用可能プロバイダー
        available_providers = ort.get_available_providers()
        print(f"📋 利用可能プロバイダー: {len(available_providers)}個")
        for i, provider in enumerate(available_providers, 1):
            print(f"  {i}. {provider}")
        
        # 重要プロバイダーの確認
        important_providers = [
            'VitisAIExecutionProvider',
            'DmlExecutionProvider', 
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]
        
        print(f"\n🎯 重要プロバイダーの状況:")
        for provider in important_providers:
            status = "✅ 利用可能" if provider in available_providers else "❌ 利用不可"
            print(f"  {provider}: {status}")
        
        # デバイス情報
        print(f"\n🖥️ デバイス情報:")
        try:
            device_count = ort.get_device()
            print(f"  デバイス: {device_count}")
        except:
            print("  デバイス情報取得不可")
        
        return True
        
    except ImportError as e:
        print(f"❌ ONNXRuntime未インストール: {e}")
        return False
    except Exception as e:
        print(f"❌ ONNXRuntime確認エラー: {e}")
        traceback.print_exc()
        return False

def check_torch_info():
    """PyTorch情報確認"""
    try:
        import torch
        
        print("\n🔥 PyTorch情報:")
        print("=" * 50)
        
        print(f"📦 PyTorchバージョン: {torch.__version__}")
        print(f"🖥️ CUDA利用可能: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"🎯 CUDAデバイス数: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                print(f"  デバイス{i}: {device_name}")
        
        # DirectML確認
        try:
            device = torch.device("cuda:0")
            test_tensor = torch.randn(1, 1).to(device)
            print(f"✅ DirectMLテスト成功: {test_tensor.device}")
        except Exception as e:
            print(f"⚠️ DirectMLテスト失敗: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ PyTorch未インストール: {e}")
        return False
    except Exception as e:
        print(f"❌ PyTorch確認エラー: {e}")
        traceback.print_exc()
        return False

def test_npu_session():
    """NPUセッションテスト"""
    try:
        import onnxruntime as ort
        import numpy as np
        
        print("\n🧪 NPUセッションテスト:")
        print("=" * 50)
        
        # VitisAI優先プロバイダー設定
        providers = [
            'VitisAIExecutionProvider',
            ('DmlExecutionProvider', {
                'device_id': 0,
                'enable_dynamic_graph_fusion': True,
                'enable_graph_optimization': True,
            }),
            'CPUExecutionProvider'
        ]
        
        print(f"🎯 テスト用プロバイダー設定:")
        for i, provider in enumerate(providers, 1):
            print(f"  {i}. {provider}")
        
        # 簡単なONNXモデル作成
        import torch
        
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        model.eval()
        
        dummy_input = torch.randn(1, 10)
        onnx_path = "./test_npu_model.onnx"
        
        # ONNX変換
        torch.onnx.export(
            model, dummy_input, onnx_path,
            export_params=True, opset_version=11,
            input_names=['input'], output_names=['output']
        )
        
        print(f"✅ テスト用ONNXモデル作成: {onnx_path}")
        
        # セッション作成テスト
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        # アクティブプロバイダー確認
        active_providers = session.get_providers()
        print(f"📋 アクティブプロバイダー: {active_providers}")
        
        # 推論テスト
        test_input = np.random.randn(1, 10).astype(np.float32)
        output = session.run(['output'], {'input': test_input})
        
        print(f"✅ NPU推論テスト成功: 出力形状{output[0].shape}")
        
        # 使用プロバイダー判定
        if 'VitisAIExecutionProvider' in active_providers:
            print("🎯 VitisAI (真のNPU) で実行中")
        elif 'DmlExecutionProvider' in active_providers:
            print("🎯 DirectML (GPU/NPU) で実行中")
        else:
            print("⚠️ CPU で実行中")
        
        return True
        
    except Exception as e:
        print(f"❌ NPUセッションテストエラー: {e}")
        traceback.print_exc()
        return False

def main():
    """メイン関数"""
    print("🚀 NPUデバッグ情報表示")
    print("🎯 プロバイダー情報とNPUアプローチの詳細確認")
    print("=" * 60)
    
    # ONNXRuntime確認
    ort_ok = check_onnxruntime_providers()
    
    # PyTorch確認
    torch_ok = check_torch_info()
    
    # NPUセッションテスト
    if ort_ok and torch_ok:
        test_npu_session()
    
    print("\n📊 デバッグ情報表示完了")
    print("💡 この情報を使って代替NPUアプローチの動作を確認してください")

if __name__ == "__main__":
    main()

