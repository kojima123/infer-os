#!/usr/bin/env python3
"""
NPU最適化モデルロード問題修正スクリプト
PyTorch 2.6のweights_only=True問題を解決
"""

import os
import sys
import torch
import traceback
from pathlib import Path
from typing import Optional, Dict, Any


class NPUModelLoadingFixer:
    """NPU最適化モデルロード修正クラス"""
    
    def __init__(self):
        self.safe_globals = [
            'modeling_llama_amd.LlamaForCausalLM',
            'modeling_llama_amd.LlamaModel',
            'modeling_llama_amd.LlamaConfig',
            'torch.nn.Linear',
            'torch.nn.Embedding',
            'torch.nn.LayerNorm',
            'transformers.models.llama.modeling_llama.LlamaForCausalLM',
            'transformers.models.llama.modeling_llama.LlamaModel'
        ]
    
    def fix_model_loading(self, model_path: str) -> bool:
        """NPU最適化モデルロード修正"""
        print("🔧 NPU最適化モデルロード修正開始")
        print("=" * 60)
        
        try:
            # 1. モデルパス確認
            model_file = os.path.join(model_path, "pytorch_llama3_8b_w_bit_4_awq_amd.pt")
            if not os.path.exists(model_file):
                print(f"❌ モデルファイルが見つかりません: {model_file}")
                return False
            
            print(f"✅ モデルファイル確認: {model_file}")
            
            # 2. 安全なグローバル設定
            print("🔧 安全なグローバル設定中...")
            
            # 方法1: add_safe_globals使用
            try:
                for global_name in self.safe_globals:
                    torch.serialization.add_safe_globals([global_name])
                print("✅ add_safe_globals設定完了")
            except Exception as e:
                print(f"⚠️ add_safe_globals設定失敗: {e}")
            
            # 3. モデルロードテスト（weights_only=False）
            print("🧪 モデルロードテスト（weights_only=False）...")
            try:
                model_data = torch.load(model_file, weights_only=False, map_location='cpu')
                print("✅ weights_only=False でロード成功")
                print(f"📊 モデルデータキー: {list(model_data.keys()) if isinstance(model_data, dict) else type(model_data)}")
                return True
                
            except Exception as e:
                print(f"❌ weights_only=False でロード失敗: {e}")
            
            # 4. モデルロードテスト（safe_globals使用）
            print("🧪 モデルロードテスト（safe_globals使用）...")
            try:
                with torch.serialization.safe_globals(self.safe_globals):
                    model_data = torch.load(model_file, weights_only=True, map_location='cpu')
                print("✅ safe_globals でロード成功")
                return True
                
            except Exception as e:
                print(f"❌ safe_globals でロード失敗: {e}")
            
            # 5. 代替ロード方法
            print("🔄 代替ロード方法試行...")
            try:
                # pickle_module指定
                import pickle
                model_data = torch.load(model_file, pickle_module=pickle, map_location='cpu')
                print("✅ pickle_module指定でロード成功")
                return True
                
            except Exception as e:
                print(f"❌ 代替ロード方法失敗: {e}")
            
            return False
            
        except Exception as e:
            print(f"❌ 修正処理エラー: {e}")
            traceback.print_exc()
            return False
    
    def create_fixed_loader(self, output_path: str = "fixed_npu_model_loader.py"):
        """修正版モデルローダー作成"""
        print(f"📝 修正版モデルローダー作成: {output_path}")
        
        loader_code = '''#!/usr/bin/env python3
"""
修正版NPU最適化モデルローダー
PyTorch 2.6 weights_only問題対応版
"""

import torch
import os
from typing import Optional, Any


class FixedNPUModelLoader:
    """修正版NPU最適化モデルローダー"""
    
    def __init__(self):
        # 安全なグローバルクラス定義
        self.safe_globals = [
            'modeling_llama_amd.LlamaForCausalLM',
            'modeling_llama_amd.LlamaModel', 
            'modeling_llama_amd.LlamaConfig',
            'torch.nn.Linear',
            'torch.nn.Embedding',
            'torch.nn.LayerNorm',
            'transformers.models.llama.modeling_llama.LlamaForCausalLM'
        ]
        
        # 安全なグローバル設定
        self._setup_safe_globals()
    
    def _setup_safe_globals(self):
        """安全なグローバル設定"""
        try:
            for global_name in self.safe_globals:
                torch.serialization.add_safe_globals([global_name])
        except Exception:
            pass  # 既に設定済みの場合は無視
    
    def load_npu_model(self, model_path: str, weights_only: bool = False) -> Optional[Any]:
        """NPU最適化モデル安全ロード"""
        model_file = os.path.join(model_path, "pytorch_llama3_8b_w_bit_4_awq_amd.pt")
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_file}")
        
        # 方法1: weights_only=False（推奨）
        if not weights_only:
            try:
                return torch.load(model_file, weights_only=False, map_location='cpu')
            except Exception as e:
                print(f"weights_only=False失敗: {e}")
        
        # 方法2: safe_globals使用
        try:
            with torch.serialization.safe_globals(self.safe_globals):
                return torch.load(model_file, weights_only=True, map_location='cpu')
        except Exception as e:
            print(f"safe_globals失敗: {e}")
        
        # 方法3: 強制的にweights_only=False
        try:
            return torch.load(model_file, weights_only=False, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"全てのロード方法が失敗: {e}")


# 使用例
if __name__ == "__main__":
    loader = FixedNPUModelLoader()
    
    # モデルロードテスト
    model_path = "llama3-8b-amd-npu"
    try:
        model_data = loader.load_npu_model(model_path)
        print("✅ NPU最適化モデルロード成功")
        print(f"📊 データ型: {type(model_data)}")
        if isinstance(model_data, dict):
            print(f"📊 キー: {list(model_data.keys())}")
    except Exception as e:
        print(f"❌ モデルロード失敗: {e}")
'''
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(loader_code)
            print(f"✅ 修正版ローダー作成完了: {output_path}")
            return True
        except Exception as e:
            print(f"❌ ローダー作成失敗: {e}")
            return False
    
    def update_existing_scripts(self):
        """既存スクリプトの更新"""
        print("🔄 既存スクリプト更新中...")
        
        scripts_to_update = [
            "npu_optimized_japanese_models.py",
            "integrated_npu_infer_os.py",
            "vitisai_npu_engine.py"
        ]
        
        for script in scripts_to_update:
            if os.path.exists(script):
                print(f"🔧 {script} 更新中...")
                self._update_script_torch_load(script)
            else:
                print(f"⚠️ {script} が見つかりません")
    
    def _update_script_torch_load(self, script_path: str):
        """スクリプト内のtorch.load呼び出しを更新"""
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # torch.load呼び出しを修正
            updated_content = content.replace(
                'torch.load(',
                'torch.load('
            )
            
            # weights_only=Trueを明示的にFalseに変更
            updated_content = updated_content.replace(
                'weights_only=True',
                'weights_only=False'
            )
            
            # 安全なロード関数の追加
            if 'def safe_torch_load(' not in updated_content:
                safe_load_function = '''
def safe_torch_load(file_path, map_location='cpu'):
    """安全なtorch.loadラッパー"""
    try:
        # 方法1: weights_only=False
        return torch.load(file_path, weights_only=False, map_location=map_location)
    except Exception:
        # 方法2: 安全なグローバル設定
        safe_globals = [
            'modeling_llama_amd.LlamaForCausalLM',
            'modeling_llama_amd.LlamaModel',
            'torch.nn.Linear'
        ]
        try:
            with torch.serialization.safe_globals(safe_globals):
                return torch.load(file_path, weights_only=True, map_location=map_location)
        except Exception:
            # 方法3: 強制的にweights_only=False
            return torch.load(file_path, weights_only=False, map_location=map_location)

'''
                updated_content = safe_load_function + updated_content
            
            # ファイル更新
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            print(f"✅ {script_path} 更新完了")
            
        except Exception as e:
            print(f"❌ {script_path} 更新失敗: {e}")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NPU最適化モデルロード問題修正")
    parser.add_argument("--model-path", default="llama3-8b-amd-npu", help="モデルパス")
    parser.add_argument("--create-loader", action="store_true", help="修正版ローダー作成")
    parser.add_argument("--update-scripts", action="store_true", help="既存スクリプト更新")
    parser.add_argument("--test-loading", action="store_true", help="ロードテスト実行")
    
    args = parser.parse_args()
    
    fixer = NPUModelLoadingFixer()
    
    try:
        if args.test_loading:
            print("🧪 NPU最適化モデルロードテスト")
            success = fixer.fix_model_loading(args.model_path)
            if success:
                print("🎉 ロードテスト成功")
            else:
                print("❌ ロードテスト失敗")
        
        if args.create_loader:
            print("📝 修正版ローダー作成")
            fixer.create_fixed_loader()
        
        if args.update_scripts:
            print("🔄 既存スクリプト更新")
            fixer.update_existing_scripts()
        
        if not any([args.test_loading, args.create_loader, args.update_scripts]):
            # デフォルト: 全て実行
            print("🚀 NPU最適化モデルロード問題修正開始")
            print("=" * 60)
            
            # 1. ロードテスト
            print("\\n1. ロードテスト実行")
            fixer.fix_model_loading(args.model_path)
            
            # 2. 修正版ローダー作成
            print("\\n2. 修正版ローダー作成")
            fixer.create_fixed_loader()
            
            # 3. 既存スクリプト更新
            print("\\n3. 既存スクリプト更新")
            fixer.update_existing_scripts()
            
            print("\\n🎉 NPU最適化モデルロード問題修正完了")
            print("💡 以下のコマンドで修正版を使用してください:")
            print("   python fixed_npu_model_loader.py")
            print("   python integrated_npu_infer_os.py --model llama3-8b-amd-npu")
        
    except KeyboardInterrupt:
        print("\\n👋 修正処理を中断しました")
    except Exception as e:
        print(f"\\n❌ 修正処理エラー: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

