"""
NPU用ONNX変換専用スクリプト
大規模LLMモデルをNPU（DirectML）向けONNX形式に変換

使用方法:
    python convert_to_onnx_npu.py --model rinna/youri-7b-chat --output ./onnx_models/
"""

import os
import sys
import argparse
import time
import traceback
from typing import Optional

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import onnx
import onnxruntime as ort

class NPUONNXConverter:
    """NPU用ONNX変換器"""
    
    def __init__(self, model_name: str, output_dir: str = "./onnx_models/"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None
        
        # 出力ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🚀 NPU用ONNX変換器初期化")
        print(f"📱 モデル: {model_name}")
        print(f"📁 出力ディレクトリ: {output_dir}")
    
    def load_model(self) -> bool:
        """モデルとトークナイザーのロード"""
        try:
            print("📝 トークナイザーをロード中...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✅ トークナイザーロード完了")
            
            print("🤖 モデルをロード中...")
            load_start = time.time()
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="cpu",  # CPU上でロード（ONNX変換用）
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            load_time = time.time() - load_start
            print(f"✅ モデルロード完了 ({load_time:.1f}秒)")
            
            # 評価モードに設定
            self.model.eval()
            
            return True
            
        except Exception as e:
            print(f"❌ モデルロードエラー: {e}")
            traceback.print_exc()
            return False
    
    def convert_to_onnx(self, max_sequence_length: int = 512) -> Optional[str]:
        """ONNX変換実行"""
        try:
            print("🔄 ONNX変換開始...")
            
            if self.model is None or self.tokenizer is None:
                print("❌ モデルまたはトークナイザーが未ロード")
                return None
            
            # サンプル入力作成
            sample_text = "こんにちは、今日はいい天気ですね。人工知能について教えてください。"
            sample_inputs = self.tokenizer(
                sample_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length
            )
            
            print(f"📊 サンプル入力形状: {sample_inputs['input_ids'].shape}")
            
            # 出力ファイルパス
            model_safe_name = self.model_name.replace('/', '_').replace('-', '_')
            onnx_path = os.path.join(self.output_dir, f"{model_safe_name}_npu.onnx")
            
            print(f"📁 出力パス: {onnx_path}")
            
            # 動的軸設定（バッチサイズとシーケンス長を動的に）
            dynamic_axes = {
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'},
                'logits': {0: 'batch_size', 1: 'sequence'}
            }
            
            print("🔧 ONNX変換実行中...")
            convert_start = time.time()
            
            # ONNX変換実行
            with torch.no_grad():
                torch.onnx.export(
                    self.model,
                    (sample_inputs['input_ids'], sample_inputs['attention_mask']),
                    onnx_path,
                    export_params=True,
                    opset_version=11,  # DirectML互換バージョン
                    do_constant_folding=True,
                    input_names=['input_ids', 'attention_mask'],
                    output_names=['logits'],
                    dynamic_axes=dynamic_axes,
                    verbose=False
                )
            
            convert_time = time.time() - convert_start
            print(f"✅ ONNX変換完了 ({convert_time:.1f}秒)")
            
            # ファイルサイズ確認
            file_size = os.path.getsize(onnx_path) / (1024**3)  # GB
            print(f"📊 ONNXファイルサイズ: {file_size:.2f}GB")
            
            # ONNX検証
            print("🔍 ONNX検証中...")
            try:
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                print("✅ ONNX検証成功")
            except Exception as e:
                print(f"⚠️ ONNX検証警告: {e}")
            
            return onnx_path
            
        except Exception as e:
            print(f"❌ ONNX変換エラー: {e}")
            traceback.print_exc()
            return None
    
    def test_npu_session(self, onnx_path: str) -> bool:
        """NPUセッションテスト"""
        try:
            print("🧪 NPUセッションテスト開始...")
            
            # DirectMLプロバイダー設定
            providers = [
                ('DmlExecutionProvider', {
                    'device_id': 0,
                    'enable_dynamic_graph_fusion': True,
                    'enable_graph_optimization': True,
                    'disable_memory_arena': False,
                    'memory_limit_mb': 4096,
                })
            ]
            
            # セッションオプション設定
            session_options = ort.SessionOptions()
            session_options.enable_mem_pattern = False
            session_options.enable_cpu_mem_arena = False
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            print("🔧 NPUセッション作成中...")
            session = ort.InferenceSession(
                onnx_path,
                providers=providers,
                sess_options=session_options
            )
            
            # プロバイダー確認
            active_providers = session.get_providers()
            print(f"📋 アクティブプロバイダー: {active_providers}")
            
            if 'DmlExecutionProvider' not in active_providers:
                print("⚠️ DirectMLプロバイダーが無効")
                return False
            
            # テスト実行
            print("🚀 NPUテスト実行中...")
            
            # テスト入力作成
            test_text = "テストプロンプト"
            test_inputs = self.tokenizer(
                test_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            )
            
            input_ids = test_inputs['input_ids'].numpy().astype(np.int64)
            attention_mask = test_inputs['attention_mask'].numpy().astype(np.int64)
            
            # NPU推論実行
            test_start = time.time()
            outputs = session.run(
                ['logits'],
                {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }
            )
            test_time = time.time() - test_start
            
            logits = outputs[0]
            print(f"✅ NPUテスト成功: logits形状{logits.shape}, 実行時間{test_time:.3f}秒")
            
            # 複数回実行でNPU負荷テスト
            print("🔥 NPU負荷テスト実行中...")
            for i in range(10):
                session.run(
                    ['logits'],
                    {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask
                    }
                )
                if i % 3 == 0:
                    print(f"  🔄 NPU負荷テスト {i+1}/10")
            
            print("✅ NPU負荷テスト完了")
            print("🎯 タスクマネージャーでNPU使用率を確認してください")
            
            return True
            
        except Exception as e:
            print(f"❌ NPUセッションテストエラー: {e}")
            traceback.print_exc()
            return False

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="NPU用ONNX変換専用スクリプト")
    parser.add_argument("--model", type=str, default="rinna/youri-7b-chat",
                       help="変換するモデル名")
    parser.add_argument("--output", type=str, default="./onnx_models/",
                       help="出力ディレクトリ")
    parser.add_argument("--max-length", type=int, default=512,
                       help="最大シーケンス長")
    parser.add_argument("--test-npu", action="store_true",
                       help="変換後にNPUセッションテストを実行")
    
    args = parser.parse_args()
    
    print("🚀 NPU用ONNX変換開始")
    print("=" * 60)
    
    # 変換器初期化
    converter = NPUONNXConverter(args.model, args.output)
    
    # モデルロード
    if not converter.load_model():
        print("❌ モデルロードに失敗しました")
        sys.exit(1)
    
    # ONNX変換実行
    onnx_path = converter.convert_to_onnx(args.max_length)
    if not onnx_path:
        print("❌ ONNX変換に失敗しました")
        sys.exit(1)
    
    print(f"✅ ONNX変換成功: {onnx_path}")
    
    # NPUセッションテスト
    if args.test_npu:
        print("\n🧪 NPUセッションテスト開始")
        print("=" * 60)
        
        if converter.test_npu_session(onnx_path):
            print("✅ NPUセッションテスト成功")
        else:
            print("❌ NPUセッションテスト失敗")
            sys.exit(1)
    
    print("\n🎉 全ての処理が完了しました")
    print(f"📁 ONNXファイル: {onnx_path}")
    print("💡 修正版デモで使用してください:")
    print(f"   python infer_os_japanese_llm_demo_fixed.py --model {args.model} --enable-npu --interactive")

if __name__ == "__main__":
    main()

