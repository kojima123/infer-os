"""
真のNPU処理エンジン
実際のLLMモデルでNPU処理を行う効率的な実装

主要機能:
- 実際のLLMモデルのONNX変換とNPU実行
- 効率的なNPU負荷率向上
- 高速なトークン生成
- 確実なDirectML統合
"""

import os
import time
import numpy as np
import torch
import onnx
import onnxruntime as ort
from typing import Dict, List, Optional, Tuple, Any
import traceback

class TrueNPUEngine:
    """真のNPU処理エンジン"""
    
    def __init__(self, model, tokenizer, device_id: int = 0):
        self.model = model
        self.tokenizer = tokenizer
        self.device_id = device_id
        
        # NPU関連
        self.npu_session = None
        self.onnx_model_path = None
        self.is_npu_ready = False
        
        # 統計情報
        self.npu_inference_count = 0
        self.total_npu_time = 0.0
        
        print(f"🚀 真のNPU処理エンジン初期化")
        print(f"🎯 デバイスID: {device_id}")
    
    def setup_npu(self) -> bool:
        """NPUセットアップ"""
        try:
            print("🔧 真のNPU処理エンジンセットアップ開始...")
            
            # ONNX変換
            if not self._convert_model_to_onnx():
                print("❌ ONNX変換失敗")
                return False
            
            # NPUセッション作成
            if not self._create_npu_session():
                print("❌ NPUセッション作成失敗")
                return False
            
            # NPU動作テスト
            if not self._test_npu_inference():
                print("❌ NPU動作テスト失敗")
                return False
            
            self.is_npu_ready = True
            print("✅ 真のNPU処理エンジンセットアップ完了")
            return True
            
        except Exception as e:
            print(f"❌ NPUセットアップエラー: {e}")
            traceback.print_exc()
            return False
    
    def _convert_model_to_onnx(self) -> bool:
        """モデルをONNXに変換"""
        try:
            print("🔄 LLMモデルONNX変換開始...")
            
            # 出力ディレクトリ作成
            os.makedirs("./onnx_models", exist_ok=True)
            
            # 出力パス設定
            model_name = self.model.config.name_or_path.replace('/', '_').replace('-', '_')
            self.onnx_model_path = f"./onnx_models/{model_name}_true_npu.onnx"
            
            # 既存ファイルチェック
            if os.path.exists(self.onnx_model_path):
                print(f"📁 既存ONNXファイル使用: {self.onnx_model_path}")
                return True
            
            # サンプル入力作成
            sample_text = "こんにちは、今日はいい天気ですね。"
            sample_inputs = self.tokenizer(
                sample_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            )
            
            print(f"📊 サンプル入力形状: {sample_inputs['input_ids'].shape}")
            
            # モデルを評価モードに設定
            self.model.eval()
            
            # 動的軸設定
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
                    self.onnx_model_path,
                    export_params=True,
                    opset_version=11,  # DirectML互換
                    do_constant_folding=True,
                    input_names=['input_ids', 'attention_mask'],
                    output_names=['logits'],
                    dynamic_axes=dynamic_axes,
                    verbose=False
                )
            
            convert_time = time.time() - convert_start
            print(f"✅ ONNX変換完了 ({convert_time:.1f}秒)")
            
            # ファイルサイズ確認
            file_size = os.path.getsize(self.onnx_model_path) / (1024**3)
            print(f"📊 ONNXファイルサイズ: {file_size:.2f}GB")
            
            return True
            
        except Exception as e:
            print(f"❌ ONNX変換エラー: {e}")
            traceback.print_exc()
            return False
    
    def _create_npu_session(self) -> bool:
        """NPUセッション作成"""
        try:
            print("🚀 NPUセッション作成中...")
            
            if not os.path.exists(self.onnx_model_path):
                print(f"❌ ONNXファイルが見つかりません: {self.onnx_model_path}")
                return False
            
            # DirectMLプロバイダー設定（最適化）
            providers = [
                ('DmlExecutionProvider', {
                    'device_id': self.device_id,
                    'enable_dynamic_graph_fusion': True,
                    'enable_graph_optimization': True,
                    'disable_memory_arena': False,
                    'memory_limit_mb': 8192,  # メモリ制限増加
                    'enable_graph_serialization': True,  # グラフシリアライゼーション有効
                })
            ]
            
            # セッションオプション設定（最適化）
            session_options = ort.SessionOptions()
            session_options.enable_mem_pattern = True  # メモリパターン有効
            session_options.enable_cpu_mem_arena = False
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL  # 並列実行
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.inter_op_num_threads = 4  # スレッド数増加
            session_options.intra_op_num_threads = 4
            
            # NPUセッション作成
            self.npu_session = ort.InferenceSession(
                self.onnx_model_path,
                providers=providers,
                sess_options=session_options
            )
            
            # プロバイダー確認
            active_providers = self.npu_session.get_providers()
            print(f"📋 アクティブプロバイダー: {active_providers}")
            
            if 'DmlExecutionProvider' not in active_providers:
                print("⚠️ DirectMLプロバイダーが無効")
                return False
            
            # 入出力情報確認
            input_info = self.npu_session.get_inputs()
            output_info = self.npu_session.get_outputs()
            
            print("✅ NPUセッション作成成功")
            print(f"📥 入力数: {len(input_info)}")
            for i, inp in enumerate(input_info):
                print(f"  {i}: {inp.name} {inp.shape} {inp.type}")
            print(f"📤 出力数: {len(output_info)}")
            for i, out in enumerate(output_info):
                print(f"  {i}: {out.name} {out.shape} {out.type}")
            
            return True
            
        except Exception as e:
            print(f"❌ NPUセッション作成エラー: {e}")
            traceback.print_exc()
            return False
    
    def _test_npu_inference(self) -> bool:
        """NPU推論テスト"""
        try:
            print("🧪 NPU推論テスト開始...")
            
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
            
            print(f"📊 テスト入力形状: input_ids{input_ids.shape}, attention_mask{attention_mask.shape}")
            
            # NPU推論実行
            test_start = time.time()
            outputs = self.npu_session.run(
                ['logits'],
                {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }
            )
            test_time = time.time() - test_start
            
            logits = outputs[0]
            print(f"✅ NPU推論テスト成功: logits形状{logits.shape}, 実行時間{test_time:.3f}秒")
            
            # 複数回実行でNPU負荷確認
            print("🔥 NPU負荷確認テスト実行中...")
            for i in range(20):  # 20回実行
                start = time.time()
                self.npu_session.run(
                    ['logits'],
                    {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask
                    }
                )
                elapsed = time.time() - start
                
                if i % 5 == 0:
                    print(f"  🔄 NPU負荷テスト {i+1}/20 ({elapsed:.3f}秒)")
            
            print("✅ NPU負荷確認テスト完了")
            print("🎯 タスクマネージャーでNPU使用率を確認してください")
            
            return True
            
        except Exception as e:
            print(f"❌ NPU推論テストエラー: {e}")
            traceback.print_exc()
            return False
    
    def generate_with_npu(self, prompt: str, max_new_tokens: int = 50, 
                         temperature: float = 0.7, top_k: int = 50, 
                         top_p: float = 0.95) -> Dict[str, Any]:
        """NPUを使用したテキスト生成"""
        if not self.is_npu_ready:
            return {"error": "NPUが準備されていません"}
        
        try:
            print(f"🚀 NPU生成開始: \"{prompt}\"")
            generation_start = time.time()
            
            # 入力トークン化
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            input_ids = inputs['input_ids'].numpy().astype(np.int64)
            attention_mask = inputs['attention_mask'].numpy().astype(np.int64)
            
            print(f"📊 入力形状: input_ids{input_ids.shape}")
            
            # 生成ループ
            generated_tokens = []
            current_input_ids = input_ids.copy()
            current_attention_mask = attention_mask.copy()
            
            for step in range(max_new_tokens):
                # NPU推論実行
                npu_start = time.time()
                outputs = self.npu_session.run(
                    ['logits'],
                    {
                        'input_ids': current_input_ids,
                        'attention_mask': current_attention_mask
                    }
                )
                npu_time = time.time() - npu_start
                
                self.npu_inference_count += 1
                self.total_npu_time += npu_time
                
                logits = outputs[0]
                next_token_logits = logits[0, -1, :]  # 最後のトークンのlogits
                
                # 温度適用
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Top-k フィルタリング
                if top_k > 0:
                    indices_to_remove = next_token_logits < np.partition(next_token_logits, -top_k)[-top_k]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # ソフトマックス適用
                exp_logits = np.exp(next_token_logits - np.max(next_token_logits))
                probabilities = exp_logits / np.sum(exp_logits)
                
                # Top-p フィルタリング
                if top_p < 1.0:
                    sorted_indices = np.argsort(probabilities)[::-1]
                    cumulative_probs = np.cumsum(probabilities[sorted_indices])
                    cutoff_index = np.searchsorted(cumulative_probs, top_p)
                    
                    # 上位p%以外を0に
                    filtered_probs = np.zeros_like(probabilities)
                    filtered_probs[sorted_indices[:cutoff_index+1]] = probabilities[sorted_indices[:cutoff_index+1]]
                    probabilities = filtered_probs / np.sum(filtered_probs)
                
                # トークン選択
                next_token_id = np.random.choice(len(probabilities), p=probabilities)
                generated_tokens.append(next_token_id)
                
                # 入力更新（次のステップ用）
                new_input_ids = np.concatenate([current_input_ids, [[next_token_id]]], axis=1)
                new_attention_mask = np.concatenate([current_attention_mask, [[1]]], axis=1)
                
                # シーケンス長制限
                if new_input_ids.shape[1] > 512:
                    new_input_ids = new_input_ids[:, -512:]
                    new_attention_mask = new_attention_mask[:, -512:]
                
                current_input_ids = new_input_ids
                current_attention_mask = new_attention_mask
                
                # 終了トークンチェック
                if next_token_id == self.tokenizer.eos_token_id:
                    print(f"🔚 終了トークン検出 (ステップ {step+1})")
                    break
                
                # 進捗表示
                if (step + 1) % 10 == 0:
                    print(f"  🔄 生成ステップ {step+1}/{max_new_tokens} (NPU: {npu_time:.3f}秒)")
            
            # 生成テキストデコード
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            full_text = prompt + generated_text
            
            generation_time = time.time() - generation_start
            avg_npu_time = self.total_npu_time / self.npu_inference_count if self.npu_inference_count > 0 else 0
            
            print(f"✅ NPU生成完了: {len(generated_tokens)}トークン, {generation_time:.2f}秒")
            print(f"📊 平均NPU推論時間: {avg_npu_time:.3f}秒")
            
            return {
                "generated_text": full_text,
                "generation_time": generation_time,
                "input_tokens": len(input_ids[0]),
                "output_tokens": len(generated_tokens),
                "tokens_per_sec": len(generated_tokens) / generation_time,
                "npu_inference_count": len(generated_tokens),
                "total_npu_time": self.total_npu_time,
                "avg_npu_time": avg_npu_time,
                "inference_method": "True NPU"
            }
            
        except Exception as e:
            print(f"❌ NPU生成エラー: {e}")
            traceback.print_exc()
            return {"error": f"NPU生成エラー: {e}"}
    
    def get_npu_stats(self) -> Dict[str, Any]:
        """NPU統計情報取得"""
        return {
            "is_npu_ready": self.is_npu_ready,
            "npu_inference_count": self.npu_inference_count,
            "total_npu_time": self.total_npu_time,
            "avg_npu_time": self.total_npu_time / self.npu_inference_count if self.npu_inference_count > 0 else 0,
            "onnx_model_path": self.onnx_model_path,
            "device_id": self.device_id
        }
    
    def cleanup(self):
        """リソースクリーンアップ"""
        if self.npu_session:
            del self.npu_session
            self.npu_session = None
        
        print("🧹 真のNPU処理エンジンクリーンアップ完了")

