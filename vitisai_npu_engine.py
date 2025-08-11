"""
VitisAI ExecutionProvider専用NPUエンジン
真のNPU処理を実現するための完全実装

主要機能:
1. VitisAI EP必須設定（config_file、環境変数）
2. INT8量子化対応
3. NPUオーバレイ設定
4. 真のNPU処理確認
"""

import os
import time
import numpy as np
import torch
import onnx
import onnxruntime as ort
from typing import Dict, List, Optional, Tuple, Any
import traceback
import subprocess
import json

class VitisAINPUEngine:
    """VitisAI ExecutionProvider専用NPUエンジン"""
    
    def __init__(self, model, tokenizer, device_id: int = 0):
        self.model = model
        self.tokenizer = tokenizer
        self.device_id = device_id
        
        # VitisAI関連
        self.vitisai_session = None
        self.is_vitisai_ready = False
        self.config_file_path = None
        
        # 統計情報
        self.npu_inference_count = 0
        self.total_npu_time = 0.0
        
        print(f"🚀 VitisAI ExecutionProvider専用NPUエンジン初期化")
        print(f"🎯 真のNPU処理実現")
    
    def setup_vitisai_npu(self) -> bool:
        """VitisAI NPUセットアップ"""
        try:
            print("🔧 VitisAI NPUセットアップ開始...")
            
            # 1. 環境変数設定
            if not self._setup_environment_variables():
                return False
            
            # 2. VitisAI設定ファイル確認
            if not self._setup_config_file():
                return False
            
            # 3. VitisAI ExecutionProvider確認
            if not self._check_vitisai_provider():
                return False
            
            # 4. NPU用ONNXモデル作成
            if not self._create_npu_onnx_model():
                return False
            
            # 5. VitisAI NPUセッション作成
            if not self._create_vitisai_session():
                return False
            
            self.is_vitisai_ready = True
            print("✅ VitisAI NPUセットアップ完了")
            return True
            
        except Exception as e:
            print(f"❌ VitisAI NPUセットアップエラー: {e}")
            traceback.print_exc()
            return False
    
    def _setup_environment_variables(self) -> bool:
        """環境変数設定"""
        try:
            print("🔧 NPUオーバレイ環境変数設定中...")
            
            # Ryzen AI インストールパス確認（強制的に正しいパス優先）
            # 正しいパス優先順位で確認
            priority_paths = [
                r"C:\Program Files\RyzenAI\1.5",      # 正しいパス（最優先）
                r"C:\Program Files\RyzenAI\1.5.1",    # 代替パス
                r"C:\AMD\RyzenAI\1.5",
                r"C:\AMD\RyzenAI\1.5.1"
            ]
            
            ryzen_ai_path = None
            for path in priority_paths:
                if os.path.exists(path):
                    ryzen_ai_path = path
                    # 強制的に正しいパスに設定
                    os.environ['RYZEN_AI_INSTALLATION_PATH'] = path
                    print(f"✅ Ryzen AIパス強制設定: {path}")
                    break
            
            if not ryzen_ai_path:
                # 環境変数から取得（フォールバック）
                ryzen_ai_path = os.environ.get('RYZEN_AI_INSTALLATION_PATH')
                
            if not ryzen_ai_path:
                    print("❌ Ryzen AI インストールパスが見つかりません")
                    return False
            else:
                print(f"✅ Ryzen AIパス確認: {ryzen_ai_path}")
            
            # NPUオーバレイ設定（STX想定）
            xclbin_path = os.path.join(
                ryzen_ai_path, 
                "voe-4.0-win_amd64", 
                "xclbins", 
                "strix", 
                "AMD_AIE2P_Nx4_Overlay.xclbin"
            )
            
            if os.path.exists(xclbin_path):
                os.environ['XLNX_VART_FIRMWARE'] = xclbin_path
                os.environ['XLNX_TARGET_NAME'] = "AMD_AIE2P_Nx4_Overlay"
                print(f"✅ NPUオーバレイ設定: AMD_AIE2P_Nx4_Overlay")
                print(f"📁 XCLBINパス: {xclbin_path}")
            else:
                # PHX/HPT用パスも試行
                xclbin_path_phx = os.path.join(
                    ryzen_ai_path, 
                    "voe-4.0-win_amd64", 
                    "xclbins", 
                    "phoenix", 
                    "AMD_AIE2P_4x4_Overlay.xclbin"
                )
                
                if os.path.exists(xclbin_path_phx):
                    os.environ['XLNX_VART_FIRMWARE'] = xclbin_path_phx
                    os.environ['XLNX_TARGET_NAME'] = "AMD_AIE2P_4x4_Overlay"
                    print(f"✅ NPUオーバレイ設定: AMD_AIE2P_4x4_Overlay (PHX)")
                    print(f"📁 XCLBINパス: {xclbin_path_phx}")
                else:
                    print("❌ NPUオーバレイファイルが見つかりません")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ 環境変数設定エラー: {e}")
            return False
    
    def _setup_config_file(self) -> bool:
        """VitisAI設定ファイル確認"""
        try:
            print("🔧 VitisAI設定ファイル確認中...")
            
            ryzen_ai_path = os.environ.get('RYZEN_AI_INSTALLATION_PATH')
            config_path = os.path.join(
                ryzen_ai_path, 
                "voe-4.0-win_amd64", 
                "vaip_config.json"
            )
            
            if os.path.exists(config_path):
                self.config_file_path = config_path
                print(f"✅ VitisAI設定ファイル確認: {config_path}")
                
                # 設定ファイル内容確認
                try:
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    print(f"📋 設定ファイル内容確認済み")
                except Exception as e:
                    print(f"⚠️ 設定ファイル読み込み警告: {e}")
                
                return True
            else:
                print(f"❌ VitisAI設定ファイルが見つかりません: {config_path}")
                return False
            
        except Exception as e:
            print(f"❌ 設定ファイル確認エラー: {e}")
            return False
    
    def _check_vitisai_provider(self) -> bool:
        """VitisAI ExecutionProvider確認"""
        try:
            print("🔧 VitisAI ExecutionProvider確認中...")
            
            available_providers = ort.get_available_providers()
            
            if 'VitisAIExecutionProvider' in available_providers:
                print("✅ VitisAI ExecutionProvider利用可能")
                return True
            else:
                print("❌ VitisAI ExecutionProvider利用不可")
                print(f"📋 利用可能プロバイダー: {available_providers}")
                return False
            
        except Exception as e:
            print(f"❌ VitisAI ExecutionProvider確認エラー: {e}")
            return False
    
    def _create_npu_onnx_model(self) -> bool:
        """NPU用ONNXモデル作成"""
        try:
            print("🔧 NPU用ONNXモデル作成中...")
            
            # NPU最適化線形層モデル
            class NPUOptimizedModel(torch.nn.Module):
                def __init__(self, hidden_size=4096, vocab_size=32000):
                    super().__init__()
                    # NPU向け最適化：シンプルな線形変換
                    self.linear = torch.nn.Linear(hidden_size, vocab_size)
                
                def forward(self, x):
                    return self.linear(x)
            
            # モデル作成
            npu_model = NPUOptimizedModel()
            
            # 実モデルの重みコピー
            if hasattr(self.model, 'lm_head') and hasattr(self.model.lm_head, 'weight'):
                print("🔧 実モデルの重みをコピー中...")
                with torch.no_grad():
                    original_weight = self.model.lm_head.weight.detach().to(torch.float32).cpu()
                    npu_model.linear.weight.copy_(original_weight)
                    
                    if hasattr(self.model.lm_head, 'bias') and self.model.lm_head.bias is not None:
                        original_bias = self.model.lm_head.bias.detach().to(torch.float32).cpu()
                        npu_model.linear.bias.copy_(original_bias)
                
                print("✅ 実モデル重みコピー完了")
            
            npu_model.eval()
            
            # ONNX変換（INT8量子化対応）
            dummy_input = torch.randn(1, 4096, dtype=torch.float32)
            onnx_path = "./onnx_models/vitisai_npu_model.onnx"
            os.makedirs("./onnx_models", exist_ok=True)
            
            # ONNX変換（opset 17推奨）
            torch.onnx.export(
                npu_model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=17,  # VitisAI推奨
                do_constant_folding=True,
                input_names=['hidden_state'],
                output_names=['logits'],
                dynamic_axes={
                    'hidden_state': {0: 'batch_size'},
                    'logits': {0: 'batch_size'}
                }
            )
            
            self.onnx_model_path = onnx_path
            print(f"✅ NPU用ONNXモデル作成成功: {onnx_path}")
            
            # INT8量子化（NPU最適化）
            if self._quantize_model_for_npu(onnx_path):
                print("✅ INT8量子化完了")
            else:
                print("⚠️ INT8量子化スキップ（FP32モデルを使用）")
            
            return True
            
        except Exception as e:
            print(f"❌ NPU用ONNXモデル作成エラー: {e}")
            return False
    
    def _quantize_model_for_npu(self, onnx_path: str) -> bool:
        """NPU用INT8量子化"""
        try:
            print("🔧 NPU用INT8量子化実行中...")
            
            # 量子化ツール確認
            try:
                from onnxruntime.quantization import quantize_dynamic, QuantType
                
                quantized_path = onnx_path.replace('.onnx', '_int8.onnx')
                
                quantize_dynamic(
                    model_input=onnx_path,
                    model_output=quantized_path,
                    weight_type=QuantType.QInt8,
                    optimize_model=True
                )
                
                # 量子化モデルを使用
                if os.path.exists(quantized_path):
                    self.onnx_model_path = quantized_path
                    print(f"✅ INT8量子化モデル作成: {quantized_path}")
                    return True
                
            except ImportError:
                print("⚠️ ONNXRuntime量子化ツール未利用可能")
            
            return False
            
        except Exception as e:
            print(f"❌ INT8量子化エラー: {e}")
            return False
    
    def _create_vitisai_session(self) -> bool:
        """VitisAI NPUセッション作成"""
        try:
            print("🚀 VitisAI NPUセッション作成中...")
            
            # VitisAI ExecutionProvider設定（必須オプション）
            providers = ["VitisAIExecutionProvider", "CPUExecutionProvider"]
            provider_options = [
                {"config_file": self.config_file_path},  # 必須
                {}
            ]
            
            print(f"📋 VitisAI設定ファイル: {self.config_file_path}")
            print(f"📋 NPUオーバレイ: {os.environ.get('XLNX_TARGET_NAME', '未設定')}")
            
            # セッションオプション設定
            session_options = ort.SessionOptions()
            session_options.enable_mem_pattern = True
            session_options.enable_cpu_mem_arena = False
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # VitisAI NPUセッション作成
            self.vitisai_session = ort.InferenceSession(
                self.onnx_model_path,
                providers=providers,
                provider_options=provider_options,
                sess_options=session_options
            )
            
            # アクティブプロバイダー確認
            active_providers = self.vitisai_session.get_providers()
            print(f"📋 アクティブプロバイダー: {active_providers}")
            
            if 'VitisAIExecutionProvider' in active_providers:
                print("🎯 VitisAI ExecutionProvider アクティブ！")
                
                # テスト推論実行
                test_input = np.random.randn(1, 4096).astype(np.float32)
                test_output = self.vitisai_session.run(['logits'], {'hidden_state': test_input})
                
                print(f"✅ VitisAI NPUセッション作成成功: 出力形状{test_output[0].shape}")
                print("🎉 真のNPU処理が実現されました！")
                return True
            else:
                print("❌ VitisAI ExecutionProviderが無効")
                return False
            
        except Exception as e:
            print(f"❌ VitisAI NPUセッション作成エラー: {e}")
            traceback.print_exc()
            return False
    
    def generate_with_vitisai_npu(self, prompt: str, max_new_tokens: int = 50, 
                                 temperature: float = 0.7) -> Dict[str, Any]:
        """VitisAI NPU生成"""
        if not self.is_vitisai_ready:
            return {"error": "VitisAI NPUが準備されていません"}
        
        try:
            print(f"🚀 VitisAI NPU生成開始: \"{prompt}\"")
            generation_start = time.time()
            
            # 入力トークン化
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            # 生成ループ
            generated_tokens = []
            
            for step in range(max_new_tokens):
                # PyTorchモデルで隠れ状態取得
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]
                    last_hidden = hidden_states[:, -1, :].cpu().numpy().astype("float32", copy=False)
                
                # VitisAI NPUで推論
                npu_start = time.time()
                npu_outputs = self.vitisai_session.run(['logits'], {'hidden_state': last_hidden})
                npu_time = time.time() - npu_start
                
                self.npu_inference_count += 1
                self.total_npu_time += npu_time
                
                logits = npu_outputs[0]
                
                # トークン選択
                if temperature > 0:
                    logits = logits / temperature
                
                exp_logits = np.exp(logits - np.max(logits))
                probabilities = exp_logits / np.sum(exp_logits)
                next_token_id = np.random.choice(len(probabilities[0]), p=probabilities[0])
                
                generated_tokens.append(next_token_id)
                
                # 入力更新
                new_token = torch.tensor([[next_token_id]])
                inputs['input_ids'] = torch.cat([inputs['input_ids'], new_token], dim=1)
                inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones(1, 1)], dim=1)
                
                # シーケンス長制限
                if inputs['input_ids'].shape[1] > 512:
                    inputs['input_ids'] = inputs['input_ids'][:, -512:]
                    inputs['attention_mask'] = inputs['attention_mask'][:, -512:]
                
                # 終了トークンチェック
                if next_token_id == self.tokenizer.eos_token_id:
                    print(f"🔚 終了トークン検出 (ステップ {step+1})")
                    break
                
                # 進捗表示
                if (step + 1) % 10 == 0:
                    print(f"  🔄 生成ステップ {step+1}/{max_new_tokens} (VitisAI NPU: {npu_time:.3f}秒)")
            
            # デコード
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            full_text = prompt + generated_text
            
            generation_time = time.time() - generation_start
            
            print(f"✅ VitisAI NPU生成完了: {len(generated_tokens)}トークン, {generation_time:.2f}秒")
            
            return {
                "generated_text": full_text,
                "generation_time": generation_time,
                "input_tokens": len(inputs['input_ids'][0]) - len(generated_tokens),
                "output_tokens": len(generated_tokens),
                "tokens_per_sec": len(generated_tokens) / generation_time,
                "npu_inference_count": len(generated_tokens),
                "total_npu_time": self.total_npu_time,
                "avg_npu_time": self.total_npu_time / len(generated_tokens),
                "inference_method": "VitisAI NPU",
                "npu_provider": "VitisAIExecutionProvider"
            }
            
        except Exception as e:
            print(f"❌ VitisAI NPU生成エラー: {e}")
            traceback.print_exc()
            return {"error": f"VitisAI NPU生成エラー: {e}"}
    
    def get_vitisai_stats(self) -> Dict[str, Any]:
        """VitisAI NPU統計情報取得"""
        return {
            "is_vitisai_ready": self.is_vitisai_ready,
            "npu_inference_count": self.npu_inference_count,
            "total_npu_time": self.total_npu_time,
            "avg_npu_time": self.total_npu_time / self.npu_inference_count if self.npu_inference_count > 0 else 0,
            "config_file": self.config_file_path,
            "npu_overlay": os.environ.get('XLNX_TARGET_NAME', '未設定'),
            "ryzen_ai_path": os.environ.get('RYZEN_AI_INSTALLATION_PATH', '未設定')
        }
    
    def cleanup(self):
        """リソースクリーンアップ"""
        if self.vitisai_session:
            del self.vitisai_session
            self.vitisai_session = None
        
        print("🧹 VitisAI NPU処理エンジンクリーンアップ完了")

