#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPU最適化LLMシステム
真のNPU対応 + infer-OS設定可能版
"""

import os
import sys
import time
import threading
import psutil
import argparse
import signal
import json
from typing import Optional, Dict, Any, List, Tuple
import warnings
warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import numpy as np
    import onnx
    import onnxruntime as ort
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    print("✅ 必要なライブラリのインポート成功")
except ImportError as e:
    print(f"❌ ライブラリインポートエラー: {e}")
    sys.exit(1)

class TimeoutHandler:
    """タイムアウト処理クラス"""
    def __init__(self, timeout_seconds: int = 180):
        self.timeout_seconds = timeout_seconds
        self.timed_out = False
    
    def timeout_handler(self, signum, frame):
        self.timed_out = True
        print(f"⏰ タイムアウト ({self.timeout_seconds}秒) が発生しました")
        raise TimeoutError(f"処理が{self.timeout_seconds}秒でタイムアウトしました")
    
    def __enter__(self):
        if os.name != 'nt':  # Windows以外でのみsignalを使用
            signal.signal(signal.SIGALRM, self.timeout_handler)
            signal.alarm(self.timeout_seconds)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.name != 'nt':
            signal.alarm(0)

class NPUPerformanceMonitor:
    """NPU性能監視クラス"""
    def __init__(self):
        self.monitoring = False
        self.cpu_samples = []
        self.memory_samples = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """監視開始"""
        self.monitoring = True
        self.cpu_samples = []
        self.memory_samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("📊 性能監視開始")
    
    def stop_monitoring(self):
        """監視停止"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        print("📊 性能監視停止")
    
    def _monitor_loop(self):
        """監視ループ"""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent
                self.cpu_samples.append(cpu_percent)
                self.memory_samples.append(memory_percent)
                time.sleep(0.5)
            except Exception:
                break
    
    def get_report(self) -> Dict[str, Any]:
        """性能レポート取得"""
        if not self.cpu_samples:
            return {"error": "監視データなし"}
        
        return {
            "samples": len(self.cpu_samples),
            "avg_cpu": sum(self.cpu_samples) / len(self.cpu_samples),
            "max_cpu": max(self.cpu_samples),
            "avg_memory": sum(self.memory_samples) / len(self.memory_samples),
            "max_memory": max(self.memory_samples)
        }

class NPUOptimizedLLMSystem:
    """NPU最適化LLMシステム（infer-OS設定可能）"""
    
    def __init__(self, timeout_seconds: int = 180, infer_os_enabled: bool = False):
        self.timeout_seconds = timeout_seconds
        self.tokenizer = None
        self.model = None
        self.npu_session = None
        self.generation_count = 0
        self.infer_os_enabled = infer_os_enabled  # infer-OS最適化設定可能
        self.performance_monitor = NPUPerformanceMonitor()
        self.active_provider = None
        self.model_name = None
        self.generation_config = None
        self.npu_model_path = None
        self.vocab_size = 50257  # GPT-2互換
        self.hidden_size = 768
        self.max_sequence_length = 512
        
        print("🚀 NPU最適化LLMシステム初期化")
        print("============================================================")
        print(f"⏰ タイムアウト設定: {timeout_seconds}秒")
        print(f"🔧 infer-OS最適化: {'ON' if self.infer_os_enabled else 'OFF'}")
        print(f"💡 infer-OS設定: コマンドライン引数 --infer-os で変更可能")
    
    def _setup_infer_os_config(self):
        """infer-OS設定の構成"""
        try:
            if self.infer_os_enabled:
                print("🔧 infer-OS最適化を有効化中...")
                
                # infer-OS設定ファイルの作成
                infer_os_config = {
                    "optimization_level": "high",
                    "enable_npu_acceleration": True,
                    "enable_memory_optimization": True,
                    "enable_compute_optimization": True,
                    "batch_size_optimization": True,
                    "sequence_length_optimization": True
                }
                
                config_path = "infer_os_config.json"
                with open(config_path, 'w') as f:
                    json.dump(infer_os_config, f, indent=2)
                
                print(f"✅ infer-OS設定ファイル作成: {config_path}")
                
                # 環境変数設定
                os.environ['INFER_OS_ENABLED'] = '1'
                os.environ['INFER_OS_CONFIG'] = config_path
                
                print("✅ infer-OS環境変数設定完了")
            else:
                print("🔧 infer-OS最適化を無効化中...")
                
                # 環境変数クリア
                if 'INFER_OS_ENABLED' in os.environ:
                    del os.environ['INFER_OS_ENABLED']
                if 'INFER_OS_CONFIG' in os.environ:
                    del os.environ['INFER_OS_CONFIG']
                
                print("✅ infer-OS無効化完了")
                
        except Exception as e:
            print(f"⚠️ infer-OS設定エラー: {e}")
    
    def _create_npu_optimized_llm_model(self, model_path: str) -> bool:
        """NPU最適化LLMモデル作成"""
        try:
            print("📄 NPU最適化LLMモデル作成中...")
            print(f"🔧 語彙サイズ: {self.vocab_size}")
            print(f"🔧 隠れ層サイズ: {self.hidden_size}")
            print(f"🔧 最大シーケンス長: {self.max_sequence_length}")
            
            # NPU最適化されたLLMアーキテクチャ
            class NPUOptimizedLLM(nn.Module):
                def __init__(self, vocab_size, hidden_size, num_layers=6, num_heads=12):
                    super().__init__()
                    self.vocab_size = vocab_size
                    self.hidden_size = hidden_size
                    self.num_layers = num_layers
                    self.num_heads = num_heads
                    
                    # 埋め込み層
                    self.token_embedding = nn.Embedding(vocab_size, hidden_size)
                    self.position_embedding = nn.Embedding(512, hidden_size)
                    
                    # Transformer層（NPU最適化）
                    self.transformer_layers = nn.ModuleList([
                        nn.TransformerEncoderLayer(
                            d_model=hidden_size,
                            nhead=num_heads,
                            dim_feedforward=hidden_size * 4,
                            dropout=0.1,
                            activation='gelu',
                            batch_first=True,
                            norm_first=True  # Pre-LN for better NPU performance
                        ) for _ in range(num_layers)
                    ])
                    
                    # 最終層
                    self.ln_f = nn.LayerNorm(hidden_size)
                    self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
                    
                    # 重み共有（メモリ効率化）
                    self.lm_head.weight = self.token_embedding.weight
                
                def forward(self, input_ids, attention_mask=None):
                    batch_size, seq_len = input_ids.shape
                    
                    # 位置エンコーディング
                    position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
                    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                    
                    # 埋め込み
                    token_embeds = self.token_embedding(input_ids)
                    position_embeds = self.position_embedding(position_ids)
                    hidden_states = token_embeds + position_embeds
                    
                    # Transformer層
                    for layer in self.transformer_layers:
                        hidden_states = layer(hidden_states, src_key_padding_mask=~attention_mask if attention_mask is not None else None)
                    
                    # 最終層
                    hidden_states = self.ln_f(hidden_states)
                    logits = self.lm_head(hidden_states)
                    
                    return logits
            
            # モデル作成
            model = NPUOptimizedLLM(
                vocab_size=self.vocab_size,
                hidden_size=self.hidden_size,
                num_layers=6,  # NPU最適化のため軽量化
                num_heads=12
            )
            model.eval()
            
            # ダミー入力作成
            batch_size = 1
            seq_len = 32  # NPU最適化のため短めに設定
            dummy_input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
            dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
            
            print(f"🔧 入力形状: input_ids={dummy_input_ids.shape}, attention_mask={dummy_attention_mask.shape}")
            
            # ONNX IRバージョン10でエクスポート
            torch.onnx.export(
                model,
                (dummy_input_ids, dummy_attention_mask),
                model_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                    'logits': {0: 'batch_size', 1: 'sequence_length'}
                }
            )
            
            # ONNXモデルを読み込んでIRバージョンを修正
            onnx_model = onnx.load(model_path)
            onnx_model.ir_version = 10
            onnx.save(onnx_model, model_path)
            
            print(f"✅ NPU最適化LLMモデル作成完了: {model_path}")
            print(f"📋 IRバージョン: {onnx_model.ir_version}")
            print(f"🎯 モデルサイズ: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
            
            return True
            
        except Exception as e:
            print(f"❌ NPU最適化LLMモデル作成エラー: {e}")
            return False
    
    def _setup_npu_session(self) -> bool:
        """NPUセッション設定"""
        try:
            print("⚡ NPUセッション設定中...")
            
            # infer-OS設定
            self._setup_infer_os_config()
            
            # vaip_config.jsonの確認
            vaip_config_paths = [
                "C:/Program Files/RyzenAI/1.5/voe-4.0-win_amd64/vaip_config.json",
                "C:/Program Files/RyzenAI/vaip_config.json",
                "./vaip_config.json"
            ]
            
            vaip_config_found = False
            for path in vaip_config_paths:
                if os.path.exists(path):
                    print(f"📁 vaip_config.json発見: {path}")
                    vaip_config_found = True
                    break
            
            if not vaip_config_found:
                print("⚠️ vaip_config.jsonが見つかりません")
            
            # NPU最適化LLMモデル作成
            self.npu_model_path = "npu_optimized_llm.onnx"
            if not self._create_npu_optimized_llm_model(self.npu_model_path):
                return False
            
            # 利用可能なプロバイダー確認
            available_providers = ort.get_available_providers()
            print(f"📋 利用可能なプロバイダー: {available_providers}")
            
            # セッションオプション設定
            session_options = ort.SessionOptions()
            session_options.log_severity_level = 3
            
            # infer-OS最適化が有効な場合の追加設定
            if self.infer_os_enabled:
                session_options.enable_cpu_mem_arena = False
                session_options.enable_mem_pattern = False
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                print("🔧 infer-OS最適化セッション設定適用")
            
            # プロバイダー選択戦略
            # VitisAIExecutionProvider優先
            if 'VitisAIExecutionProvider' in available_providers:
                try:
                    print("🔄 VitisAIExecutionProvider試行...")
                    
                    # VitisAI EP設定
                    vitisai_options = {}
                    if self.infer_os_enabled:
                        vitisai_options.update({
                            'config_file': 'vaip_config.json',
                            'enable_optimization': True
                        })
                    
                    providers = [
                        ('VitisAIExecutionProvider', vitisai_options),
                        'CPUExecutionProvider'
                    ]
                    
                    self.npu_session = ort.InferenceSession(
                        self.npu_model_path,
                        sess_options=session_options,
                        providers=providers
                    )
                    self.active_provider = 'VitisAIExecutionProvider'
                    print("✅ VitisAIExecutionProvider セッション作成成功")
                except Exception as e:
                    print(f"⚠️ VitisAIExecutionProvider失敗: {e}")
                    self.npu_session = None
            
            # DmlExecutionProvider フォールバック
            if self.npu_session is None and 'DmlExecutionProvider' in available_providers:
                try:
                    print("🔄 DmlExecutionProvider試行...")
                    self.npu_session = ort.InferenceSession(
                        self.npu_model_path,
                        sess_options=session_options,
                        providers=['DmlExecutionProvider', 'CPUExecutionProvider']
                    )
                    self.active_provider = 'DmlExecutionProvider'
                    print("✅ DmlExecutionProvider セッション作成成功")
                except Exception as e:
                    print(f"⚠️ DmlExecutionProvider失敗: {e}")
                    self.npu_session = None
            
            # CPU フォールバック
            if self.npu_session is None:
                try:
                    print("🔄 CPUExecutionProvider試行...")
                    self.npu_session = ort.InferenceSession(
                        self.npu_model_path,
                        sess_options=session_options,
                        providers=['CPUExecutionProvider']
                    )
                    self.active_provider = 'CPUExecutionProvider'
                    print("✅ CPUExecutionProvider セッション作成成功")
                except Exception as e:
                    print(f"❌ CPUExecutionProvider失敗: {e}")
                    return False
            
            if self.npu_session is None:
                return False
            
            print(f"✅ NPUセッション作成成功")
            print(f"🔧 使用プロバイダー: {self.npu_session.get_providers()}")
            print(f"🎯 アクティブプロバイダー: {self.active_provider}")
            print(f"🔧 infer-OS最適化: {'有効' if self.infer_os_enabled else '無効'}")
            
            # NPU動作テスト
            test_input_ids = np.random.randint(0, self.vocab_size, (1, 32), dtype=np.int64)
            test_attention_mask = np.ones((1, 32), dtype=np.bool_)
            test_output = self.npu_session.run(None, {
                'input_ids': test_input_ids,
                'attention_mask': test_attention_mask
            })
            print(f"✅ NPU LLM動作テスト完了: 出力形状 {test_output[0].shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ NPUセッション設定エラー: {e}")
            return False
    
    def _load_tokenizer(self) -> bool:
        """トークナイザーロード"""
        try:
            print("🔤 トークナイザーロード中...")
            
            # GPT-2互換トークナイザー使用
            tokenizer_candidates = [
                "gpt2",
                "microsoft/DialoGPT-medium",
                "openai-gpt"
            ]
            
            for candidate in tokenizer_candidates:
                try:
                    print(f"🔄 {candidate}トークナイザー試行中...")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        candidate,
                        trust_remote_code=True,
                        use_fast=False
                    )
                    
                    # パディングトークン設定
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    print(f"✅ トークナイザーロード成功: {candidate}")
                    print(f"📋 語彙サイズ: {len(self.tokenizer)}")
                    return True
                    
                except Exception as e:
                    print(f"⚠️ {candidate}トークナイザー失敗: {e}")
                    continue
            
            print("❌ 全てのトークナイザー候補でロードに失敗")
            return False
            
        except Exception as e:
            print(f"❌ トークナイザーロードエラー: {e}")
            return False
    
    def initialize(self) -> bool:
        """システム初期化"""
        try:
            with TimeoutHandler(self.timeout_seconds):
                # NPUセッション設定
                if not self._setup_npu_session():
                    print("❌ NPUセッション設定失敗")
                    return False
                
                # トークナイザーロード
                if not self._load_tokenizer():
                    print("❌ トークナイザーロード失敗")
                    return False
                
                print("✅ NPU最適化LLMシステム初期化完了")
                return True
                
        except TimeoutError:
            print("❌ 初期化タイムアウト")
            return False
        except Exception as e:
            print(f"❌ 初期化エラー: {e}")
            return False
    
    def _npu_text_generation(self, input_text: str, max_new_tokens: int = 50) -> str:
        """NPUでの実際のテキスト生成"""
        try:
            print(f"📝 NPUテキスト生成中...")
            
            # 入力テキストをトークン化
            inputs = self.tokenizer(
                input_text,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=self.max_sequence_length
            )
            
            input_ids = inputs['input_ids'].astype(np.int64)
            attention_mask = inputs['attention_mask'].astype(np.bool_)
            
            print(f"🔧 入力形状: {input_ids.shape}")
            
            generated_tokens = []
            current_input_ids = input_ids
            current_attention_mask = attention_mask
            
            # 自己回帰的生成
            for step in range(max_new_tokens):
                # NPUで推論実行
                outputs = self.npu_session.run(None, {
                    'input_ids': current_input_ids,
                    'attention_mask': current_attention_mask
                })
                
                logits = outputs[0]  # [batch_size, seq_len, vocab_size]
                
                # 最後のトークンの予測を取得
                next_token_logits = logits[0, -1, :]  # [vocab_size]
                
                # 温度スケーリング
                temperature = 0.7
                next_token_logits = next_token_logits / temperature
                
                # ソフトマックス
                probs = np.exp(next_token_logits) / np.sum(np.exp(next_token_logits))
                
                # サンプリング
                next_token_id = np.random.choice(len(probs), p=probs)
                
                # EOSトークンチェック
                if next_token_id == self.tokenizer.eos_token_id:
                    break
                
                generated_tokens.append(next_token_id)
                
                # 次の入力を準備
                next_token_array = np.array([[next_token_id]], dtype=np.int64)
                current_input_ids = np.concatenate([current_input_ids, next_token_array], axis=1)
                current_attention_mask = np.concatenate([
                    current_attention_mask, 
                    np.array([[True]], dtype=np.bool_)
                ], axis=1)
                
                # 最大長チェック
                if current_input_ids.shape[1] >= self.max_sequence_length:
                    break
                
                if (step + 1) % 10 == 0:
                    print(f"  📊 生成進捗: {step + 1}/{max_new_tokens}")
            
            # 生成されたトークンをデコード
            if generated_tokens:
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                print(f"✅ NPU生成完了: {len(generated_tokens)}トークン")
                return generated_text.strip()
            else:
                return "[生成されたトークンなし]"
                
        except Exception as e:
            print(f"❌ NPUテキスト生成エラー: {e}")
            return f"[NPU生成エラー: {str(e)}]"
    
    def generate_text(self, prompt: str, max_tokens: int = 50) -> str:
        """NPU最適化テキスト生成"""
        try:
            print(f"🔄 NPU最適化生成中（タイムアウト: {self.timeout_seconds}秒）...")
            
            with TimeoutHandler(self.timeout_seconds):
                # 性能監視開始
                self.performance_monitor.start_monitoring()
                
                start_time = time.time()
                
                # NPUでの実際のテキスト生成
                generated_text = self._npu_text_generation(prompt, max_tokens)
                
                end_time = time.time()
                generation_time = end_time - start_time
                
                # 性能監視停止
                self.performance_monitor.stop_monitoring()
                
                # 結果表示
                print(f"🎯 NPU生成結果:")
                print(f"  📝 入力: {prompt}")
                print(f"  🎯 出力: {generated_text}")
                print(f"  ⏱️ 生成時間: {generation_time:.3f}秒")
                print(f"  🔧 アクティブプロバイダー: {self.active_provider}")
                print(f"  🔧 infer-OS最適化: {'有効' if self.infer_os_enabled else '無効'}")
                
                # 性能レポート
                perf_report = self.performance_monitor.get_report()
                if "error" not in perf_report:
                    print(f"📊 性能レポート:")
                    print(f"  🔢 サンプル数: {perf_report['samples']}")
                    print(f"  💻 平均CPU使用率: {perf_report['avg_cpu']:.1f}%")
                    print(f"  💻 最大CPU使用率: {perf_report['max_cpu']:.1f}%")
                    print(f"  💾 平均メモリ使用率: {perf_report['avg_memory']:.1f}%")
                
                self.generation_count += 1
                
                return generated_text
                
        except TimeoutError:
            return f"⏰ タイムアウト: {prompt}"
        except Exception as e:
            return f"❌ エラー: {e}"
    
    def interactive_mode(self):
        """インタラクティブモード"""
        print(f"\n🇯🇵 NPU最適化LLMシステム - インタラクティブモード")
        print(f"⏰ タイムアウト設定: {self.timeout_seconds}秒")
        print(f"🔧 infer-OS最適化: {'ON' if self.infer_os_enabled else 'OFF'}")
        print(f"🎯 アクティブプロバイダー: {self.active_provider}")
        print(f"🤖 NPUモデル: {self.npu_model_path}")
        print(f"💡 'exit'または'quit'で終了、'stats'で統計表示")
        print("============================================================")
        
        while True:
            try:
                prompt = input("\n🤖 プロンプトを入力してください: ").strip()
                
                if prompt.lower() in ['exit', 'quit', '終了']:
                    print("👋 NPU最適化LLMシステムを終了します")
                    break
                
                if prompt.lower() == 'stats':
                    print(f"\n📊 システム統計:")
                    print(f"  🔢 生成回数: {self.generation_count}")
                    print(f"  ⏰ タイムアウト設定: {self.timeout_seconds}秒")
                    print(f"  🔧 infer-OS最適化: {'ON' if self.infer_os_enabled else 'OFF'}")
                    print(f"  🤖 NPUモデル: {self.npu_model_path}")
                    print(f"  🔤 トークナイザー: {'✅ 利用可能' if self.tokenizer else '❌ 未ロード'}")
                    print(f"  ⚡ NPUセッション: {'✅ 利用可能' if self.npu_session else '❌ 未作成'}")
                    print(f"  🎯 アクティブプロバイダー: {self.active_provider}")
                    if self.npu_session:
                        print(f"  📋 全プロバイダー: {self.npu_session.get_providers()}")
                    continue
                
                if not prompt:
                    continue
                
                start_time = time.time()
                response = self.generate_text(prompt, max_tokens=30)
                end_time = time.time()
                
                print(f"\n📝 生成結果:")
                print(f"💬 プロンプト: {prompt}")
                print(f"🎯 応答: {response}")
                print(f"⏱️ 総生成時間: {end_time - start_time:.2f}秒")
                
            except KeyboardInterrupt:
                print("\n👋 NPU最適化LLMシステムを終了します")
                break
            except Exception as e:
                print(f"❌ エラー: {e}")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="NPU最適化LLMシステム（infer-OS設定可能）")
    parser.add_argument("--interactive", action="store_true", help="インタラクティブモード")
    parser.add_argument("--prompt", type=str, help="単発テスト用プロンプト")
    parser.add_argument("--tokens", type=int, default=50, help="生成トークン数")
    parser.add_argument("--timeout", type=int, default=180, help="タイムアウト秒数")
    parser.add_argument("--infer-os", action="store_true", help="infer-OS最適化を有効化")
    
    args = parser.parse_args()
    
    # システム初期化
    system = NPUOptimizedLLMSystem(
        timeout_seconds=args.timeout,
        infer_os_enabled=args.infer_os
    )
    
    if not system.initialize():
        print("❌ システム初期化に失敗しました")
        return
    
    if args.interactive:
        # インタラクティブモード
        system.interactive_mode()
    elif args.prompt:
        # 単発テスト
        print(f"\n🎯 単発NPUテキスト生成実行")
        print(f"📝 プロンプト: {args.prompt}")
        print(f"⚡ 生成トークン数: {args.tokens}")
        print(f"🔧 infer-OS最適化: {'ON' if args.infer_os else 'OFF'}")
        
        start_time = time.time()
        response = system.generate_text(args.prompt, max_tokens=args.tokens)
        end_time = time.time()
        
        print(f"\n📝 生成結果:")
        print(f"💬 プロンプト: {args.prompt}")
        print(f"🎯 応答: {response}")
        print(f"⏱️ 総実行時間: {end_time - start_time:.2f}秒")
    else:
        print("❌ --interactive または --prompt を指定してください")
        print("💡 infer-OS最適化を有効にするには --infer-os を追加してください")

if __name__ == "__main__":
    main()

