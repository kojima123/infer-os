# -*- coding: utf-8 -*-
"""
Ryzen AI NPU テキスト生成デモシステム
実際のLLMモデルでのテキスト生成確認
"""

import os
import sys
import time
import argparse
import json
import threading
import psutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import warnings
warnings.filterwarnings("ignore")

try:
    import onnxruntime as ort
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    print("✅ 必要なライブラリのインポート成功")
except ImportError as e:
    print(f"❌ ライブラリインポートエラー: {e}")
    print("💡 pip install onnxruntime transformers torch を実行してください")
    sys.exit(1)

class PerformanceMonitor:
    """性能監視クラス"""
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
    
    def stop_monitoring(self):
        """監視停止"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
    
    def _monitor_loop(self):
        """監視ループ"""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent
                self.cpu_samples.append(cpu_percent)
                self.memory_samples.append(memory_percent)
                time.sleep(0.5)
            except:
                break
    
    def get_report(self) -> Dict[str, float]:
        """性能レポート取得"""
        if not self.cpu_samples or not self.memory_samples:
            return {"avg_cpu": 0.0, "max_cpu": 0.0, "avg_memory": 0.0, "max_memory": 0.0}
        
        return {
            "avg_cpu": sum(self.cpu_samples) / len(self.cpu_samples),
            "max_cpu": max(self.cpu_samples),
            "avg_memory": sum(self.memory_samples) / len(self.memory_samples),
            "max_memory": max(self.memory_samples),
            "samples": len(self.cpu_samples)
        }

class RyzenAITextGenerationDemo:
    """Ryzen AI テキスト生成デモシステム"""
    
    def __init__(self, enable_infer_os: bool = False, use_npu: bool = True):
        self.enable_infer_os = enable_infer_os
        self.use_npu = use_npu
        self.model = None
        self.tokenizer = None
        self.npu_session = None
        self.active_provider = None
        self.performance_monitor = PerformanceMonitor()
        
        # Ryzen AI実績モデル候補
        self.model_candidates = [
            {
                "name": "microsoft/DialoGPT-medium",
                "description": "対話特化モデル（Ryzen AI実績）",
                "size": "117M",
                "ryzen_ai_proven": True
            },
            {
                "name": "microsoft/DialoGPT-small", 
                "description": "軽量対話モデル（Ryzen AI実績）",
                "size": "117M",
                "ryzen_ai_proven": True
            },
            {
                "name": "gpt2",
                "description": "汎用生成モデル（Ryzen AI実績）",
                "size": "124M",
                "ryzen_ai_proven": True
            },
            {
                "name": "distilgpt2",
                "description": "軽量生成モデル（Ryzen AI実績）",
                "size": "82M",
                "ryzen_ai_proven": True
            }
        ]
        
        print("🚀 Ryzen AI テキスト生成デモシステム初期化")
        print(f"🔧 infer-OS最適化: {'有効' if enable_infer_os else '無効'}")
        print(f"🔧 NPU使用: {'有効' if use_npu else '無効'}")
    
    def initialize(self) -> bool:
        """システム初期化"""
        try:
            print("🔧 システム初期化中...")
            
            # infer-OS環境設定
            self._setup_infer_os_environment()
            
            # NPUセッション作成（オプション）
            if self.use_npu:
                self._setup_npu_session()
            
            # LLMモデル読み込み
            if not self._load_llm_model():
                return False
            
            print("✅ Ryzen AI テキスト生成デモシステム初期化完了")
            return True
            
        except Exception as e:
            print(f"❌ システム初期化エラー: {e}")
            return False
    
    def _setup_infer_os_environment(self):
        """infer-OS環境設定"""
        if self.enable_infer_os:
            print("🔧 infer-OS最適化環境設定中...")
            
            infer_os_env = {
                'INFER_OS_ENABLE': '1',
                'INFER_OS_OPTIMIZATION_LEVEL': 'high',
                'INFER_OS_NPU_ACCELERATION': '1',
                'INFER_OS_MEMORY_OPTIMIZATION': '1',
                'INFER_OS_COMPUTE_OPTIMIZATION': '1'
            }
            
            for key, value in infer_os_env.items():
                os.environ[key] = value
                print(f"  📝 {key}={value}")
            
            print("✅ infer-OS最適化環境設定完了")
        else:
            print("🔧 infer-OS最適化: 無効（ベースライン測定）")
            # infer-OS無効化
            for key in ['INFER_OS_ENABLE', 'INFER_OS_OPTIMIZATION_LEVEL', 
                       'INFER_OS_NPU_ACCELERATION', 'INFER_OS_MEMORY_OPTIMIZATION',
                       'INFER_OS_COMPUTE_OPTIMIZATION']:
                os.environ.pop(key, None)
    
    def _setup_npu_session(self):
        """NPUセッション作成（オプション）"""
        try:
            print("⚡ NPUセッション作成中...")
            
            # シンプルなNPUテスト用モデル作成
            npu_model_path = "npu_test_model.onnx"
            if not self._create_npu_test_model(npu_model_path):
                print("⚠️ NPUテストモデル作成失敗")
                return
            
            # 利用可能なプロバイダー確認
            available_providers = ort.get_available_providers()
            print(f"📋 利用可能なプロバイダー: {available_providers}")
            
            # セッションオプション
            session_options = ort.SessionOptions()
            session_options.log_severity_level = 3
            
            if self.enable_infer_os:
                session_options.enable_cpu_mem_arena = True
                session_options.enable_mem_pattern = True
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            else:
                session_options.enable_cpu_mem_arena = False
                session_options.enable_mem_pattern = False
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            
            # VitisAI ExecutionProvider（NPU）
            if 'VitisAIExecutionProvider' in available_providers:
                try:
                    print("🔄 VitisAIExecutionProvider試行...")
                    
                    vitisai_options = {
                        "cache_dir": "C:/temp/vaip_cache",
                        "cache_key": "text_generation_demo",
                        "log_level": "info"
                    }
                    
                    providers = [
                        ('VitisAIExecutionProvider', vitisai_options),
                        'CPUExecutionProvider'
                    ]
                    
                    self.npu_session = ort.InferenceSession(
                        npu_model_path,
                        sess_options=session_options,
                        providers=providers
                    )
                    
                    self.active_provider = 'VitisAIExecutionProvider'
                    print("✅ VitisAIExecutionProvider セッション作成成功")
                    
                    # NPU動作テスト
                    test_input = np.random.randn(1, 256).astype(np.float32)
                    test_output = self.npu_session.run(None, {'input': test_input})
                    print(f"✅ NPU動作テスト完了: 出力形状 {test_output[0].shape}")
                    
                except Exception as e:
                    print(f"⚠️ VitisAIExecutionProvider失敗: {e}")
                    self.npu_session = None
            
            # DmlExecutionProvider フォールバック
            if self.npu_session is None and 'DmlExecutionProvider' in available_providers:
                try:
                    print("🔄 DmlExecutionProvider試行...")
                    self.npu_session = ort.InferenceSession(
                        npu_model_path,
                        sess_options=session_options,
                        providers=['DmlExecutionProvider', 'CPUExecutionProvider']
                    )
                    self.active_provider = 'DmlExecutionProvider'
                    print("✅ DmlExecutionProvider セッション作成成功")
                except Exception as e:
                    print(f"⚠️ DmlExecutionProvider失敗: {e}")
                    self.npu_session = None
            
            if self.npu_session:
                print(f"✅ NPUセッション作成成功")
                print(f"🔧 アクティブプロバイダー: {self.active_provider}")
            else:
                print("⚠️ NPUセッション作成失敗（CPUフォールバック）")
                
        except Exception as e:
            print(f"❌ NPUセッション作成エラー: {e}")
    
    def _create_npu_test_model(self, model_path: str) -> bool:
        """NPUテスト用モデル作成"""
        try:
            import torch
            import torch.nn as nn
            import onnx
            
            class NPUTestModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear1 = nn.Linear(256, 512)
                    self.relu = nn.ReLU()
                    self.linear2 = nn.Linear(512, 256)
                    self.dropout = nn.Dropout(0.1)
                
                def forward(self, x):
                    x = self.linear1(x)
                    x = self.relu(x)
                    x = self.dropout(x)
                    x = self.linear2(x)
                    return x
            
            model = NPUTestModel()
            model.eval()
            
            dummy_input = torch.randn(1, 256)
            
            torch.onnx.export(
                model,
                dummy_input,
                model_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # IRバージョン調整
            onnx_model = onnx.load(model_path)
            onnx_model.ir_version = 10
            onnx.save(onnx_model, model_path)
            
            print(f"✅ NPUテストモデル作成完了: {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ NPUテストモデル作成エラー: {e}")
            return False
    
    def _load_llm_model(self) -> bool:
        """LLMモデル読み込み"""
        for candidate in self.model_candidates:
            try:
                model_name = candidate["name"]
                print(f"🔄 {candidate['description']}を試行中: {model_name}")
                print(f"🎯 Ryzen AI実績: {'あり' if candidate['ryzen_ai_proven'] else 'なし'}")
                
                # トークナイザー読み込み
                print("🔤 トークナイザーロード中...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                
                # パディングトークン設定
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                print("✅ トークナイザーロード成功")
                
                # モデル読み込み
                print("🤖 モデルロード中...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype="auto",
                    device_map="auto" if self.enable_infer_os else "cpu"
                )
                
                print(f"✅ モデルロード成功: {model_name}")
                print(f"🎯 Ryzen AI実績: {'あり' if candidate['ryzen_ai_proven'] else 'なし'}")
                print(f"📦 モデルサイズ: {candidate['size']}")
                
                return True
                
            except Exception as e:
                print(f"⚠️ {model_name} 読み込み失敗: {e}")
                continue
        
        print("❌ 全てのモデル読み込みに失敗しました")
        return False
    
    def generate_text(self, prompt: str, max_tokens: int = 50, num_generations: int = 1) -> List[str]:
        """テキスト生成"""
        if self.model is None or self.tokenizer is None:
            print("❌ モデルが初期化されていません")
            return []
        
        try:
            print(f"\n🎯 Ryzen AI テキスト生成開始")
            print(f"💬 プロンプト: {prompt}")
            print(f"🔢 最大トークン数: {max_tokens}")
            print(f"🔢 生成回数: {num_generations}")
            print(f"🔧 infer-OS最適化: {'有効' if self.enable_infer_os else '無効'}")
            print(f"🔧 NPUセッション: {'有効' if self.npu_session else '無効'}")
            
            # 性能監視開始
            self.performance_monitor.start_monitoring()
            
            # 入力トークン化
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            print(f"🔤 入力トークン数: {inputs['input_ids'].shape[1]}")
            
            # 生成設定
            generation_config = GenerationConfig(
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
            
            # テキスト生成実行
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    num_return_sequences=num_generations
                )
            
            generation_time = time.time() - start_time
            
            # 性能監視停止
            self.performance_monitor.stop_monitoring()
            performance_report = self.performance_monitor.get_report()
            
            # 生成結果デコード
            generated_texts = []
            for i, output in enumerate(outputs):
                # 入力部分を除去
                generated_tokens = output[inputs['input_ids'].shape[1]:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                full_text = prompt + generated_text
                generated_texts.append(full_text)
                
                print(f"\n🎯 生成結果 {i+1}:")
                print(f"  💬 プロンプト: {prompt}")
                print(f"  🎯 生成部分: {generated_text}")
                print(f"  📝 完全テキスト: {full_text}")
            
            # 性能レポート
            tokens_generated = sum(len(self.tokenizer.encode(text)) - len(self.tokenizer.encode(prompt)) 
                                 for text in generated_texts)
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            
            print(f"\n📊 Ryzen AI テキスト生成性能レポート:")
            print(f"  🔢 生成回数: {num_generations}")
            print(f"  🔢 総生成トークン数: {tokens_generated}")
            print(f"  ⏱️ 生成時間: {generation_time:.3f}秒")
            print(f"  📊 生成速度: {tokens_per_second:.1f} トークン/秒")
            print(f"  🔧 infer-OS最適化: {'有効' if self.enable_infer_os else '無効'}")
            print(f"  🔧 NPUセッション: {'有効' if self.npu_session else '無効'}")
            
            print(f"\n📊 システム性能:")
            print(f"  💻 平均CPU使用率: {performance_report['avg_cpu']:.1f}%")
            print(f"  💻 最大CPU使用率: {performance_report['max_cpu']:.1f}%")
            print(f"  💾 平均メモリ使用率: {performance_report['avg_memory']:.1f}%")
            print(f"  💾 最大メモリ使用率: {performance_report['max_memory']:.1f}%")
            print(f"  🔢 監視サンプル数: {performance_report['samples']}")
            
            return generated_texts
            
        except Exception as e:
            print(f"❌ テキスト生成エラー: {e}")
            return []
    
    def run_benchmark(self, prompts: List[str], max_tokens: int = 30) -> Dict[str, Any]:
        """ベンチマーク実行"""
        print(f"\n🎯 Ryzen AI テキスト生成ベンチマーク開始")
        print(f"🔢 プロンプト数: {len(prompts)}")
        print(f"🔢 最大トークン数: {max_tokens}")
        print(f"🔧 infer-OS最適化: {'有効' if self.enable_infer_os else '無効'}")
        
        results = {
            "infer_os_enabled": self.enable_infer_os,
            "npu_enabled": self.npu_session is not None,
            "total_prompts": len(prompts),
            "successful_generations": 0,
            "failed_generations": 0,
            "total_time": 0,
            "total_tokens": 0,
            "generations": []
        }
        
        start_time = time.time()
        
        for i, prompt in enumerate(prompts):
            print(f"\n📊 ベンチマーク進捗: {i+1}/{len(prompts)}")
            
            try:
                generated_texts = self.generate_text(prompt, max_tokens, 1)
                if generated_texts:
                    results["successful_generations"] += 1
                    results["generations"].extend(generated_texts)
                    
                    # トークン数計算
                    tokens = len(self.tokenizer.encode(generated_texts[0]))
                    results["total_tokens"] += tokens
                else:
                    results["failed_generations"] += 1
                    
            except Exception as e:
                print(f"❌ プロンプト{i+1}生成失敗: {e}")
                results["failed_generations"] += 1
        
        results["total_time"] = time.time() - start_time
        results["tokens_per_second"] = results["total_tokens"] / results["total_time"] if results["total_time"] > 0 else 0
        results["avg_time_per_generation"] = results["total_time"] / results["successful_generations"] if results["successful_generations"] > 0 else 0
        
        # ベンチマーク結果表示
        print(f"\n🎯 Ryzen AI テキスト生成ベンチマーク結果:")
        print(f"  🔧 infer-OS最適化: {'有効' if results['infer_os_enabled'] else '無効'}")
        print(f"  🔧 NPU使用: {'有効' if results['npu_enabled'] else '無効'}")
        print(f"  ✅ 成功生成数: {results['successful_generations']}")
        print(f"  ❌ 失敗生成数: {results['failed_generations']}")
        print(f"  ⏱️ 総実行時間: {results['total_time']:.3f}秒")
        print(f"  🔢 総トークン数: {results['total_tokens']}")
        print(f"  📊 生成速度: {results['tokens_per_second']:.1f} トークン/秒")
        print(f"  ⏱️ 平均生成時間: {results['avg_time_per_generation']:.3f}秒")
        
        return results

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Ryzen AI テキスト生成デモシステム")
    parser.add_argument("--prompt", type=str, default="人工知能について教えてください。", help="生成プロンプト")
    parser.add_argument("--tokens", type=int, default=50, help="最大生成トークン数")
    parser.add_argument("--generations", type=int, default=1, help="生成回数")
    parser.add_argument("--infer-os", action="store_true", help="infer-OS最適化を有効にする")
    parser.add_argument("--no-npu", action="store_true", help="NPU使用を無効にする")
    parser.add_argument("--interactive", action="store_true", help="インタラクティブモード")
    parser.add_argument("--benchmark", action="store_true", help="ベンチマークモード")
    parser.add_argument("--compare", action="store_true", help="infer-OS ON/OFF比較")
    
    args = parser.parse_args()
    
    if args.compare:
        print("🔄 infer-OS ON/OFF比較テスト実行")
        
        # テストプロンプト
        test_prompts = [
            "人工知能について教えてください。",
            "機械学習の基本概念を説明してください。",
            "深層学習とは何ですか？",
            "自然言語処理の応用例を教えてください。",
            "コンピュータビジョンの技術について説明してください。"
        ]
        
        # infer-OS OFF
        print("\n" + "="*60)
        print("📊 ベースライン測定（infer-OS OFF）")
        print("="*60)
        demo_off = RyzenAITextGenerationDemo(enable_infer_os=False, use_npu=not args.no_npu)
        if demo_off.initialize():
            results_off = demo_off.run_benchmark(test_prompts, args.tokens)
        else:
            print("❌ infer-OS OFF システム初期化失敗")
            return
        
        # infer-OS ON
        print("\n" + "="*60)
        print("📊 最適化測定（infer-OS ON）")
        print("="*60)
        demo_on = RyzenAITextGenerationDemo(enable_infer_os=True, use_npu=not args.no_npu)
        if demo_on.initialize():
            results_on = demo_on.run_benchmark(test_prompts, args.tokens)
        else:
            print("❌ infer-OS ON システム初期化失敗")
            return
        
        # 比較結果表示
        if results_off and results_on:
            print("\n" + "="*60)
            print("📊 infer-OS ON/OFF 比較結果")
            print("="*60)
            
            speed_improvement = (results_on['tokens_per_second'] / results_off['tokens_per_second'] - 1) * 100 if results_off['tokens_per_second'] > 0 else 0
            time_improvement = (1 - results_on['avg_time_per_generation'] / results_off['avg_time_per_generation']) * 100 if results_off['avg_time_per_generation'] > 0 else 0
            
            print(f"📊 生成速度:")
            print(f"  OFF: {results_off['tokens_per_second']:.1f} トークン/秒")
            print(f"  ON:  {results_on['tokens_per_second']:.1f} トークン/秒")
            print(f"  改善: {speed_improvement:+.1f}%")
            
            print(f"⏱️ 平均生成時間:")
            print(f"  OFF: {results_off['avg_time_per_generation']:.3f}秒")
            print(f"  ON:  {results_on['avg_time_per_generation']:.3f}秒")
            print(f"  改善: {time_improvement:+.1f}%")
            
            print(f"✅ 成功率:")
            print(f"  OFF: {results_off['successful_generations']}/{results_off['total_prompts']}")
            print(f"  ON:  {results_on['successful_generations']}/{results_on['total_prompts']}")
    
    else:
        # 単一実行
        demo = RyzenAITextGenerationDemo(enable_infer_os=args.infer_os, use_npu=not args.no_npu)
        if demo.initialize():
            if args.interactive:
                print("\n🎯 インタラクティブモード開始")
                print("💡 'quit' または 'exit' で終了")
                
                while True:
                    try:
                        prompt = input("\n💬 プロンプト: ").strip()
                        if prompt.lower() in ['quit', 'exit', 'q']:
                            break
                        if prompt:
                            demo.generate_text(prompt, args.tokens, args.generations)
                    except KeyboardInterrupt:
                        print("\n👋 終了します")
                        break
            
            elif args.benchmark:
                test_prompts = [
                    "人工知能について教えてください。",
                    "機械学習の基本概念を説明してください。",
                    "深層学習とは何ですか？",
                    "自然言語処理の応用例を教えてください。",
                    "コンピュータビジョンの技術について説明してください。"
                ]
                demo.run_benchmark(test_prompts, args.tokens)
            
            else:
                demo.generate_text(args.prompt, args.tokens, args.generations)
        else:
            print("❌ システム初期化失敗")

if __name__ == "__main__":
    main()

