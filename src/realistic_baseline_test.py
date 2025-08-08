#!/usr/bin/env python3
"""
現実的なベースラインテスト
実際のLLMモデルを使用した性能測定

作成者: Manus AI
"""

import time
import torch
import psutil
import gc
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

class RealisticPerformanceMetrics:
    """現実的な性能メトリクス"""
    def __init__(self):
        self.tokens_per_second = 0.0
        self.latency_ms = 0.0
        self.memory_usage_mb = 0.0
        self.gpu_memory_mb = 0.0
        self.total_tokens = 0
        self.processing_time = 0.0

class RealisticLLMBenchmark:
    """現実的なLLMベンチマーク"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Using device: {self.device}")
    
    def load_model(self) -> bool:
        """モデルの読み込み"""
        try:
            print(f"📥 Loading model: {self.model_name}")
            start_time = time.time()
            
            # トークナイザーの読み込み
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side='left'
            )
            
            # パディングトークンの設定
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # モデルの読み込み
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            load_time = time.time() - start_time
            print(f"✅ Model loaded in {load_time:.2f} seconds")
            
            # モデル情報の表示
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"📊 Model parameters: {total_params:,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    def measure_baseline_performance(self, prompts: List[str], max_new_tokens: int = 50) -> List[RealisticPerformanceMetrics]:
        """ベースライン性能測定"""
        if not self.model or not self.tokenizer:
            print("❌ Model not loaded")
            return []
        
        print(f"🧪 Measuring baseline performance with {len(prompts)} prompts")
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"\nTest {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            try:
                # メモリ使用量測定開始
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                gpu_memory_before = 0
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gpu_memory_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                
                # 推論実行
                start_time = time.time()
                
                # トークン化
                inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                input_length = inputs.shape[1]
                
                # 生成
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id,
                        attention_mask=torch.ones_like(inputs)
                    )
                
                end_time = time.time()
                
                # 結果の処理
                generated_tokens = outputs[0][input_length:]
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # メトリクス計算
                total_time = end_time - start_time
                tokens_generated = len(generated_tokens)
                tokens_per_second = tokens_generated / total_time if total_time > 0 else 0
                
                # メモリ使用量測定終了
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                gpu_memory_after = 0
                if torch.cuda.is_available():
                    gpu_memory_after = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                
                # メトリクス作成
                metrics = RealisticPerformanceMetrics()
                metrics.tokens_per_second = tokens_per_second
                metrics.latency_ms = total_time * 1000
                metrics.memory_usage_mb = memory_after - memory_before
                metrics.gpu_memory_mb = gpu_memory_after - gpu_memory_before
                metrics.total_tokens = tokens_generated
                metrics.processing_time = total_time
                
                results.append(metrics)
                
                # 結果表示
                print(f"  Generated: {response[:100]}...")
                print(f"  Tokens: {tokens_generated}")
                print(f"  Performance: {tokens_per_second:.2f} tok/s")
                print(f"  Latency: {total_time*1000:.1f} ms")
                print(f"  Memory: {memory_after - memory_before:.1f} MB")
                if torch.cuda.is_available():
                    print(f"  GPU Memory: {gpu_memory_after - gpu_memory_before:.1f} MB")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
            
            # メモリクリーンアップ
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        return results
    
    def analyze_results(self, results: List[RealisticPerformanceMetrics]) -> Dict[str, float]:
        """結果分析"""
        if not results:
            return {"error": "No results to analyze"}
        
        # 統計計算
        avg_tokens_per_second = sum(r.tokens_per_second for r in results) / len(results)
        avg_latency = sum(r.latency_ms for r in results) / len(results)
        avg_memory = sum(r.memory_usage_mb for r in results) / len(results)
        avg_gpu_memory = sum(r.gpu_memory_mb for r in results) / len(results)
        total_tokens = sum(r.total_tokens for r in results)
        total_time = sum(r.processing_time for r in results)
        
        min_tokens_per_second = min(r.tokens_per_second for r in results)
        max_tokens_per_second = max(r.tokens_per_second for r in results)
        
        return {
            "total_tests": len(results),
            "avg_tokens_per_second": avg_tokens_per_second,
            "min_tokens_per_second": min_tokens_per_second,
            "max_tokens_per_second": max_tokens_per_second,
            "avg_latency_ms": avg_latency,
            "avg_memory_usage_mb": avg_memory,
            "avg_gpu_memory_mb": avg_gpu_memory,
            "total_tokens": total_tokens,
            "total_time": total_time,
            "overall_throughput": total_tokens / total_time if total_time > 0 else 0
        }

def run_realistic_baseline_test():
    """現実的なベースラインテスト実行"""
    print("🧪 Realistic LLM Baseline Performance Test")
    print("=" * 60)
    
    # システム情報表示
    print(f"💻 System Info:")
    print(f"  CPU: {psutil.cpu_count()} cores")
    print(f"  Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 軽量モデルでテスト（リソース制約を考慮）
    model_name = "microsoft/DialoGPT-small"  # 約117M parameters
    
    benchmark = RealisticLLMBenchmark(model_name)
    
    if not benchmark.load_model():
        print("❌ Failed to load model")
        return
    
    # テストプロンプト
    test_prompts = [
        "Hello, how are you today?",
        "What is artificial intelligence?",
        "Explain machine learning in simple terms.",
        "How does a neural network work?",
        "What are the benefits of AI?"
    ]
    
    print(f"\n🔄 Running baseline tests with {len(test_prompts)} prompts...")
    
    # ベースライン測定
    results = benchmark.measure_baseline_performance(test_prompts, max_new_tokens=30)
    
    if not results:
        print("❌ No successful tests")
        return
    
    # 結果分析
    analysis = benchmark.analyze_results(results)
    
    print("\n📊 Baseline Performance Analysis")
    print("=" * 60)
    print(f"Total tests: {analysis['total_tests']}")
    print(f"Average tokens/second: {analysis['avg_tokens_per_second']:.2f}")
    print(f"Min tokens/second: {analysis['min_tokens_per_second']:.2f}")
    print(f"Max tokens/second: {analysis['max_tokens_per_second']:.2f}")
    print(f"Average latency: {analysis['avg_latency_ms']:.1f} ms")
    print(f"Average memory usage: {analysis['avg_memory_usage_mb']:.1f} MB")
    if torch.cuda.is_available():
        print(f"Average GPU memory: {analysis['avg_gpu_memory_mb']:.1f} MB")
    print(f"Overall throughput: {analysis['overall_throughput']:.2f} tok/s")
    
    # NPU統合による期待される改善の予測
    print("\n🚀 Expected Improvements with NPU Integration")
    print("=" * 60)
    
    baseline_performance = analysis['avg_tokens_per_second']
    
    # 理論的改善予測
    improvements = {
        "Memory Hierarchy Optimization": 1.3,  # 30% improvement
        "NPU SRAM Utilization": 1.4,          # 40% improvement  
        "Dynamic Layer Skipping": 1.6,        # 60% improvement
        "FFN Pruning": 1.5,                   # 50% improvement
        "Token Halting": 1.2,                 # 20% improvement
        "KV Cache Pruning": 1.3,              # 30% improvement
    }
    
    cumulative_improvement = 1.0
    for technique, improvement in improvements.items():
        cumulative_improvement *= improvement
        predicted_performance = baseline_performance * cumulative_improvement
        print(f"{technique:.<30} {improvement:.1f}x -> {predicted_performance:.1f} tok/s")
    
    final_predicted = baseline_performance * cumulative_improvement
    print(f"\n🎯 Final Predicted Performance: {final_predicted:.1f} tok/s")
    print(f"📈 Total Improvement: {cumulative_improvement:.1f}x ({(cumulative_improvement-1)*100:.0f}%)")
    
    # 目標達成評価
    target_performance = 24.0
    if final_predicted >= target_performance:
        print(f"✅ Target of {target_performance} tok/s is achievable!")
    else:
        gap = target_performance - final_predicted
        print(f"⚠️ Target of {target_performance} tok/s requires additional {gap:.1f} tok/s improvement")
    
    print(f"\n📝 Baseline established: {baseline_performance:.2f} tok/s")
    print("✅ Realistic baseline test completed")

if __name__ == "__main__":
    run_realistic_baseline_test()

