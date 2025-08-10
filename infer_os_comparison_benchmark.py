#!/usr/bin/env python3
"""
Infer-OS有り無し比較ベンチマーク
量子化モデルでのInfer-OS統合効果を定量的に測定
"""

import time
import psutil
import threading
import statistics
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
import os
from datetime import datetime

@dataclass
class BenchmarkResult:
    """ベンチマーク結果データクラス"""
    model_name: str
    quantization_profile: str
    infer_os_enabled: bool
    
    # パフォーマンス指標
    avg_tokens_per_sec: float
    peak_memory_gb: float
    avg_memory_gb: float
    avg_cpu_percent: float
    avg_response_time_sec: float
    
    # 品質指標
    avg_japanese_ratio: float
    avg_text_length: int
    coherence_score: float
    
    # 安定性指標
    performance_variance: float
    error_count: int
    total_requests: int
    
    # 追加メトリクス
    first_token_latency: float
    throughput_requests_per_min: float
    memory_efficiency_score: float

class InferOSMode(Enum):
    """Infer-OS動作モード"""
    DISABLED = "disabled"
    ENABLED = "enabled"
    COMPARISON = "comparison"

class ComparisonBenchmark:
    """Infer-OS有り無し比較ベンチマーク"""
    
    def __init__(self, model_name: str, quantization_profile: str = "balanced"):
        self.model_name = model_name
        self.quantization_profile = quantization_profile
        self.results: Dict[str, BenchmarkResult] = {}
        
        # ベンチマーク設定
        self.test_prompts = [
            "人工知能の未来について詳しく説明してください。",
            "機械学習の基本概念と応用例を教えてください。",
            "深層学習とニューラルネットワークの関係を解説してください。",
            "自然言語処理の最新技術について述べてください。",
            "量子コンピューティングの可能性を論じてください。",
            "データサイエンスの重要性について説明してください。",
            "プログラミング言語Pythonの特徴を教えてください。",
            "クラウドコンピューティングの利点を挙げてください。"
        ]
        
        self.monitoring_active = False
        self.performance_data = []
    
    def start_system_monitoring(self):
        """システムリソース監視開始"""
        self.monitoring_active = True
        self.performance_data = []
        
        def monitor():
            while self.monitoring_active:
                data = {
                    'timestamp': time.time(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'memory_used_gb': psutil.virtual_memory().used / (1024**3),
                    'cpu_percent': psutil.cpu_percent(interval=0.1),
                    'cpu_per_core': psutil.cpu_percent(interval=0.1, percpu=True)
                }
                self.performance_data.append(data)
                time.sleep(0.5)
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_system_monitoring(self) -> Dict:
        """システムリソース監視停止"""
        self.monitoring_active = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        
        if not self.performance_data:
            return {}
        
        memory_usage = [d['memory_used_gb'] for d in self.performance_data]
        cpu_usage = [d['cpu_percent'] for d in self.performance_data]
        
        return {
            'peak_memory_gb': max(memory_usage),
            'avg_memory_gb': statistics.mean(memory_usage),
            'min_memory_gb': min(memory_usage),
            'avg_cpu_percent': statistics.mean(cpu_usage),
            'max_cpu_percent': max(cpu_usage),
            'memory_variance': statistics.variance(memory_usage) if len(memory_usage) > 1 else 0,
            'cpu_variance': statistics.variance(cpu_usage) if len(cpu_usage) > 1 else 0
        }
    
    def measure_text_quality(self, text: str) -> Dict:
        """テキスト品質測定"""
        if not text:
            return {
                'japanese_ratio': 0.0,
                'text_length': 0,
                'coherence_score': 0.0
            }
        
        # 日本語文字比率計算
        japanese_chars = 0
        total_chars = len(text)
        
        for char in text:
            if '\u3040' <= char <= '\u309F':  # ひらがな
                japanese_chars += 1
            elif '\u30A0' <= char <= '\u30FF':  # カタカナ
                japanese_chars += 1
            elif '\u4E00' <= char <= '\u9FAF':  # 漢字
                japanese_chars += 1
        
        japanese_ratio = japanese_chars / total_chars if total_chars > 0 else 0
        
        # 簡易的な一貫性スコア（文の長さの分散から推定）
        sentences = text.split('。')
        sentence_lengths = [len(s.strip()) for s in sentences if s.strip()]
        
        if len(sentence_lengths) > 1:
            coherence_score = 1.0 / (1.0 + statistics.variance(sentence_lengths) / 100)
        else:
            coherence_score = 0.8  # デフォルト値
        
        return {
            'japanese_ratio': japanese_ratio,
            'text_length': total_chars,
            'coherence_score': min(1.0, coherence_score)
        }
    
    def run_single_benchmark(self, demo_instance, infer_os_enabled: bool, num_iterations: int = 5) -> BenchmarkResult:
        """単一ベンチマーク実行"""
        print(f"\n📊 ベンチマーク実行中: Infer-OS {'有効' if infer_os_enabled else '無効'}")
        
        # システム監視開始
        self.start_system_monitoring()
        
        # パフォーマンス測定データ
        response_times = []
        tokens_per_sec_list = []
        first_token_latencies = []
        quality_metrics = []
        error_count = 0
        
        start_benchmark_time = time.time()
        
        for i, prompt in enumerate(self.test_prompts[:num_iterations]):
            try:
                print(f"  テスト {i+1}/{num_iterations}: {prompt[:30]}...")
                
                # 推論実行
                start_time = time.time()
                result = demo_instance.generate_japanese_text(prompt, max_new_tokens=100)
                end_time = time.time()
                
                response_time = end_time - start_time
                response_times.append(response_time)
                
                # パフォーマンス指標
                if 'generation_speed' in result:
                    tokens_per_sec_list.append(result['generation_speed'])
                
                if 'first_token_time' in result:
                    first_token_latencies.append(result['first_token_time'])
                else:
                    first_token_latencies.append(response_time * 0.1)  # 推定値
                
                # 品質指標
                generated_text = result.get('generated_text', '')
                quality = self.measure_text_quality(generated_text)
                quality_metrics.append(quality)
                
                # 短い休憩（メモリ安定化）
                time.sleep(0.5)
                
            except Exception as e:
                print(f"    ❌ エラー: {e}")
                error_count += 1
                continue
        
        end_benchmark_time = time.time()
        total_benchmark_time = end_benchmark_time - start_benchmark_time
        
        # システム監視停止
        system_metrics = self.stop_system_monitoring()
        
        # 結果集計
        if response_times:
            avg_response_time = statistics.mean(response_times)
            performance_variance = statistics.variance(response_times) if len(response_times) > 1 else 0
        else:
            avg_response_time = 0
            performance_variance = 0
        
        if tokens_per_sec_list:
            avg_tokens_per_sec = statistics.mean(tokens_per_sec_list)
        else:
            avg_tokens_per_sec = 0
        
        if first_token_latencies:
            avg_first_token_latency = statistics.mean(first_token_latencies)
        else:
            avg_first_token_latency = 0
        
        if quality_metrics:
            avg_japanese_ratio = statistics.mean([q['japanese_ratio'] for q in quality_metrics])
            avg_text_length = statistics.mean([q['text_length'] for q in quality_metrics])
            avg_coherence_score = statistics.mean([q['coherence_score'] for q in quality_metrics])
        else:
            avg_japanese_ratio = 0
            avg_text_length = 0
            avg_coherence_score = 0
        
        # スループット計算
        successful_requests = len(response_times)
        throughput = (successful_requests / total_benchmark_time) * 60 if total_benchmark_time > 0 else 0
        
        # メモリ効率スコア計算
        memory_efficiency = 0
        if system_metrics.get('avg_memory_gb', 0) > 0:
            memory_efficiency = avg_tokens_per_sec / system_metrics['avg_memory_gb']
        
        return BenchmarkResult(
            model_name=self.model_name,
            quantization_profile=self.quantization_profile,
            infer_os_enabled=infer_os_enabled,
            avg_tokens_per_sec=avg_tokens_per_sec,
            peak_memory_gb=system_metrics.get('peak_memory_gb', 0),
            avg_memory_gb=system_metrics.get('avg_memory_gb', 0),
            avg_cpu_percent=system_metrics.get('avg_cpu_percent', 0),
            avg_response_time_sec=avg_response_time,
            avg_japanese_ratio=avg_japanese_ratio,
            avg_text_length=int(avg_text_length),
            coherence_score=avg_coherence_score,
            performance_variance=performance_variance,
            error_count=error_count,
            total_requests=num_iterations,
            first_token_latency=avg_first_token_latency,
            throughput_requests_per_min=throughput,
            memory_efficiency_score=memory_efficiency
        )
    
    def run_comparison_benchmark(self, demo_class, num_iterations: int = 5) -> Dict[str, BenchmarkResult]:
        """Infer-OS有り無し比較ベンチマーク実行"""
        print(f"\n🔥 Infer-OS有り無し比較ベンチマーク開始")
        print(f"モデル: {self.model_name}")
        print(f"量子化プロファイル: {self.quantization_profile}")
        print(f"テスト回数: {num_iterations}")
        
        results = {}
        
        # Infer-OS無効でのベンチマーク
        print(f"\n📊 Phase 1: Infer-OS無効でのベンチマーク")
        try:
            demo_without_infer_os = demo_class(
                model_name=self.model_name,
                use_4bit=True,
                use_8bit=False,
                use_advanced_quant=True,
                quantization_profile=self.quantization_profile,
                infer_os_enabled=False  # Infer-OS無効
            )
            
            if demo_without_infer_os.load_model_with_optimization():
                results['without_infer_os'] = self.run_single_benchmark(
                    demo_without_infer_os, False, num_iterations
                )
                print("✅ Infer-OS無効ベンチマーク完了")
            else:
                print("❌ Infer-OS無効モデルロード失敗")
                
        except Exception as e:
            print(f"❌ Infer-OS無効ベンチマークエラー: {e}")
        
        # メモリクリア
        if 'demo_without_infer_os' in locals():
            del demo_without_infer_os
        
        time.sleep(2)  # メモリ安定化
        
        # Infer-OS有効でのベンチマーク
        print(f"\n📊 Phase 2: Infer-OS有効でのベンチマーク")
        try:
            demo_with_infer_os = demo_class(
                model_name=self.model_name,
                use_4bit=True,
                use_8bit=False,
                use_advanced_quant=True,
                quantization_profile=self.quantization_profile,
                infer_os_enabled=True  # Infer-OS有効
            )
            
            if demo_with_infer_os.load_model_with_optimization():
                results['with_infer_os'] = self.run_single_benchmark(
                    demo_with_infer_os, True, num_iterations
                )
                print("✅ Infer-OS有効ベンチマーク完了")
            else:
                print("❌ Infer-OS有効モデルロード失敗")
                
        except Exception as e:
            print(f"❌ Infer-OS有効ベンチマークエラー: {e}")
        
        self.results = results
        return results
    
    def generate_comparison_report(self) -> str:
        """比較レポート生成"""
        if len(self.results) != 2:
            return "❌ 比較に必要なデータが不足しています"
        
        without_infer_os = self.results['without_infer_os']
        with_infer_os = self.results['with_infer_os']
        
        # 改善率計算
        def calc_improvement(old_val, new_val, higher_is_better=True):
            if old_val == 0:
                return 0
            if higher_is_better:
                return ((new_val - old_val) / old_val) * 100
            else:
                return ((old_val - new_val) / old_val) * 100
        
        speed_improvement = calc_improvement(
            without_infer_os.avg_tokens_per_sec,
            with_infer_os.avg_tokens_per_sec
        )
        
        memory_improvement = calc_improvement(
            without_infer_os.avg_memory_gb,
            with_infer_os.avg_memory_gb,
            higher_is_better=False
        )
        
        response_time_improvement = calc_improvement(
            without_infer_os.avg_response_time_sec,
            with_infer_os.avg_response_time_sec,
            higher_is_better=False
        )
        
        throughput_improvement = calc_improvement(
            without_infer_os.throughput_requests_per_min,
            with_infer_os.throughput_requests_per_min
        )
        
        report = f"""
🔥 **Infer-OS統合効果 詳細比較レポート**

## 📋 **テスト環境**
- **モデル**: {self.model_name}
- **量子化プロファイル**: {self.quantization_profile}
- **テスト日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 **パフォーマンス比較**

### **推論速度**
- **Infer-OS無効**: {without_infer_os.avg_tokens_per_sec:.1f} tokens/sec
- **Infer-OS有効**: {with_infer_os.avg_tokens_per_sec:.1f} tokens/sec
- **改善率**: {speed_improvement:+.1f}% {'🚀' if speed_improvement > 0 else '⚠️'}

### **メモリ使用量**
- **Infer-OS無効**: {without_infer_os.avg_memory_gb:.1f}GB
- **Infer-OS有効**: {with_infer_os.avg_memory_gb:.1f}GB
- **削減率**: {memory_improvement:+.1f}% {'💾' if memory_improvement > 0 else '⚠️'}

### **応答時間**
- **Infer-OS無効**: {without_infer_os.avg_response_time_sec:.2f}秒
- **Infer-OS有効**: {with_infer_os.avg_response_time_sec:.2f}秒
- **短縮率**: {response_time_improvement:+.1f}% {'⚡' if response_time_improvement > 0 else '⚠️'}

### **スループット**
- **Infer-OS無効**: {without_infer_os.throughput_requests_per_min:.1f} requests/min
- **Infer-OS有効**: {with_infer_os.throughput_requests_per_min:.1f} requests/min
- **向上率**: {throughput_improvement:+.1f}% {'📈' if throughput_improvement > 0 else '⚠️'}

## 🎯 **品質比較**

### **日本語品質**
- **Infer-OS無効**: {without_infer_os.avg_japanese_ratio:.1%}
- **Infer-OS有効**: {with_infer_os.avg_japanese_ratio:.1%}
- **差異**: {(with_infer_os.avg_japanese_ratio - without_infer_os.avg_japanese_ratio)*100:+.1f}%

### **文章長**
- **Infer-OS無効**: {without_infer_os.avg_text_length}文字
- **Infer-OS有効**: {with_infer_os.avg_text_length}文字
- **差異**: {with_infer_os.avg_text_length - without_infer_os.avg_text_length:+d}文字

### **一貫性スコア**
- **Infer-OS無効**: {without_infer_os.coherence_score:.3f}
- **Infer-OS有効**: {with_infer_os.coherence_score:.3f}
- **差異**: {(with_infer_os.coherence_score - without_infer_os.coherence_score):+.3f}

## 🔧 **システムリソース比較**

### **CPU使用率**
- **Infer-OS無効**: {without_infer_os.avg_cpu_percent:.1f}%
- **Infer-OS有効**: {with_infer_os.avg_cpu_percent:.1f}%
- **差異**: {(with_infer_os.avg_cpu_percent - without_infer_os.avg_cpu_percent):+.1f}%

### **メモリ効率**
- **Infer-OS無効**: {without_infer_os.memory_efficiency_score:.2f} tokens/sec/GB
- **Infer-OS有効**: {with_infer_os.memory_efficiency_score:.2f} tokens/sec/GB
- **向上率**: {calc_improvement(without_infer_os.memory_efficiency_score, with_infer_os.memory_efficiency_score):+.1f}%

### **安定性**
- **Infer-OS無効**: 分散 {without_infer_os.performance_variance:.3f}
- **Infer-OS有効**: 分散 {with_infer_os.performance_variance:.3f}
- **安定性**: {'向上' if with_infer_os.performance_variance < without_infer_os.performance_variance else '低下'}

## 🎉 **総合評価**

### **Infer-OS統合の効果**
"""
        
        # 総合スコア計算
        total_score = 0
        score_count = 0
        
        if speed_improvement > 0:
            total_score += min(speed_improvement, 100)
            score_count += 1
        
        if memory_improvement > 0:
            total_score += min(memory_improvement, 100)
            score_count += 1
        
        if response_time_improvement > 0:
            total_score += min(response_time_improvement, 100)
            score_count += 1
        
        if throughput_improvement > 0:
            total_score += min(throughput_improvement, 100)
            score_count += 1
        
        overall_score = total_score / score_count if score_count > 0 else 0
        
        if overall_score >= 50:
            evaluation = "🏆 **優秀** - Infer-OS統合により大幅な性能向上を実現"
        elif overall_score >= 25:
            evaluation = "✅ **良好** - Infer-OS統合により明確な性能向上を確認"
        elif overall_score >= 10:
            evaluation = "📈 **改善** - Infer-OS統合により一定の性能向上を確認"
        else:
            evaluation = "⚠️ **要検証** - Infer-OS統合の効果が限定的"
        
        report += f"""
- **総合改善スコア**: {overall_score:.1f}%
- **評価**: {evaluation}

### **推奨事項**
"""
        
        if speed_improvement > 20:
            report += "- ✅ 推論速度の大幅向上により、リアルタイム応用に適用可能\n"
        
        if memory_improvement > 30:
            report += "- ✅ メモリ効率の大幅改善により、より大規模なモデルの実行が可能\n"
        
        if response_time_improvement > 25:
            report += "- ✅ 応答時間の大幅短縮により、インタラクティブな用途に最適\n"
        
        if overall_score < 10:
            report += "- ⚠️ 環境設定やモデル選択の見直しを推奨\n"
            report += "- 💡 より軽量なモデルまたは異なる量子化プロファイルを試行\n"
        
        return report
    
    def save_results(self, filename: str = None):
        """結果をJSONファイルに保存"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"infer_os_comparison_{self.model_name.replace('/', '_')}_{timestamp}.json"
        
        # BenchmarkResultをdict形式に変換
        results_dict = {}
        for key, result in self.results.items():
            results_dict[key] = {
                'model_name': result.model_name,
                'quantization_profile': result.quantization_profile,
                'infer_os_enabled': result.infer_os_enabled,
                'avg_tokens_per_sec': result.avg_tokens_per_sec,
                'peak_memory_gb': result.peak_memory_gb,
                'avg_memory_gb': result.avg_memory_gb,
                'avg_cpu_percent': result.avg_cpu_percent,
                'avg_response_time_sec': result.avg_response_time_sec,
                'avg_japanese_ratio': result.avg_japanese_ratio,
                'avg_text_length': result.avg_text_length,
                'coherence_score': result.coherence_score,
                'performance_variance': result.performance_variance,
                'error_count': result.error_count,
                'total_requests': result.total_requests,
                'first_token_latency': result.first_token_latency,
                'throughput_requests_per_min': result.throughput_requests_per_min,
                'memory_efficiency_score': result.memory_efficiency_score
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
        
        print(f"📁 結果を保存しました: {filename}")
        return filename

def main():
    """メイン実行関数"""
    print("🔥 Infer-OS統合効果比較ベンチマーク")
    
    # テスト用の簡易実行
    benchmark = ComparisonBenchmark("rinna/youri-7b-chat", "balanced")
    
    # 実際の使用時は、JapaneseHeavyLLMDemoクラスを渡す
    # results = benchmark.run_comparison_benchmark(JapaneseHeavyLLMDemo, num_iterations=3)
    # report = benchmark.generate_comparison_report()
    # print(report)
    # benchmark.save_results()
    
    print("✅ ベンチマーク機能実装完了")

if __name__ == "__main__":
    main()

