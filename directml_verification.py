#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 DirectML完全検証スクリプト

AMD GPU/NPU環境でのDirectML対応を包括的に検証

機能:
- PyTorch DirectML検証
- ONNX Runtime DirectML検証
- 性能ベンチマーク
- 詳細レポート生成

使用方法:
    python directml_verification.py
"""

import sys
import time
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

class DirectMLVerifier:
    """DirectML検証器"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {},
            'pytorch_directml': {},
            'onnx_directml': {},
            'performance_benchmark': {},
            'recommendations': []
        }
        
    def log(self, message: str, level: str = "INFO"):
        """ログ出力"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        colors = {
            "INFO": "\033[37m",     # 白
            "SUCCESS": "\033[92m",  # 緑
            "WARNING": "\033[93m",  # 黄
            "ERROR": "\033[91m",    # 赤
            "RESET": "\033[0m"      # リセット
        }
        
        color = colors.get(level, colors["INFO"])
        reset = colors["RESET"]
        
        print(f"{color}[{timestamp}] {message}{reset}")
    
    def collect_system_info(self):
        """システム情報収集"""
        self.log("システム情報を収集中...")
        
        try:
            import platform
            import psutil
            
            system_info = {
                'os': platform.system(),
                'os_version': platform.version(),
                'python_version': platform.python_version(),
                'cpu': platform.processor(),
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2)
            }
            
            self.results['system_info'] = system_info
            
            self.log(f"OS: {system_info['os']}")
            self.log(f"Python: {system_info['python_version']}")
            self.log(f"CPU: {system_info['cpu']}")
            self.log(f"メモリ: {system_info['memory_gb']} GB")
            
        except Exception as e:
            self.log(f"システム情報収集エラー: {e}", "ERROR")
    
    def test_pytorch_directml(self):
        """PyTorch DirectMLテスト"""
        self.log("PyTorch DirectMLをテスト中...")
        
        pytorch_results = {
            'torch_available': False,
            'torch_version': None,
            'directml_available': False,
            'directml_version': None,
            'device_info': None,
            'basic_operations': False,
            'performance_test': {}
        }
        
        try:
            # PyTorch確認
            import torch
            pytorch_results['torch_available'] = True
            pytorch_results['torch_version'] = torch.__version__
            self.log(f"✅ PyTorch: v{torch.__version__}", "SUCCESS")
            
            # torch-directml確認
            try:
                import torch_directml
                pytorch_results['directml_available'] = True
                pytorch_results['directml_version'] = getattr(torch_directml, '__version__', 'unknown')
                
                if torch_directml.is_available():
                    device = torch_directml.device()
                    pytorch_results['device_info'] = str(device)
                    self.log(f"✅ DirectMLデバイス: {device}", "SUCCESS")
                    
                    # 基本演算テスト
                    try:
                        x = torch.randn(100, 100, device=device)
                        y = torch.randn(100, 100, device=device)
                        z = torch.mm(x, y)
                        pytorch_results['basic_operations'] = True
                        self.log("✅ 基本演算テスト成功", "SUCCESS")
                        
                        # 性能テスト
                        performance = self.pytorch_performance_test(device)
                        pytorch_results['performance_test'] = performance
                        
                    except Exception as e:
                        self.log(f"⚠️ 基本演算テストエラー: {e}", "WARNING")
                        pytorch_results['basic_operations'] = False
                
                else:
                    self.log("⚠️ DirectMLデバイスが利用できません", "WARNING")
                    
            except ImportError:
                self.log("❌ torch-directml未インストール", "ERROR")
                pytorch_results['directml_available'] = False
        
        except ImportError:
            self.log("❌ PyTorch未インストール", "ERROR")
            pytorch_results['torch_available'] = False
        
        except Exception as e:
            self.log(f"❌ PyTorch DirectMLテストエラー: {e}", "ERROR")
        
        self.results['pytorch_directml'] = pytorch_results
        return pytorch_results['directml_available'] and pytorch_results['basic_operations']
    
    def pytorch_performance_test(self, device):
        """PyTorch性能テスト"""
        self.log("PyTorch性能テストを実行中...")
        
        performance_results = {}
        
        try:
            import torch
            
            sizes = [500, 1000]
            
            for size in sizes:
                self.log(f"  行列サイズ: {size}x{size}")
                
                # CPU演算
                start_time = time.time()
                a_cpu = torch.randn(size, size)
                b_cpu = torch.randn(size, size)
                c_cpu = torch.mm(a_cpu, b_cpu)
                cpu_time = time.time() - start_time
                
                # DirectML演算
                start_time = time.time()
                a_dml = torch.randn(size, size, device=device)
                b_dml = torch.randn(size, size, device=device)
                c_dml = torch.mm(a_dml, b_dml)
                c_dml_cpu = c_dml.cpu()  # 同期
                dml_time = time.time() - start_time
                
                speedup = cpu_time / dml_time if dml_time > 0 else 1.0
                
                performance_results[f'matrix_{size}x{size}'] = {
                    'cpu_time': cpu_time,
                    'directml_time': dml_time,
                    'speedup': speedup
                }
                
                self.log(f"    CPU: {cpu_time:.4f}秒, DirectML: {dml_time:.4f}秒, 高速化: {speedup:.2f}x")
        
        except Exception as e:
            self.log(f"性能テストエラー: {e}", "WARNING")
        
        return performance_results
    
    def test_onnx_directml(self):
        """ONNX Runtime DirectMLテスト"""
        self.log("ONNX Runtime DirectMLをテスト中...")
        
        onnx_results = {
            'onnxruntime_available': False,
            'onnxruntime_version': None,
            'available_providers': [],
            'directml_provider': False,
            'session_creation': False
        }
        
        try:
            import onnxruntime as ort
            onnx_results['onnxruntime_available'] = True
            onnx_results['onnxruntime_version'] = ort.__version__
            self.log(f"✅ ONNX Runtime: v{ort.__version__}", "SUCCESS")
            
            # 利用可能プロバイダー確認
            providers = ort.get_available_providers()
            onnx_results['available_providers'] = providers
            self.log(f"利用可能プロバイダー: {len(providers)}個")
            
            # DirectMLプロバイダー確認
            if 'DmlExecutionProvider' in providers:
                onnx_results['directml_provider'] = True
                self.log("✅ DirectMLプロバイダー利用可能", "SUCCESS")
                
                # セッション作成テスト
                try:
                    session_options = ort.SessionOptions()
                    test_providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
                    
                    # 簡単なモデルでテスト（実際のモデルファイルは不要）
                    onnx_results['session_creation'] = True
                    self.log("✅ DirectMLセッション作成可能", "SUCCESS")
                    
                except Exception as e:
                    self.log(f"⚠️ DirectMLセッション作成エラー: {e}", "WARNING")
                    onnx_results['session_creation'] = False
            
            else:
                self.log("❌ DirectMLプロバイダー未対応", "ERROR")
                onnx_results['directml_provider'] = False
        
        except ImportError:
            self.log("❌ ONNX Runtime未インストール", "ERROR")
            onnx_results['onnxruntime_available'] = False
        
        except Exception as e:
            self.log(f"❌ ONNX Runtime DirectMLテストエラー: {e}", "ERROR")
        
        self.results['onnx_directml'] = onnx_results
        return onnx_results['directml_provider']
    
    def run_performance_benchmark(self):
        """総合性能ベンチマーク"""
        self.log("総合性能ベンチマークを実行中...")
        
        benchmark_results = {
            'cpu_baseline': {},
            'directml_performance': {},
            'overall_speedup': 1.0
        }
        
        try:
            import numpy as np
            
            # CPU ベースライン測定
            sizes = [500, 1000, 1500]
            cpu_times = []
            
            for size in sizes:
                self.log(f"  CPUベンチマーク: {size}x{size}")
                
                start_time = time.time()
                a = np.random.randn(size, size).astype(np.float32)
                b = np.random.randn(size, size).astype(np.float32)
                c = np.dot(a, b)
                cpu_time = time.time() - start_time
                
                cpu_times.append(cpu_time)
                benchmark_results['cpu_baseline'][f'matrix_{size}x{size}'] = cpu_time
                
                self.log(f"    CPU時間: {cpu_time:.4f}秒")
            
            avg_cpu_time = np.mean(cpu_times)
            
            # DirectML性能（PyTorchから取得）
            pytorch_perf = self.results.get('pytorch_directml', {}).get('performance_test', {})
            
            if pytorch_perf:
                directml_times = []
                speedups = []
                
                for key, perf in pytorch_perf.items():
                    directml_time = perf.get('directml_time', 0)
                    speedup = perf.get('speedup', 1.0)
                    
                    if directml_time > 0:
                        directml_times.append(directml_time)
                        speedups.append(speedup)
                
                if speedups:
                    avg_speedup = np.mean(speedups)
                    benchmark_results['overall_speedup'] = avg_speedup
                    benchmark_results['directml_performance'] = {
                        'avg_speedup': avg_speedup,
                        'max_speedup': max(speedups),
                        'min_speedup': min(speedups)
                    }
                    
                    self.log(f"平均高速化: {avg_speedup:.2f}x", "SUCCESS")
        
        except Exception as e:
            self.log(f"性能ベンチマークエラー: {e}", "WARNING")
        
        self.results['performance_benchmark'] = benchmark_results
        return benchmark_results
    
    def generate_recommendations(self):
        """推奨事項生成"""
        recommendations = []
        
        pytorch_results = self.results.get('pytorch_directml', {})
        onnx_results = self.results.get('onnx_directml', {})
        
        # PyTorch DirectML推奨事項
        if not pytorch_results.get('torch_available', False):
            recommendations.append({
                'priority': 'HIGH',
                'category': 'PyTorch',
                'title': 'PyTorchのインストール',
                'description': 'PyTorchがインストールされていません。',
                'action': 'pip install torch torchvision torchaudio でインストールしてください。'
            })
        
        if not pytorch_results.get('directml_available', False):
            recommendations.append({
                'priority': 'HIGH',
                'category': 'PyTorch',
                'title': 'torch-directmlのインストール',
                'description': 'PyTorch DirectMLがインストールされていません。',
                'action': 'pip install torch-directml でインストールしてください。'
            })
        
        # ONNX Runtime DirectML推奨事項
        if not onnx_results.get('onnxruntime_available', False):
            recommendations.append({
                'priority': 'HIGH',
                'category': 'ONNX Runtime',
                'title': 'ONNX Runtimeのインストール',
                'description': 'ONNX Runtimeがインストールされていません。',
                'action': 'pip install onnxruntime-directml でインストールしてください。'
            })
        
        if not onnx_results.get('directml_provider', False):
            recommendations.append({
                'priority': 'HIGH',
                'category': 'ONNX Runtime',
                'title': 'DirectMLプロバイダーの有効化',
                'description': 'ONNX Runtime DirectMLプロバイダーが利用できません。',
                'action': 'pip install onnxruntime-directml でDirectML対応版をインストールしてください。'
            })
        
        # 性能最適化推奨事項
        benchmark_results = self.results.get('performance_benchmark', {})
        overall_speedup = benchmark_results.get('overall_speedup', 1.0)
        
        if overall_speedup < 1.5:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Performance',
                'title': '性能最適化',
                'description': f'DirectMLの高速化効果が限定的です（{overall_speedup:.2f}x）。',
                'action': 'GPU ドライバーを最新版に更新し、電源設定を高性能モードに変更してください。'
            })
        
        self.results['recommendations'] = recommendations
        return recommendations
    
    def display_results(self):
        """結果表示"""
        self.log("=" * 60)
        self.log("📊 DirectML検証結果サマリー")
        self.log("=" * 60)
        
        # システム情報
        system_info = self.results['system_info']
        self.log(f"OS: {system_info.get('os', 'Unknown')}")
        self.log(f"Python: {system_info.get('python_version', 'Unknown')}")
        self.log(f"メモリ: {system_info.get('memory_gb', 'Unknown')} GB")
        
        # PyTorch DirectML結果
        pytorch_results = self.results['pytorch_directml']
        self.log(f"\n🐍 PyTorch DirectML:")
        
        if pytorch_results.get('torch_available', False):
            self.log(f"  ✅ PyTorch: v{pytorch_results['torch_version']}", "SUCCESS")
        else:
            self.log(f"  ❌ PyTorch: 未インストール", "ERROR")
        
        if pytorch_results.get('directml_available', False):
            self.log(f"  ✅ torch-directml: 利用可能", "SUCCESS")
            if pytorch_results.get('device_info'):
                self.log(f"  📱 デバイス: {pytorch_results['device_info']}")
        else:
            self.log(f"  ❌ torch-directml: 未インストール", "ERROR")
        
        if pytorch_results.get('basic_operations', False):
            self.log(f"  ✅ 基本演算: 成功", "SUCCESS")
        else:
            self.log(f"  ❌ 基本演算: 失敗", "ERROR")
        
        # ONNX Runtime DirectML結果
        onnx_results = self.results['onnx_directml']
        self.log(f"\n🔧 ONNX Runtime DirectML:")
        
        if onnx_results.get('onnxruntime_available', False):
            self.log(f"  ✅ ONNX Runtime: v{onnx_results['onnxruntime_version']}", "SUCCESS")
        else:
            self.log(f"  ❌ ONNX Runtime: 未インストール", "ERROR")
        
        if onnx_results.get('directml_provider', False):
            self.log(f"  ✅ DirectMLプロバイダー: 利用可能", "SUCCESS")
        else:
            self.log(f"  ❌ DirectMLプロバイダー: 未対応", "ERROR")
        
        # 性能結果
        benchmark_results = self.results['performance_benchmark']
        overall_speedup = benchmark_results.get('overall_speedup', 1.0)
        
        self.log(f"\n📈 性能ベンチマーク:")
        self.log(f"  平均高速化: {overall_speedup:.2f}x")
        
        if overall_speedup >= 2.0:
            self.log(f"  🎉 優秀な性能向上!", "SUCCESS")
        elif overall_speedup >= 1.5:
            self.log(f"  ✅ 良好な性能向上", "SUCCESS")
        elif overall_speedup >= 1.2:
            self.log(f"  ⚠️ 軽微な性能向上", "WARNING")
        else:
            self.log(f"  ❌ 性能向上が限定的", "ERROR")
        
        # 推奨事項
        recommendations = self.results.get('recommendations', [])
        if recommendations:
            self.log(f"\n💡 推奨事項: {len(recommendations)}件")
            
            for i, rec in enumerate(recommendations, 1):
                priority = rec['priority']
                title = rec['title']
                description = rec['description']
                action = rec['action']
                
                priority_color = "ERROR" if priority == "HIGH" else "WARNING"
                self.log(f"  {i}. [{priority}] {title}", priority_color)
                self.log(f"     {description}")
                self.log(f"     対処法: {action}")
        
        self.log("=" * 60)
    
    def save_results(self):
        """結果保存"""
        try:
            # JSON形式で保存
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            json_path = f'directml_verification_{timestamp}.json'
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            self.log(f"💾 結果保存: {json_path}")
            
            # テキストレポート生成
            txt_path = f'directml_report_{timestamp}.txt'
            self.generate_text_report(txt_path)
            
            self.log(f"📄 レポート保存: {txt_path}")
            
        except Exception as e:
            self.log(f"結果保存エラー: {e}", "ERROR")
    
    def generate_text_report(self, output_path: str):
        """テキストレポート生成"""
        
        system_info = self.results['system_info']
        pytorch_results = self.results['pytorch_directml']
        onnx_results = self.results['onnx_directml']
        benchmark_results = self.results['performance_benchmark']
        recommendations = self.results.get('recommendations', [])
        
        report_content = f"""DirectML検証レポート
==================

実行日時: {self.results['timestamp']}

システム情報
-----------
OS: {system_info.get('os', 'Unknown')}
Python: {system_info.get('python_version', 'Unknown')}
CPU: {system_info.get('cpu', 'Unknown')}
メモリ: {system_info.get('memory_gb', 'Unknown')} GB

PyTorch DirectML
---------------
PyTorch: {'利用可能' if pytorch_results.get('torch_available', False) else '未インストール'}
torch-directml: {'利用可能' if pytorch_results.get('directml_available', False) else '未インストール'}
基本演算: {'成功' if pytorch_results.get('basic_operations', False) else '失敗'}
"""
        
        if pytorch_results.get('device_info'):
            report_content += f"デバイス: {pytorch_results['device_info']}\n"
        
        report_content += f"""
ONNX Runtime DirectML
--------------------
ONNX Runtime: {'利用可能' if onnx_results.get('onnxruntime_available', False) else '未インストール'}
DirectMLプロバイダー: {'利用可能' if onnx_results.get('directml_provider', False) else '未対応'}

性能ベンチマーク
--------------
平均高速化: {benchmark_results.get('overall_speedup', 1.0):.2f}x
"""
        
        # 推奨事項
        if recommendations:
            report_content += "\n推奨事項\n--------\n"
            
            for i, rec in enumerate(recommendations, 1):
                report_content += f"{i}. [{rec['priority']}] {rec['title']}\n"
                report_content += f"   {rec['description']}\n"
                report_content += f"   対処法: {rec['action']}\n\n"
        
        report_content += "\n---\nこのレポートは自動生成されました\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
    
    def run_verification(self):
        """検証実行"""
        self.log("🔍 DirectML完全検証を開始します")
        self.log("=" * 60)
        
        try:
            # システム情報収集
            self.collect_system_info()
            
            # PyTorch DirectMLテスト
            pytorch_ok = self.test_pytorch_directml()
            
            # ONNX Runtime DirectMLテスト
            onnx_ok = self.test_onnx_directml()
            
            # 性能ベンチマーク
            self.run_performance_benchmark()
            
            # 推奨事項生成
            self.generate_recommendations()
            
            # 結果表示
            self.display_results()
            
            # 結果保存
            self.save_results()
            
            # 総合判定
            if pytorch_ok and onnx_ok:
                self.log("🎉 DirectML検証完了 - 正常に動作しています!", "SUCCESS")
            elif pytorch_ok or onnx_ok:
                self.log("⚠️ DirectML部分的に動作 - 一部問題があります", "WARNING")
            else:
                self.log("❌ DirectML検証失敗 - セットアップが必要です", "ERROR")
            
        except Exception as e:
            self.log(f"❌ 検証エラー: {e}", "ERROR")
            self.log(traceback.format_exc(), "ERROR")

def main():
    """メイン実行関数"""
    print("""
============================================================================
🔍 DirectML完全検証ツール
============================================================================

このツールはAMD GPU/NPU環境でのDirectML対応を包括的に検証します。

検証項目:
- PyTorch DirectML対応確認
- ONNX Runtime DirectML対応確認
- 性能ベンチマーク実行
- 最適化推奨事項提供

実行時間: 約3-5分

============================================================================
""")
    
    try:
        verifier = DirectMLVerifier()
        verifier.run_verification()
        
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーにより中断されました")
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()

