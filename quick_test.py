#!/usr/bin/env python3
"""
🚀 Infer-OS Windows NPU クイックテストスクリプト

使用方法:
    python quick_test.py

機能:
    - システム環境検証
    - NPU デバイス検出
    - Infer-OS 基本機能テスト
    - 性能ベンチマーク実行
"""

import sys
import os
import time
import platform
import subprocess
import json
from pathlib import Path

def print_header(title):
    """セクションヘッダーを表示"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def print_result(test_name, success, details=""):
    """テスト結果を表示"""
    status = "✅ 成功" if success else "❌ 失敗"
    print(f"{test_name}: {status}")
    if details:
        print(f"   詳細: {details}")

def test_system_info():
    """システム情報テスト"""
    print_header("システム情報")
    
    try:
        import psutil
        
        # 基本情報
        print(f"OS: {platform.system()} {platform.release()}")
        print(f"CPU: {platform.processor()}")
        print(f"Python: {platform.python_version()}")
        
        # メモリ情報
        memory = psutil.virtual_memory()
        print(f"メモリ: {memory.total // (1024**3)} GB (使用率: {memory.percent}%)")
        
        # ディスク情報
        disk = psutil.disk_usage('.')
        print(f"ディスク空き容量: {disk.free // (1024**3)} GB")
        
        return True
    except Exception as e:
        print(f"システム情報取得エラー: {e}")
        return False

def test_npu_detection():
    """NPU デバイス検出テスト"""
    print_header("NPU デバイス検出")
    
    npu_detected = False
    
    try:
        # Windows GPU情報取得
        if platform.system() == "Windows":
            result = subprocess.run(
                ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                capture_output=True, text=True, timeout=10
            )
            
            gpu_names = result.stdout
            print("検出されたGPUデバイス:")
            
            for line in gpu_names.split('\n'):
                line = line.strip()
                if line and line != "Name":
                    print(f"  - {line}")
                    if 'AMD' in line.upper() or 'RADEON' in line.upper():
                        npu_detected = True
        
        # PyTorch CUDA確認
        try:
            import torch
            print(f"\nPyTorch バージョン: {torch.__version__}")
            print(f"CUDA利用可能: {torch.cuda.is_available()}")
            
            if torch.cuda.is_available():
                print(f"CUDA デバイス数: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"  デバイス {i}: {torch.cuda.get_device_name(i)}")
        except ImportError:
            print("PyTorch が見つかりません")
        
        print_result("NPU/GPU検出", npu_detected, "AMD デバイス検出" if npu_detected else "AMD デバイス未検出")
        return npu_detected
        
    except Exception as e:
        print_result("NPU検出", False, str(e))
        return False

def test_dependencies():
    """依存関係テスト"""
    print_header("依存関係チェック")
    
    required_packages = [
        'torch', 'numpy', 'pandas', 'matplotlib', 
        'flask', 'requests', 'psutil', 'onnxruntime'
    ]
    
    results = {}
    
    for package in required_packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            results[package] = {'success': True, 'version': version}
            print_result(f"{package}", True, f"v{version}")
        except ImportError:
            results[package] = {'success': False, 'version': None}
            print_result(f"{package}", False, "未インストール")
    
    success_count = sum(1 for r in results.values() if r['success'])
    total_count = len(required_packages)
    
    print(f"\n依存関係: {success_count}/{total_count} 成功")
    return success_count == total_count

def test_infer_os_modules():
    """Infer-OS モジュールテスト"""
    print_header("Infer-OS モジュール")
    
    # パス設定
    src_path = Path('src')
    if src_path.exists():
        sys.path.insert(0, str(src_path))
    
    modules_to_test = [
        ('runtime.enhanced_iobinding', 'EnhancedIOBinding'),
        ('optim.kv_quantization', 'KVQuantizationManager'),
        ('optim.speculative_generation', 'SpeculativeGenerator'),
        ('optim.gpu_npu_pipeline', 'GPUNPUPipeline')
    ]
    
    results = {}
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            
            # 基本初期化テスト
            instance = cls()
            results[module_name] = True
            print_result(f"{module_name}", True, f"{class_name} 初期化成功")
            
        except Exception as e:
            results[module_name] = False
            print_result(f"{module_name}", False, str(e))
    
    success_count = sum(1 for r in results.values() if r)
    total_count = len(modules_to_test)
    
    print(f"\nInfer-OSモジュール: {success_count}/{total_count} 成功")
    return success_count > 0

def test_performance_benchmark():
    """性能ベンチマークテスト"""
    print_header("性能ベンチマーク")
    
    try:
        import numpy as np
        
        # CPU ベンチマーク
        print("CPU 行列演算ベンチマーク実行中...")
        
        sizes = [100, 500, 1000]
        results = {}
        
        for size in sizes:
            start_time = time.time()
            
            # 行列演算
            a = np.random.randn(size, size).astype(np.float32)
            b = np.random.randn(size, size).astype(np.float32)
            c = np.dot(a, b)
            
            elapsed = time.time() - start_time
            results[f"matrix_{size}x{size}"] = elapsed
            
            print(f"  {size}x{size} 行列: {elapsed:.4f}秒")
        
        # メモリ使用量測定
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            results['memory_usage_mb'] = memory_mb
            print(f"  メモリ使用量: {memory_mb:.1f} MB")
        except:
            pass
        
        return results
        
    except Exception as e:
        print_result("性能ベンチマーク", False, str(e))
        return {}

def test_web_demo():
    """Webデモテスト"""
    print_header("Webデモ起動テスト")
    
    try:
        # Flask アプリケーションの存在確認
        demo_path = Path('infer-os-demo/src/main.py')
        if not demo_path.exists():
            print_result("Webデモファイル", False, "main.py が見つかりません")
            return False
        
        print_result("Webデモファイル", True, "main.py 確認")
        
        # Flask インポートテスト
        try:
            import flask
            print_result("Flask", True, f"v{flask.__version__}")
        except ImportError:
            print_result("Flask", False, "未インストール")
            return False
        
        print("\n注意: Webデモの実際の起動は手動で行ってください:")
        print("  python infer-os-demo/src/main.py")
        print("  ブラウザで http://localhost:5000 にアクセス")
        
        return True
        
    except Exception as e:
        print_result("Webデモテスト", False, str(e))
        return False

def run_integrated_test():
    """統合テスト実行"""
    print_header("統合テスト")
    
    try:
        # 統合テストスクリプトの存在確認
        test_script = Path('benchmarks/integrated_performance_test.py')
        if test_script.exists():
            print("統合テストスクリプト実行中...")
            result = subprocess.run([sys.executable, str(test_script)], 
                                  capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print_result("統合テスト", True, "実行成功")
                print("出力:")
                print(result.stdout)
                return True
            else:
                print_result("統合テスト", False, f"終了コード: {result.returncode}")
                print("エラー:")
                print(result.stderr)
                return False
        else:
            print_result("統合テストスクリプト", False, "ファイルが見つかりません")
            return False
            
    except subprocess.TimeoutExpired:
        print_result("統合テスト", False, "タイムアウト")
        return False
    except Exception as e:
        print_result("統合テスト", False, str(e))
        return False

def generate_report(results):
    """テスト結果レポート生成"""
    print_header("テスト結果サマリー")
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'system': {
            'os': platform.system(),
            'python': platform.python_version(),
            'architecture': platform.architecture()[0]
        },
        'tests': results
    }
    
    # 成功率計算
    total_tests = len([k for k in results.keys() if k != 'performance'])
    successful_tests = len([k for k, v in results.items() 
                           if k != 'performance' and v])
    
    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"総合成功率: {success_rate:.1f}% ({successful_tests}/{total_tests})")
    
    # 推奨アクション
    print("\n📋 推奨アクション:")
    
    if results.get('dependencies', False):
        print("✅ 依存関係OK - Infer-OS実行準備完了")
    else:
        print("❌ 依存関係不足 - pip install -r requirements.txt を実行")
    
    if results.get('npu_detection', False):
        print("✅ NPU検出OK - ハードウェア最適化利用可能")
    else:
        print("⚠️  NPU未検出 - CPUモードで動作")
    
    if results.get('infer_os_modules', False):
        print("✅ Infer-OSモジュールOK - 最適化機能利用可能")
    else:
        print("❌ Infer-OSモジュール不足 - プロジェクト構成を確認")
    
    # レポートファイル保存
    try:
        with open('test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n📄 詳細レポート: test_report.json に保存")
    except Exception as e:
        print(f"レポート保存エラー: {e}")
    
    return success_rate

def main():
    """メイン実行関数"""
    print("🚀 Infer-OS Windows NPU クイックテスト開始")
    print(f"実行時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # テスト実行
    results = {}
    
    # システム情報（常に実行）
    test_system_info()
    
    # 各テスト実行
    results['npu_detection'] = test_npu_detection()
    results['dependencies'] = test_dependencies()
    results['infer_os_modules'] = test_infer_os_modules()
    results['web_demo'] = test_web_demo()
    
    # 性能テスト
    performance_results = test_performance_benchmark()
    if performance_results:
        results['performance'] = performance_results
    
    # 統合テスト（オプション）
    try:
        results['integrated_test'] = run_integrated_test()
    except:
        results['integrated_test'] = False
    
    # 結果レポート
    success_rate = generate_report(results)
    
    # 終了メッセージ
    if success_rate >= 80:
        print("\n🎉 テスト完了 - Infer-OS実行準備完了！")
        print("\n次のステップ:")
        print("1. python infer-os-demo/src/main.py  # Webデモ起動")
        print("2. ブラウザで http://localhost:5000 にアクセス")
        print("3. 各最適化技術をテスト")
    else:
        print("\n⚠️  テスト完了 - 一部問題があります")
        print("上記の推奨アクションを実行してください")
    
    return success_rate >= 80

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  テスト中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        sys.exit(1)

