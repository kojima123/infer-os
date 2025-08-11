"""
NPU負荷分析ツール
VitisAI NPU動作時の詳細な負荷分析

使用方法:
    python npu_load_analyzer.py
"""

import os
import time
import threading
import subprocess
import psutil
from typing import Dict, List, Any
import json

class NPULoadAnalyzer:
    """NPU負荷分析ツール"""
    
    def __init__(self):
        self.monitoring = False
        self.load_data = []
        self.monitor_thread = None
        
        print("🔍 NPU負荷分析ツール初期化")
        print("🎯 VitisAI NPU vs ハードウェアNPU負荷の詳細分析")
    
    def start_monitoring(self, duration: int = 30):
        """NPU負荷監視開始"""
        print(f"📊 NPU負荷監視開始 ({duration}秒間)")
        
        self.monitoring = True
        self.load_data = []
        
        # 監視スレッド開始
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(duration,)
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """NPU負荷監視停止"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        print("📊 NPU負荷監視停止")
        return self.analyze_results()
    
    def _monitor_loop(self, duration: int):
        """監視ループ"""
        start_time = time.time()
        
        while self.monitoring and (time.time() - start_time) < duration:
            try:
                # システム負荷取得
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_info = psutil.virtual_memory()
                
                # NPU関連プロセス検索
                npu_processes = self._find_npu_processes()
                
                # VitisAI関連プロセス検索
                vitisai_processes = self._find_vitisai_processes()
                
                # データ記録
                data_point = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_info.percent,
                    'npu_processes': npu_processes,
                    'vitisai_processes': vitisai_processes
                }
                
                self.load_data.append(data_point)
                
                # 短い間隔で監視
                time.sleep(0.1)
                
            except Exception as e:
                print(f"⚠️ 監視エラー: {e}")
                time.sleep(0.5)
    
    def _find_npu_processes(self) -> List[Dict]:
        """NPU関連プロセス検索"""
        npu_processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    proc_info = proc.info
                    proc_name = proc_info['name'].lower()
                    
                    # NPU関連キーワード
                    npu_keywords = [
                        'npu', 'neural', 'aie', 'vitis', 'xilinx', 
                        'ryzen', 'amd', 'accelerator'
                    ]
                    
                    if any(keyword in proc_name for keyword in npu_keywords):
                        npu_processes.append({
                            'pid': proc_info['pid'],
                            'name': proc_info['name'],
                            'cpu_percent': proc_info['cpu_percent'],
                            'memory_percent': proc_info['memory_percent']
                        })
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            print(f"⚠️ NPUプロセス検索エラー: {e}")
        
        return npu_processes
    
    def _find_vitisai_processes(self) -> List[Dict]:
        """VitisAI関連プロセス検索"""
        vitisai_processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'cmdline']):
                try:
                    proc_info = proc.info
                    proc_name = proc_info['name'].lower()
                    cmdline = ' '.join(proc_info['cmdline'] or []).lower()
                    
                    # VitisAI関連キーワード
                    vitisai_keywords = [
                        'vitisai', 'vitis', 'onnxruntime', 'python',
                        'run_vitisai_demo', 'vitisai_npu_engine'
                    ]
                    
                    if any(keyword in proc_name or keyword in cmdline for keyword in vitisai_keywords):
                        vitisai_processes.append({
                            'pid': proc_info['pid'],
                            'name': proc_info['name'],
                            'cpu_percent': proc_info['cpu_percent'],
                            'memory_percent': proc_info['memory_percent'],
                            'cmdline': cmdline[:100]  # 最初の100文字
                        })
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            print(f"⚠️ VitisAIプロセス検索エラー: {e}")
        
        return vitisai_processes
    
    def analyze_results(self) -> Dict[str, Any]:
        """結果分析"""
        if not self.load_data:
            return {"error": "監視データなし"}
        
        print("\n📊 NPU負荷分析結果:")
        print("=" * 50)
        
        # 基本統計
        cpu_values = [d['cpu_percent'] for d in self.load_data]
        memory_values = [d['memory_percent'] for d in self.load_data]
        
        avg_cpu = sum(cpu_values) / len(cpu_values)
        max_cpu = max(cpu_values)
        avg_memory = sum(memory_values) / len(memory_values)
        
        print(f"📈 CPU使用率: 平均{avg_cpu:.1f}%, 最大{max_cpu:.1f}%")
        print(f"💾 メモリ使用率: 平均{avg_memory:.1f}%")
        
        # NPU関連プロセス分析
        all_npu_processes = set()
        all_vitisai_processes = set()
        
        for data_point in self.load_data:
            for proc in data_point['npu_processes']:
                all_npu_processes.add((proc['pid'], proc['name']))
            
            for proc in data_point['vitisai_processes']:
                all_vitisai_processes.add((proc['pid'], proc['name']))
        
        print(f"\n🔍 検出されたNPU関連プロセス: {len(all_npu_processes)}個")
        for pid, name in all_npu_processes:
            print(f"  - PID {pid}: {name}")
        
        print(f"\n🔍 検出されたVitisAI関連プロセス: {len(all_vitisai_processes)}個")
        for pid, name in all_vitisai_processes:
            print(f"  - PID {pid}: {name}")
        
        # NPU負荷分析
        self._analyze_npu_load()
        
        return {
            "monitoring_duration": len(self.load_data) * 0.1,
            "avg_cpu": avg_cpu,
            "max_cpu": max_cpu,
            "avg_memory": avg_memory,
            "npu_processes": len(all_npu_processes),
            "vitisai_processes": len(all_vitisai_processes)
        }
    
    def _analyze_npu_load(self):
        """NPU負荷詳細分析"""
        print(f"\n🎯 NPU負荷詳細分析:")
        
        # Windows NPU監視コマンド試行
        try:
            # NPU使用率取得試行
            result = subprocess.run(
                ['wmic', 'path', 'Win32_PerfRawData_Counters_ProcessorInformation', 'get', 'Name,PercentProcessorTime'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                print("📊 プロセッサー情報取得成功")
                lines = result.stdout.strip().split('\n')
                for line in lines[1:3]:  # 最初の数行のみ表示
                    if line.strip():
                        print(f"  {line.strip()}")
            
        except Exception as e:
            print(f"⚠️ NPU監視コマンドエラー: {e}")
        
        # NPU負荷が0%の理由分析
        print(f"\n💡 NPU負荷0%の可能性:")
        print("  1. 🔄 処理時間が短すぎる（0.037秒）")
        print("  2. 🧠 VAIMLがソフトウェア層で動作")
        print("  3. 📊 タスクマネージャーの更新間隔問題")
        print("  4. 🔧 部分的NPU処理（最終層のみ）")
        print("  5. ⚡ NPUが瞬間的にのみ動作")
    
    def continuous_load_test(self, iterations: int = 100):
        """継続的負荷テスト"""
        print(f"\n🔄 継続的NPU負荷テスト開始 ({iterations}回)")
        
        try:
            # VitisAI NPUエンジンインポート
            from vitisai_npu_engine import VitisAINPUEngine
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # 軽量セットアップ
            print("🔧 軽量セットアップ中...")
            tokenizer = AutoTokenizer.from_pretrained(
                "rinna/youri-7b-chat",
                trust_remote_code=True
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                "rinna/youri-7b-chat",
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            vitisai_engine = VitisAINPUEngine(model, tokenizer)
            
            if vitisai_engine.setup_vitisai_npu():
                print("✅ VitisAI NPUエンジン準備完了")
                
                # 監視開始
                self.start_monitoring(duration=iterations * 2)
                
                # 継続的推論実行
                for i in range(iterations):
                    try:
                        # NPU推論実行
                        result = vitisai_engine.generate_with_vitisai_npu(
                            f"テスト{i}",
                            max_new_tokens=5,
                            temperature=0.8
                        )
                        
                        if i % 10 == 0:
                            print(f"  🔄 推論 {i+1}/{iterations}")
                        
                        # 短い間隔
                        time.sleep(0.1)
                        
                    except Exception as e:
                        print(f"⚠️ 推論エラー {i}: {e}")
                
                # 監視停止・分析
                analysis = self.stop_monitoring()
                print(f"\n📊 継続的負荷テスト完了")
                
            else:
                print("❌ VitisAI NPUエンジンセットアップ失敗")
                
        except Exception as e:
            print(f"❌ 継続的負荷テストエラー: {e}")
            import traceback
            traceback.print_exc()

def main():
    """メイン関数"""
    analyzer = NPULoadAnalyzer()
    
    print("🔍 NPU負荷分析ツール")
    print("🎯 VitisAI NPU vs ハードウェアNPU負荷の詳細分析")
    print("=" * 60)
    
    # 基本監視テスト
    print("\n1️⃣ 基本NPU負荷監視テスト (10秒)")
    analyzer.start_monitoring(duration=10)
    
    print("💡 この間にVitisAI NPUデモを実行してください:")
    print("   python run_vitisai_demo.py --interactive")
    print("   プロンプト: 人参について")
    
    time.sleep(10)
    basic_analysis = analyzer.stop_monitoring()
    
    # 継続的負荷テスト
    print("\n2️⃣ 継続的NPU負荷テスト")
    response = input("継続的負荷テストを実行しますか？ (y/n): ")
    
    if response.lower() == 'y':
        analyzer.continuous_load_test(iterations=50)
    
    print("\n🏁 NPU負荷分析完了")

if __name__ == "__main__":
    main()

