"""
集中的NPU負荷デモ
NPU負荷率を確実に上げるための集中的処理

使用方法:
    python intensive_npu_demo.py
"""

import os
import time
import threading
import numpy as np
from typing import Dict, Any

# 環境変数設定
os.environ['RYZEN_AI_INSTALLATION_PATH'] = r"C:\Program Files\RyzenAI\1.5"

class IntensiveNPUDemo:
    """集中的NPU負荷デモ"""
    
    def __init__(self):
        self.vitisai_engine = None
        self.model = None
        self.tokenizer = None
        self.running = False
        
        print("⚡ 集中的NPU負荷デモ初期化")
        print("🎯 NPU負荷率を確実に上げる")
    
    def setup(self) -> bool:
        """セットアップ"""
        try:
            print("🔧 集中的NPU処理セットアップ中...")
            
            # 必要なライブラリインポート
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from vitisai_npu_engine import VitisAINPUEngine
            
            # トークナイザーロード
            print("📝 トークナイザーロード中...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "rinna/youri-7b-chat",
                trust_remote_code=True,
                use_fast=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # モデルロード
            print("🤖 モデルロード中...")
            self.model = AutoModelForCausalLM.from_pretrained(
                "rinna/youri-7b-chat",
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # VitisAI NPUエンジンセットアップ
            print("🚀 VitisAI NPUエンジンセットアップ中...")
            self.vitisai_engine = VitisAINPUEngine(self.model, self.tokenizer)
            
            if self.vitisai_engine.setup_vitisai_npu():
                print("✅ 集中的NPU処理セットアップ完了")
                return True
            else:
                print("❌ VitisAI NPUエンジンセットアップ失敗")
                return False
                
        except Exception as e:
            print(f"❌ セットアップエラー: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def intensive_npu_processing(self, duration: int = 60):
        """集中的NPU処理"""
        print(f"⚡ 集中的NPU処理開始 ({duration}秒間)")
        print("📊 タスクマネージャーでNPU負荷率を確認してください")
        
        self.running = True
        start_time = time.time()
        iteration = 0
        
        # 複数の処理パターン
        prompts = [
            "人参について",
            "野菜の栄養",
            "健康的な食事",
            "料理のレシピ",
            "日本の文化"
        ]
        
        try:
            while self.running and (time.time() - start_time) < duration:
                iteration += 1
                prompt = prompts[iteration % len(prompts)]
                
                # VitisAI NPU推論実行
                result = self.vitisai_engine.generate_with_vitisai_npu(
                    prompt,
                    max_new_tokens=10,
                    temperature=0.8
                )
                
                # 進捗表示
                if iteration % 5 == 0:
                    elapsed = time.time() - start_time
                    print(f"  ⚡ 推論 {iteration} 完了 ({elapsed:.1f}秒経過)")
                
                # 短い間隔で継続実行
                time.sleep(0.05)  # 50ms間隔
            
            total_time = time.time() - start_time
            print(f"✅ 集中的NPU処理完了: {iteration}回推論, {total_time:.1f}秒")
            
        except KeyboardInterrupt:
            print("\n⏹️ ユーザーによる中断")
        except Exception as e:
            print(f"❌ 集中的NPU処理エラー: {e}")
    
    def parallel_npu_processing(self, num_threads: int = 4, duration: int = 30):
        """並列NPU処理"""
        print(f"🔄 並列NPU処理開始 ({num_threads}スレッド, {duration}秒間)")
        
        self.running = True
        threads = []
        
        def worker_thread(thread_id: int):
            """ワーカースレッド"""
            iteration = 0
            start_time = time.time()
            
            while self.running and (time.time() - start_time) < duration:
                try:
                    # スレッド固有のプロンプト
                    prompt = f"スレッド{thread_id}テスト{iteration}"
                    
                    # VitisAI NPU推論
                    result = self.vitisai_engine.generate_with_vitisai_npu(
                        prompt,
                        max_new_tokens=5,
                        temperature=0.8
                    )
                    
                    iteration += 1
                    
                    # 短い間隔
                    time.sleep(0.02)  # 20ms間隔
                    
                except Exception as e:
                    print(f"⚠️ スレッド{thread_id}エラー: {e}")
                    time.sleep(0.1)
            
            print(f"  ✅ スレッド{thread_id}完了: {iteration}回推論")
        
        try:
            # スレッド開始
            for i in range(num_threads):
                thread = threading.Thread(target=worker_thread, args=(i,))
                threads.append(thread)
                thread.start()
                time.sleep(0.1)  # スレッド開始間隔
            
            # 完了待機
            for thread in threads:
                thread.join()
            
            print("✅ 並列NPU処理完了")
            
        except KeyboardInterrupt:
            print("\n⏹️ ユーザーによる中断")
            self.running = False
            
            # スレッド終了待機
            for thread in threads:
                thread.join(timeout=1)
    
    def continuous_load_generation(self, duration: int = 120):
        """継続的負荷生成"""
        print(f"🔥 継続的NPU負荷生成開始 ({duration}秒間)")
        print("📊 この間にタスクマネージャーでNPU負荷率を確認してください")
        print("⏹️ Ctrl+Cで中断可能")
        
        try:
            # 段階的負荷増加
            phases = [
                {"name": "軽負荷", "interval": 0.2, "tokens": 5, "duration": 20},
                {"name": "中負荷", "interval": 0.1, "tokens": 10, "duration": 30},
                {"name": "高負荷", "interval": 0.05, "tokens": 15, "duration": 40},
                {"name": "最大負荷", "interval": 0.02, "tokens": 20, "duration": 30}
            ]
            
            for phase in phases:
                print(f"\n🔄 {phase['name']}フェーズ開始 ({phase['duration']}秒)")
                
                phase_start = time.time()
                iteration = 0
                
                while (time.time() - phase_start) < phase['duration']:
                    # NPU推論実行
                    result = self.vitisai_engine.generate_with_vitisai_npu(
                        f"負荷テスト{iteration}",
                        max_new_tokens=phase['tokens'],
                        temperature=0.8
                    )
                    
                    iteration += 1
                    
                    # 進捗表示
                    if iteration % 10 == 0:
                        elapsed = time.time() - phase_start
                        print(f"    ⚡ {phase['name']}: {iteration}回推論 ({elapsed:.1f}秒)")
                    
                    time.sleep(phase['interval'])
                
                print(f"  ✅ {phase['name']}フェーズ完了: {iteration}回推論")
            
            print("\n🎉 継続的NPU負荷生成完了")
            
        except KeyboardInterrupt:
            print("\n⏹️ ユーザーによる中断")
    
    def stop(self):
        """処理停止"""
        self.running = False

def main():
    """メイン関数"""
    demo = IntensiveNPUDemo()
    
    print("⚡ 集中的NPU負荷デモ")
    print("🎯 NPU負荷率を確実に上げる")
    print("=" * 60)
    
    # セットアップ
    if not demo.setup():
        print("❌ セットアップ失敗")
        return
    
    print("\n📊 タスクマネージャーを開いてNPU負荷率を確認してください")
    print("💡 以下のテストから選択してください:")
    print("  1. 集中的NPU処理 (60秒)")
    print("  2. 並列NPU処理 (4スレッド, 30秒)")
    print("  3. 継続的負荷生成 (段階的, 120秒)")
    print("  4. 全テスト実行")
    
    try:
        choice = input("\n選択してください (1-4): ").strip()
        
        if choice == "1":
            demo.intensive_npu_processing(duration=60)
        elif choice == "2":
            demo.parallel_npu_processing(num_threads=4, duration=30)
        elif choice == "3":
            demo.continuous_load_generation(duration=120)
        elif choice == "4":
            print("\n🔄 全テスト実行開始")
            demo.intensive_npu_processing(duration=30)
            time.sleep(2)
            demo.parallel_npu_processing(num_threads=2, duration=20)
            time.sleep(2)
            demo.continuous_load_generation(duration=60)
        else:
            print("❌ 無効な選択")
    
    except KeyboardInterrupt:
        print("\n⏹️ デモ中断")
    finally:
        demo.stop()
    
    print("\n🏁 集中的NPU負荷デモ完了")

if __name__ == "__main__":
    main()

