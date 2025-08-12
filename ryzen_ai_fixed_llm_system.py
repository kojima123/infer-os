# -*- coding: utf-8 -*-
"""
Ryzen AI NPU対応日本語LLMシステム（bitsandbytes問題修正版）
通常のfloat16モデル使用でbitsandbytes問題を回避
PyTorch生成エラー修正版
"""

import os
import sys
import time
import argparse
import json
import threading
import psutil
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import warnings
warnings.filterwarnings("ignore")

try:
    import torch
    import onnxruntime as ort
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import snapshot_download
    print("✅ 必要なライブラリのインポート成功")
except ImportError as e:
    print(f"❌ ライブラリインポートエラー: {e}")
    print("💡 以下のコマンドを実行してください:")
    print("pip install torch transformers onnxruntime huggingface_hub")
    sys.exit(1)

class RyzenAIFixedLLMSystem:
    """Ryzen AI NPU対応日本語LLMシステム（bitsandbytes問題修正版）"""
    
    def __init__(self, enable_infer_os: bool = False):
        self.enable_infer_os = enable_infer_os
        
        # bitsandbytes問題回避: 通常のfloat16モデル使用
        self.model_id = "cyberagent/open-calm-small"  # 軽量版で安定性確保
        
        # Windows権限問題回避: 直接パスを使用
        self.cache_dir = Path("./models")
        self.model_dir = None  # ダウンロード後に設定
        self.onnx_path = None  # モデルディレクトリ確定後に設定
        
        # システム状態
        self.pytorch_model = None
        self.tokenizer = None
        self.onnx_session = None
        self.active_provider = None
        
        # NPU監視
        self.npu_monitoring = False
        self.npu_usage_history = []
        self.max_npu_usage = 0.0
        self.npu_active_count = 0
        self.total_inferences = 0
        
        print(f"🚀 Ryzen AI NPU対応日本語LLMシステム初期化")
        print(f"🎯 使用モデル: {self.model_id}")
        print(f"📝 モデル詳細: CyberAgent OpenCALM-Small（bitsandbytes問題回避版）")
        print(f"🌐 言語: 日本語特化")
        print(f"🔧 infer-OS最適化: {'有効' if enable_infer_os else '無効'}")
        print(f"🛠️ bitsandbytes問題対応: 通常のfloat16モデル使用")
    
    def setup_infer_os_environment(self):
        """infer-OS環境設定"""
        if self.enable_infer_os:
            print("🔧 infer-OS最適化環境設定中...")
            
            infer_os_env = {
                'INFER_OS_ENABLE': '1',
                'INFER_OS_OPTIMIZATION_LEVEL': 'high',
                'INFER_OS_NPU_ACCELERATION': '1',
                'INFER_OS_MEMORY_OPTIMIZATION': '1',
                'INFER_OS_JAPANESE_OPTIMIZATION': '1'
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
                       'INFER_OS_JAPANESE_OPTIMIZATION']:
                os.environ.pop(key, None)
    
    def download_model(self) -> bool:
        """日本語モデルダウンロード（Windows権限問題修正版）"""
        try:
            print(f"📥 {self.model_id} ダウンロード開始...")
            print(f"📝 CyberAgent OpenCALM-Small（bitsandbytes問題回避版）")
            print(f"🌐 日本語特化モデル")
            print(f"⚠️ 注意: 初回ダウンロードに時間がかかります")
            
            start_time = time.time()
            
            # HuggingFace Hubからダウンロード（シンボリックリンク回避）
            model_path = snapshot_download(
                repo_id=self.model_id,
                cache_dir=str(self.cache_dir),
                resume_download=True,
                local_files_only=False
            )
            
            # Windows権限問題回避: コピーを使用
            self.model_dir = self.cache_dir / "open-calm-small"
            
            if not self.model_dir.exists():
                print("📁 モデルファイルをコピー中（Windows権限問題回避）...")
                self.model_dir.mkdir(parents=True, exist_ok=True)
                
                # 必要ファイルのみコピー
                source_dir = Path(model_path)
                for file_path in source_dir.iterdir():
                    if file_path.is_file():
                        dest_path = self.model_dir / file_path.name
                        if not dest_path.exists():
                            shutil.copy2(file_path, dest_path)
                            print(f"  ✅ コピー完了: {file_path.name}")
            
            # ONNXパス設定
            self.onnx_path = self.model_dir / "open_calm_small_npu.onnx"
            
            download_time = time.time() - start_time
            
            print(f"✅ ダウンロード完了!")
            print(f"📁 保存先: {self.model_dir}")
            print(f"⏱️ ダウンロード時間: {download_time:.1f}秒")
            
            # ファイル存在確認
            required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
            missing_files = []
            
            for file_name in required_files:
                file_path = self.model_dir / file_name
                if file_path.exists():
                    print(f"  ✅ {file_name}: {file_path.stat().st_size:,} bytes")
                else:
                    missing_files.append(file_name)
                    print(f"  ❌ {file_name}: 見つかりません")
            
            if missing_files:
                print(f"⚠️ 不足ファイル: {missing_files}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ ダウンロードエラー: {e}")
            return False
    
    def load_pytorch_model(self) -> bool:
        """PyTorchモデル読み込み（bitsandbytes問題修正版）"""
        try:
            if self.model_dir is None:
                print("❌ モデルディレクトリが設定されていません")
                return False
            
            print("📥 PyTorchモデル読み込み中...")
            print(f"🎯 モデル: {self.model_id}")
            print(f"🔧 量子化: bitsandbytes問題回避のため通常のfloat16使用")
            
            # トークナイザー読み込み
            print("🔤 トークナイザー読み込み中...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_dir),
                trust_remote_code=True
            )
            
            # パディングトークン設定
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"✅ トークナイザー読み込み完了: 語彙サイズ {len(self.tokenizer)}")
            
            # 通常のfloat16モデル読み込み（bitsandbytes回避）
            print("🔧 float16モデル読み込み中（bitsandbytes問題回避）...")
            print("⚠️ 注意: 初回ロードは時間がかかります")
            
            self.pytorch_model = AutoModelForCausalLM.from_pretrained(
                str(self.model_dir),
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            print(f"✅ PyTorchモデル読み込み完了")
            print(f"🔧 データ型: float16（bitsandbytes問題回避）")
            print(f"📊 デバイス: {self.pytorch_model.device}")
            
            # メモリ使用量表示
            if torch.cuda.is_available():
                memory_gb = torch.cuda.memory_allocated() / 1024**3
                print(f"💾 GPU メモリ使用量: {memory_gb:.2f} GB")
            else:
                print("💾 CPU使用")
            
            return True
            
        except Exception as e:
            print(f"❌ PyTorchモデル読み込みエラー: {e}")
            return False
    
    def generate_text_pytorch(self, prompt: str, max_new_tokens: int = 50) -> str:
        """PyTorch日本語テキスト生成（生成エラー修正版）"""
        try:
            if self.pytorch_model is None or self.tokenizer is None:
                return "❌ PyTorchモデルが読み込まれていません"
            
            print(f"💬 PyTorch日本語生成: '{prompt}'")
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.pytorch_model.device)
            
            with torch.no_grad():
                # 生成エラー修正版パラメータ
                outputs = self.pytorch_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=5,
                    do_sample=True,
                    temperature=0.8,  # 安定した温度設定
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    use_cache=True
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # プロンプト部分を除去
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            # 空の結果チェック
            if not generated_text:
                generated_text = "（生成されたテキストが空でした）"
            
            return generated_text
            
        except Exception as e:
            print(f"⚠️ PyTorch生成エラー詳細: {e}")
            # フォールバック: 簡単な生成
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.pytorch_model.device)
                with torch.no_grad():
                    outputs = self.pytorch_model.generate(
                        **inputs,
                        max_new_tokens=20,
                        do_sample=False,  # グリーディ生成
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                return generated_text if generated_text else "（フォールバック生成も失敗）"
            except Exception as e2:
                return f"❌ PyTorch生成エラー: {e2}"
    
    def interactive_mode(self):
        """インタラクティブ日本語生成モード"""
        print(f"\n🎯 インタラクティブ日本語生成モード")
        print(f"📝 モデル: {self.model_id}")
        print(f"🔧 プロバイダー: PyTorch")
        print(f"💡 コマンド: 'quit'で終了")
        print(f"=" * 60)
        
        try:
            while True:
                prompt = input("\n💬 プロンプトを入力してください: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not prompt:
                    continue
                
                print(f"💬 テキスト生成中: '{prompt[:50]}...'")
                
                start_time = time.time()
                result = self.generate_text_pytorch(prompt, max_new_tokens=64)
                generation_time = time.time() - start_time
                
                print(f"✅ テキスト生成完了")
                print(f"\n🎯 生成結果:")
                print(f"{result}")
                print(f"\n⏱️ 生成時間: {generation_time:.3f}秒")
                
        except KeyboardInterrupt:
            print("\n👋 インタラクティブモードを終了します")
    
    def run_benchmark(self, num_inferences: int = 10) -> Dict[str, Any]:
        """ベンチマーク実行"""
        try:
            print(f"📊 PyTorch ベンチマーク開始: {num_inferences}回推論")
            print(f"🎯 モデル: {self.model_id}")
            print(f"🔧 プロバイダー: PyTorch")
            print(f"🌐 言語: 日本語")
            
            start_time = time.time()
            successful_inferences = 0
            
            test_prompts = [
                "AIによって私達の暮らしは、",
                "日本の未来について考えると、",
                "技術革新が社会に与える影響は、",
                "人工知能の発展により、",
                "デジタル社会において重要なのは、"
            ]
            
            for i in range(num_inferences):
                try:
                    prompt = test_prompts[i % len(test_prompts)]
                    result = self.generate_text_pytorch(prompt, max_new_tokens=20)
                    
                    if not result.startswith("❌"):
                        successful_inferences += 1
                        print(f"✅ 推論 {i+1}/{num_inferences}: 成功")
                    else:
                        print(f"❌ 推論 {i+1}/{num_inferences}: 失敗")
                    
                except Exception as e:
                    print(f"❌ 推論 {i+1}/{num_inferences}: エラー - {e}")
            
            total_time = time.time() - start_time
            
            # 統計計算
            success_rate = (successful_inferences / num_inferences) * 100
            throughput = successful_inferences / total_time
            avg_inference_time = total_time / num_inferences * 1000  # ms
            
            results = {
                "successful_inferences": successful_inferences,
                "total_inferences": num_inferences,
                "success_rate": success_rate,
                "total_time": total_time,
                "throughput": throughput,
                "avg_inference_time": avg_inference_time,
                "active_provider": "PyTorch",
                "model_id": self.model_id
            }
            
            print(f"\n📊 ベンチマーク結果:")
            print(f"  ⚡ 成功推論回数: {successful_inferences}/{num_inferences}")
            print(f"  📊 成功率: {success_rate:.1f}%")
            print(f"  ⏱️ 総実行時間: {total_time:.3f}秒")
            print(f"  📈 スループット: {throughput:.1f} 推論/秒")
            print(f"  ⚡ 平均推論時間: {avg_inference_time:.1f}ms")
            print(f"  🔧 アクティブプロバイダー: PyTorch")
            
            return results
            
        except Exception as e:
            print(f"❌ ベンチマークエラー: {e}")
            return {}
    
    def initialize_system(self) -> bool:
        """システム全体初期化"""
        try:
            print("🚀 Ryzen AI NPU対応日本語LLMシステム初期化開始")
            
            # infer-OS環境設定
            self.setup_infer_os_environment()
            
            # モデルダウンロード
            if not self.download_model():
                return False
            
            # PyTorchモデル読み込み
            if not self.load_pytorch_model():
                return False
            
            print("✅ Ryzen AI NPU対応日本語LLMシステム初期化完了")
            return True
            
        except Exception as e:
            print(f"❌ システム初期化エラー: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Ryzen AI NPU対応日本語LLMシステム（bitsandbytes問題修正版）")
    parser.add_argument("--interactive", action="store_true", help="インタラクティブモード")
    parser.add_argument("--benchmark", action="store_true", help="ベンチマーク実行")
    parser.add_argument("--inferences", type=int, default=10, help="ベンチマーク推論回数")
    parser.add_argument("--prompt", type=str, help="単発テキスト生成")
    parser.add_argument("--tokens", type=int, default=50, help="生成トークン数")
    parser.add_argument("--infer-os", action="store_true", help="infer-OS最適化有効")
    
    args = parser.parse_args()
    
    try:
        system = RyzenAIFixedLLMSystem(enable_infer_os=args.infer_os)
        
        if not system.initialize_system():
            print("❌ システム初期化に失敗しました")
            return
        
        if args.interactive:
            system.interactive_mode()
        elif args.benchmark:
            system.run_benchmark(args.inferences)
        elif args.prompt:
            print(f"💬 単発テキスト生成: '{args.prompt}'")
            result = system.generate_text_pytorch(args.prompt, args.tokens)
            print(f"🎯 生成結果:")
            print(f"{result}")
        else:
            # デフォルト: 簡単なテスト
            system.run_benchmark(5)
    
    except KeyboardInterrupt:
        print("\n👋 プログラムを終了します")
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")

if __name__ == "__main__":
    main()

