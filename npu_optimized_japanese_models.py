#!/usr/bin/env python3
"""
NPU最適化日本語モデル実装
真のNPU活用を実現する日本語LLMデモ

対応モデル:
1. llama3-8b-amd-npu (8B) - NPU完全対応済み
2. Llama-3.1-70B-Japanese-Instruct-2407 (70B) - ONNX変換チャレンジ
"""

import os
import sys
import time
import torch
import psutil
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

# Transformers関連
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    set_seed
)

# ONNX関連
try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️ ONNX関連ライブラリが見つかりません")

# NPU関連
try:
    import qlinear
    QLINEAR_AVAILABLE = True
except ImportError:
    QLINEAR_AVAILABLE = False
    print("⚠️ qlinearライブラリが見つかりません")


class NPUOptimizedJapaneseModel:
    """NPU最適化日本語モデルクラス"""
    
    def __init__(self, model_name: str = "llama3-8b-amd-npu"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.npu_session = None
        self.model_type = self._detect_model_type()
        
        # NPU環境設定
        self._setup_npu_environment()
        
    def _detect_model_type(self) -> str:
        """モデルタイプを検出"""
        if "llama3-8b-amd-npu" in self.model_name:
            return "npu_optimized"
        elif "Llama-3.1-70B-Japanese" in self.model_name:
            return "large_japanese"
        elif "ALMA-Ja-V3-amd-npu" in self.model_name:
            return "translation_optimized"
        else:
            return "unknown"
    
    def _setup_npu_environment(self):
        """NPU環境変数設定"""
        print("🔧 NPU環境設定中...")
        
        # Ryzen AIパス設定
        ryzen_ai_paths = [
            "C:\\Program Files\\RyzenAI\\1.5",
            "C:\\Program Files\\RyzenAI\\1.5.1",
            "C:\\Program Files\\RyzenAI\\1.2"
        ]
        
        for path in ryzen_ai_paths:
            if os.path.exists(path):
                os.environ["RYZEN_AI_INSTALLATION_PATH"] = path
                print(f"✅ Ryzen AIパス設定: {path}")
                break
        
        # NPUオーバレイ設定
        if "RYZEN_AI_INSTALLATION_PATH" in os.environ:
            base_path = os.environ["RYZEN_AI_INSTALLATION_PATH"]
            xclbin_path = os.path.join(base_path, "voe-4.0-win_amd64", "xclbins", "strix", "AMD_AIE2P_Nx4_Overlay.xclbin")
            
            if os.path.exists(xclbin_path):
                os.environ["XLNX_VART_FIRMWARE"] = xclbin_path
                os.environ["XLNX_TARGET_NAME"] = "AMD_AIE2P_Nx4_Overlay"
                os.environ["NUM_OF_DPU_RUNNERS"] = "1"
                print("✅ NPUオーバレイ設定完了")
            else:
                print("❌ NPUオーバレイファイルが見つかりません")
    
    def setup_model(self):
        """モデルセットアップ"""
        print(f"🚀 {self.model_name} セットアップ開始...")
        
        if self.model_type == "npu_optimized":
            return self._setup_npu_optimized_model()
        elif self.model_type == "large_japanese":
            return self._setup_large_japanese_model()
        elif self.model_type == "translation_optimized":
            return self._setup_translation_model()
        else:
            print(f"❌ 未対応のモデルタイプ: {self.model_type}")
            return False
    
    def _setup_npu_optimized_model(self) -> bool:
        """NPU最適化済みモデル（8B）のセットアップ"""
        print("🔧 NPU最適化済みモデルセットアップ中...")
        
        try:
            # CPU設定
            p = psutil.Process()
            p.cpu_affinity([0, 1, 2, 3])
            torch.set_num_threads(4)
            
            # ログ設定
            transformers.logging.set_verbosity_error()
            logging.disable(logging.CRITICAL)
            
            # トークナイザーロード
            print("📝 トークナイザーロード中...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # モデルロード（事前訓練済みNPUモデル）
            print("🤖 NPU最適化モデルロード中...")
            model_path = os.path.join(self.model_name, "pytorch_llama3_8b_w_bit_4_awq_amd.pt")
            
            if os.path.exists(model_path):
                self.model = torch.load(model_path)
                self.model.eval()
                self.model = self.model.to(torch.bfloat16)
                
                # NPU量子化設定
                if QLINEAR_AVAILABLE:
                    print("⚡ NPU量子化設定中...")
                    for n, m in self.model.named_modules():
                        if isinstance(m, qlinear.QLinearPerGrp):
                            print(f"  📊 量子化レイヤー: {n}")
                            m.device = "aie"
                            m.quantize_weights()
                    print("✅ NPU量子化完了")
                else:
                    print("⚠️ qlinearライブラリが利用できません")
                
                print("✅ NPU最適化モデルセットアップ完了")
                return True
            else:
                print(f"❌ モデルファイルが見つかりません: {model_path}")
                return False
                
        except Exception as e:
            print(f"❌ NPU最適化モデルセットアップエラー: {e}")
            return False
    
    def _setup_large_japanese_model(self) -> bool:
        """大規模日本語モデル（70B）のセットアップ"""
        print("🔧 大規模日本語モデル（70B）セットアップ中...")
        
        try:
            # トークナイザーロード
            print("📝 トークナイザーロード中...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # モデルロード（メモリ効率化）
            print("🤖 大規模モデルロード中（メモリ効率化）...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            print("🔄 ONNX変換準備中...")
            # ONNX変換は次のステップで実装
            
            print("✅ 大規模日本語モデルセットアップ完了")
            return True
            
        except Exception as e:
            print(f"❌ 大規模日本語モデルセットアップエラー: {e}")
            return False
    
    def _setup_translation_model(self) -> bool:
        """翻訳特化モデルのセットアップ"""
        print("🔧 翻訳特化モデルセットアップ中...")
        
        try:
            # トークナイザーロード
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # モデルロード
            model_path = os.path.join(self.model_name, "alma_w_bit_4_awq_fa_amd.pt")
            
            if os.path.exists(model_path):
                self.model = torch.load(model_path)
                self.model.eval()
                self.model = self.model.to(torch.bfloat16)
                
                # NPU量子化設定
                if QLINEAR_AVAILABLE:
                    for n, m in self.model.named_modules():
                        if isinstance(m, qlinear.QLinearPerGrp):
                            print(f"📊 量子化レイヤー: {n}")
                            m.device = "aie"
                            m.quantize_weights()
                
                print("✅ 翻訳特化モデルセットアップ完了")
                return True
            else:
                print(f"❌ モデルファイルが見つかりません: {model_path}")
                return False
                
        except Exception as e:
            print(f"❌ 翻訳特化モデルセットアップエラー: {e}")
            return False
    
    def generate_text(self, prompt: str, max_new_tokens: int = 200) -> str:
        """テキスト生成"""
        if not self.model or not self.tokenizer:
            return "❌ モデルが初期化されていません"
        
        try:
            start_time = time.time()
            
            if self.model_type == "npu_optimized":
                return self._generate_npu_optimized(prompt, max_new_tokens)
            elif self.model_type == "large_japanese":
                return self._generate_large_japanese(prompt, max_new_tokens)
            elif self.model_type == "translation_optimized":
                return self._generate_translation(prompt, max_new_tokens)
            else:
                return "❌ 未対応のモデルタイプ"
                
        except Exception as e:
            return f"❌ 生成エラー: {e}"
    
    def _generate_npu_optimized(self, prompt: str, max_new_tokens: int) -> str:
        """NPU最適化モデルでの生成"""
        print("⚡ NPU最適化生成開始...")
        
        # メッセージ形式
        messages = [
            {"role": "system", "content": "あなたは親切で知識豊富な日本語アシスタントです。"},
            {"role": "user", "content": prompt}
        ]
        
        # トークナイザー適用
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )
        
        # 終了トークン設定
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        # NPU生成
        start_time = time.time()
        outputs = self.model.generate(
            input_ids['input_ids'],
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            attention_mask=input_ids['attention_mask'],
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # 結果デコード
        response = outputs[0][input_ids['input_ids'].shape[-1]:]
        response_text = self.tokenizer.decode(response, skip_special_tokens=True)
        
        generation_time = time.time() - start_time
        tokens_generated = len(response)
        
        print(f"⚡ NPU生成完了: {tokens_generated}トークン, {generation_time:.2f}秒")
        print(f"🚀 生成速度: {tokens_generated/generation_time:.2f} トークン/秒")
        
        return response_text
    
    def _generate_large_japanese(self, prompt: str, max_new_tokens: int) -> str:
        """大規模日本語モデルでの生成"""
        print("🔥 大規模日本語モデル生成開始...")
        
        # メッセージ形式
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # トークナイザー適用
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # 生成
        start_time = time.time()
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # 結果デコード
        response = output_ids[0][input_ids.shape[-1]:]
        response_text = self.tokenizer.decode(response, skip_special_tokens=True)
        
        generation_time = time.time() - start_time
        tokens_generated = len(response)
        
        print(f"🔥 大規模モデル生成完了: {tokens_generated}トークン, {generation_time:.2f}秒")
        print(f"🚀 生成速度: {tokens_generated/generation_time:.2f} トークン/秒")
        
        return response_text
    
    def _generate_translation(self, prompt: str, max_new_tokens: int) -> str:
        """翻訳特化モデルでの生成"""
        print("🌐 翻訳特化生成開始...")
        
        # 翻訳プロンプト形式
        system = """あなたは高度な技能を持つプロの日本語-英語および英語-日本語翻訳者です。与えられたテキストを正確に翻訳してください。"""
        
        full_prompt = f"""{system}

### 指示:
{prompt}

### 応答:
"""
        
        # トークナイザー適用
        tokenized_input = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True,
            max_length=1600,
            truncation=True
        )
        
        # 生成
        start_time = time.time()
        outputs = self.model.generate(
            tokenized_input['input_ids'],
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            attention_mask=tokenized_input['attention_mask'],
            do_sample=True,
            temperature=0.3,
            top_p=0.5
        )
        
        # 結果デコード
        response = outputs[0][tokenized_input['input_ids'].shape[-1]:]
        response_text = self.tokenizer.decode(response, skip_special_tokens=True)
        
        generation_time = time.time() - start_time
        tokens_generated = len(response)
        
        print(f"🌐 翻訳生成完了: {tokens_generated}トークン, {generation_time:.2f}秒")
        print(f"🚀 生成速度: {tokens_generated/generation_time:.2f} トークン/秒")
        
        return response_text
    
    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報取得"""
        info = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "npu_optimized": self.model_type in ["npu_optimized", "translation_optimized"],
            "japanese_support": True,
            "estimated_parameters": self._get_parameter_count(),
            "memory_usage": self._get_memory_usage()
        }
        return info
    
    def _get_parameter_count(self) -> str:
        """パラメータ数推定"""
        if "8b" in self.model_name.lower():
            return "8B"
        elif "70b" in self.model_name.lower():
            return "70B"
        elif "7b" in self.model_name.lower():
            return "7B"
        else:
            return "Unknown"
    
    def _get_memory_usage(self) -> str:
        """メモリ使用量取得"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_gb = memory_info.rss / (1024 ** 3)
            return f"{memory_gb:.1f}GB"
        except:
            return "Unknown"


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="NPU最適化日本語モデルデモ")
    parser.add_argument("--model", default="llama3-8b-amd-npu", 
                       choices=["llama3-8b-amd-npu", "cyberagent/Llama-3.1-70B-Japanese-Instruct-2407", "ALMA-Ja-V3-amd-npu"],
                       help="使用するモデル")
    parser.add_argument("--prompt", default="人工知能の未来について教えてください。", help="生成プロンプト")
    parser.add_argument("--max-tokens", type=int, default=200, help="最大生成トークン数")
    parser.add_argument("--interactive", action="store_true", help="インタラクティブモード")
    
    args = parser.parse_args()
    
    print("🚀 NPU最適化日本語モデルデモ開始")
    print("🎯 真のNPU活用実現版")
    print("=" * 60)
    
    # モデル初期化
    model = NPUOptimizedJapaneseModel(args.model)
    
    # セットアップ
    if not model.setup_model():
        print("❌ モデルセットアップに失敗しました")
        return
    
    # モデル情報表示
    info = model.get_model_info()
    print(f"📱 モデル: {info['model_name']}")
    print(f"🔧 タイプ: {info['model_type']}")
    print(f"⚡ NPU最適化: {'✅' if info['npu_optimized'] else '❌'}")
    print(f"🇯🇵 日本語対応: {'✅' if info['japanese_support'] else '❌'}")
    print(f"📊 パラメータ数: {info['estimated_parameters']}")
    print(f"💾 メモリ使用量: {info['memory_usage']}")
    print("=" * 60)
    
    if args.interactive:
        # インタラクティブモード
        print("🇯🇵 NPU最適化日本語モデル - インタラクティブモード")
        print("💡 'exit'または'quit'で終了")
        print("=" * 60)
        
        while True:
            try:
                prompt = input("\n🤖 プロンプトを入力してください: ")
                
                if prompt.lower() in ['exit', 'quit', '終了']:
                    print("👋 インタラクティブモードを終了します")
                    break
                
                if not prompt.strip():
                    continue
                
                print("\n🔄 生成中...")
                response = model.generate_text(prompt, args.max_tokens)
                
                print(f"\n✅ 生成完了:")
                print(f"📝 応答: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 インタラクティブモードを終了します")
                break
            except Exception as e:
                print(f"\n❌ エラー: {e}")
    else:
        # 単発実行
        print(f"🤖 プロンプト: {args.prompt}")
        print("\n🔄 生成中...")
        
        response = model.generate_text(args.prompt, args.max_tokens)
        
        print(f"\n✅ 生成完了:")
        print(f"📝 応答: {response}")
    
    print("\n🏁 NPU最適化日本語モデルデモ完了")


if __name__ == "__main__":
    main()

