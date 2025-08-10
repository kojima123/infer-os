#!/usr/bin/env python3
"""
RyzenAI統合日本語LLMデモ
RyzenAI 1.5.1 + rinna/youri-7b-chat統合実装
"""

import os
import sys
import time
import torch
import numpy as np
from typing import Optional, List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

# RyzenAI NPU推論エンジンインポート
from ryzenai_npu_engine import RyzenAINPUEngine, RYZENAI_AVAILABLE

class RyzenAIJapaneseLLM:
    """RyzenAI統合日本語LLM"""
    
    def __init__(self, model_name: str = "rinna/youri-7b-chat"):
        """
        RyzenAI統合日本語LLM初期化
        
        Args:
            model_name: 使用するモデル名
        """
        self.model_name = model_name
        self.tokenizer = None
        self.pytorch_model = None
        self.ryzenai_engine = None
        self.npu_layers = None
        
        print("🇯🇵 RyzenAI統合日本語LLM初期化開始")
        print(f"📦 モデル: {model_name}")
        
        # 初期化実行
        self._initialize_components()
    
    def _initialize_components(self):
        """コンポーネント初期化"""
        try:
            # 1. トークナイザー初期化
            print("🔤 トークナイザー初期化中...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✅ トークナイザー初期化完了")
            
            # 2. PyTorchモデル初期化
            print("🧠 PyTorchモデル初期化中...")
            self.pytorch_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="cpu",  # CPUで初期化
                trust_remote_code=True
            )
            
            print("✅ PyTorchモデル初期化完了")
            
            # 3. RyzenAI NPU推論エンジン初期化
            print("⚡ RyzenAI NPU推論エンジン初期化中...")
            self.ryzenai_engine = RyzenAINPUEngine()
            
            if self.ryzenai_engine.npu_available:
                print("✅ RyzenAI NPU推論エンジン初期化成功")
                
                # NPU用LLM推論レイヤー作成
                self._setup_npu_layers()
            else:
                print("⚠️ RyzenAI NPU推論エンジンが利用できません")
                print("💡 CPUフォールバックモードで動作します")
            
        except Exception as e:
            print(f"❌ コンポーネント初期化エラー: {e}")
            raise
    
    def _setup_npu_layers(self):
        """NPU用LLM推論レイヤーセットアップ"""
        try:
            print("🔧 NPU用LLM推論レイヤーセットアップ中...")
            
            # モデル設定取得
            config = self.pytorch_model.config
            vocab_size = config.vocab_size
            hidden_size = config.hidden_size
            
            print(f"  📊 語彙サイズ: {vocab_size}")
            print(f"  📊 隠れ層サイズ: {hidden_size}")
            
            # RyzenAI用LLM推論レイヤー作成
            self.npu_layers = self.ryzenai_engine.create_simple_llm_inference(
                vocab_size=vocab_size,
                hidden_dim=hidden_size
            )
            
            if self.npu_layers:
                print("✅ NPU用LLM推論レイヤーセットアップ完了")
            else:
                print("❌ NPU用LLM推論レイヤーセットアップ失敗")
                
        except Exception as e:
            print(f"❌ NPU推論レイヤーセットアップエラー: {e}")
            self.npu_layers = None
    
    def generate_with_ryzenai(self, prompt: str, max_length: int = 300) -> str:
        """
        RyzenAI NPU使用テキスト生成
        
        Args:
            prompt: 入力プロンプト
            max_length: 最大生成長
            
        Returns:
            str: 生成されたテキスト
        """
        if not self.ryzenai_engine.npu_available or not self.npu_layers:
            print("⚠️ RyzenAI NPU推論が利用できません、CPUフォールバック")
            return self._generate_cpu_fallback(prompt, max_length)
        
        try:
            print("⚡ RyzenAI NPU推論による日本語生成開始")
            print(f"📝 プロンプト: \"{prompt}\"")
            
            # トークン化
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]
            
            print(f"🔤 入力トークン数: {input_ids.shape[1]}")
            
            generated_tokens = []
            current_ids = input_ids
            
            start_time = time.time()
            
            # 逐次生成
            for step in range(max_length):
                # PyTorchモデルで隠れ状態取得
                with torch.no_grad():
                    outputs = self.pytorch_model(current_ids, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]  # 最後の隠れ状態
                    last_hidden = hidden_states[:, -1, :].cpu().numpy()  # 最後のトークンの隠れ状態
                
                # RyzenAI NPUで推論実行
                if step % 10 == 0:
                    print(f"  🔄 生成ステップ {step+1}/{max_length}")
                
                # NPU推論
                npu_logits = self._npu_inference(last_hidden)
                
                if npu_logits is not None:
                    # 次トークン選択
                    next_token_id = self._sample_next_token(npu_logits)
                    
                    # 生成終了判定
                    if next_token_id == self.tokenizer.eos_token_id:
                        print("🏁 生成終了トークン検出")
                        break
                    
                    generated_tokens.append(next_token_id)
                    
                    # 次の入力準備
                    next_token_tensor = torch.tensor([[next_token_id]])
                    current_ids = torch.cat([current_ids, next_token_tensor], dim=1)
                    
                    # メモリ効率のため、コンテキスト長制限
                    if current_ids.shape[1] > 512:
                        current_ids = current_ids[:, -256:]  # 後半256トークンを保持
                else:
                    print("⚠️ NPU推論失敗、CPUフォールバック")
                    return self._generate_cpu_fallback(prompt, max_length)
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # 生成結果デコード
            if generated_tokens:
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                full_text = prompt + generated_text
                
                print(f"✅ RyzenAI NPU生成完了")
                print(f"  ⏱️ 生成時間: {generation_time:.2f}秒")
                print(f"  🔤 生成トークン数: {len(generated_tokens)}")
                print(f"  🚀 生成速度: {len(generated_tokens)/generation_time:.1f}トークン/秒")
                
                return full_text
            else:
                print("⚠️ トークン生成なし")
                return prompt
                
        except Exception as e:
            print(f"❌ RyzenAI NPU生成エラー: {e}")
            return self._generate_cpu_fallback(prompt, max_length)
    
    def _npu_inference(self, hidden_state: np.ndarray) -> Optional[np.ndarray]:
        """NPU推論実行"""
        try:
            # RMSNorm実行
            normalized = self.npu_layers['rms_norm'](hidden_state)
            
            # Linear層（言語モデルヘッド）実行
            logits = self.npu_layers['lm_head'](normalized)
            
            return logits
            
        except Exception as e:
            print(f"❌ NPU推論エラー: {e}")
            return None
    
    def _sample_next_token(self, logits: np.ndarray, temperature: float = 0.7) -> int:
        """次トークンサンプリング"""
        try:
            # 温度スケーリング
            logits = logits / temperature
            
            # ソフトマックス
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            
            # サンプリング
            next_token_id = np.random.choice(len(probs), p=probs)
            
            return int(next_token_id)
            
        except Exception as e:
            print(f"❌ トークンサンプリングエラー: {e}")
            return self.tokenizer.eos_token_id
    
    def _generate_cpu_fallback(self, prompt: str, max_length: int = 300) -> str:
        """CPUフォールバック生成"""
        try:
            print("🖥️ CPU推論による日本語生成")
            
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.pytorch_model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text
            
        except Exception as e:
            print(f"❌ CPU生成エラー: {e}")
            return prompt
    
    def interactive_mode(self):
        """インタラクティブモード"""
        print("\n🇯🇵 RyzenAI統合日本語LLMインタラクティブモード")
        print("=" * 60)
        
        # NPU状態表示
        if self.ryzenai_engine.npu_available:
            status = self.ryzenai_engine.get_npu_status()
            print("⚡ NPU状態:")
            print(f"  📱 デバイス: {status.get('device', 'Unknown')}")
            print(f"  🚀 パフォーマンス: {status.get('performance_stats', {}).get('throughput', 'Unknown')}回/秒")
        else:
            print("🖥️ CPU推論モード")
        
        print("\n💡 'exit'または'quit'で終了")
        print("=" * 60)
        
        while True:
            try:
                prompt = input("\n🤖 プロンプトを入力してください: ").strip()
                
                if prompt.lower() in ['exit', 'quit', '終了']:
                    print("👋 RyzenAI統合日本語LLMを終了します")
                    break
                
                if not prompt:
                    print("⚠️ プロンプトを入力してください")
                    continue
                
                print("\n🔄 生成中...")
                
                # RyzenAI NPU生成実行
                result = self.generate_with_ryzenai(prompt, max_length=300)
                
                print(f"\n📝 生成結果:")
                print("-" * 40)
                print(result)
                print("-" * 40)
                
            except KeyboardInterrupt:
                print("\n\n👋 RyzenAI統合日本語LLMを終了します")
                break
            except Exception as e:
                print(f"\n❌ エラーが発生しました: {e}")

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="RyzenAI統合日本語LLMデモ")
    parser.add_argument("--model", default="rinna/youri-7b-chat", help="使用するモデル名")
    parser.add_argument("--interactive", action="store_true", help="インタラクティブモード")
    parser.add_argument("--prompt", type=str, help="単発プロンプト実行")
    
    args = parser.parse_args()
    
    print("🎯 RyzenAI統合日本語LLMデモ")
    print("=" * 50)
    
    if not RYZENAI_AVAILABLE:
        print("⚠️ RyzenAI SDK が利用できません")
        print("💡 RyzenAI 1.5.1 SDKをインストールしてください")
        print("📦 インストール方法:")
        print("  pip install ryzenai")
        print("  または AMD公式サイトからSDKをダウンロード")
        return
    
    # RyzenAI統合日本語LLM初期化
    llm = RyzenAIJapaneseLLM(model_name=args.model)
    
    if args.interactive:
        # インタラクティブモード
        llm.interactive_mode()
    elif args.prompt:
        # 単発プロンプト実行
        print(f"\n📝 プロンプト: {args.prompt}")
        print("🔄 生成中...")
        
        result = llm.generate_with_ryzenai(args.prompt)
        
        print(f"\n📝 生成結果:")
        print("-" * 40)
        print(result)
        print("-" * 40)
    else:
        # デフォルトテスト
        test_prompt = "人参について教えてください。"
        print(f"\n🧪 テスト実行: {test_prompt}")
        
        result = llm.generate_with_ryzenai(test_prompt)
        
        print(f"\n📝 生成結果:")
        print("-" * 40)
        print(result)
        print("-" * 40)

if __name__ == "__main__":
    main()

