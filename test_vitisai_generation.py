"""
VitisAI NPU生成テスト専用スクリプト
1トークン問題の解決確認用

使用方法:
    python test_vitisai_generation.py
"""

import os
import sys

# 環境変数設定（テスト用）
os.environ['RYZEN_AI_INSTALLATION_PATH'] = r"C:\Program Files\RyzenAI\1.5"

def test_vitisai_generation():
    """VitisAI NPU生成テスト"""
    print("🧪 VitisAI NPU生成テスト開始")
    print("🎯 1トークン問題解決確認")
    print("=" * 50)
    
    try:
        # 必要なライブラリインポート
        import torch
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from vitisai_npu_engine import VitisAINPUEngine
        
        print("✅ ライブラリインポート成功")
        
        # 軽量テスト用設定
        model_name = "rinna/youri-7b-chat"
        test_prompt = "人参について"
        max_tokens = 20  # テスト用に短く
        
        print(f"📱 モデル: {model_name}")
        print(f"💬 テストプロンプト: {test_prompt}")
        print(f"🔢 最大トークン数: {max_tokens}")
        
        # トークナイザーロード
        print("\n📝 トークナイザーロード中...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ トークナイザーロード完了")
        print(f"📊 語彙サイズ: {len(tokenizer)}")
        print(f"🔚 終了トークンID: {tokenizer.eos_token_id}")
        
        # モデルロード（軽量設定）
        print("\n🤖 モデルロード中...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✅ モデルロード完了")
        
        # VitisAI NPUエンジンテスト
        print("\n🚀 VitisAI NPUエンジンテスト開始...")
        vitisai_engine = VitisAINPUEngine(model, tokenizer)
        
        if vitisai_engine.setup_vitisai_npu():
            print("✅ VitisAI NPUエンジンセットアップ成功")
            
            # 生成テスト実行
            print(f"\n⚡ VitisAI NPU生成テスト実行...")
            print(f"💬 プロンプト: \"{test_prompt}\"")
            
            result = vitisai_engine.generate_with_vitisai_npu(
                test_prompt, 
                max_new_tokens=max_tokens,
                temperature=0.8
            )
            
            if "error" not in result:
                print(f"\n✅ 生成テスト成功！")
                print(f"📝 生成テキスト: {result['generated_text']}")
                print(f"📊 出力トークン数: {result['output_tokens']}")
                print(f"⏱️ 生成時間: {result['generation_time']:.2f}秒")
                print(f"🚀 生成速度: {result['tokens_per_sec']:.1f} トークン/秒")
                print(f"🔧 推論方法: {result['inference_method']}")
                
                # 成功判定
                if result['output_tokens'] > 1:
                    print(f"\n🎉 1トークン問題解決成功！")
                    print(f"✅ {result['output_tokens']}トークン生成")
                else:
                    print(f"\n⚠️ 1トークン問題継続")
                    print(f"❌ {result['output_tokens']}トークンのみ")
            else:
                print(f"❌ 生成テストエラー: {result['error']}")
        else:
            print("❌ VitisAI NPUエンジンセットアップ失敗")
            
            # CPU推論テスト（比較用）
            print("\n🖥️ CPU推論テスト（比較用）...")
            inputs = tokenizer(test_prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"📝 CPU生成結果: {generated_text}")
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()

def show_token_analysis():
    """トークン分析表示"""
    print("\n🔍 トークン分析:")
    
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(
            "rinna/youri-7b-chat",
            trust_remote_code=True
        )
        
        test_text = "人参について教えてください。"
        tokens = tokenizer.encode(test_text)
        
        print(f"📝 テキスト: {test_text}")
        print(f"🔢 トークンID: {tokens}")
        print(f"📊 トークン数: {len(tokens)}")
        print(f"🔚 終了トークンID: {tokenizer.eos_token_id}")
        
        # 各トークンの詳細
        for i, token_id in enumerate(tokens):
            token_text = tokenizer.decode([token_id])
            print(f"  {i}: {token_id} → '{token_text}'")
        
    except Exception as e:
        print(f"❌ トークン分析エラー: {e}")

if __name__ == "__main__":
    print("🧪 VitisAI NPU生成テスト")
    print("🎯 1トークン問題解決確認")
    print("=" * 60)
    
    # トークン分析
    show_token_analysis()
    
    # 生成テスト
    test_vitisai_generation()
    
    print("\n🏁 テスト完了")

