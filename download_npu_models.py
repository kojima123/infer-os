#!/usr/bin/env python3
"""
NPU最適化日本語モデルダウンロードスクリプト
Hugging Faceから最適化済みモデルを自動ダウンロード
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


class NPUModelDownloader:
    """NPU最適化モデルダウンローダー"""
    
    def __init__(self):
        self.models = {
            "llama3-8b-amd-npu": {
                "repo": "dahara1/llama3-8b-amd-npu",
                "size": "8B",
                "description": "Llama3 8B NPU最適化版（多言語対応）",
                "npu_ready": True,
                "japanese_support": True,
                "estimated_size": "4.5GB"
            },
            "ALMA-Ja-V3-amd-npu": {
                "repo": "dahara1/ALMA-Ja-V3-amd-npu", 
                "size": "7B",
                "description": "ALMA 7B 日本語翻訳特化NPU版",
                "npu_ready": True,
                "japanese_support": True,
                "estimated_size": "4.0GB"
            },
            "llama3.1-8b-instruct-amd-npu": {
                "repo": "dahara1/llama3.1-8b-Instruct-amd-npu",
                "size": "8B", 
                "description": "Llama3.1 8B NPU最適化版",
                "npu_ready": True,
                "japanese_support": True,
                "estimated_size": "4.5GB"
            },
            "llama-translate-amd-npu": {
                "repo": "dahara1/llama-translate-amd-npu",
                "size": "7B",
                "description": "Llama翻訳特化NPU版",
                "npu_ready": True,
                "japanese_support": True,
                "estimated_size": "4.0GB"
            },
            "Llama-3.1-70B-Japanese": {
                "repo": "cyberagent/Llama-3.1-70B-Japanese-Instruct-2407",
                "size": "70B",
                "description": "Llama3.1 70B 日本語特化版（要ONNX変換）",
                "npu_ready": False,
                "japanese_support": True,
                "estimated_size": "140GB"
            }
        }
    
    def list_models(self):
        """利用可能なモデル一覧表示"""
        print("🔍 利用可能なNPU最適化日本語モデル")
        print("=" * 80)
        
        for model_key, info in self.models.items():
            npu_status = "✅ NPU対応済み" if info["npu_ready"] else "🔄 ONNX変換必要"
            jp_status = "🇯🇵 日本語対応" if info["japanese_support"] else "🌐 多言語"
            
            print(f"📱 {model_key}")
            print(f"   📊 サイズ: {info['size']} ({info['estimated_size']})")
            print(f"   📝 説明: {info['description']}")
            print(f"   ⚡ NPU: {npu_status}")
            print(f"   🗣️ 言語: {jp_status}")
            print(f"   📦 リポジトリ: {info['repo']}")
            print()
    
    def check_requirements(self) -> bool:
        """必要な依存関係チェック"""
        print("🔍 依存関係チェック中...")
        
        # huggingface-hubチェック
        try:
            import huggingface_hub
            print("✅ huggingface-hub: インストール済み")
        except ImportError:
            print("❌ huggingface-hub: 未インストール")
            print("💡 インストール: pip install huggingface-hub")
            return False
        
        # huggingface-cliチェック
        try:
            result = subprocess.run(["huggingface-cli", "--help"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ huggingface-cli: 利用可能")
            else:
                print("❌ huggingface-cli: 利用不可")
                return False
        except FileNotFoundError:
            print("❌ huggingface-cli: 見つかりません")
            print("💡 インストール: pip install -U 'huggingface_hub[cli]'")
            return False
        
        return True
    
    def download_model(self, model_key: str, local_dir: str = None) -> bool:
        """モデルダウンロード"""
        if model_key not in self.models:
            print(f"❌ 未知のモデル: {model_key}")
            self.list_models()
            return False
        
        model_info = self.models[model_key]
        repo = model_info["repo"]
        
        if local_dir is None:
            local_dir = model_key
        
        print(f"📥 モデルダウンロード開始")
        print(f"📱 モデル: {model_key}")
        print(f"📦 リポジトリ: {repo}")
        print(f"📊 サイズ: {model_info['size']} ({model_info['estimated_size']})")
        print(f"📁 保存先: {local_dir}")
        print("=" * 60)
        
        # ダウンロードディレクトリ作成
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        
        # huggingface-cliでダウンロード
        cmd = [
            "huggingface-cli",
            "download",
            repo,
            "--revision", "main",
            "--local-dir", local_dir
        ]
        
        try:
            print("🔄 ダウンロード実行中...")
            print(f"💻 コマンド: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, check=True, text=True)
            
            print("✅ ダウンロード完了")
            
            # ダウンロード後の確認
            self._verify_download(local_dir, model_info)
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ ダウンロードエラー: {e}")
            return False
        except Exception as e:
            print(f"❌ 予期しないエラー: {e}")
            return False
    
    def _verify_download(self, local_dir: str, model_info: dict):
        """ダウンロード確認"""
        print("🔍 ダウンロード確認中...")
        
        local_path = Path(local_dir)
        
        # 基本ファイル確認
        required_files = ["config.json", "tokenizer.json"]
        optional_files = ["README.md", "tokenizer_config.json"]
        
        for file_name in required_files:
            file_path = local_path / file_name
            if file_path.exists():
                print(f"✅ {file_name}: 存在")
            else:
                print(f"⚠️ {file_name}: 見つかりません")
        
        # NPU最適化ファイル確認
        if model_info["npu_ready"]:
            npu_files = [
                "pytorch_llama3_8b_w_bit_4_awq_amd.pt",
                "alma_w_bit_4_awq_fa_amd.pt"
            ]
            
            npu_file_found = False
            for npu_file in npu_files:
                npu_path = local_path / npu_file
                if npu_path.exists():
                    print(f"⚡ NPU最適化ファイル: {npu_file} ✅")
                    npu_file_found = True
                    break
            
            if not npu_file_found:
                print("⚠️ NPU最適化ファイルが見つかりません")
        
        # ディスク使用量確認
        try:
            total_size = sum(f.stat().st_size for f in local_path.rglob('*') if f.is_file())
            size_gb = total_size / (1024 ** 3)
            print(f"💾 ダウンロードサイズ: {size_gb:.1f}GB")
        except Exception as e:
            print(f"⚠️ サイズ計算エラー: {e}")
        
        print("✅ ダウンロード確認完了")
    
    def download_all_npu_ready(self):
        """NPU対応済みモデルを全てダウンロード"""
        print("📥 NPU対応済みモデル一括ダウンロード開始")
        print("=" * 60)
        
        npu_ready_models = [key for key, info in self.models.items() if info["npu_ready"]]
        
        print(f"🎯 対象モデル数: {len(npu_ready_models)}")
        for model in npu_ready_models:
            print(f"  📱 {model} ({self.models[model]['size']})")
        print()
        
        success_count = 0
        for model_key in npu_ready_models:
            print(f"\n📥 {model_key} ダウンロード中...")
            if self.download_model(model_key):
                success_count += 1
                print(f"✅ {model_key} ダウンロード成功")
            else:
                print(f"❌ {model_key} ダウンロード失敗")
        
        print(f"\n🏁 一括ダウンロード完了")
        print(f"✅ 成功: {success_count}/{len(npu_ready_models)}")
    
    def get_download_instructions(self, model_key: str):
        """手動ダウンロード手順表示"""
        if model_key not in self.models:
            print(f"❌ 未知のモデル: {model_key}")
            return
        
        model_info = self.models[model_key]
        repo = model_info["repo"]
        
        print(f"📋 {model_key} 手動ダウンロード手順")
        print("=" * 60)
        print("1. 必要なライブラリインストール:")
        print("   pip install -U 'huggingface_hub[cli]'")
        print()
        print("2. ダウンロードコマンド:")
        print(f"   huggingface-cli download {repo} --revision main --local-dir {model_key}")
        print()
        print("3. 環境設定（NPU対応モデルの場合）:")
        if model_info["npu_ready"]:
            print("   set XLNX_VART_FIRMWARE=<ryzen_ai_path>\\voe-4.0-win_amd64\\1x4.xclbin")
            print("   set NUM_OF_DPU_RUNNERS=1")
        else:
            print("   ONNX変換が必要です")
        print()
        print("4. 実行:")
        print(f"   python npu_optimized_japanese_models.py --model {model_key} --interactive")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="NPU最適化日本語モデルダウンローダー")
    parser.add_argument("--list", action="store_true", help="利用可能なモデル一覧表示")
    parser.add_argument("--download", help="指定モデルをダウンロード")
    parser.add_argument("--download-all-npu", action="store_true", help="NPU対応済みモデルを全てダウンロード")
    parser.add_argument("--local-dir", help="ダウンロード先ディレクトリ")
    parser.add_argument("--instructions", help="指定モデルの手動ダウンロード手順表示")
    parser.add_argument("--check", action="store_true", help="依存関係チェック")
    
    args = parser.parse_args()
    
    downloader = NPUModelDownloader()
    
    if args.list:
        downloader.list_models()
    elif args.check:
        if downloader.check_requirements():
            print("✅ 全ての依存関係が満たされています")
        else:
            print("❌ 依存関係に問題があります")
    elif args.instructions:
        downloader.get_download_instructions(args.instructions)
    elif args.download:
        if not downloader.check_requirements():
            print("❌ 依存関係を先に解決してください")
            return
        
        downloader.download_model(args.download, args.local_dir)
    elif args.download_all_npu:
        if not downloader.check_requirements():
            print("❌ 依存関係を先に解決してください")
            return
        
        downloader.download_all_npu_ready()
    else:
        print("🚀 NPU最適化日本語モデルダウンローダー")
        print("=" * 60)
        print("使用方法:")
        print("  --list                 : モデル一覧表示")
        print("  --check                : 依存関係チェック")
        print("  --download MODEL       : 指定モデルダウンロード")
        print("  --download-all-npu     : NPU対応済みモデル一括ダウンロード")
        print("  --instructions MODEL   : 手動ダウンロード手順表示")
        print()
        print("例:")
        print("  python download_npu_models.py --list")
        print("  python download_npu_models.py --download llama3-8b-amd-npu")
        print("  python download_npu_models.py --download-all-npu")


if __name__ == "__main__":
    main()

