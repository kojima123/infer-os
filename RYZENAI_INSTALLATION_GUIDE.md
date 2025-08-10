# RyzenAI 1.5.1 インストールガイド（旧版）

## ⚠️ 重要な更新

**この文書は旧版です。最新の公式仕様に基づいたガイドは `RYZENAI_OFFICIAL_SETUP_GUIDE.md` をご覧ください。**

## 🎯 概要

AMD Ryzen AI 9 365のNPUを使用するためのRyzenAI 1.5.1 SDKインストールガイドです。

**注意**: この実装は独自のryzenaiモジュールを想定していましたが、実際にはAMD公式のLemonade SDKを使用する必要があります。

## 📋 前提条件

### ハードウェア要件
- **CPU**: AMD Ryzen AI 9 365 (NPU搭載)
- **メモリ**: 16GB以上推奨
- **OS**: Windows 11 (22H2以降)

### ソフトウェア要件
- **Python**: 3.8-3.11
- **Visual Studio**: 2019または2022 (C++ビルドツール)
- **Git**: 最新版

## 🚀 インストール手順

### 1. AMD公式サイトからSDKダウンロード

```powershell
# AMD Developer サイトにアクセス
# https://www.amd.com/en/developer/ryzen-ai.html
```

**ダウンロードファイル**:
- `ryzenai-1.5.1-windows-x64.zip`
- `ryzenai-python-1.5.1.whl`

### 2. RyzenAI SDKインストール

```powershell
# 1. SDKファイル展開
Expand-Archive -Path "ryzenai-1.5.1-windows-x64.zip" -DestinationPath "C:\RyzenAI"

# 2. 環境変数設定
$env:RYZENAI_ROOT = "C:\RyzenAI"
$env:PATH += ";C:\RyzenAI\bin"

# 3. Python パッケージインストール
pip install ryzenai-python-1.5.1.whl

# 4. 依存関係インストール
pip install numpy onnx onnxruntime torch transformers
```

### 3. NPUドライバー確認

```powershell
# デバイスマネージャーでNPU確認
devmgmt.msc

# NPU Compute Accelerator Device が表示されることを確認
```

### 4. RyzenAI動作確認

```python
# テストスクリプト実行
python ryzenai_npu_engine.py
```

**期待される出力**:
```
✅ RyzenAI 1.5.1 SDK インポート成功
🚀 RyzenAI 1.5.1 NPU推論エンジン初期化開始
🔧 RyzenAI SDK初期化中...
📱 利用可能デバイス: X個
🎯 NPUデバイス発見: 1個
  📱 NPU 0: AMD Ryzen AI NPU
✅ 使用NPUデバイス: AMD Ryzen AI NPU
⚡ RyzenAI Runtime初期化中...
✅ RyzenAI Runtime初期化成功
🧪 NPU性能テスト実行中...
✅ NPU性能テスト完了
🎉 RyzenAI NPU推論エンジン準備完了！
```

## 🔧 トラブルシューティング

### 問題1: RyzenAI SDKインポートエラー

**症状**:
```
ImportError: No module named 'ryzenai'
```

**解決方法**:
```powershell
# 1. Python環境確認
python --version

# 2. 仮想環境作成（推奨）
python -m venv ryzenai_env
ryzenai_env\Scripts\activate

# 3. RyzenAI再インストール
pip install --force-reinstall ryzenai-python-1.5.1.whl
```

### 問題2: NPUデバイス未認識

**症状**:
```
❌ NPUデバイスが見つかりません
```

**解決方法**:
```powershell
# 1. NPUドライバー更新
# デバイスマネージャー → NPU Compute Accelerator Device → ドライバー更新

# 2. Windows Update実行
# 設定 → Windows Update → 更新プログラムのチェック

# 3. システム再起動
shutdown /r /t 0
```

### 問題3: 権限エラー

**症状**:
```
PermissionError: Access denied
```

**解決方法**:
```powershell
# 管理者権限でPowerShell実行
# スタートメニュー → PowerShell → 右クリック → 管理者として実行

# RyzenAI フォルダー権限設定
icacls "C:\RyzenAI" /grant Users:F /T
```

## 📦 パッケージ依存関係

### 必須パッケージ
```
ryzenai>=1.5.1
numpy>=1.21.0
onnx>=1.12.0
onnxruntime>=1.15.0
torch>=2.0.0
transformers>=4.30.0
```

### オプションパッケージ
```
matplotlib>=3.5.0  # 可視化用
jupyter>=1.0.0     # ノートブック用
tensorboard>=2.10.0  # ログ用
```

## 🎯 使用方法

### 基本的な使用方法

```python
# RyzenAI NPU推論エンジンテスト
python ryzenai_npu_engine.py

# RyzenAI統合日本語LLMテスト
python ryzenai_japanese_llm.py --prompt "人参について教えてください。"

# インタラクティブモード
python ryzenai_japanese_llm.py --interactive
```

### 高度な使用方法

```python
from ryzenai_npu_engine import RyzenAINPUEngine
from ryzenai_japanese_llm import RyzenAIJapaneseLLM

# NPU推論エンジン初期化
engine = RyzenAINPUEngine()

# 日本語LLM初期化
llm = RyzenAIJapaneseLLM("rinna/youri-7b-chat")

# NPU推論実行
result = llm.generate_with_ryzenai("こんにちは")
```

## 📊 パフォーマンス期待値

### NPU使用時
- **推論速度**: 50-100トークン/秒
- **メモリ使用量**: 8-12GB
- **NPU使用率**: 70-90%
- **消費電力**: 低消費電力

### CPU比較
- **速度向上**: 2-3倍
- **消費電力**: 50-70%削減
- **並列処理**: 高効率

## 🔗 参考リンク

- [AMD RyzenAI 公式サイト](https://www.amd.com/en/developer/ryzen-ai.html)
- [RyzenAI SDK ドキュメント](https://ryzenai.docs.amd.com/)
- [AMD Developer フォーラム](https://community.amd.com/t5/ai-ml/ct-p/ai-ml)
- [GitHub RyzenAI サンプル](https://github.com/amd/ryzenai)

## 📞 サポート

### 技術サポート
- **AMD Developer サポート**: [support@amd.com](mailto:support@amd.com)
- **コミュニティフォーラム**: AMD Developer Community
- **GitHub Issues**: RyzenAI リポジトリ

### よくある質問
1. **Q**: RyzenAI 1.5.1は他のAMD CPUでも動作しますか？
   **A**: Ryzen AI シリーズ（NPU搭載）でのみ動作します。

2. **Q**: Windows 10でも使用できますか？
   **A**: Windows 11 (22H2以降) が推奨です。

3. **Q**: 他のLLMモデルでも使用できますか？
   **A**: はい、Transformers対応モデルであれば使用可能です。

