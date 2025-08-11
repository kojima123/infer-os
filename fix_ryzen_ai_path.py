"""
Ryzen AIパス修正スクリプト
正しいパス（C:\Program Files\RyzenAI\1.5）に環境変数を設定

使用方法:
    python fix_ryzen_ai_path.py
"""

import os
import subprocess

def fix_ryzen_ai_path():
    """Ryzen AIパス修正"""
    print("🔧 Ryzen AIパス修正スクリプト")
    print("=" * 50)
    
    # 正しいパス
    correct_path = r"C:\Program Files\RyzenAI\1.5"
    
    print(f"🎯 正しいパス: {correct_path}")
    
    # パス存在確認
    if os.path.exists(correct_path):
        print("✅ 正しいパスが存在します")
        
        # 環境変数設定（現在のセッション用）
        os.environ['RYZEN_AI_INSTALLATION_PATH'] = correct_path
        print(f"✅ 環境変数設定完了: RYZEN_AI_INSTALLATION_PATH={correct_path}")
        
        # NPUオーバレイ設定
        xclbin_path = os.path.join(
            correct_path, 
            "voe-4.0-win_amd64", 
            "xclbins", 
            "strix", 
            "AMD_AIE2P_Nx4_Overlay.xclbin"
        )
        
        if os.path.exists(xclbin_path):
            os.environ['XLNX_VART_FIRMWARE'] = xclbin_path
            os.environ['XLNX_TARGET_NAME'] = "AMD_AIE2P_Nx4_Overlay"
            print(f"✅ NPUオーバレイ設定完了: AMD_AIE2P_Nx4_Overlay")
            print(f"📁 XCLBINパス: {xclbin_path}")
        else:
            # PHX用パスも試行
            xclbin_path_phx = os.path.join(
                correct_path, 
                "voe-4.0-win_amd64", 
                "xclbins", 
                "phoenix", 
                "AMD_AIE2P_4x4_Overlay.xclbin"
            )
            
            if os.path.exists(xclbin_path_phx):
                os.environ['XLNX_VART_FIRMWARE'] = xclbin_path_phx
                os.environ['XLNX_TARGET_NAME'] = "AMD_AIE2P_4x4_Overlay"
                print(f"✅ NPUオーバレイ設定完了: AMD_AIE2P_4x4_Overlay (PHX)")
                print(f"📁 XCLBINパス: {xclbin_path_phx}")
            else:
                print("❌ NPUオーバレイファイルが見つかりません")
                return False
        
        # vaip_config.json確認
        config_path = os.path.join(correct_path, "voe-4.0-win_amd64", "vaip_config.json")
        if os.path.exists(config_path):
            print(f"✅ VitisAI設定ファイル確認: {config_path}")
        else:
            print(f"❌ VitisAI設定ファイルが見つかりません: {config_path}")
            return False
        
        print("\n🎉 Ryzen AIパス修正完了！")
        print("💡 以下のコマンドでVitisAI NPUデモを再実行してください:")
        print("   python run_vitisai_demo.py --interactive")
        
        return True
        
    else:
        print(f"❌ 正しいパスが存在しません: {correct_path}")
        
        # 利用可能なパス検索
        search_paths = [
            r"C:\Program Files\RyzenAI\1.5",
            r"C:\Program Files\RyzenAI\1.5.1", 
            r"C:\Program Files\RyzenAI",
            r"C:\AMD\RyzenAI\1.5",
            r"C:\AMD\RyzenAI\1.5.1",
            r"C:\AMD\RyzenAI"
        ]
        
        print("\n🔍 利用可能なパス検索:")
        found_paths = []
        for path in search_paths:
            if os.path.exists(path):
                found_paths.append(path)
                print(f"  ✅ 発見: {path}")
        
        if found_paths:
            print(f"\n💡 利用可能なパス: {found_paths[0]}")
            print("   手動で環境変数を設定してください:")
            print(f"   set RYZEN_AI_INSTALLATION_PATH={found_paths[0]}")
        else:
            print("\n❌ Ryzen AIインストールが見つかりません")
        
        return False

def show_current_environment():
    """現在の環境変数表示"""
    print("\n📊 現在の環境変数:")
    
    ryzen_ai_path = os.environ.get('RYZEN_AI_INSTALLATION_PATH', '未設定')
    xlnx_firmware = os.environ.get('XLNX_VART_FIRMWARE', '未設定')
    xlnx_target = os.environ.get('XLNX_TARGET_NAME', '未設定')
    
    print(f"  RYZEN_AI_INSTALLATION_PATH: {ryzen_ai_path}")
    print(f"  XLNX_VART_FIRMWARE: {xlnx_firmware}")
    print(f"  XLNX_TARGET_NAME: {xlnx_target}")

if __name__ == "__main__":
    print("🚀 Ryzen AIパス修正開始")
    
    # 現在の環境表示
    show_current_environment()
    
    # パス修正実行
    success = fix_ryzen_ai_path()
    
    if success:
        # 修正後の環境表示
        show_current_environment()
        print("\n✅ 修正完了！VitisAI NPUデモを再実行してください。")
    else:
        print("\n❌ 修正失敗。手動でパス設定が必要です。")

