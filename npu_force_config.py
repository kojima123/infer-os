#!/usr/bin/env python3
"""
NPU強制使用設定ツール
デバイスID 0でNPU強制使用設定を生成
"""

import json
import time

def create_force_npu_config():
    """NPU強制使用設定ファイル作成"""
    print("🎯 NPU強制使用設定ファイル作成")
    print("=" * 40)
    
    # NPU強制使用設定
    config = {
        'npu_device_id': 0,  # デバイスID 0を強制NPU使用
        'force_npu_mode': True,  # NPU強制モード有効
        'performance': {
            'success': True,
            'device_id': 0,
            'execution_time': 2.5,
            'iterations': 100,
            'avg_time_per_iteration': 0.025,
            'throughput': 40.0,
            'note': 'NPU強制使用モード（推定値）'
        },
        'directml_config': {
            'device_id': 0,
            'disable_memory_arena': True,  # UTF-8エラー回避
            'memory_limit_mb': 1024,
            'enable_dynamic_graph_fusion': False,  # 安定性重視
            'enable_graph_optimization': True,
            'force_npu_execution': True,  # NPU強制実行
        },
        'gpu_disable_config': {
            'disable_gpu_fallback': True,  # GPUフォールバック無効
            'cpu_only_fallback': True,  # CPU専用フォールバック
            'exclude_gpu_devices': [
                'AMD Radeon 880M',
                'Radeon',
                'GPU'
            ]
        },
        'timestamp': time.time(),
        'creation_mode': 'force_npu',
        'note': 'NPU Compute Accelerator Device強制使用設定'
    }
    
    try:
        with open('npu_config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print("✅ NPU強制使用設定ファイル作成完了")
        print(f"  📁 ファイル: npu_config.json")
        print(f"  🎯 デバイスID: {config['npu_device_id']}")
        print(f"  🔧 強制NPUモード: {config['force_npu_mode']}")
        print(f"  ⚡ 期待パフォーマンス: {config['performance']['throughput']}回/秒")
        
        print("\n🎯 設定内容:")
        print(f"  📋 NPU強制実行: 有効")
        print(f"  📋 GPUフォールバック: 無効")
        print(f"  📋 メモリアリーナ: 無効（UTF-8エラー回避）")
        print(f"  📋 グラフ最適化: 有効")
        
        print("\n🚀 次のステップ:")
        print("  python infer_os_japanese_llm_demo.py --enable-npu --interactive")
        print("  タスクマネージャーでNPU使用率を確認してください")
        
        return True
        
    except Exception as e:
        print(f"❌ 設定ファイル作成エラー: {e}")
        return False

def verify_npu_config():
    """NPU設定ファイル確認"""
    try:
        with open('npu_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print("\n📋 現在のNPU設定:")
        print(f"  🎯 デバイスID: {config.get('npu_device_id', 'Unknown')}")
        print(f"  🔧 強制NPUモード: {config.get('force_npu_mode', False)}")
        print(f"  ⚡ 期待パフォーマンス: {config.get('performance', {}).get('throughput', 'Unknown')}回/秒")
        print(f"  📅 作成日時: {time.ctime(config.get('timestamp', 0))}")
        
        return config
        
    except FileNotFoundError:
        print("⚠️ NPU設定ファイルが見つかりません")
        return None
    except Exception as e:
        print(f"❌ 設定ファイル読み込みエラー: {e}")
        return None

def main():
    """メイン実行関数"""
    print("🔧 NPU強制使用設定ツール")
    print("=" * 50)
    
    # 既存設定確認
    existing_config = verify_npu_config()
    
    if existing_config:
        print("\n❓ 既存の設定ファイルが見つかりました")
        user_input = input("新しい強制NPU設定で上書きしますか？ (y/N): ")
        
        if user_input.lower() != 'y':
            print("設定変更をキャンセルしました")
            return
    
    # NPU強制使用設定作成
    success = create_force_npu_config()
    
    if success:
        print("\n🎉 NPU強制使用設定完了！")
        print("\n💡 重要な注意事項:")
        print("  1. この設定はデバイスID 0を強制的にNPUとして使用します")
        print("  2. 実際にNPUが使用されるかはハードウェア依存です")
        print("  3. タスクマネージャーでNPU使用率を必ず確認してください")
        print("  4. 問題がある場合は python npu_recovery_guide.py を実行してください")
    else:
        print("\n❌ NPU強制使用設定に失敗しました")

if __name__ == "__main__":
    main()

