# ===================================================================
# 🚀 Infer-OS Windows NPU環境 自動セットアップスクリプト
# ===================================================================
# 
# 使用方法:
# 1. PowerShellを管理者権限で起動
# 2. Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# 3. .\windows_setup_scripts.ps1
#
# 対象: AMD Ryzen AI搭載Windows PC
# 所要時間: 約20-30分
# ===================================================================

param(
    [switch]$SkipDrivers,
    [switch]$SkipPython,
    [switch]$SkipGit,
    [string]$InstallPath = "C:\infer-os-test"
)

# エラーハンドリング設定
$ErrorActionPreference = "Stop"

# ログ関数
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $color = switch($Level) {
        "ERROR" { "Red" }
        "WARN"  { "Yellow" }
        "SUCCESS" { "Green" }
        default { "White" }
    }
    Write-Host "[$timestamp] [$Level] $Message" -ForegroundColor $color
}

# システム要件チェック
function Test-SystemRequirements {
    Write-Log "システム要件をチェック中..."
    
    # Windows バージョンチェック
    $osVersion = [System.Environment]::OSVersion.Version
    if ($osVersion.Major -lt 10 -or ($osVersion.Major -eq 10 -and $osVersion.Build -lt 22000)) {
        Write-Log "Windows 11 22H2以降が必要です。現在: $($osVersion)" "ERROR"
        exit 1
    }
    
    # メモリチェック
    $memory = Get-CimInstance -ClassName Win32_ComputerSystem
    $memoryGB = [math]::Round($memory.TotalPhysicalMemory / 1GB, 1)
    if ($memoryGB -lt 16) {
        Write-Log "16GB以上のメモリを推奨します。現在: ${memoryGB}GB" "WARN"
    }
    
    # AMD CPU チェック
    $cpu = Get-CimInstance -ClassName Win32_Processor
    if ($cpu.Name -notlike "*AMD*" -or $cpu.Name -notlike "*Ryzen*") {
        Write-Log "AMD Ryzen AI CPUが検出されませんでした: $($cpu.Name)" "WARN"
    }
    
    Write-Log "システム要件チェック完了" "SUCCESS"
}

# Chocolatey インストール
function Install-Chocolatey {
    if (Get-Command choco -ErrorAction SilentlyContinue) {
        Write-Log "Chocolatey は既にインストール済み"
        return
    }
    
    Write-Log "Chocolatey をインストール中..."
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    
    # パス更新
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    
    Write-Log "Chocolatey インストール完了" "SUCCESS"
}

# Git インストール
function Install-Git {
    if (-not $SkipGit -and (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Log "Git は既にインストール済み"
        return
    }
    
    if (-not $SkipGit) {
        Write-Log "Git をインストール中..."
        choco install git -y
        
        # パス更新
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        
        Write-Log "Git インストール完了" "SUCCESS"
    }
}

# Python インストール
function Install-Python {
    if (-not $SkipPython -and (Get-Command python -ErrorAction SilentlyContinue)) {
        $pythonVersion = python --version 2>&1
        if ($pythonVersion -match "3\.(9|10|11)") {
            Write-Log "Python は既にインストール済み: $pythonVersion"
            return
        }
    }
    
    if (-not $SkipPython) {
        Write-Log "Python 3.11 をインストール中..."
        choco install python311 -y
        
        # パス更新
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        
        Write-Log "Python インストール完了" "SUCCESS"
    }
}

# AMD ドライバー インストール
function Install-AMDDrivers {
    if ($SkipDrivers) {
        Write-Log "ドライバーインストールをスキップ"
        return
    }
    
    Write-Log "AMD ドライバーをインストール中..."
    try {
        choco install amd-ryzen-chipset -y
        Write-Log "AMD ドライバー インストール完了" "SUCCESS"
    } catch {
        Write-Log "AMD ドライバー自動インストール失敗。手動インストールを推奨" "WARN"
        Write-Log "ダウンロード先: https://www.amd.com/support" "INFO"
    }
}

# プロジェクトセットアップ
function Setup-Project {
    Write-Log "Infer-OS プロジェクトをセットアップ中..."
    
    # 作業ディレクトリ作成
    if (-not (Test-Path $InstallPath)) {
        New-Item -ItemType Directory -Path $InstallPath -Force | Out-Null
    }
    
    Set-Location $InstallPath
    
    # GitHubからクローン
    if (Test-Path "infer-os") {
        Write-Log "既存のinfer-osディレクトリを削除中..."
        Remove-Item -Recurse -Force "infer-os"
    }
    
    Write-Log "GitHubからクローン中..."
    git clone https://github.com/kojima123/infer-os.git
    Set-Location "infer-os"
    
    Write-Log "プロジェクトセットアップ完了" "SUCCESS"
}

# Python環境セットアップ
function Setup-PythonEnvironment {
    Write-Log "Python仮想環境をセットアップ中..."
    
    # 仮想環境作成
    if (Test-Path "venv") {
        Write-Log "既存の仮想環境を削除中..."
        Remove-Item -Recurse -Force "venv"
    }
    
    python -m venv venv
    
    # 仮想環境アクティベート
    & ".\venv\Scripts\Activate.ps1"
    
    # pip アップグレード
    python -m pip install --upgrade pip
    
    Write-Log "Python仮想環境セットアップ完了" "SUCCESS"
}

# 依存関係インストール
function Install-Dependencies {
    Write-Log "依存関係をインストール中..."
    
    # PyTorch (CPU版)
    Write-Log "PyTorch をインストール中..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # ONNX Runtime
    Write-Log "ONNX Runtime をインストール中..."
    pip install onnxruntime
    
    # 基本ライブラリ
    Write-Log "基本ライブラリをインストール中..."
    pip install numpy pandas matplotlib seaborn
    pip install flask requests beautifulsoup4
    pip install psutil
    
    # プロジェクト固有依存関係
    if (Test-Path "requirements.txt") {
        Write-Log "プロジェクト固有依存関係をインストール中..."
        pip install -r requirements.txt
    }
    
    Write-Log "依存関係インストール完了" "SUCCESS"
}

# システム検証
function Test-Installation {
    Write-Log "インストール検証中..."
    
    # Python バージョン確認
    $pythonVersion = python --version
    Write-Log "Python バージョン: $pythonVersion"
    
    # PyTorch 確認
    $torchTest = python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>&1
    Write-Log "PyTorch テスト: $torchTest"
    
    # ONNX Runtime 確認
    $onnxTest = python -c "import onnxruntime; print(f'ONNX Runtime: {onnxruntime.__version__}')" 2>&1
    Write-Log "ONNX Runtime テスト: $onnxTest"
    
    # NPU デバイス検出
    Write-Log "NPU デバイス検出中..."
    $gpuInfo = Get-CimInstance -ClassName Win32_VideoController | Where-Object { $_.Name -like "*AMD*" }
    if ($gpuInfo) {
        Write-Log "AMD GPU/NPU 検出: $($gpuInfo.Name)" "SUCCESS"
    } else {
        Write-Log "AMD GPU/NPU が検出されませんでした" "WARN"
    }
    
    Write-Log "インストール検証完了" "SUCCESS"
}

# メイン実行
function Main {
    Write-Log "=== Infer-OS Windows NPU環境セットアップ開始 ===" "SUCCESS"
    Write-Log "インストールパス: $InstallPath"
    
    try {
        Test-SystemRequirements
        Install-Chocolatey
        Install-Git
        Install-Python
        Install-AMDDrivers
        Setup-Project
        Setup-PythonEnvironment
        Install-Dependencies
        Test-Installation
        
        Write-Log "=== セットアップ完了 ===" "SUCCESS"
        Write-Log "次のステップ:"
        Write-Log "1. .\venv\Scripts\Activate.ps1  # 仮想環境アクティベート"
        Write-Log "2. python benchmarks/integrated_performance_test.py  # テスト実行"
        Write-Log "3. python infer-os-demo/src/main.py  # Webデモ起動"
        
    } catch {
        Write-Log "セットアップエラー: $($_.Exception.Message)" "ERROR"
        Write-Log "詳細: $($_.ScriptStackTrace)" "ERROR"
        exit 1
    }
}

# スクリプト実行
Main

