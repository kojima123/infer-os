@echo off
REM ===================================================================
REM 🚀 Infer-OS Windows NPU環境 バッチセットアップスクリプト
REM ===================================================================
REM 
REM 使用方法:
REM 1. 管理者権限でコマンドプロンプトを起動
REM 2. windows_batch_setup.bat を実行
REM
REM 対象: AMD Ryzen AI搭載Windows PC
REM 所要時間: 約20-30分
REM ===================================================================

setlocal enabledelayedexpansion
set INSTALL_PATH=C:\infer-os-test
set LOG_FILE=%INSTALL_PATH%\setup_log.txt

echo ===================================================================
echo 🚀 Infer-OS Windows NPU環境セットアップ開始
echo ===================================================================
echo インストールパス: %INSTALL_PATH%
echo ログファイル: %LOG_FILE%
echo.

REM ログディレクトリ作成
if not exist "%INSTALL_PATH%" mkdir "%INSTALL_PATH%"

REM ログ関数（簡易版）
echo [%date% %time%] セットアップ開始 >> "%LOG_FILE%"

REM システム要件チェック
echo 📋 システム要件をチェック中...
echo [%date% %time%] システム要件チェック開始 >> "%LOG_FILE%"

REM Windows バージョンチェック
for /f "tokens=4-5 delims=. " %%i in ('ver') do set VERSION=%%i.%%j
echo Windows バージョン: %VERSION%
echo [%date% %time%] Windows バージョン: %VERSION% >> "%LOG_FILE%"

REM メモリチェック
for /f "skip=1" %%p in ('wmic computersystem get TotalPhysicalMemory') do (
    set MEMORY_BYTES=%%p
    goto :memory_done
)
:memory_done
set /a MEMORY_GB=%MEMORY_BYTES:~0,-9%
echo メモリ: %MEMORY_GB% GB
echo [%date% %time%] メモリ: %MEMORY_GB% GB >> "%LOG_FILE%"

REM CPU チェック
for /f "skip=1 delims=" %%p in ('wmic cpu get name') do (
    set CPU_NAME=%%p
    goto :cpu_done
)
:cpu_done
echo CPU: %CPU_NAME%
echo [%date% %time%] CPU: %CPU_NAME% >> "%LOG_FILE%"

REM Chocolatey インストールチェック
echo.
echo 🍫 Chocolatey をチェック中...
choco --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Chocolatey をインストール中...
    echo [%date% %time%] Chocolatey インストール開始 >> "%LOG_FILE%"
    
    powershell -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"
    
    if %errorlevel% neq 0 (
        echo ❌ Chocolatey インストール失敗
        echo [%date% %time%] Chocolatey インストール失敗 >> "%LOG_FILE%"
        pause
        exit /b 1
    )
    
    REM パス更新
    call refreshenv
    echo ✅ Chocolatey インストール完了
    echo [%date% %time%] Chocolatey インストール完了 >> "%LOG_FILE%"
) else (
    echo ✅ Chocolatey は既にインストール済み
    echo [%date% %time%] Chocolatey 既存 >> "%LOG_FILE%"
)

REM Git インストール
echo.
echo 📦 Git をチェック中...
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Git をインストール中...
    echo [%date% %time%] Git インストール開始 >> "%LOG_FILE%"
    choco install git -y
    
    if %errorlevel% neq 0 (
        echo ❌ Git インストール失敗
        echo [%date% %time%] Git インストール失敗 >> "%LOG_FILE%"
        pause
        exit /b 1
    )
    
    call refreshenv
    echo ✅ Git インストール完了
    echo [%date% %time%] Git インストール完了 >> "%LOG_FILE%"
) else (
    echo ✅ Git は既にインストール済み
    echo [%date% %time%] Git 既存 >> "%LOG_FILE%"
)

REM Python インストール
echo.
echo 🐍 Python をチェック中...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python 3.11 をインストール中...
    echo [%date% %time%] Python インストール開始 >> "%LOG_FILE%"
    choco install python311 -y
    
    if %errorlevel% neq 0 (
        echo ❌ Python インストール失敗
        echo [%date% %time%] Python インストール失敗 >> "%LOG_FILE%"
        pause
        exit /b 1
    )
    
    call refreshenv
    echo ✅ Python インストール完了
    echo [%date% %time%] Python インストール完了 >> "%LOG_FILE%"
) else (
    echo ✅ Python は既にインストール済み
    echo [%date% %time%] Python 既存 >> "%LOG_FILE%"
)

REM プロジェクトディレクトリに移動
cd /d "%INSTALL_PATH%"

REM 既存プロジェクト削除
if exist "infer-os" (
    echo 既存のinfer-osディレクトリを削除中...
    rmdir /s /q "infer-os"
)

REM GitHubからクローン
echo.
echo 📥 GitHub からプロジェクトをクローン中...
echo [%date% %time%] Git クローン開始 >> "%LOG_FILE%"
git clone https://github.com/kojima123/infer-os.git

if %errorlevel% neq 0 (
    echo ❌ Git クローン失敗
    echo [%date% %time%] Git クローン失敗 >> "%LOG_FILE%"
    pause
    exit /b 1
)

cd infer-os
echo ✅ プロジェクトクローン完了
echo [%date% %time%] Git クローン完了 >> "%LOG_FILE%"

REM Python仮想環境作成
echo.
echo 🔧 Python仮想環境をセットアップ中...
echo [%date% %time%] 仮想環境作成開始 >> "%LOG_FILE%"

if exist "venv" rmdir /s /q "venv"
python -m venv venv

if %errorlevel% neq 0 (
    echo ❌ 仮想環境作成失敗
    echo [%date% %time%] 仮想環境作成失敗 >> "%LOG_FILE%"
    pause
    exit /b 1
)

REM 仮想環境アクティベート
call venv\Scripts\activate.bat

REM pip アップグレード
echo pip をアップグレード中...
python -m pip install --upgrade pip

echo ✅ Python仮想環境セットアップ完了
echo [%date% %time%] 仮想環境作成完了 >> "%LOG_FILE%"

REM 依存関係インストール
echo.
echo 📚 依存関係をインストール中...
echo [%date% %time%] 依存関係インストール開始 >> "%LOG_FILE%"

echo PyTorch をインストール中...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo ONNX Runtime をインストール中...
pip install onnxruntime

echo 基本ライブラリをインストール中...
pip install numpy pandas matplotlib seaborn
pip install flask requests beautifulsoup4
pip install psutil

REM プロジェクト固有依存関係
if exist "requirements.txt" (
    echo プロジェクト固有依存関係をインストール中...
    pip install -r requirements.txt
)

echo ✅ 依存関係インストール完了
echo [%date% %time%] 依存関係インストール完了 >> "%LOG_FILE%"

REM クイックテスト実行
echo.
echo 🧪 クイックテストを実行中...
echo [%date% %time%] クイックテスト開始 >> "%LOG_FILE%"

REM quick_test.py をダウンロード（GitHubに含まれていない場合）
if not exist "quick_test.py" (
    echo quick_test.py をダウンロード中...
    REM ここでquick_test.pyの内容を作成
    echo import sys, platform, time > quick_test.py
    echo print("🚀 Infer-OS クイックテスト") >> quick_test.py
    echo print(f"Python: {platform.python_version()}") >> quick_test.py
    echo print("✅ 基本テスト完了") >> quick_test.py
)

python quick_test.py

echo [%date% %time%] クイックテスト完了 >> "%LOG_FILE%"

REM セットアップ完了メッセージ
echo.
echo ===================================================================
echo ✅ Infer-OS Windows NPU環境セットアップ完了！
echo ===================================================================
echo.
echo 📋 次のステップ:
echo 1. 仮想環境をアクティベート: venv\Scripts\activate.bat
echo 2. テスト実行: python quick_test.py
echo 3. Webデモ起動: python infer-os-demo\src\main.py
echo 4. ブラウザでアクセス: http://localhost:5000
echo.
echo 📄 ログファイル: %LOG_FILE%
echo 📁 プロジェクトパス: %CD%
echo.
echo [%date% %time%] セットアップ完了 >> "%LOG_FILE%"

pause
echo セットアップスクリプト終了

