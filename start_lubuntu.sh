#!/bin/bash
[ -z "$BASH_VERSION" ] && exec bash "$0" "$@"

# ============================================================
# start_lubuntu.sh — Dentphoto Document Translator
# ============================================================

# 스크립트가 위치한 폴더로 이동 (데스크톱 바로가기 실행 지원)
cd "$(dirname "$0")" || exit 1

echo "=============================================="
echo "      Dentphoto AI Document Translator"
echo "=============================================="
echo ""

# ----------------------------------------------------------
# 1. Python 버전 확인 및 가상환경 생성 (첫 실행 시에만)
# ----------------------------------------------------------
PYTHON_CMD=""
for cmd in python3.14 python3.13 python3.12 python3.11; do
    if command -v $cmd >/dev/null 2>&1; then
        PYTHON_CMD=$cmd
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    if python3 -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null; then
        PYTHON_CMD=python3
    fi
fi

if [ -z "$PYTHON_CMD" ]; then
    CURRENT_VER=$(python3 --version 2>/dev/null || echo "not installed")
    echo ""
    echo "[Error] Python 3.11 or higher is required. (current: $CURRENT_VER)"
    echo ""
    echo "Install: sudo apt update && sudo apt install -y python3.11 python3.11-venv"
    echo ""
    exit 1
fi

echo "Python version: $($PYTHON_CMD --version) ✅"
echo ""

if [ ! -d "dentphoto_env" ]; then
    echo "First run: setting up Dentphoto environment."
    echo "Installing required packages... (this may take a few minutes)"
    echo ""

    $PYTHON_CMD -m venv dentphoto_env
    . dentphoto_env/bin/activate

    dentphoto_env/bin/pip install --upgrade pip --no-cache-dir

    # requirements.txt 기반 패키지 설치 단일화
    if [ -f "requirements.txt" ]; then
        dentphoto_env/bin/pip install -r requirements.txt --no-cache-dir
    else
        # requirements.txt 파일이 없을 경우 필수 라이브러리만 최소한으로 설치
        echo "[Warning] requirements.txt not found. Installing minimal default packages..."
        dentphoto_env/bin/pip install gradio requests pymupdf beautifulsoup4 --no-cache-dir
    fi

    if [ $? -ne 0 ]; then
        echo ""
        echo "[Error] Package installation failed."
        echo "Delete the dentphoto_env folder and run start_ubuntu.sh again."
        deactivate
        exit 1
    fi

    echo ""
    echo "Dentphoto environment created successfully!"
    echo ""
else
    . dentphoto_env/bin/activate
fi

if [ "$DENTPHOTO_SETUP_ONLY" = "1" ]; then
    echo "Setup complete (DENTPHOTO_SETUP_ONLY=1): Dentphoto was not started."
    deactivate
    exit 0
fi

# ----------------------------------------------------------
# 2. Dentphoto 번역기 실행
# ----------------------------------------------------------
echo "Starting Dentphoto AI Translator..."
echo "Please wait..."
echo ""

dentphoto_env/bin/python3 dentphoto.py

deactivate
