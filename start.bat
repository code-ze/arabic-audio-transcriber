@echo off
rem Python env, model and transcripts all live on D: to save space on C:
cd /d "%~dp0"
set ASR_MODEL=D:\AI\models\voxtral-arabic
set DATA_DIR=D:\AI\arabic-transcriber-data
set TORCH_HOME=D:\AI\torch
set HF_HUB_OFFLINE=1
start "" http://127.0.0.1:8765
D:\AI\venvs\arabic-transcriber\Scripts\python.exe app.py
pause
