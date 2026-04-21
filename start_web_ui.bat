@echo off
cd /d %~dp0
python -m pip install -r requirements.txt
start "Process Assistant Web UI" python web_ui.py
timeout /t 1 /nobreak >nul
start "" http://127.0.0.1:5050
exit
