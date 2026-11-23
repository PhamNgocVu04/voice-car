@echo off
cd /d %~dp0
echo Dang khoi dong VoiceCar AI, cho khoang 15 giay...
start "" cmd /c "timeout /t 15 >nul & start chrome http://localhost:8000"
py -3.14 server.py
pause