@echo off
chcp 65001 >nul
title Naga Agent
cd /d %~dp0
call .venv\Scripts\activate.bat
python main.py
pause 