@echo off
chcp 65001 >nul
title 智能配送优化系统

cd /d "%~dp0"

echo.
echo ═══════════════════════════════════════
echo   智能配送优化系统 — 启动中...
echo ═══════════════════════════════════════
echo.
echo   浏览器将自动打开, 等待约 5 秒...
echo   关闭此窗口即停止服务
echo.

start "" http://localhost:8501
streamlit run app.py --server.port 8501 --server.headless true

pause
