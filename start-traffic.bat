@echo off
chcp 65001 >nul
title Traffic Terminal
call "D:\miniconda3\Scripts\activate.bat" D:\miniconda3
call conda activate yolo
cd /d D:\multi-road-crossing-vehicle-count
echo.
echo  ==========================================
echo   环境 [yolo] 已就绪
echo   目录 D:\multi-road-crossing-vehicle-count
echo.
echo   跑脚本示例：
echo     python src\01_detect_video.py
echo  ==========================================
echo.


