@echo off
rem Start meeting-copilot in its conda env and open the UI in the browser.
rem Double-click it, or pass extra flags:  run.bat --dry-run   /   run.bat --simulate demo_meeting.wav
setlocal
set "ENV_NAME=VideoVoiceText"
cd /d "%~dp0"

rem Make sure conda is on PATH (double-clicked .bat files may not have it)
where conda >nul 2>nul
if errorlevel 1 (
    if exist "%LOCALAPPDATA%\anaconda3\condabin\conda.bat" (
        set "PATH=%LOCALAPPDATA%\anaconda3\condabin;%PATH%"
    ) else (
        echo conda not found. Open an Anaconda Prompt and run:  conda activate %ENV_NAME% ^&^& python copilot.py --open
        pause
        exit /b 1
    )
)

call conda activate %ENV_NAME%
if errorlevel 1 (
    echo Could not activate the conda env "%ENV_NAME%".
    pause
    exit /b 1
)

echo Starting meeting-copilot in "%ENV_NAME%"  (stop with Ctrl+C)
echo.
python copilot.py --open %*

echo.
echo meeting-copilot has stopped.
pause
endlocal
