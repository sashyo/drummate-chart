@echo off
setlocal EnableDelayedExpansion
rem DrumMate Chart - one-click start for Windows. First run installs everything into this folder.
cd /d "%~dp0"
title DrumMate Chart
echo.
echo   DrumMate Chart
echo.

rem --- Python 3.10+ (installs via winget if missing) ---
set "PY="
for %%v in (3.12 3.11 3.10 3.13) do (
  if not defined PY (
    py -%%v -c "import sys" >nul 2>&1 && set "PY=py -%%v"
  )
)
if not defined PY (
  python -c "import sys; assert sys.version_info>=(3,10)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo   Python 3.12 is needed. Installing it with winget ^(a few minutes^)...
  winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
  set "PY=py -3.12"
)

rem --- virtual environment + packages (first run only) ---
if not exist ".venv\Scripts\python.exe" (
  echo   Creating the Python environment...
  %PY% -m venv .venv || goto :fail
  echo   Installing packages ^(first run only, 5-10 minutes, ~1.5 GB^)...
  .venv\Scripts\python -m pip install --quiet --upgrade pip
  .venv\Scripts\python -m pip install --quiet -r requirements.txt || goto :fail
  .venv\Scripts\python -m pip install --quiet torch torchaudio --index-url https://download.pytorch.org/whl/cpu || goto :fail
  .venv\Scripts\python -m pip install --quiet demucs || goto :fail
)

rem --- ffmpeg (downloaded into tools\ffmpeg if not on PATH) ---
where ffmpeg >nul 2>&1
if errorlevel 1 (
  if not exist "tools\ffmpeg\bin\ffmpeg.exe" (
    echo   Downloading ffmpeg ^(~90 MB^)...
    powershell -NoProfile -Command "Invoke-WebRequest -Uri https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip -OutFile ffmpeg.zip; Expand-Archive -Force ffmpeg.zip tools\ffmpeg_tmp; $d = Get-ChildItem tools\ffmpeg_tmp -Directory | Select-Object -First 1; Move-Item -Force $d.FullName tools\ffmpeg; Remove-Item -Recurse -Force tools\ffmpeg_tmp; Remove-Item ffmpeg.zip" || goto :fail
  )
  set "PATH=%cd%\tools\ffmpeg\bin;%PATH%"
)

rem --- Node.js (yt-dlp needs a JavaScript runtime for YouTube links) ---
where node >nul 2>&1
if errorlevel 1 (
  echo   Installing Node.js LTS with winget ^(needed for YouTube links^)...
  winget install -e --id OpenJS.NodeJS.LTS --accept-source-agreements --accept-package-agreements
  set "PATH=%ProgramFiles%\nodejs;%PATH%"
)

rem --- run (YouTube links allowed: this is your own machine, personal use) ---
set "DRUMS_LINKS=1"
set "DRUMS_ALLOW_YOUTUBE=1"
if "%DRUMS_PORT%"=="" set "DRUMS_PORT=8000"
echo.
echo   Starting on http://127.0.0.1:%DRUMS_PORT%  ^(close this window to stop^)
echo   The first chart also downloads the separation models ^(~500 MB^).
echo.
if not "%DRUMS_NO_BROWSER%"=="1" start "" "http://127.0.0.1:%DRUMS_PORT%"
.venv\Scripts\python -m uvicorn backend.server:app --host 127.0.0.1 --port %DRUMS_PORT%
goto :eof

:fail
echo.
echo   Something failed above. Delete the .venv folder and run this again, or open an issue with the text of this window.
pause
