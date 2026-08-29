@echo off
setlocal
cd /d "%~dp0"

set "PYEXE="
if exist "%USERPROFILE%\AppData\Local\Programs\Python\Python313\python.exe" (
    set "PYEXE=%USERPROFILE%\AppData\Local\Programs\Python\Python313\python.exe"
) else (
    where python >nul 2>nul && set "PYEXE=python"
)
if not defined PYEXE (
    where py >nul 2>nul && set "PYEXE=py"
)
if not defined PYEXE (
    echo ERROR: cannot find Python interpreter.
    pause
    exit /b 1
)

set "PORT=8765"
echo Starting Text Eraser at http://127.0.0.1:%PORT%/
echo.
"%PYEXE%" -m uvicorn textpatch.webapp:app --host 127.0.0.1 --port %PORT%
pause
