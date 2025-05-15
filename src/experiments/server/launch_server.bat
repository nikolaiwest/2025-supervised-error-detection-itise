@echo off
REM MLflow server launcher for Windows

REM Get port from first argument or use default
set PORT=%1
if "%PORT%"=="" set PORT=5000

echo Starting MLflow server...
echo.
echo MLflow UI will be available at http://127.0.0.1:%PORT%
echo (Press Ctrl+C followed by Y to stop the server)
echo.

REM Activate the virtual environment from project root
REM Since this script is in src/experiments/server/, we need to go up 3 levels
call ..\..\..\venv\Scripts\activate.bat

if %errorlevel% neq 0 (
    echo Failed to activate virtual environment. 
    echo The script is looking for: ..\..\..\venv\Scripts\activate.bat
    echo Please ensure your venv is in the project root.
    pause
    exit /b 1
)

REM Start MLflow server
mlflow server ^
  --backend-store-uri sqlite:///.\mlflow.db ^
  --default-artifact-root /src/experiments/server/artifacts ^
  --host 127.0.0.1 ^
  --port %PORT%

REM Deactivate the environment when done
call deactivate

pause