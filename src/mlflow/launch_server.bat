@echo off
REM MLflow server launcher for Windows
REM Get port from first argument or use default
set PORT=%1
if "%PORT%"=="" set PORT=5000

REM Activate the virtual environment from project root
REM Since this script is in src/mlflow/, we need to go up 1 levels
call ..\..\venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment.
    echo The script is looking for: ..\..\venv\Scripts\activate.bat
    echo Please ensure your venv is in the project root.
    exit /b 1
)

REM Start MLflow server
mlflow server ^
  --backend-store-uri sqlite:///.\database\mlflow.db ^
  --default-artifact-root /src/mlflow/artifacts ^
  --host 127.0.0.1 ^
  --port %PORT%

REM Deactivate the environment when done
call deactivate