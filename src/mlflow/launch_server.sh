#!/bin/bash
# MLflow server launcher for Linux
# Get port from first argument or use default
PORT=$1
if [ -z "$PORT" ]; then
    PORT=5000
fi

echo "Starting MLflow server..."
echo
echo "MLflow UI will be available at http://127.0.0.1:$PORT"
echo "(Press Ctrl+C to stop the server)"
echo

# Activate the virtual environment from project root
# Since this script is in src/experiments/server/, we need to go up 3 levels
source ../../../venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment."
    echo "The script is looking for: ../../../venv/bin/activate"
    echo "Please ensure your venv is in the project root."
    read -p "Press any key to continue..."
    exit 1
fi

# Get the project root directory (3 levels up from the script)
pushd ../../.. > /dev/null
PROJECT_ROOT=$(pwd)
popd > /dev/null

# Start MLflow server with artifacts directory in the server directory
mlflow server \
  --backend-store-uri sqlite:///./mlflow.db \
  --default-artifact-root file://$PROJECT_ROOT/src/experiments/server/artifacts \
  --host 127.0.0.1 \
  --port $PORT

# Deactivate the environment when done
deactivate
read -p "Press any key to continue..."