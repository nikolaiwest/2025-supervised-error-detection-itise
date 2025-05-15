import atexit
import os
import subprocess
import sys
import time

# Store the mlflow process globally so we can terminate it properly
_mlflow_process = None


def launch_server(port=5000, wait=True):
    """
    Launch the MLflow server programmatically.

    Args:
        port (int): Port to run the server on
        wait (bool): If True, wait for server to start before returning

    Returns:
        subprocess.Popen: The process object for the server
    """
    global _mlflow_process

    # If server is already running, return the process
    if _mlflow_process is not None and _mlflow_process.poll() is None:
        print(f"MLflow server already running at http://127.0.0.1:{port}")
        return _mlflow_process

    # Get the directory where this module is located
    module_dir = os.path.dirname(os.path.abspath(__file__))

    # Determine which script to run based on the OS
    if sys.platform.startswith("win"):
        script_path = os.path.join(module_dir, "launch_server.bat")
        # On Windows, we need shell=True to execute batch files
        _mlflow_process = subprocess.Popen(
            f"{script_path} {port}", shell=True, cwd=module_dir
        )
    else:
        script_path = os.path.join(module_dir, "launch_server.sh")
        # Make sure the script is executable
        os.chmod(script_path, 0o755)  # rwxr-xr-x
        _mlflow_process = subprocess.Popen([script_path, str(port)], cwd=module_dir)

    # Register function to terminate server on exit
    atexit.register(stop_server)

    if wait:
        # Wait a moment for the server to start
        print(f"Starting MLflow server on port {port}...")
        time.sleep(3)
        print(f"MLflow UI available at http://127.0.0.1:{port}")

    return _mlflow_process


def stop_server():
    """Stop the MLflow server if it's running."""
    global _mlflow_process

    if _mlflow_process is not None:
        if _mlflow_process.poll() is None:  # If process is still running
            print("Stopping MLflow server...")
            _mlflow_process.terminate()
            _mlflow_process.wait(timeout=5)
            print("MLflow server stopped")
        _mlflow_process = None
