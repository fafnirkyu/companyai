import subprocess
import time
import requests
import atexit
import os
import sys

# ---------------------------
# Backend configuration
# ---------------------------
FASTAPI_PORT = 8000
SERVER_IP = "192.168.0.230"
BACKEND_URL = f"http://{SERVER_IP}:{FASTAPI_PORT}"
BACKEND_CMD = [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", str(FASTAPI_PORT)]

backend_process = None


def start_backend():
    """Start FastAPI backend as subprocess."""
    global backend_process
    backend_process = subprocess.Popen(BACKEND_CMD)
    print(f"🚀 Backend started (PID {backend_process.pid})")


def stop_backend():
    """Stop backend when main app exits."""
    global backend_process
    if backend_process and backend_process.poll() is None:
        backend_process.terminate()
        backend_process.wait(timeout=10)
        print("🛑 Backend terminated.")


def wait_for_backend(url, timeout=180):
    """Wait for backend health check to respond."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                print("✅ Backend is ready.")
                return
        except requests.RequestException:
            pass
        print("⏳ Waiting for backend to start...")
        time.sleep(5)
    raise TimeoutError("Backend did not start in time.")


def run():
    """Start backend and then Streamlit frontend."""
    start_backend()
    try:
        wait_for_backend(f"{BACKEND_URL}/")  # hit FastAPI root
        subprocess.run([sys.executable, "-m", "streamlit", "run", "frontend/streamlit_app.py"], check=True)
    finally:
        stop_backend()


if __name__ == "__main__":
    atexit.register(stop_backend)
    run()
