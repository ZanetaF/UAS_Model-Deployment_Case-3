import subprocess
import time
import sys
import os
from threading import Thread


BASE_DIR = "C:/COOLYEAH/SEM 4/Model Deployment/UAS"
FASTAPI_FILE = os.path.join(BASE_DIR, "main.py")
STREAMLIT_FILE = os.path.join(BASE_DIR, "app.py")
MODEL_FILE = os.path.join(BASE_DIR, "best_obesity_model.pkl")

def run_fastapi():
    print("🚀 Starting FastAPI backend...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "main:app",
            "--host", "127.0.0.1",
            "--port", "8000",
            "--reload"
        ], check=True, cwd=BASE_DIR)
    except Exception as e:
        print(f"❌ Error running FastAPI: {e}")

def run_streamlit():
    print("🚀 Starting Streamlit frontend...")
    time.sleep(3)  
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit",
            "run", STREAMLIT_FILE,
            "--server.port", "8501"
        ], check=True)
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")

def main():
    print("🏁 Starting Obesity Classification Application")
    print("=" * 50)

    required_files = [FASTAPI_FILE, STREAMLIT_FILE, MODEL_FILE]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return

    fastapi_thread = Thread(target=run_fastapi, daemon=True)
    streamlit_thread = Thread(target=run_streamlit, daemon=True)

    fastapi_thread.start()
    print("⏳ Waiting for FastAPI to initialize...")
    time.sleep(2)

    streamlit_thread.start()

    print("\n🎉 Applications started successfully!")
    print("📊 FastAPI: http://localhost:8000/docs")
    print("🖥️  Streamlit: http://localhost:8501")
    print("=" * 50)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping applications...")
        print("👋 Applications stopped")

if __name__ == "__main__":
    main()
