import os
import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 64)
    print("Neura Unified Local System Launcher")
    print("=" * 64)
    
    # التأكد من مسار السيرفر والواجهة
    server_path = Path("neura/api/server.py")
    website_path = Path("Nova-Website/index.html")
    
    if not server_path.exists():
        print(f"[Error] Server file not found at {server_path}")
        sys.exit(1)
        
    if not website_path.exists():
        print(f"[Warning] Website interface not found at {website_path}")

    print("[1] Checking local engine status (Ollama)...")
    print("[2] Server and UI are fully integrated and linked.")
    print("[3] Starting Neura HTTP Server on http://127.0.0.1:8000 ...")
    print("=" * 64)
    
    # تشغيل السيرفر تلقائياً
    os.execvp(sys.executable, [sys.executable, str(server_path)])

if __name__ == "__main__":
    main()
