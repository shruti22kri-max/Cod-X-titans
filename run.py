"""
Launcher Script for Underwater Sonar Object Detection Prototype
"""

import sys
import webbrowser
import threading
import time
import uvicorn

def open_browser(url: str):
    time.sleep(1.2)
    print(f"Opening browser at {url} ...")
    webbrowser.open(url)

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"
    
    print("\n" + "=" * 65)
    print("  UNDERWATER SONAR OBJECT DETECTION PROTOTYPE")
    print(f"  Server starting at: {url}")
    print("  API Endpoint:       POST /predict")
    print("  Documentation:      /docs (Swagger UI)")
    print("=" * 65 + "\n")
    
    # Automatically open the browser
    threading.Thread(target=open_browser, args=(url,), daemon=True).start()
    
    # Run uvicorn server
    uvicorn.run("backend.app:app", host=host, port=port, reload=False)

