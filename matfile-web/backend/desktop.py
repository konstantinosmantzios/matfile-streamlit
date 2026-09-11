import threading
import time
import uvicorn
import webview
from main import app

def start_server():
    # Start the FastAPI server on localhost:8000
    # log_level="error" reduces the terminal noise from uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

if __name__ == '__main__':
    # 1. Start the server in a separate daemon thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # 2. Give the server a moment to start
    time.sleep(1.5)
    
    # 3. Create and start the PyWebView window
    window = webview.create_window(
        title="Matfile Web App",
        url="http://127.0.0.1:8000",
        width=1200,
        height=800,
        resizable=True
    )
    
    # Start the webview application loop
    webview.start()
