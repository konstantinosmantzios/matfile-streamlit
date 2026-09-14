import threading
import time
import socket
import uvicorn
import webview
from main import app, cleanup_all_temp_files

def find_free_port(start_port=8000):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('127.0.0.1', start_port))
        return start_port
    except OSError:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]
    finally:
        sock.close()

def start_server(port):
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

if __name__ == '__main__':
    port = find_free_port(8000)

    # 1. Start the server in a separate daemon thread
    server_thread = threading.Thread(target=start_server, args=(port,), daemon=True)
    server_thread.start()
    
    # 2. Give the server a moment to start
    time.sleep(1.2)
    
    # 3. Create and start the PyWebView window
    window = webview.create_window(
        title="FAMELab Mat File Viewer",
        url=f"http://127.0.0.1:{port}",
        width=1280,
        height=850,
        min_size=(900, 600),
        resizable=True
    )
    
    try:
        # Start the webview application loop
        webview.start()
    finally:
        print("[DESKTOP] Window closed by user. Cleaning up session files...")
        cleanup_all_temp_files()
