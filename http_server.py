import http.server
import socketserver
import logging
import os
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="static", **kwargs)
    
    def log_message(self, format, *args):
        logger.info(format % args)

def run_server(port=8080):
    """运行HTTP服务器"""
    handler = MyHttpRequestHandler
    
    with socketserver.TCPServer(("", port), handler) as httpd:
        logger.info(f"HTTP服务器运行在 http://localhost:{port}")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
