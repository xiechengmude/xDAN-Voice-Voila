import os
import sys
import subprocess
import time
import signal
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# 进程列表
processes = []

def signal_handler(sig, frame):
    """处理终止信号"""
    logger.info("接收到终止信号，正在关闭服务...")
    for process in processes:
        if process.poll() is None:  # 如果进程仍在运行
            process.terminate()
            logger.info(f"已终止进程 {process.pid}")
    sys.exit(0)

def start_services():
    """启动所有服务"""
    # 确保使用项目根目录的虚拟环境
    venv_path = Path(".venv/bin/python")
    if not venv_path.exists():
        logger.warning("未找到虚拟环境，使用系统Python")
        python_cmd = "python3.11.11"
    else:
        python_cmd = str(venv_path)
    
    # 启动WebSocket服务器（全双工音频聊天服务）
    logger.info("启动WebSocket服务器...")
    ws_process = subprocess.Popen(
        [python_cmd, "autonomous_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    processes.append(ws_process)
    logger.info(f"WebSocket服务器进程ID: {ws_process.pid}")
    
    # 等待WebSocket服务器启动
    time.sleep(2)
    
    # 启动HTTP服务器（提供Web界面）
    logger.info("启动HTTP服务器...")
    http_process = subprocess.Popen(
        [python_cmd, "http_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    processes.append(http_process)
    logger.info(f"HTTP服务器进程ID: {http_process.pid}")
    
    # 打印访问信息
    logger.info("全双工音频聊天服务已启动!")
    logger.info("请访问: http://localhost:8080")
    logger.info("按Ctrl+C停止服务")
    
    # 监控进程输出
    try:
        while True:
            # 检查WebSocket服务器输出
            ws_output = ws_process.stdout.readline()
            if ws_output:
                logger.info(f"[WebSocket] {ws_output.strip()}")
            
            ws_error = ws_process.stderr.readline()
            if ws_error:
                logger.error(f"[WebSocket] {ws_error.strip()}")
            
            # 检查HTTP服务器输出
            http_output = http_process.stdout.readline()
            if http_output:
                logger.info(f"[HTTP] {http_output.strip()}")
            
            http_error = http_process.stderr.readline()
            if http_error:
                logger.error(f"[HTTP] {http_error.strip()}")
            
            # 检查进程是否仍在运行
            if ws_process.poll() is not None:
                logger.error("WebSocket服务器已停止")
                break
            
            if http_process.poll() is not None:
                logger.error("HTTP服务器已停止")
                break
            
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("接收到终止信号，正在关闭服务...")
        for process in processes:
            if process.poll() is None:
                process.terminate()
                logger.info(f"已终止进程 {process.pid}")

if __name__ == "__main__":
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 启动服务
    start_services()
