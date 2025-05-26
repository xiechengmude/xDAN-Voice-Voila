import os
import sys
import subprocess
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def run_api_server():
    """运行 API 服务器"""
    # 检查环境
    # 首先检查是否在 conda 环境中
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and "voice" in conda_prefix:
        logger.info(f"检测到 conda 环境: {conda_prefix}")
        python_cmd = "python"  # conda 激活后直接使用 python 命令
    else:
        # 检查项目根目录的虚拟环境
        venv_path = Path(".venv/bin/python")
        if venv_path.exists():
            logger.info(f"使用项目虚拟环境: {venv_path}")
            python_cmd = str(venv_path)
        else:
            logger.warning("未找到虚拟环境，使用系统 Python")
            python_cmd = "python3.11"
    
    # 加载环境变量
    env_path = Path(".env")
    if env_path.exists():
        logger.info("加载 .env 文件")
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ[key] = value
    
    # 运行 API 服务器
    logger.info("启动 API 服务器...")
    cmd = [python_cmd, "-m", "app.main"]
    
    try:
        # 使用 subprocess 运行命令
        process = subprocess.run(
            cmd,
            check=True,
            text=True,
        )
        return process.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"运行 API 服务器时出错: {e}")
        return e.returncode
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭...")
        return 0

if __name__ == "__main__":
    sys.exit(run_api_server())
