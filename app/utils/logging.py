import logging
from app.config import settings

def setup_logger(name: str = None) -> logging.Logger:
    """
    设置并返回一个配置好的日志记录器
    
    Args:
        name: 日志记录器名称，默认为根记录器
        
    Returns:
        配置好的日志记录器
    """
    logger_name = name or "voila_api"
    logger = logging.getLogger(logger_name)
    
    # 设置日志级别
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # 如果已经有处理器，不再添加
    if logger.handlers:
        return logger
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # 设置格式
    formatter = logging.Formatter(settings.LOG_FORMAT)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(console_handler)
    
    return logger

# 创建默认日志记录器
logger = setup_logger()
