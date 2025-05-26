import os
from pathlib import Path
from typing import Optional, Dict, Any, List

# 基础路径
BASE_DIR = Path(__file__).parent.parent

# 环境变量配置
class Settings:
    # 应用配置
    APP_NAME: str = "Voila 全双工语音 API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "基于 Voila-Autonomous 模型的全双工语音 API 服务"
    
    # 服务器配置
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    
    # CORS 配置
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # 模型配置
    MODEL_NAME: str = os.getenv("MODEL_NAME", "maitrix-org/Voila-autonomous-preview")
    AUDIO_TOKENIZER_PATH: str = os.getenv("AUDIO_TOKENIZER_PATH", "maitrix-org/Voila-Tokenizer")
    REF_VOICE_PATH: str = os.getenv("REF_VOICE_PATH", str(BASE_DIR / "examples/character_ref_emb_demo.pkl"))
    DEFAULT_REF_NAME: str = os.getenv("DEFAULT_REF_NAME", "Homer Simpson")
    
    # 会话配置
    SESSION_TIMEOUT: int = int(os.getenv("SESSION_TIMEOUT", "1800"))  # 30分钟
    CLEANUP_INTERVAL: int = int(os.getenv("CLEANUP_INTERVAL", "300"))  # 5分钟
    
    # 音频配置
    SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))
    AUDIO_CHUNK_SIZE: int = int(os.getenv("AUDIO_CHUNK_SIZE", "4000"))
    
    # 模型推理配置
    MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "128"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))
    TOP_K: int = int(os.getenv("TOP_K", "50"))
    AUDIO_TEMPERATURE: float = float(os.getenv("AUDIO_TEMPERATURE", "0.8"))
    AUDIO_TOP_K: int = int(os.getenv("AUDIO_TOP_K", "50"))
    
    # 临时文件配置
    TEMP_DIR: str = os.getenv("TEMP_DIR", str(BASE_DIR / "temp"))
    
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 创建设置实例
settings = Settings()

# 确保临时目录存在
os.makedirs(settings.TEMP_DIR, exist_ok=True)
