from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class SessionResponse(BaseModel):
    """会话响应模型"""
    session_id: str = Field(..., description="会话ID")
    message: str = Field(..., description="响应消息")

class AudioResponse(BaseModel):
    """音频处理响应模型"""
    text: Optional[str] = Field(None, description="文本响应")
    audio_base64: Optional[str] = Field(None, description="Base64编码的音频数据")

class VoiceResponse(BaseModel):
    """声音设置响应模型"""
    message: str = Field(..., description="响应消息")

class VoiceListResponse(BaseModel):
    """声音列表响应模型"""
    voices: List[str] = Field(..., description="可用声音列表")

class ErrorResponse(BaseModel):
    """错误响应模型"""
    detail: str = Field(..., description="错误详情")

class WebSocketResponse(BaseModel):
    """WebSocket响应基类"""
    type: str = Field(..., description="响应类型")

class WelcomeResponse(WebSocketResponse):
    """欢迎响应模型"""
    type: str = Field("welcome", description="响应类型")
    message: str = Field(..., description="欢迎消息")

class AudioResponseWS(WebSocketResponse):
    """音频响应模型 (WebSocket)"""
    type: str = Field("response", description="响应类型")
    data: Dict[str, Any] = Field(..., description="响应数据")

class InfoResponse(WebSocketResponse):
    """信息响应模型"""
    type: str = Field("info", description="响应类型")
    message: str = Field(..., description="信息消息")

class ErrorResponseWS(WebSocketResponse):
    """错误响应模型 (WebSocket)"""
    type: str = Field("error", description="响应类型")
    error: str = Field(..., description="错误消息")
