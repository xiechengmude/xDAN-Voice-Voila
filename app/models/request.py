from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class SessionRequest(BaseModel):
    """会话创建请求模型"""
    instruction: Optional[str] = Field(
        default="你是一个由 Maitrix.org 创建的智能 AI 助手。",
        description="会话指令"
    )
    voice_name: Optional[str] = Field(
        default=None,
        description="预设声音名称"
    )

class AudioRequest(BaseModel):
    """音频处理请求模型"""
    session_id: str = Field(..., description="会话ID")
    audio_base64: str = Field(..., description="Base64编码的音频数据")

class TextRequest(BaseModel):
    """文本处理请求模型"""
    session_id: str = Field(..., description="会话ID")
    text: str = Field(..., description="文本内容")

class VoiceRequest(BaseModel):
    """声音设置请求模型"""
    session_id: str = Field(..., description="会话ID")
    voice_name: str = Field(..., description="预设声音名称")

class CustomVoiceRequest(BaseModel):
    """自定义声音请求模型 - 用于表单提交，不直接使用"""
    session_id: str = Field(..., description="会话ID")
    # 注意：文件上传通过表单处理，不在这里定义

class WebSocketMessage(BaseModel):
    """WebSocket消息基类"""
    type: str = Field(..., description="消息类型")

class AudioMessage(WebSocketMessage):
    """音频消息模型"""
    type: str = Field("audio", description="消息类型")
    data: List[float] = Field(..., description="音频数据")

class VoiceChangeMessage(WebSocketMessage):
    """声音更改消息模型"""
    type: str = Field("change_voice", description="消息类型")
    voice_name: Optional[str] = Field(None, description="预设声音名称")
    voice_file: Optional[str] = Field(None, description="Base64编码的声音文件")

class EndMessage(WebSocketMessage):
    """结束消息模型"""
    type: str = Field("end", description="消息类型")
