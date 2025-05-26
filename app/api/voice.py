from fastapi import APIRouter, Depends, HTTPException, status, Form, UploadFile, File
from fastapi.responses import JSONResponse
import os
from typing import List

from app.models.response import VoiceResponse, VoiceListResponse
from app.core.session import session_manager
from app.services.model_service import get_available_voices
from app.utils.logging import logger
from app.utils.errors import NotFoundError
from app.config import settings

router = APIRouter(prefix="/voice", tags=["声音管理"])

@router.post("/set", response_model=VoiceResponse)
async def set_voice(session_id: str = Form(...), voice_name: str = Form(...)):
    """
    设置预定义声音
    
    Args:
        session_id: 会话ID
        voice_name: 声音名称
        
    Returns:
        声音设置响应
    """
    try:
        session = session_manager.get_session(session_id)
        session.voice_name = voice_name
        session._load_voice()
        
        return {"message": f"已设置声音: {voice_name}"}
    except NotFoundError as e:
        logger.error(f"设置声音时出错: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"设置声音时出错: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"设置声音时出错: {str(e)}"
        )

@router.post("/custom", response_model=VoiceResponse)
async def set_custom_voice(session_id: str = Form(...), audio_file: UploadFile = File(...)):
    """
    设置自定义声音
    
    Args:
        session_id: 会话ID
        audio_file: 音频文件
        
    Returns:
        声音设置响应
    """
    try:
        session = session_manager.get_session(session_id)
        
        # 保存上传的音频文件
        temp_file = os.path.join(settings.TEMP_DIR, f"temp_voice_{session_id}.wav")
        with open(temp_file, "wb") as f:
            f.write(await audio_file.read())
        
        # 设置自定义声音
        success = session.set_custom_voice(temp_file)
        
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        if success:
            return {"message": "已设置自定义声音"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="设置自定义声音失败"
            )
    except NotFoundError as e:
        logger.error(f"设置自定义声音时出错: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"设置自定义声音时出错: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"设置自定义声音时出错: {str(e)}"
        )

@router.get("/list", response_model=VoiceListResponse)
async def list_voices():
    """
    获取可用声音列表
    
    Returns:
        声音列表响应
    """
    try:
        voices = get_available_voices()
        return {"voices": voices}
    except Exception as e:
        logger.error(f"获取声音列表时出错: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取声音列表时出错: {str(e)}"
        )
