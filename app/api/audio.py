from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
import numpy as np

from app.models.request import AudioRequest
from app.models.response import AudioResponse
from app.core.session import session_manager
from app.services.model_service import process_audio
from app.utils.logging import logger
from app.utils.errors import NotFoundError
from app.utils.audio import decode_audio_base64

router = APIRouter(prefix="/audio", tags=["音频处理"])

@router.post("/process", response_model=AudioResponse)
async def process_audio_api(request: AudioRequest, background_tasks: BackgroundTasks):
    """
    处理音频并返回响应
    
    Args:
        request: 音频处理请求
        background_tasks: 后台任务
        
    Returns:
        音频处理响应
    """
    try:
        # 获取会话
        session = session_manager.get_session(request.session_id)
        
        # 解码 base64 音频
        audio_data, _ = decode_audio_base64(request.audio_base64)
        
        # 处理音频
        response = await process_audio(session, audio_data)
        return response
    except NotFoundError as e:
        logger.error(f"处理音频时出错: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"处理音频时出错: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理音频时出错: {str(e)}"
        )
