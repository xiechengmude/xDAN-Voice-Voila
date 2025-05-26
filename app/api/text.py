from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from app.models.request import TextRequest
from app.models.response import AudioResponse
from app.core.session import session_manager
from app.services.model_service import process_text
from app.utils.logging import logger
from app.utils.errors import NotFoundError

router = APIRouter(prefix="/text", tags=["文本处理"])

@router.post("/process", response_model=AudioResponse)
async def process_text_api(request: TextRequest):
    """
    处理文本并返回语音响应
    
    Args:
        request: 文本处理请求
        
    Returns:
        音频处理响应
    """
    try:
        # 获取会话
        session = session_manager.get_session(request.session_id)
        
        # 处理文本
        response = await process_text(session, request.text)
        return response
    except NotFoundError as e:
        logger.error(f"处理文本时出错: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"处理文本时出错: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理文本时出错: {str(e)}"
        )
