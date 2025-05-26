from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.responses import JSONResponse

from app.models.request import SessionRequest
from app.models.response import SessionResponse
from app.core.session import session_manager
from app.utils.logging import logger
from app.utils.errors import NotFoundError

router = APIRouter(prefix="/session", tags=["会话管理"])

@router.post("/create", response_model=SessionResponse)
async def create_session(request: SessionRequest):
    """
    创建新会话
    
    Args:
        request: 会话创建请求
        
    Returns:
        会话响应
    """
    try:
        session = session_manager.create_session(
            instruction=request.instruction,
            voice_name=request.voice_name
        )
        return {"session_id": session.session_id, "message": "会话已创建"}
    except Exception as e:
        logger.error(f"创建会话时出错: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建会话时出错: {str(e)}"
        )

@router.post("/close")
async def close_session(session_id: str = Form(...)):
    """
    关闭会话
    
    Args:
        session_id: 会话ID
        
    Returns:
        成功消息
    """
    try:
        session_manager.close_session(session_id)
        return {"message": "会话已关闭"}
    except NotFoundError as e:
        logger.error(f"关闭会话时出错: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"关闭会话时出错: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"关闭会话时出错: {str(e)}"
        )
