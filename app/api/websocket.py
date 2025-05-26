from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, status
import json
import numpy as np
from typing import Dict, Any

from app.core.session import session_manager
from app.services.model_service import process_audio
from app.utils.logging import logger
from app.utils.errors import NotFoundError

router = APIRouter(tags=["WebSocket"])

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket 端点，用于实时全双工音频交互
    
    Args:
        websocket: WebSocket 连接
        session_id: 会话ID
    """
    await websocket.accept()
    
    # 检查会话是否存在
    try:
        session = session_manager.get_session(session_id)
    except NotFoundError:
        await websocket.send_json({"type": "error", "error": "会话不存在"})
        await websocket.close()
        return
    
    logger.info(f"WebSocket 连接已建立: {session_id}")
    
    # 发送欢迎消息
    await websocket.send_json({
        "type": "welcome",
        "message": "已连接到全双工音频聊天服务"
    })
    
    try:
        while True:
            # 接收消息
            data = await websocket.receive_json()
            
            if data["type"] == "audio":
                # 接收音频数据
                audio_data = np.array(data["data"], dtype=np.float32)
                
                # 处理音频并生成响应
                response = await process_audio(session, audio_data)
                
                # 发送响应
                await websocket.send_json({
                    "type": "response",
                    "data": response
                })
                
            elif data["type"] == "change_voice":
                # 更改语音
                if "voice_name" in data:
                    session.voice_name = data["voice_name"]
                    session._load_voice()
                    await websocket.send_json({
                        "type": "info",
                        "message": f"已切换到 {data['voice_name']} 的声音"
                    })
                elif "voice_file" in data:
                    # 处理自定义声音
                    # 这里需要额外处理，因为 WebSocket 不能直接上传文件
                    # 可以通过 base64 编码传输
                    pass
                
            elif data["type"] == "end":
                # 结束会话
                break
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket 连接已断开: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket 处理时出错: {e}")
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except:
            pass
    finally:
        # 不关闭会话，只关闭 WebSocket 连接
        logger.info(f"WebSocket 连接已关闭: {session_id}")
