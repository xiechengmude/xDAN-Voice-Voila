import asyncio
import os
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import time
import uvicorn

from app.config import settings
from app.api import session, voice, audio, text, websocket
from app.services.model_service import load_models
from app.core.session import session_manager
from app.utils.logging import logger
from app.utils.errors import APIError

# 创建 FastAPI 应用
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# 注册路由
app.include_router(session.router, prefix="/api")
app.include_router(voice.router, prefix="/api")
app.include_router(audio.router, prefix="/api")
app.include_router(text.router, prefix="/api")
app.include_router(websocket.router)

# 挂载静态文件
# 获取项目根目录
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
static_dir = os.path.join(root_dir, 'static')

# 添加静态文件支持
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 异常处理
@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    """API 错误处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

# 请求中间件
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """添加处理时间头"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# 定期清理过期会话
async def cleanup_sessions():
    """清理过期会话"""
    while True:
        session_manager.cleanup_expired_sessions()
        await asyncio.sleep(settings.CLEANUP_INTERVAL)

@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    # 加载模型
    load_models()
    
    # 启动会话清理任务
    asyncio.create_task(cleanup_sessions())
    
    logger.info(f"{settings.APP_NAME} 已启动")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行"""
    logger.info(f"{settings.APP_NAME} 已关闭")

@app.get("/")
async def root():
    """API 根路径"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": settings.APP_DESCRIPTION
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}

# 启动服务器
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
