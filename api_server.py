import os
import asyncio
import json
import logging
import base64
import io
import uuid
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from threading import Thread
from queue import Queue
import pickle

import numpy as np
import torch
import torchaudio
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from transformers import AutoTokenizer

from model import VoilaAutonomousModel
from spkr import SpeakerEmbedding
from voila_tokenizer import VoilaTokenizer
from tokenize_func import (
    voila_input_format,
    AUDIO_TOKEN_FORMAT,
    DEFAULT_AUDIO_TOKEN,
    DEFAULT_ASSISTANT_TOKEN,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# 全局变量
model = None
tokenizer = None
tokenizer_voila = None
spkr_model = None
active_sessions = {}  # 存储活跃会话

# 模型配置
MODEL_NAME = "maitrix-org/Voila-autonomous-preview"
AUDIO_TOKENIZER_PATH = "maitrix-org/Voila-Tokenizer"
REF_VOICE_PATH = "examples/character_ref_emb_demo.pkl"
DEFAULT_REF_NAME = "Homer Simpson"

# 创建 FastAPI 应用
app = FastAPI(
    title="Voila 全双工语音 API",
    description="基于 Voila-Autonomous 模型的全双工语音 API 服务",
    version="1.0.0",
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境中应该限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义请求和响应模型
class SessionRequest(BaseModel):
    instruction: Optional[str] = "你是一个由 Maitrix.org 创建的智能 AI 助手。"
    voice_name: Optional[str] = DEFAULT_REF_NAME

class AudioRequest(BaseModel):
    session_id: str
    audio_base64: Optional[str] = None

class TextRequest(BaseModel):
    session_id: str
    text: str

class SessionResponse(BaseModel):
    session_id: str
    message: str

class AudioResponse(BaseModel):
    text: Optional[str] = None
    audio_base64: Optional[str] = None

# 会话管理类
class Session:
    def __init__(self, session_id, instruction="你是一个由 Maitrix.org 创建的智能 AI 助手。", voice_name=DEFAULT_REF_NAME):
        self.session_id = session_id
        self.instruction = instruction
        self.voice_name = voice_name
        self.ref_embs = None
        self.ref_embs_mask = None
        self.history = {
            "instruction": instruction,
            "conversations": [],
        }
        self.audio_buffer = []
        self.last_activity = time.time()
        
        # 加载参考声音
        self._load_voice()
    
    def _load_voice(self):
        """加载参考声音嵌入"""
        try:
            ref_emb_mask_list = pickle.load(open(REF_VOICE_PATH, "rb"))
            if self.voice_name in ref_emb_mask_list:
                self.ref_embs = torch.tensor(ref_emb_mask_list[self.voice_name], dtype=torch.float32, device="cuda")
                self.ref_embs_mask = torch.tensor([1], device="cuda")
                logger.info(f"已加载声音: {self.voice_name}")
            else:
                logger.warning(f"未找到声音: {self.voice_name}，使用默认声音")
                self.ref_embs = torch.tensor(ref_emb_mask_list[DEFAULT_REF_NAME], dtype=torch.float32, device="cuda")
                self.ref_embs_mask = torch.tensor([1], device="cuda")
        except Exception as e:
            logger.error(f"加载声音时出错: {e}")
            raise
    
    def set_custom_voice(self, audio_file):
        """设置自定义声音"""
        try:
            wav, sr = torchaudio.load(audio_file)
            self.ref_embs = spkr_model(wav, sr)
            self.ref_embs_mask = torch.tensor([1], device="cuda")
            self.voice_name = "自定义声音"
            logger.info(f"已设置自定义声音")
            return True
        except Exception as e:
            logger.error(f"设置自定义声音时出错: {e}")
            return False
    
    def add_user_audio(self, audio_data):
        """添加用户音频到历史记录"""
        # 保存临时音频文件
        temp_file = f"temp_{self.session_id}_{int(time.time())}.wav"
        sf.write(temp_file, audio_data, 16000)
        
        # 更新对话历史
        self.history["conversations"].append({"from": "user", "audio": {"file": temp_file}})
        self.last_activity = time.time()
        
        return temp_file
    
    def add_user_text(self, text):
        """添加用户文本到历史记录"""
        self.history["conversations"].append({"from": "user", "text": text})
        self.last_activity = time.time()
    
    def prepare_for_response(self):
        """准备生成响应"""
        self.history["conversations"].append({"from": "assistant"})
    
    def cleanup_temp_files(self):
        """清理临时文件"""
        for conv in self.history["conversations"]:
            if "audio" in conv and "file" in conv["audio"]:
                file_path = conv["audio"]["file"]
                if os.path.exists(file_path) and file_path.startswith("temp_"):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.error(f"删除临时文件时出错: {file_path}, {e}")

def load_models():
    """加载所有必要的模型"""
    global model, tokenizer, tokenizer_voila, spkr_model
    
    logger.info("正在加载模型...")
    
    # 禁用 PyTorch 默认初始化以加速模型创建
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    
    # 加载自主对话模型
    model = VoilaAutonomousModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
        use_cache=True,
    )
    model = model.cuda()
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer_voila = VoilaTokenizer(model_path=AUDIO_TOKENIZER_PATH, device="cuda")
    
    # 加载说话人嵌入模型
    spkr_model = SpeakerEmbedding(device="cuda")
    
    logger.info("模型加载完成")

async def process_audio(session_id, audio_data):
    """处理音频并生成响应"""
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    # 添加用户音频到历史记录
    temp_file = session.add_user_audio(audio_data)
    session.prepare_for_response()
    
    # 准备模型所需的 token ID
    num_codebooks = model.config.num_codebooks
    codebook_size = model.config.codebook_size
    
    AUDIO_MIN_TOKEN_ID = tokenizer.convert_tokens_to_ids(AUDIO_TOKEN_FORMAT.format(0))
    AUDIO_MAX_TOKEN_ID = tokenizer.convert_tokens_to_ids(AUDIO_TOKEN_FORMAT.format(codebook_size*num_codebooks-1))
    AUDIO_TOKEN_ID = tokenizer.convert_tokens_to_ids(DEFAULT_AUDIO_TOKEN)
    ASSISTANT_TOKEN_ID = tokenizer.convert_tokens_to_ids(DEFAULT_ASSISTANT_TOKEN)
    
    # 配置数据
    data_cfg = {
        "input_type": "autonomous",
        "task_type": "chat_aiao_auto",
        "num_codebooks": num_codebooks,
        "codebook_size": codebook_size,
    }
    
    try:
        # 准备输入
        input_ids, _, _, streaming_user_input_audio_tokens = voila_input_format(
            session.history, tokenizer, tokenizer_voila, data_cfg
        )
        
        # 创建输入生成器
        def get_input_generator(all_tokens):
            for i in range(len(all_tokens[0])):
                yield all_tokens[:,i]
        
        input_generator = get_input_generator(torch.as_tensor(streaming_user_input_audio_tokens).cuda())
        input_ids = [torch.as_tensor([input]).transpose(1,2).cuda() for input in input_ids]
        input_ids = torch.cat(input_ids, dim=2)
        
        # 生成参数
        gen_params = {
            "input_ids": input_ids,
            "input_generator": input_generator,
            "ref_embs": session.ref_embs,
            "ref_embs_mask": session.ref_embs_mask,
            "max_new_tokens": 128,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "llm_audio_token_id": AUDIO_TOKEN_ID,
            "min_audio_token_id": AUDIO_MIN_TOKEN_ID,
            "llm_assistant_token_id": ASSISTANT_TOKEN_ID,
            "temperature": 0.2,
            "top_k": 50,
            "audio_temperature": 0.8,
            "audio_top_k": 50,
        }
        
        # 生成响应
        with torch.inference_mode():
            outputs = model.run_generate(**gen_params)
            
            # 处理输出
            outputs = outputs.chunk(2, dim=2)[1]
            outputs = outputs[0].cpu().tolist()
            
            predict_outputs = outputs[input_ids.shape[1]:]
            text_outputs = []
            audio_outputs = []
            for _ in range(num_codebooks):
                audio_outputs.append([])
            
            for item in predict_outputs:
                if item[0] >= AUDIO_MIN_TOKEN_ID and item[0] <= AUDIO_MAX_TOKEN_ID:
                    for n, at in enumerate(item):
                        audio_outputs[n].append((at - AUDIO_MIN_TOKEN_ID)%codebook_size)
                elif item[0] != tokenizer.eos_token_id:
                    text_outputs.append(item[0])
            
            # 解码文本
            response_text = tokenizer.decode(text_outputs)
            
            # 解码音频
            audio_values = tokenizer_voila.decode(torch.tensor(audio_outputs).cuda())
            audio_numpy = audio_values.detach().cpu().numpy()
            
            # 将音频转换为 base64
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, audio_numpy, 16000, format='WAV')
            audio_bytes.seek(0)
            audio_base64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
            
            return {
                "text": response_text,
                "audio_base64": audio_base64
            }
            
    except Exception as e:
        logger.error(f"处理音频时出错: {e}")
        raise HTTPException(status_code=500, detail=f"处理音频时出错: {str(e)}")
    finally:
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)

# 定期清理过期会话
async def cleanup_sessions():
    """清理超过30分钟未活动的会话"""
    while True:
        current_time = time.time()
        sessions_to_remove = []
        
        for session_id, session in active_sessions.items():
            if current_time - session.last_activity > 1800:  # 30分钟
                sessions_to_remove.append(session_id)
                session.cleanup_temp_files()
        
        for session_id in sessions_to_remove:
            del active_sessions[session_id]
            logger.info(f"已清理过期会话: {session_id}")
        
        await asyncio.sleep(300)  # 每5分钟检查一次

@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    # 加载模型
    load_models()
    
    # 启动会话清理任务
    asyncio.create_task(cleanup_sessions())

@app.post("/api/session/create", response_model=SessionResponse)
async def create_session(request: SessionRequest):
    """创建新会话"""
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = Session(
        session_id=session_id,
        instruction=request.instruction,
        voice_name=request.voice_name
    )
    logger.info(f"已创建新会话: {session_id}")
    return {"session_id": session_id, "message": "会话已创建"}

@app.post("/api/session/close")
async def close_session(session_id: str = Form(...)):
    """关闭会话"""
    if session_id in active_sessions:
        session = active_sessions[session_id]
        session.cleanup_temp_files()
        del active_sessions[session_id]
        logger.info(f"已关闭会话: {session_id}")
        return {"message": "会话已关闭"}
    else:
        raise HTTPException(status_code=404, detail="会话不存在")

@app.post("/api/voice/set")
async def set_voice(session_id: str = Form(...), voice_name: str = Form(...)):
    """设置预定义声音"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    session = active_sessions[session_id]
    session.voice_name = voice_name
    session._load_voice()
    
    return {"message": f"已设置声音: {voice_name}"}

@app.post("/api/voice/custom")
async def set_custom_voice(session_id: str = Form(...), audio_file: UploadFile = File(...)):
    """设置自定义声音"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    session = active_sessions[session_id]
    
    # 保存上传的音频文件
    temp_file = f"temp_voice_{session_id}.wav"
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
        raise HTTPException(status_code=500, detail="设置自定义声音失败")

@app.post("/api/audio/process", response_model=AudioResponse)
async def process_audio_api(request: AudioRequest, background_tasks: BackgroundTasks):
    """处理音频并返回响应"""
    if request.session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    # 解码 base64 音频
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
        audio_io = io.BytesIO(audio_bytes)
        audio_data, sample_rate = sf.read(audio_io)
        
        # 重采样到 16kHz (如果需要)
        if sample_rate != 16000:
            audio_tensor = torch.tensor(audio_data).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio_tensor = resampler(audio_tensor)
            audio_data = audio_tensor.squeeze(0).numpy()
        
        # 处理音频
        response = await process_audio(request.session_id, audio_data)
        return response
    except Exception as e:
        logger.error(f"处理音频请求时出错: {e}")
        raise HTTPException(status_code=500, detail=f"处理音频请求时出错: {str(e)}")

@app.post("/api/text/process", response_model=AudioResponse)
async def process_text_api(request: TextRequest):
    """处理文本并返回语音响应"""
    if request.session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    session = active_sessions[request.session_id]
    
    # 添加用户文本到历史记录
    session.add_user_text(request.text)
    session.prepare_for_response()
    
    # 准备模型所需的 token ID
    num_codebooks = model.config.num_codebooks
    codebook_size = model.config.codebook_size
    
    AUDIO_MIN_TOKEN_ID = tokenizer.convert_tokens_to_ids(AUDIO_TOKEN_FORMAT.format(0))
    AUDIO_MAX_TOKEN_ID = tokenizer.convert_tokens_to_ids(AUDIO_TOKEN_FORMAT.format(codebook_size*num_codebooks-1))
    AUDIO_TOKEN_ID = tokenizer.convert_tokens_to_ids(DEFAULT_AUDIO_TOKEN)
    ASSISTANT_TOKEN_ID = tokenizer.convert_tokens_to_ids(DEFAULT_ASSISTANT_TOKEN)
    
    # 配置数据
    data_cfg = {
        "input_type": "autonomous",
        "task_type": "chat_tts",  # 使用 TTS 任务类型
        "num_codebooks": num_codebooks,
        "codebook_size": codebook_size,
    }
    
    try:
        # 准备输入
        input_ids, _, _, _ = voila_input_format(
            session.history, tokenizer, tokenizer_voila, data_cfg
        )
        
        input_ids = torch.as_tensor([input_ids]).transpose(1,2).cuda()
        
        # 生成参数
        gen_params = {
            "input_ids": input_ids,
            "ref_embs": session.ref_embs,
            "ref_embs_mask": session.ref_embs_mask,
            "max_new_tokens": 128,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "llm_audio_token_id": AUDIO_TOKEN_ID,
            "min_audio_token_id": AUDIO_MIN_TOKEN_ID,
            "temperature": 0.2,
            "top_k": 50,
            "audio_temperature": 0.8,
            "audio_top_k": 50,
        }
        
        # 生成响应
        with torch.inference_mode():
            outputs = model.run_generate(**gen_params)
            outputs = outputs[0].cpu().tolist()
            
            predict_outputs = outputs[input_ids.shape[1]:]
            text_outputs = []
            audio_outputs = []
            for _ in range(num_codebooks):
                audio_outputs.append([])
            
            for item in predict_outputs:
                if item[0] >= AUDIO_MIN_TOKEN_ID and item[0] <= AUDIO_MAX_TOKEN_ID:
                    for n, at in enumerate(item):
                        audio_outputs[n].append((at - AUDIO_MIN_TOKEN_ID)%codebook_size)
                elif item[0] != tokenizer.eos_token_id:
                    text_outputs.append(item[0])
            
            # 解码文本
            response_text = tokenizer.decode(text_outputs)
            
            # 解码音频
            audio_values = tokenizer_voila.decode(torch.tensor(audio_outputs).cuda())
            audio_numpy = audio_values.detach().cpu().numpy()
            
            # 将音频转换为 base64
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, audio_numpy, 16000, format='WAV')
            audio_bytes.seek(0)
            audio_base64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
            
            return {
                "text": response_text,
                "audio_base64": audio_base64
            }
            
    except Exception as e:
        logger.error(f"处理文本时出错: {e}")
        raise HTTPException(status_code=500, detail=f"处理文本时出错: {str(e)}")

# WebSocket 路由
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket 端点，用于实时全双工音频交互"""
    await websocket.accept()
    
    # 检查会话是否存在
    if session_id not in active_sessions:
        await websocket.send_json({"error": "会话不存在"})
        await websocket.close()
        return
    
    session = active_sessions[session_id]
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
                response = await process_audio(session_id, audio_data)
                
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
                    pass
                
            elif data["type"] == "end":
                # 结束会话
                break
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket 连接已断开: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket 处理时出错: {e}")
        await websocket.send_json({"error": str(e)})
    finally:
        # 不关闭会话，只关闭 WebSocket 连接
        pass

@app.get("/api/voices/list")
async def list_voices():
    """获取可用声音列表"""
    try:
        ref_emb_mask_list = pickle.load(open(REF_VOICE_PATH, "rb"))
        voices = list(ref_emb_mask_list.keys())
        return {"voices": voices}
    except Exception as e:
        logger.error(f"获取声音列表时出错: {e}")
        raise HTTPException(status_code=500, detail=f"获取声音列表时出错: {str(e)}")

@app.get("/")
async def root():
    """API 根路径"""
    return {
        "name": "Voila 全双工语音 API",
        "version": "1.0.0",
        "description": "基于 Voila-Autonomous 模型的全双工语音 API 服务"
    }

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
