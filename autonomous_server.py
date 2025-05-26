import os
import asyncio
import json
import logging
import websockets
import torch
import torchaudio
import numpy as np
import pickle
from pathlib import Path
from threading import Thread
from queue import Queue

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
audio_queue = Queue()
response_queue = Queue()

# 模型配置
MODEL_NAME = "maitrix-org/Voila-autonomous-preview"
AUDIO_TOKENIZER_PATH = "maitrix-org/Voila-Tokenizer"
REF_VOICE_PATH = "examples/character_ref_emb_demo.pkl"
DEFAULT_REF_NAME = "Homer Simpson"

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

def get_ref_embs(ref_name=None, ref_audio=None):
    """获取参考声音嵌入"""
    if ref_audio:
        # 从音频文件获取声音嵌入
        wav, sr = torchaudio.load(ref_audio)
        ref_embs = spkr_model(wav, sr)
    else:
        # 从预设声音库获取声音嵌入
        ref_emb_mask_list = pickle.load(open(REF_VOICE_PATH, "rb"))
        ref_name = ref_name or DEFAULT_REF_NAME
        ref_embs = torch.tensor(ref_emb_mask_list[ref_name], dtype=torch.float32, device="cuda")
    
    ref_embs_mask = torch.tensor([1], device="cuda")
    return ref_embs, ref_embs_mask

class AudioProcessor(Thread):
    """音频处理线程，处理用户输入并生成响应"""
    def __init__(self, ref_name=None, ref_audio=None):
        super().__init__()
        self.daemon = True
        self.running = True
        self.ref_name = ref_name
        self.ref_audio = ref_audio
        self.audio_buffer = []
        self.sample_rate = 16000  # 采样率
        self.chunk_size = 4000    # 每个音频块的大小
        self.history = {
            "instruction": "你是一个由 Maitrix.org 创建的智能 AI 助手。",
            "conversations": [],
        }
        
    def run(self):
        """线程主循环"""
        logger.info("音频处理线程启动")
        
        # 获取参考声音嵌入
        ref_embs, ref_embs_mask = get_ref_embs(self.ref_name, self.ref_audio)
        
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
        
        while self.running:
            # 从队列获取音频数据
            try:
                audio_chunk = audio_queue.get(timeout=0.1)
                self.audio_buffer.extend(audio_chunk)
                
                # 当缓冲区达到一定大小时处理
                if len(self.audio_buffer) >= self.chunk_size:
                    # 处理音频块
                    audio_data = np.array(self.audio_buffer[:self.chunk_size], dtype=np.float32)
                    self.audio_buffer = self.audio_buffer[self.chunk_size:]
                    
                    # 保存音频用于处理
                    temp_file = "temp_input.wav"
                    torchaudio.save(temp_file, torch.tensor(audio_data).unsqueeze(0), self.sample_rate)
                    
                    # 更新对话历史
                    self.history["conversations"].append({"from": "user", "audio": {"file": temp_file}})
                    self.history["conversations"].append({"from": "assistant"})
                    
                    # 准备输入
                    input_ids, _, _, streaming_user_input_audio_tokens = voila_input_format(
                        self.history, tokenizer, tokenizer_voila, data_cfg
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
                        "ref_embs": ref_embs,
                        "ref_embs_mask": ref_embs_mask,
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
                        
                        # 解码音频
                        audio_values = tokenizer_voila.decode(torch.tensor(audio_outputs).cuda())
                        audio_numpy = audio_values.detach().cpu().numpy()
                        
                        # 将响应放入队列
                        response_queue.put({
                            "text": tokenizer.decode(text_outputs),
                            "audio": audio_numpy.tolist()
                        })
                    
                    # 清理临时文件
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                logger.error(f"处理音频时出错: {e}")
        
        logger.info("音频处理线程停止")

async def websocket_handler(websocket, path):
    """WebSocket 处理函数"""
    logger.info(f"新的客户端连接: {websocket.remote_address}")
    
    # 启动音频处理线程
    processor = AudioProcessor()
    processor.start()
    
    try:
        # 发送欢迎消息
        await websocket.send(json.dumps({
            "type": "welcome",
            "message": "已连接到全双工音频聊天服务"
        }))
        
        # 创建响应发送任务
        async def send_responses():
            while True:
                try:
                    if not response_queue.empty():
                        response = response_queue.get()
                        await websocket.send(json.dumps({
                            "type": "response",
                            "data": response
                        }))
                    await asyncio.sleep(0.05)
                except Exception as e:
                    logger.error(f"发送响应时出错: {e}")
                    break
        
        # 启动响应发送任务
        response_task = asyncio.create_task(send_responses())
        
        # 接收客户端消息
        async for message in websocket:
            try:
                data = json.loads(message)
                if data["type"] == "audio":
                    # 接收音频数据
                    audio_data = data["data"]
                    audio_queue.put(audio_data)
                elif data["type"] == "change_voice":
                    # 更改语音
                    processor.running = False
                    processor.join()
                    processor = AudioProcessor(ref_name=data.get("voice_name"), ref_audio=data.get("voice_file"))
                    processor.start()
                elif data["type"] == "end":
                    # 结束会话
                    break
            except json.JSONDecodeError:
                logger.error("无效的 JSON 数据")
            except Exception as e:
                logger.error(f"处理消息时出错: {e}")
        
        # 取消响应任务
        response_task.cancel()
        
    except websockets.exceptions.ConnectionClosed:
        logger.info("客户端断开连接")
    finally:
        # 停止处理线程
        processor.running = False
        processor.join()
        logger.info(f"客户端连接关闭: {websocket.remote_address}")

async def main():
    """主函数"""
    # 加载模型
    load_models()
    
    # 启动 WebSocket 服务器
    host = "0.0.0.0"
    port = 8765
    
    logger.info(f"启动 WebSocket 服务器: ws://{host}:{port}")
    async with websockets.serve(websocket_handler, host, port):
        await asyncio.Future()  # 运行直到被中断

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("服务器关闭")
