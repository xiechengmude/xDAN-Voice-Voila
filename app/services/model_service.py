import torch
import pickle
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

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

from app.config import settings
from app.utils.logging import logger
from app.utils.errors import ModelError

# 全局模型变量
model = None
tokenizer = None
tokenizer_voila = None
spkr_model = None

# 模型配置
model_config = {
    "audio_min_token_id": None,
    "audio_max_token_id": None,
    "audio_token_id": None,
    "assistant_token_id": None,
    "num_codebooks": None,
    "codebook_size": None,
}

def load_models():
    """加载所有必要的模型"""
    global model, tokenizer, tokenizer_voila, spkr_model, model_config
    
    logger.info("正在加载模型...")
    
    # 禁用 PyTorch 默认初始化以加速模型创建
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    
    try:
        # 加载自主对话模型
        model = VoilaAutonomousModel.from_pretrained(
            settings.MODEL_NAME,
            torch_dtype=torch.bfloat16,
            use_flash_attention_2=True,
            use_cache=True,
        )
        model = model.cuda()
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
        tokenizer_voila = VoilaTokenizer(model_path=settings.AUDIO_TOKENIZER_PATH, device="cuda")
        
        # 加载说话人嵌入模型
        spkr_model = SpeakerEmbedding(device="cuda")
        
        # 设置模型配置
        num_codebooks = model.config.num_codebooks
        codebook_size = model.config.codebook_size
        
        model_config["num_codebooks"] = num_codebooks
        model_config["codebook_size"] = codebook_size
        model_config["audio_min_token_id"] = tokenizer.convert_tokens_to_ids(AUDIO_TOKEN_FORMAT.format(0))
        model_config["audio_max_token_id"] = tokenizer.convert_tokens_to_ids(AUDIO_TOKEN_FORMAT.format(codebook_size*num_codebooks-1))
        model_config["audio_token_id"] = tokenizer.convert_tokens_to_ids(DEFAULT_AUDIO_TOKEN)
        model_config["assistant_token_id"] = tokenizer.convert_tokens_to_ids(DEFAULT_ASSISTANT_TOKEN)
        
        logger.info("模型加载完成")
    except Exception as e:
        logger.error(f"加载模型时出错: {e}")
        raise ModelError(f"加载模型时出错: {str(e)}")

def get_available_voices() -> List[str]:
    """
    获取可用声音列表
    
    Returns:
        声音列表
    """
    try:
        ref_emb_mask_list = pickle.load(open(settings.REF_VOICE_PATH, "rb"))
        voices = list(ref_emb_mask_list.keys())
        return voices
    except Exception as e:
        logger.error(f"获取声音列表时出错: {e}")
        raise ModelError(f"获取声音列表时出错: {str(e)}")

async def process_audio(session, audio_data: np.ndarray) -> Dict[str, Any]:
    """
    处理音频并生成响应
    
    Args:
        session: 会话对象
        audio_data: 音频数据
        
    Returns:
        响应数据
    """
    # 添加用户音频到历史记录
    temp_file = session.add_user_audio(audio_data)
    session.prepare_for_response()
    
    # 配置数据
    data_cfg = {
        "input_type": "autonomous",
        "task_type": "chat_aiao_auto",
        "num_codebooks": model_config["num_codebooks"],
        "codebook_size": model_config["codebook_size"],
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
            "max_new_tokens": settings.MAX_NEW_TOKENS,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "llm_audio_token_id": model_config["audio_token_id"],
            "min_audio_token_id": model_config["audio_min_token_id"],
            "llm_assistant_token_id": model_config["assistant_token_id"],
            "temperature": settings.TEMPERATURE,
            "top_k": settings.TOP_K,
            "audio_temperature": settings.AUDIO_TEMPERATURE,
            "audio_top_k": settings.AUDIO_TOP_K,
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
            for _ in range(model_config["num_codebooks"]):
                audio_outputs.append([])
            
            for item in predict_outputs:
                if item[0] >= model_config["audio_min_token_id"] and item[0] <= model_config["audio_max_token_id"]:
                    for n, at in enumerate(item):
                        audio_outputs[n].append((at - model_config["audio_min_token_id"]) % model_config["codebook_size"])
                elif item[0] != tokenizer.eos_token_id:
                    text_outputs.append(item[0])
            
            # 解码文本
            response_text = tokenizer.decode(text_outputs)
            
            # 解码音频
            audio_values = tokenizer_voila.decode(torch.tensor(audio_outputs).cuda())
            audio_numpy = audio_values.detach().cpu().numpy()
            
            from app.utils.audio import encode_audio_base64
            audio_base64 = encode_audio_base64(audio_numpy)
            
            return {
                "text": response_text,
                "audio_base64": audio_base64
            }
            
    except Exception as e:
        logger.error(f"处理音频时出错: {e}")
        raise ModelError(f"处理音频时出错: {str(e)}")
    finally:
        # 清理临时文件
        import os
        if os.path.exists(temp_file):
            os.remove(temp_file)

async def process_text(session, text: str) -> Dict[str, Any]:
    """
    处理文本并生成语音响应
    
    Args:
        session: 会话对象
        text: 文本内容
        
    Returns:
        响应数据
    """
    # 添加用户文本到历史记录
    session.add_user_text(text)
    session.prepare_for_response()
    
    # 配置数据
    data_cfg = {
        "input_type": "autonomous",
        "task_type": "chat_tts",  # 使用 TTS 任务类型
        "num_codebooks": model_config["num_codebooks"],
        "codebook_size": model_config["codebook_size"],
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
            "max_new_tokens": settings.MAX_NEW_TOKENS,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "llm_audio_token_id": model_config["audio_token_id"],
            "min_audio_token_id": model_config["audio_min_token_id"],
            "temperature": settings.TEMPERATURE,
            "top_k": settings.TOP_K,
            "audio_temperature": settings.AUDIO_TEMPERATURE,
            "audio_top_k": settings.AUDIO_TOP_K,
        }
        
        # 生成响应
        with torch.inference_mode():
            outputs = model.run_generate(**gen_params)
            outputs = outputs[0].cpu().tolist()
            
            predict_outputs = outputs[input_ids.shape[1]:]
            text_outputs = []
            audio_outputs = []
            for _ in range(model_config["num_codebooks"]):
                audio_outputs.append([])
            
            for item in predict_outputs:
                if item[0] >= model_config["audio_min_token_id"] and item[0] <= model_config["audio_max_token_id"]:
                    for n, at in enumerate(item):
                        audio_outputs[n].append((at - model_config["audio_min_token_id"]) % model_config["codebook_size"])
                elif item[0] != tokenizer.eos_token_id:
                    text_outputs.append(item[0])
            
            # 解码文本
            response_text = tokenizer.decode(text_outputs)
            
            # 解码音频
            audio_values = tokenizer_voila.decode(torch.tensor(audio_outputs).cuda())
            audio_numpy = audio_values.detach().cpu().numpy()
            
            from app.utils.audio import encode_audio_base64
            audio_base64 = encode_audio_base64(audio_numpy)
            
            return {
                "text": response_text,
                "audio_base64": audio_base64
            }
            
    except Exception as e:
        logger.error(f"处理文本时出错: {e}")
        raise ModelError(f"处理文本时出错: {str(e)}")
