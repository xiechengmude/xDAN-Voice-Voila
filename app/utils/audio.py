import io
import base64
import numpy as np
import torch
import torchaudio
import soundfile as sf
from typing import Tuple, Optional, Union

from app.config import settings
from app.utils.logging import logger

def load_audio_file(file_path: str) -> Tuple[np.ndarray, int]:
    """
    加载音频文件
    
    Args:
        file_path: 音频文件路径
        
    Returns:
        音频数据和采样率
    """
    try:
        wav, sr = torchaudio.load(file_path)
        wav = wav.squeeze(0).numpy()
        
        # 如果采样率不是目标采样率，进行重采样
        if sr != settings.SAMPLE_RATE:
            wav = resample_audio(wav, sr, settings.SAMPLE_RATE)
            sr = settings.SAMPLE_RATE
            
        return wav, sr
    except Exception as e:
        logger.error(f"加载音频文件出错: {e}")
        raise

def decode_audio_base64(audio_base64: str) -> Tuple[np.ndarray, int]:
    """
    解码 base64 编码的音频数据
    
    Args:
        audio_base64: base64 编码的音频数据
        
    Returns:
        音频数据和采样率
    """
    try:
        audio_bytes = base64.b64decode(audio_base64)
        audio_io = io.BytesIO(audio_bytes)
        audio_data, sample_rate = sf.read(audio_io)
        
        # 如果采样率不是目标采样率，进行重采样
        if sample_rate != settings.SAMPLE_RATE:
            audio_data = resample_audio(audio_data, sample_rate, settings.SAMPLE_RATE)
            sample_rate = settings.SAMPLE_RATE
            
        return audio_data, sample_rate
    except Exception as e:
        logger.error(f"解码音频数据出错: {e}")
        raise

def encode_audio_base64(audio_data: np.ndarray, sample_rate: int = settings.SAMPLE_RATE) -> str:
    """
    将音频数据编码为 base64
    
    Args:
        audio_data: 音频数据
        sample_rate: 采样率
        
    Returns:
        base64 编码的音频数据
    """
    try:
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, audio_data, sample_rate, format='WAV')
        audio_bytes.seek(0)
        audio_base64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
        return audio_base64
    except Exception as e:
        logger.error(f"编码音频数据出错: {e}")
        raise

def resample_audio(audio_data: Union[np.ndarray, torch.Tensor], orig_sr: int, target_sr: int) -> np.ndarray:
    """
    重采样音频数据
    
    Args:
        audio_data: 音频数据
        orig_sr: 原始采样率
        target_sr: 目标采样率
        
    Returns:
        重采样后的音频数据
    """
    if orig_sr == target_sr:
        return audio_data
    
    try:
        if isinstance(audio_data, np.ndarray):
            audio_tensor = torch.tensor(audio_data)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
        else:
            audio_tensor = audio_data
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
        
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        resampled_tensor = resampler(audio_tensor)
        
        return resampled_tensor.squeeze(0).numpy()
    except Exception as e:
        logger.error(f"重采样音频数据出错: {e}")
        raise

def save_audio_file(audio_data: np.ndarray, file_path: str, sample_rate: int = settings.SAMPLE_RATE) -> None:
    """
    保存音频数据到文件
    
    Args:
        audio_data: 音频数据
        file_path: 文件路径
        sample_rate: 采样率
    """
    try:
        sf.write(file_path, audio_data, sample_rate)
    except Exception as e:
        logger.error(f"保存音频文件出错: {e}")
        raise
