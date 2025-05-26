import os
import time
import uuid
import pickle
from typing import Dict, Optional, List, Any
import torch

from app.config import settings
from app.utils.logging import logger
from app.utils.errors import NotFoundError

class Session:
    """会话管理类"""
    
    def __init__(self, session_id: str, instruction: str = None, voice_name: str = None):
        """
        初始化会话
        
        Args:
            session_id: 会话ID
            instruction: 会话指令
            voice_name: 声音名称
        """
        self.session_id = session_id
        self.instruction = instruction or "你是一个由 Maitrix.org 创建的智能 AI 助手。"
        self.voice_name = voice_name or settings.DEFAULT_REF_NAME
        self.ref_embs = None
        self.ref_embs_mask = None
        self.history = {
            "instruction": self.instruction,
            "conversations": [],
        }
        self.audio_buffer = []
        self.last_activity = time.time()
        
        # 加载参考声音
        self._load_voice()
    
    def _load_voice(self):
        """加载参考声音嵌入"""
        try:
            ref_emb_mask_list = pickle.load(open(settings.REF_VOICE_PATH, "rb"))
            if self.voice_name in ref_emb_mask_list:
                self.ref_embs = torch.tensor(ref_emb_mask_list[self.voice_name], dtype=torch.float32, device="cuda")
                self.ref_embs_mask = torch.tensor([1], device="cuda")
                logger.info(f"已加载声音: {self.voice_name}")
            else:
                logger.warning(f"未找到声音: {self.voice_name}，使用默认声音")
                self.ref_embs = torch.tensor(ref_emb_mask_list[settings.DEFAULT_REF_NAME], dtype=torch.float32, device="cuda")
                self.ref_embs_mask = torch.tensor([1], device="cuda")
        except Exception as e:
            logger.error(f"加载声音时出错: {e}")
            raise
    
    def set_custom_voice(self, audio_file: str) -> bool:
        """
        设置自定义声音
        
        Args:
            audio_file: 音频文件路径
            
        Returns:
            是否成功
        """
        try:
            from app.services.model_service import spkr_model
            
            wav, sr = torch.load(audio_file)
            self.ref_embs = spkr_model(wav, sr)
            self.ref_embs_mask = torch.tensor([1], device="cuda")
            self.voice_name = "自定义声音"
            logger.info(f"已设置自定义声音")
            return True
        except Exception as e:
            logger.error(f"设置自定义声音时出错: {e}")
            return False
    
    def add_user_audio(self, audio_data, sample_rate: int = settings.SAMPLE_RATE) -> str:
        """
        添加用户音频到历史记录
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            临时文件路径
        """
        from app.utils.audio import save_audio_file
        
        # 保存临时音频文件
        temp_file = os.path.join(settings.TEMP_DIR, f"temp_{self.session_id}_{int(time.time())}.wav")
        save_audio_file(audio_data, temp_file, sample_rate)
        
        # 更新对话历史
        self.history["conversations"].append({"from": "user", "audio": {"file": temp_file}})
        self.last_activity = time.time()
        
        return temp_file
    
    def add_user_text(self, text: str) -> None:
        """
        添加用户文本到历史记录
        
        Args:
            text: 文本内容
        """
        self.history["conversations"].append({"from": "user", "text": text})
        self.last_activity = time.time()
    
    def prepare_for_response(self) -> None:
        """准备生成响应"""
        self.history["conversations"].append({"from": "assistant"})
    
    def cleanup_temp_files(self) -> None:
        """清理临时文件"""
        for conv in self.history["conversations"]:
            if "audio" in conv and "file" in conv["audio"]:
                file_path = conv["audio"]["file"]
                if os.path.exists(file_path) and os.path.basename(file_path).startswith("temp_"):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.error(f"删除临时文件时出错: {file_path}, {e}")
    
    def is_expired(self) -> bool:
        """
        检查会话是否过期
        
        Returns:
            是否过期
        """
        return time.time() - self.last_activity > settings.SESSION_TIMEOUT


class SessionManager:
    """会话管理器"""
    
    def __init__(self):
        """初始化会话管理器"""
        self.sessions: Dict[str, Session] = {}
    
    def create_session(self, instruction: str = None, voice_name: str = None) -> Session:
        """
        创建新会话
        
        Args:
            instruction: 会话指令
            voice_name: 声音名称
            
        Returns:
            创建的会话
        """
        session_id = str(uuid.uuid4())
        session = Session(
            session_id=session_id,
            instruction=instruction,
            voice_name=voice_name
        )
        self.sessions[session_id] = session
        logger.info(f"已创建新会话: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Session:
        """
        获取会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话对象
            
        Raises:
            NotFoundError: 会话不存在
        """
        session = self.sessions.get(session_id)
        if not session:
            raise NotFoundError(f"会话不存在: {session_id}")
        
        # 更新最后活动时间
        session.last_activity = time.time()
        return session
    
    def close_session(self, session_id: str) -> bool:
        """
        关闭会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否成功
            
        Raises:
            NotFoundError: 会话不存在
        """
        session = self.get_session(session_id)
        session.cleanup_temp_files()
        del self.sessions[session_id]
        logger.info(f"已关闭会话: {session_id}")
        return True
    
    def cleanup_expired_sessions(self) -> None:
        """清理过期会话"""
        sessions_to_remove = []
        
        for session_id, session in self.sessions.items():
            if session.is_expired():
                sessions_to_remove.append(session_id)
                session.cleanup_temp_files()
        
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
            logger.info(f"已清理过期会话: {session_id}")


# 创建全局会话管理器实例
session_manager = SessionManager()
