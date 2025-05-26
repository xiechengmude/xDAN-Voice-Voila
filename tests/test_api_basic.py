#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Voila 全双工语音 API 服务基本功能测试用例
"""

import os
import sys
import json
import time
import base64
import pytest
import requests
import numpy as np
import websocket
from dotenv import load_dotenv

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载环境变量
load_dotenv()

# 测试配置
API_BASE_URL = f"http://{os.getenv('HOST', 'localhost')}:{os.getenv('PORT', '8000')}"
WS_BASE_URL = f"ws://{os.getenv('HOST', 'localhost')}:{os.getenv('PORT', '8000')}"


class TestVoilaAPI:
    """测试 Voila API 的基本功能"""
    
    session_id = None
    
    def setup_method(self):
        """每个测试方法前执行，创建会话"""
        response = requests.post(
            f"{API_BASE_URL}/api/session/create",
            json={"instruction": "这是一个测试会话", "voice_name": "Homer Simpson"}
        )
        assert response.status_code == 200
        self.session_id = response.json()["session_id"]
        print(f"创建会话: {self.session_id}")
    
    def teardown_method(self):
        """每个测试方法后执行，关闭会话"""
        if self.session_id:
            response = requests.post(
                f"{API_BASE_URL}/api/session/close",
                data={"session_id": self.session_id}
            )
            assert response.status_code == 200
            print(f"关闭会话: {self.session_id}")
            self.session_id = None
    
    def test_health_check(self):
        """测试健康检查端点"""
        response = requests.get(f"{API_BASE_URL}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        print("健康检查通过")
    
    def test_voice_list(self):
        """测试获取可用声音列表"""
        response = requests.get(f"{API_BASE_URL}/api/voice/list")
        assert response.status_code == 200
        assert "voices" in response.json()
        voices = response.json()["voices"]
        assert isinstance(voices, list)
        assert len(voices) > 0
        print(f"获取到 {len(voices)} 个可用声音")
    
    def test_set_voice(self):
        """测试设置声音"""
        # 首先获取可用声音
        voices_response = requests.get(f"{API_BASE_URL}/api/voice/list")
        voices = voices_response.json()["voices"]
        
        if len(voices) > 0:
            test_voice = voices[0]
            response = requests.post(
                f"{API_BASE_URL}/api/voice/set",
                data={"session_id": self.session_id, "voice_name": test_voice}
            )
            assert response.status_code == 200
            assert f"已设置声音: {test_voice}" in response.json()["message"]
            print(f"成功设置声音: {test_voice}")
    
    def test_text_processing(self):
        """测试文本处理"""
        response = requests.post(
            f"{API_BASE_URL}/api/text/process",
            json={
                "session_id": self.session_id,
                "text": "你好，这是一个测试文本"
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert "text" in result
        assert "audio_base64" in result
        assert len(result["audio_base64"]) > 0
        print(f"文本处理成功，响应文本: {result['text'][:30]}...")
    
    def create_test_audio(self):
        """创建测试音频数据"""
        # 创建一个简单的正弦波作为测试音频
        sample_rate = 16000
        duration = 1  # 1秒
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz 正弦波
        
        # 转换为 WAV 格式的 base64 字符串
        import io
        import soundfile as sf
        
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV')
        buffer.seek(0)
        
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    def test_audio_processing(self):
        """测试音频处理"""
        audio_base64 = self.create_test_audio()
        
        response = requests.post(
            f"{API_BASE_URL}/api/audio/process",
            json={
                "session_id": self.session_id,
                "audio_base64": audio_base64
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert "text" in result
        assert "audio_base64" in result
        print(f"音频处理成功，响应文本: {result['text'][:30]}...")
    
    def test_websocket_connection(self):
        """测试 WebSocket 连接"""
        # 创建 WebSocket 连接
        ws = websocket.create_connection(f"{WS_BASE_URL}/ws/{self.session_id}")
        
        # 接收欢迎消息
        welcome = json.loads(ws.recv())
        assert welcome["type"] == "welcome"
        print("WebSocket 连接成功，收到欢迎消息")
        
        # 发送音频数据
        audio_data = np.zeros(16000).tolist()  # 1秒静音
        ws.send(json.dumps({
            "type": "audio",
            "data": audio_data
        }))
        
        # 接收响应（设置超时以防止无限等待）
        ws.settimeout(10)
        try:
            response = json.loads(ws.recv())
            assert response["type"] == "response"
            assert "text" in response["data"]
            assert "audio_base64" in response["data"]
            print(f"WebSocket 音频处理成功，响应文本: {response['data']['text'][:30]}...")
        except websocket.WebSocketTimeoutException:
            pytest.fail("WebSocket 响应超时")
        finally:
            # 关闭连接
            ws.close()


class TestErrorHandling:
    """测试错误处理"""
    
    def test_invalid_session(self):
        """测试无效会话ID"""
        response = requests.post(
            f"{API_BASE_URL}/api/text/process",
            json={
                "session_id": "invalid_session_id",
                "text": "测试文本"
            }
        )
        assert response.status_code == 404
        assert "错误" in response.json()["detail"].lower()
        print("无效会话ID测试通过")
    
    def test_invalid_audio(self):
        """测试无效音频数据"""
        # 创建有效会话
        session_response = requests.post(
            f"{API_BASE_URL}/api/session/create",
            json={"instruction": "测试会话"}
        )
        session_id = session_response.json()["session_id"]
        
        try:
            # 发送无效音频
            response = requests.post(
                f"{API_BASE_URL}/api/audio/process",
                json={
                    "session_id": session_id,
                    "audio_base64": "invalid_base64_data"
                }
            )
            assert response.status_code in [400, 422]
            print("无效音频数据测试通过")
        finally:
            # 清理会话
            requests.post(
                f"{API_BASE_URL}/api/session/close",
                data={"session_id": session_id}
            )


def run_tests():
    """运行所有测试"""
    # 检查服务是否在运行
    try:
        health_check = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if health_check.status_code != 200:
            print(f"错误: API 服务未正常运行，状态码: {health_check.status_code}")
            return False
    except requests.RequestException as e:
        print(f"错误: 无法连接到 API 服务 ({e})，请确保服务已启动")
        return False
    
    # 运行测试
    pytest.main(["-xvs", __file__])
    return True


if __name__ == "__main__":
    run_tests()
