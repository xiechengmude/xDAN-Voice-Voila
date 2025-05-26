import requests
import json
import base64
import time
import asyncio
import websockets
import pyaudio
import wave
import numpy as np
import soundfile as sf
import io
from threading import Thread
from queue import Queue

# API 服务配置
API_BASE_URL = "http://localhost:8000/api"
WS_BASE_URL = "ws://localhost:8000/ws"

# 音频配置
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000

# 全局变量
audio_queue = Queue()
response_queue = Queue()
is_recording = False
websocket = None
session_id = None

async def create_session(instruction="你是一个由 Maitrix.org 创建的智能 AI 助手。", voice_name="Homer Simpson"):
    """创建新会话"""
    url = f"{API_BASE_URL}/session/create"
    data = {
        "instruction": instruction,
        "voice_name": voice_name
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        print(f"会话已创建: {result['session_id']}")
        return result["session_id"]
    except Exception as e:
        print(f"创建会话时出错: {e}")
        return None

async def close_session(session_id):
    """关闭会话"""
    url = f"{API_BASE_URL}/session/close"
    data = {"session_id": session_id}
    
    try:
        response = requests.post(url, data=data)
        response.raise_for_status()
        print(f"会话已关闭: {session_id}")
        return True
    except Exception as e:
        print(f"关闭会话时出错: {e}")
        return False

async def set_voice(session_id, voice_name):
    """设置预定义声音"""
    url = f"{API_BASE_URL}/voice/set"
    data = {
        "session_id": session_id,
        "voice_name": voice_name
    }
    
    try:
        response = requests.post(url, data=data)
        response.raise_for_status()
        print(f"已设置声音: {voice_name}")
        return True
    except Exception as e:
        print(f"设置声音时出错: {e}")
        return False

async def set_custom_voice(session_id, audio_file_path):
    """设置自定义声音"""
    url = f"{API_BASE_URL}/voice/custom"
    files = {"audio_file": open(audio_file_path, "rb")}
    data = {"session_id": session_id}
    
    try:
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        print("已设置自定义声音")
        return True
    except Exception as e:
        print(f"设置自定义声音时出错: {e}")
        return False

async def process_audio(session_id, audio_data):
    """处理音频并获取响应"""
    url = f"{API_BASE_URL}/audio/process"
    
    # 将音频数据转换为 base64
    audio_bytes = io.BytesIO()
    sf.write(audio_bytes, audio_data, RATE, format='WAV')
    audio_bytes.seek(0)
    audio_base64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
    
    data = {
        "session_id": session_id,
        "audio_base64": audio_base64
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        
        # 处理文本响应
        if "text" in result and result["text"]:
            print(f"AI: {result['text']}")
        
        # 处理音频响应
        if "audio_base64" in result and result["audio_base64"]:
            audio_data = base64.b64decode(result["audio_base64"])
            audio_array, _ = sf.read(io.BytesIO(audio_data))
            
            # 播放音频
            play_audio(audio_array)
        
        return result
    except Exception as e:
        print(f"处理音频时出错: {e}")
        return None

async def process_text(session_id, text):
    """处理文本并获取语音响应"""
    url = f"{API_BASE_URL}/text/process"
    
    data = {
        "session_id": session_id,
        "text": text
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        
        # 处理文本响应
        if "text" in result and result["text"]:
            print(f"AI: {result['text']}")
        
        # 处理音频响应
        if "audio_base64" in result and result["audio_base64"]:
            audio_data = base64.b64decode(result["audio_base64"])
            audio_array, _ = sf.read(io.BytesIO(audio_data))
            
            # 播放音频
            play_audio(audio_array)
        
        return result
    except Exception as e:
        print(f"处理文本时出错: {e}")
        return None

async def list_voices():
    """获取可用声音列表"""
    url = f"{API_BASE_URL}/voices/list"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        result = response.json()
        print("可用声音:")
        for voice in result["voices"]:
            print(f"- {voice}")
        return result["voices"]
    except Exception as e:
        print(f"获取声音列表时出错: {e}")
        return []

def play_audio(audio_data):
    """播放音频数据"""
    p = pyaudio.PyAudio()
    
    # 打开音频流
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        output=True
    )
    
    # 播放音频
    stream.write(audio_data.astype(np.float32).tobytes())
    
    # 关闭流
    stream.stop_stream()
    stream.close()
    p.terminate()

def record_audio_callback(in_data, frame_count, time_info, status):
    """录音回调函数"""
    if is_recording:
        # 将字节数据转换为 numpy 数组
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        audio_queue.put(audio_data)
    
    return (in_data, pyaudio.paContinue)

def start_recording():
    """开始录音"""
    global is_recording
    
    p = pyaudio.PyAudio()
    
    # 打开音频流
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=record_audio_callback
    )
    
    is_recording = True
    print("开始录音...")
    
    return p, stream

def stop_recording(p, stream):
    """停止录音"""
    global is_recording
    
    is_recording = False
    
    # 关闭流
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    print("录音已停止")

async def websocket_client(session_id):
    """WebSocket 客户端，用于全双工音频交互"""
    global websocket
    
    uri = f"{WS_BASE_URL}/{session_id}"
    
    async with websockets.connect(uri) as ws:
        websocket = ws
        print("WebSocket 连接已建立")
        
        # 创建发送和接收任务
        send_task = asyncio.create_task(send_audio_data(ws))
        receive_task = asyncio.create_task(receive_responses(ws))
        
        # 等待任务完成
        await asyncio.gather(send_task, receive_task)

async def send_audio_data(ws):
    """发送音频数据到服务器"""
    while True:
        if not audio_queue.empty():
            # 获取音频数据
            audio_data = audio_queue.get()
            
            # 发送到服务器
            await ws.send(json.dumps({
                "type": "audio",
                "data": audio_data.tolist()
            }))
        
        await asyncio.sleep(0.05)

async def receive_responses(ws):
    """接收服务器响应"""
    while True:
        try:
            # 接收响应
            response = await ws.recv()
            data = json.loads(response)
            
            if data["type"] == "welcome":
                print(f"服务器: {data['message']}")
            elif data["type"] == "response":
                response_data = data["data"]
                
                # 处理文本响应
                if "text" in response_data and response_data["text"]:
                    print(f"AI: {response_data['text']}")
                
                # 处理音频响应
                if "audio_base64" in response_data and response_data["audio_base64"]:
                    audio_data = base64.b64decode(response_data["audio_base64"])
                    audio_array, _ = sf.read(io.BytesIO(audio_data))
                    
                    # 将响应放入队列，由主线程播放
                    response_queue.put(audio_array)
            elif data["type"] == "info":
                print(f"服务器: {data['message']}")
            elif data["type"] == "error":
                print(f"错误: {data['error']}")
        except Exception as e:
            print(f"接收响应时出错: {e}")
            break

def response_player():
    """响应播放线程"""
    while True:
        if not response_queue.empty():
            # 获取音频数据
            audio_data = response_queue.get()
            
            # 播放音频
            play_audio(audio_data)
        
        time.sleep(0.05)

async def rest_api_demo():
    """REST API 演示"""
    # 创建会话
    session_id = await create_session()
    if not session_id:
        return
    
    try:
        # 获取可用声音列表
        await list_voices()
        
        # 设置声音
        await set_voice(session_id, "Lisa Simpson")
        
        # 发送文本并获取语音响应
        await process_text(session_id, "你好，我是一个用户。")
        
        # 录制一段音频
        p, stream = start_recording()
        await asyncio.sleep(5)  # 录制5秒
        stop_recording(p, stream)
        
        # 处理录制的音频
        audio_data = np.concatenate([audio_queue.get() for _ in range(audio_queue.qsize())])
        await process_audio(session_id, audio_data)
        
        # 关闭会话
        await close_session(session_id)
        
    except Exception as e:
        print(f"演示过程中出错: {e}")
        await close_session(session_id)

async def websocket_demo():
    """WebSocket 演示"""
    # 创建会话
    session_id = await create_session()
    if not session_id:
        return
    
    try:
        # 启动响应播放线程
        player_thread = Thread(target=response_player)
        player_thread.daemon = True
        player_thread.start()
        
        # 开始录音
        p, stream = start_recording()
        
        # 连接 WebSocket
        await websocket_client(session_id)
        
    except Exception as e:
        print(f"演示过程中出错: {e}")
    finally:
        # 停止录音
        if 'p' in locals() and 'stream' in locals():
            stop_recording(p, stream)
        
        # 关闭会话
        await close_session(session_id)

async def interactive_demo():
    """交互式演示"""
    print("=== Voila 全双工语音 API 客户端演示 ===")
    print("1. REST API 演示")
    print("2. WebSocket 演示 (全双工)")
    print("3. 退出")
    
    choice = input("请选择演示类型: ")
    
    if choice == "1":
        await rest_api_demo()
    elif choice == "2":
        await websocket_demo()
    elif choice == "3":
        print("退出演示")
    else:
        print("无效的选择")

if __name__ == "__main__":
    asyncio.run(interactive_demo())
