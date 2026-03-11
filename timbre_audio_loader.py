"""
@title: Timbre Audio Loader
@author: ComfyUI-Index-TTS
@description: 用于加载Timbre模型目录下的音频文件的节点
"""

import os
import sys
import hashlib
import torchaudio
import torch
import glob
from pathlib import Path
import folder_paths

# # 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# if current_dir not in sys.path:
#     sys.path.append(current_dir)

# 直接使用torchaudio功能，不再导入额外函数

class TimbreAudioLoader:
    """
    ComfyUI节点: 从Timbre模型目录加载音频样本文件，支持刷新列表
    """
    
    # 保存扫描的音频文件缓存
    audio_files_cache = []
    
    @classmethod
    def INPUT_TYPES(cls):
        # 定义Timbre模型目录路径 - 使用项目内的目录
        timbre_dir = os.path.join(current_dir, "TimbreModel")
        
        # 确保目录存在
        os.makedirs(timbre_dir, exist_ok=True)
        
        # 扫描所有支持的音频文件
        cls.scan_audio_files(timbre_dir)
        
        return {
            "required": {
                "audio_file": (cls.audio_files_cache, ),
                "refresh": ("BOOLEAN", {"default": False, "label": "刷新音频列表"})
            }
        }
    
    @classmethod
    def scan_audio_files(cls, directory):
        """扫描目录下的所有音频文件"""
        # 支持的音频格式模式（Windows不区分大小写）
        audio_patterns = ["**/*.wav", "**/*.mp3", "**/*.ogg", "**/*.flac"]
        
        # 初始化音频文件缓存
        cls.audio_files_cache = ["无音频文件"] # 默认选项
        
        # 检查目录是否存在
        if not os.path.exists(directory):
            print(f"[TimbreAudioLoader] 警告: 目录不存在: {directory}")
            return
        
        # 使用集合来确保文件名唯一性
        unique_filenames = set()
        audio_files = []
        
        # 扫描所有音频文件
        for pattern in audio_patterns:
            # 使用递归模式搜索
            matches = glob.glob(os.path.join(directory, pattern), recursive=True)
            for file_path in matches:
                # 提取文件名（不包含路径）
                file_name = os.path.basename(file_path)
                # 只添加尚未添加的文件名
                if file_name.lower() not in unique_filenames:
                    unique_filenames.add(file_name.lower())
                    audio_files.append(file_path)
        
        # 将收集到的文件添加到缓存
        if audio_files:
            # 按文件名排序
            audio_files.sort(key=lambda x: os.path.basename(x).lower())
            
            # 添加文件名到缓存
            for file_path in audio_files:
                file_name = os.path.basename(file_path)
                cls.audio_files_cache.append(file_name)
            
            print(f"[TimbreAudioLoader] 已加载 {len(cls.audio_files_cache)-1} 个音频文件")
        else:
            print(f"[TimbreAudioLoader] 警告: 未找到音频文件，路径: {directory}")
    
    RETURN_TYPES = ("AUDIO", )
    FUNCTION = "load_timbre_audio"
    CATEGORY = "audio"
    
    def load_timbre_audio(self, audio_file, refresh):
        """加载选择的音频文件或刷新列表"""
        # 定义Timbre模型目录路径 - 使用项目内的目录
        timbre_dir = os.path.join(current_dir, "TimbreModel")
        
        # 如果用户点击了刷新按钮
        if refresh:
            self.__class__.scan_audio_files(timbre_dir)
            print("[TimbreAudioLoader] 已刷新音频文件列表")
        
        # 如果选择了"无音频文件"或列表为空，返回空的音频数据
        if audio_file == "无音频文件" or not audio_file:
            # 创建一个小的空音频样本
            waveform = torch.zeros((1, 16000))  # 1秒静音
            sample_rate = 16000
            return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}, )
        
        # 构建完整的文件路径
        file_path = os.path.join(timbre_dir, audio_file)
        
        try:
            # 使用torchaudio加载音频
            waveform, sample_rate = torchaudio.load(file_path)
            
            # 返回ComfyUI音频格式
            return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}, )
        except Exception as e:
            print(f"[TimbreAudioLoader] 加载音频失败: {e}")
            # 发生错误时返回空音频
            waveform = torch.zeros((1, 16000))
            sample_rate = 16000
            return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}, )
    
    @classmethod
    def IS_CHANGED(cls, audio_file, refresh):
        """当输入变化时通知ComfyUI"""
        # 如果点击了刷新按钮，返回随机值以触发节点更新
        if refresh:
            return str(os.urandom(8).hex())
        
        # 如果选择了有效的音频文件，返回文件路径作为变化标识
        if audio_file != "无音频文件":
            timbre_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), "models", "Index-TTS", "timbre")
            file_path = os.path.join(timbre_dir, audio_file)
            
            # 检查文件是否存在
            if os.path.exists(file_path):
                # 计算文件哈希值，用于标识变化
                m = hashlib.sha256()
                with open(file_path, 'rb') as f:
                    m.update(f.read())
                return m.digest().hex()
        
        return audio_file

class RefreshTimbreAudio:
    """
    简单的刷新Timbre音频列表节点
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "refresh": ("BOOLEAN", {"default": True, "label": "刷新音频列表"})
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "refresh"
    CATEGORY = "audio"
    OUTPUT_NODE = True
    
    def refresh(self, refresh):
        if refresh:
            # 定义Timbre模型目录路径
            timbre_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), "models", "Index-TTS", "timbre")
            
            # 刷新TimbreAudioLoader中的缓存
            TimbreAudioLoader.scan_audio_files(timbre_dir)
            print("[RefreshTimbreAudio] 已刷新音频文件列表")
        
        return {}
