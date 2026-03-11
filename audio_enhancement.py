"""
@title: 音频增强处理模块
@author: ComfyUI-Index-TTS
@description: 用于处理和增强TTS生成的音频质量
"""

import os
import sys
import numpy as np
import torch
import tempfile
import time
import librosa
import soundfile as sf
from scipy import signal as scipy_signal

# # 确保当前目录在导入路径中
# current_dir = os.path.dirname(os.path.abspath(__file__))
# if current_dir not in sys.path:
#     sys.path.append(current_dir)


class AudioCleanupNode:
    """
    ComfyUI的音频清理节点，用于去除混响和杂音，提高人声质量
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "denoise_strength": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
                "dereverb_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "high_pass_freq": ("FLOAT", {"default": 100.0, "min": 20.0, "max": 500.0, "step": 10.0}),
                "low_pass_freq": ("FLOAT", {"default": 8000.0, "min": 1000.0, "max": 16000.0, "step": 100.0}),
                "normalize": (["true", "false"], {"default": "true"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("enhanced_audio",)
    FUNCTION = "enhance_audio"
    CATEGORY = "audio"
    
    def __init__(self):
        print("[AudioCleanup] 初始化音频清理节点")
    
    def enhance_audio(self, audio, denoise_strength=0.5, dereverb_strength=0.7, 
                     high_pass_freq=100.0, low_pass_freq=8000.0, normalize="true"):
        """
        增强音频质量，去除杂音和混响
        
        参数:
            audio: ComfyUI音频格式，字典包含 "waveform" 和 "sample_rate"
            denoise_strength: 降噪强度，0.1到1.0
            dereverb_strength: 去混响强度，0.0到1.0
            high_pass_freq: 高通滤波频率，20到500
            low_pass_freq: 低通滤波频率，1000到16000
            normalize: 是否归一化音频，"true"或"false"
            
        返回:
            enhanced_audio: 增强后的音频，ComfyUI音频格式
        """
        try:
            print(f"[AudioCleanup] 开始处理音频")
            
            # 处理ComfyUI的音频格式
            if isinstance(audio, dict) and "waveform" in audio and "sample_rate" in audio:
                waveform = audio["waveform"]
                sample_rate = audio["sample_rate"]
                
                print(f"[AudioCleanup] 输入音频格式: ComfyUI字典格式, sample_rate={sample_rate}")
                print(f"[AudioCleanup] waveform类型: {type(waveform)}, 形状: {waveform.shape if hasattr(waveform, 'shape') else '未知'}")
                
                # 如果是tensor，转换为numpy
                if isinstance(waveform, torch.Tensor):
                    # 确保我们处理的是一个二维数组 [通道, 样本]
                    if waveform.dim() == 3:  # [batch, 通道, 样本]
                        waveform = waveform.squeeze(0)  # 移除batch维度
                    waveform = waveform.cpu().numpy()
                    
                    # 如果是多通道，取第一个通道
                    if waveform.ndim > 1 and waveform.shape[0] > 1:
                        print(f"[AudioCleanup] 检测到多通道音频({waveform.shape[0]}通道)，使用第一个通道")
                        audio_data = waveform[0]
                    else:
                        audio_data = waveform.squeeze()  # 确保是一维数组
                else:
                    audio_data = waveform
                    
                print(f"[AudioCleanup] 处理前的音频形状: {audio_data.shape}")
                print(f"[AudioCleanup] 处理参数: 降噪强度={denoise_strength}, 去混响强度={dereverb_strength}")
                print(f"[AudioCleanup] 滤波设置: 高通={high_pass_freq}Hz, 低通={low_pass_freq}Hz, 归一化={normalize}")
                
                # 开始音频处理流程
                enhanced_audio = audio_data.copy()
                
                # 1. 应用高通滤波器去除低频噪音
                if high_pass_freq > 20.0:
                    print(f"[AudioCleanup] 应用高通滤波器，截止频率: {high_pass_freq}Hz")
                    b, a = scipy_signal.butter(4, high_pass_freq / (sample_rate / 2), 'highpass')
                    enhanced_audio = scipy_signal.filtfilt(b, a, enhanced_audio)
                
                # 2. 应用低通滤波器去除高频噪音
                if low_pass_freq < 16000.0:
                    print(f"[AudioCleanup] 应用低通滤波器，截止频率: {low_pass_freq}Hz")
                    b, a = scipy_signal.butter(4, low_pass_freq / (sample_rate / 2), 'lowpass')
                    enhanced_audio = scipy_signal.filtfilt(b, a, enhanced_audio)
                
                # 3. 降噪处理
                if denoise_strength > 0.1:
                    print(f"[AudioCleanup] 应用降噪处理，强度: {denoise_strength}")
                    # 使用谱减法降噪
                    # 计算短时傅里叶变换(STFT)
                    stft = librosa.stft(enhanced_audio)
                    
                    # 估计噪声谱
                    noise_stft = np.abs(stft[:, :int(stft.shape[1] * 0.1)])  # 使用前10%作为噪声估计
                    noise_spec = np.mean(noise_stft, axis=1)
                    
                    # 谱减法
                    spec = np.abs(stft)
                    phase = np.angle(stft)
                    spec_sub = np.maximum(spec - denoise_strength * np.expand_dims(noise_spec, 1), 0)
                    
                    # 重建信号
                    enhanced_stft = spec_sub * np.exp(1j * phase)
                    enhanced_audio = librosa.istft(enhanced_stft, length=len(enhanced_audio))
                
                # 4. 去混响处理
                if dereverb_strength > 0.0:
                    print(f"[AudioCleanup] 应用去混响处理，强度: {dereverb_strength}")
                    # 简化的去混响方法 - 使用谱包络增强
                    D = librosa.stft(enhanced_audio)
                    S_db = librosa.amplitude_to_db(np.abs(D))
                    
                    # 应用谱包络增强
                    percentile = int((1 - dereverb_strength) * 100)
                    percentile = max(1, min(percentile, 99))  # 确保在有效范围内
                    S_enhanced = np.percentile(S_db, percentile, axis=1)
                    S_enhanced = np.expand_dims(S_enhanced, 1)
                    
                    # 应用增强
                    gain = np.repeat(S_enhanced, S_db.shape[1], axis=1)
                    S_db_enhanced = S_db * dereverb_strength + gain * (1 - dereverb_strength)
                    
                    # 转回时域
                    S_enhanced = librosa.db_to_amplitude(S_db_enhanced)
                    phase = np.angle(D)
                    D_enhanced = S_enhanced * np.exp(1j * phase)
                    enhanced_audio = librosa.istft(D_enhanced, length=len(enhanced_audio))
                
                # 5. 归一化
                if normalize == "true":
                    print(f"[AudioCleanup] 应用音频归一化")
                    enhanced_audio = librosa.util.normalize(enhanced_audio)
                
                # 输出处理结果信息
                original_rms = np.sqrt(np.mean(audio_data ** 2))
                enhanced_rms = np.sqrt(np.mean(enhanced_audio ** 2))
                print(f"[AudioCleanup] 原始音频RMS: {original_rms:.6f}")
                print(f"[AudioCleanup] 增强后音频RMS: {enhanced_rms:.6f}")
                print(f"[AudioCleanup] RMS变化比例: {enhanced_rms/original_rms if original_rms > 0 else 'N/A'}")
                
                # 转换为torch tensor并设置为ComfyUI期望的格式
                enhanced_tensor = torch.tensor(enhanced_audio, dtype=torch.float32)
                
                # 确保是3D张量 [batch, channels, samples]
                if enhanced_tensor.dim() == 1:
                    enhanced_tensor = enhanced_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
                
                # 返回ComfyUI音频格式
                enhanced_dict = {
                    "waveform": enhanced_tensor,
                    "sample_rate": sample_rate
                }
                
                print(f"[AudioCleanup] 音频增强完成，输出形状: {enhanced_tensor.shape}")
                return (enhanced_dict,)
            else:
                print(f"[AudioCleanup] 错误: 输入音频格式不正确: {type(audio)}")
                raise ValueError("输入音频格式不支持，应为ComfyUI的AUDIO类型")
                
        except Exception as e:
            import traceback
            print(f"[AudioCleanup] 处理音频失败: {e}")
            print(f"[AudioCleanup] 错误详情:")
            traceback.print_exc()
            
            # 生成一个简单的错误提示音频
            sample_rate = 24000
            duration = 1.0  # 1秒
            t = np.linspace(0, duration, int(sample_rate * duration))
            warning_tone = np.sin(2 * np.pi * 880 * t).astype(np.float32)  # 880Hz警告音
            print(f"[AudioCleanup] 生成警告音频作为错误处理")
            
            # 转换为ComfyUI音频格式
            signal_tensor = torch.tensor(warning_tone, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
            audio_dict = {
                "waveform": signal_tensor,
                "sample_rate": sample_rate
            }
            
            return (audio_dict,)
