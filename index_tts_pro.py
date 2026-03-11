import os
import sys
import numpy as np
import torch
import tempfile
import soundfile as sf
import time
import re
import json
from pathlib import Path
import random

# # 确保当前目录在导入路径中
# current_dir = os.path.dirname(os.path.abspath(__file__))
# if current_dir not in sys.path:
#     sys.path.append(current_dir)

# 导入TTS模型
from .tts_models import IndexTTSModel


class IndexTTSProNode:
    """
    ComfyUI的IndexTTS Pro节点，专用于小说阅读，支持多角色语音合成
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "structured_text": ("STRING", {"multiline": True, "default": "<Narrator>This is a sample narrative text.<Character1>Hello.<Narrator>He said."}),
                "narrator_audio": ("AUDIO", {"description": "正文/旁白的参考音频"}),
                "model_version": (["Index-TTS", "IndexTTS-1.5"], {"default": "Index-TTS"}),
                "language": (["auto", "zh", "en"], {"default": "auto"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                "character1_audio": ("AUDIO", {"description": "角色1的参考音频"}),
                "character2_audio": ("AUDIO", {"description": "角色2的参考音频"}),
                "character3_audio": ("AUDIO", {"description": "角色3的参考音频"}),
                "character4_audio": ("AUDIO", {"description": "角色4的参考音频"}),
                "character5_audio": ("AUDIO", {"description": "角色5的参考音频"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.5, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 30, "min": 1, "max": 100, "step": 1}),
                "repetition_penalty": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 15.0, "step": 0.5}),
                "length_penalty": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "max_mel_tokens": ("INT", {"default": 600, "min": 100, "max": 1500, "step": 50}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "INT", "STRING", "STRING",)
    RETURN_NAMES = ("audio", "seed", "Subtitle", "SimplifiedSubtitle",)
    FUNCTION = "generate_multi_voice_speech"
    CATEGORY = "audio"
    
    def __init__(self):
        # 根路径
        self.models_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
        # 可用模型版本
        self.model_versions = {
            "Index-TTS": os.path.join(self.models_root, "Index-TTS"),
            "IndexTTS-1.5": os.path.join(self.models_root, "IndexTTS-1.5")
        }
        # 默认使用 Index-TTS 版本
        self.current_version = "Index-TTS"
        self.model_dir = self.model_versions[self.current_version]
        self.tts_model = None
        
        print(f"[IndexTTS Pro] 初始化节点，可用模型版本: {list(self.model_versions.keys())}")
        print(f"[IndexTTS Pro] 默认模型目录: {self.model_dir}")
    
    def _init_model(self, model_version="Index-TTS"):
        """初始化TTS模型（延迟加载）
        
        Args:
            model_version: 模型版本，默认为 "Index-TTS"
        """
        # 如果版本发生变化或模型未加载，重新加载模型
        if self.tts_model is None or self.current_version != model_version:
            # 更新当前版本和模型目录
            if model_version in self.model_versions:
                self.current_version = model_version
                self.model_dir = self.model_versions[model_version]
                print(f"[IndexTTS Pro] 切换到模型版本: {model_version}, 目录: {self.model_dir}")
            else:
                print(f"[IndexTTS Pro] 警告: 未知模型版本 {model_version}，使用默认版本 {self.current_version}")
            
            # 如果已有模型，先释放资源
            if self.tts_model is not None:
                print(f"[IndexTTS Pro] 卸载现有模型...")
                self.tts_model = None
                # 强制垃圾回收
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            print(f"[IndexTTS Pro] 开始加载模型版本: {self.current_version}...")
            # 检查必要的模型文件
            required_files = ["gpt.pth", "config.yaml"]
            missing_files = []
            for file in required_files:
                file_path = os.path.join(self.model_dir, file)
                if not os.path.exists(file_path):
                    missing_files.append(file)
                else:
                    file_size = os.path.getsize(file_path) / (1024*1024)  # 转换为MB
                    print(f"[IndexTTS Pro] 找到模型文件: {file} ({file_size:.2f}MB)")
            
            if missing_files:
                error_msg = f"模型 {self.current_version} 缺少必要的文件: {', '.join(missing_files)}"
                print(f"[IndexTTS Pro] 错误: {error_msg}")
                raise FileNotFoundError(error_msg)
                
            try:
                # 记录开始加载时间
                start_time = time.time()
                
                # 使用tts_models.py中的IndexTTSModel实现
                self.tts_model = IndexTTSModel(model_dir=self.model_dir)
                
                # 记录加载完成时间
                load_time = time.time() - start_time
                print(f"[IndexTTS Pro] 模型 {self.current_version} 已成功加载，耗时: {load_time:.2f}秒")
            except Exception as e:
                import traceback
                print(f"[IndexTTS Pro] 初始化模型 {self.current_version} 失败: {e}")
                print(f"[IndexTTS Pro] 错误详情:")
                traceback.print_exc()
                raise RuntimeError(f"初始化IndexTTS模型 {self.current_version} 失败: {e}")
    
    def _process_audio_input(self, audio_input):
        """处理ComfyUI的音频格式
        
        Args:
            audio_input: ComfyUI的音频格式
            
        Returns:
            tuple: (waveform, sample_rate) 元组
        """
        if audio_input is None:
            return None
            
        if isinstance(audio_input, dict) and "waveform" in audio_input and "sample_rate" in audio_input:
            waveform = audio_input["waveform"]
            sample_rate = audio_input["sample_rate"]
            
            # 如果waveform是torch.Tensor，转换为numpy
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()
                
            return (waveform, sample_rate)
            
        # 如果已经是元组格式
        elif isinstance(audio_input, tuple) and len(audio_input) == 2:
            return audio_input
            
        # 如果都不是，报错
        raise ValueError("参考音频格式不支持，应为 AUDIO 类型")
    
    def _parse_structured_text(self, structured_text):
        """解析结构化文本
        
        Args:
            structured_text: 结构化文本，如 "<Narrator>This is narrative text<Character1>This is Character1's line"
            
        Returns:
            list: 解析后的文本段落列表，每个元素为 (role, text)
        """
        segments = []
        # 标签匹配模式
        pattern = re.compile(r'<(Narrator|Character\d+)>([^<]+)')
        
        # 查找所有匹配
        matches = pattern.findall(structured_text)
        
        # 如果找不到任何匹配，将整个文本作为旁白处理
        if not matches:
            segments.append(("Narrator", structured_text))
        else:
            for role, text in matches:
                segments.append((role, text))
                
        return segments
    
    def _concatenate_audio(self, audio_segments):
        """连接多个音频段落
        
        Args:
            audio_segments: 音频段落列表，每个元素为 (waveform, sample_rate)
            
        Returns:
            tuple: 连接后的 (waveform, sample_rate)
        """
        if not audio_segments:
            return None
            
        # 确保所有段落的采样率相同
        sample_rate = audio_segments[0][1]
        
        # 过滤有效的音频段落
        valid_segments = []
        for idx, segment in enumerate(audio_segments):
            try:
                audio_data, seg_sample_rate = segment
                
                # 如果是第一个有效的音频段，设置采样率
                if not valid_segments:
                    sample_rate = seg_sample_rate
                    
                # 检查音频数据是否有效
                if audio_data is not None and isinstance(audio_data, np.ndarray):
                    # 确保是有效的numpy数组
                    if audio_data.size > 0:
                        valid_segments.append(audio_data)
                        print(f"[IndexTTS Pro] Added segment {idx+1}: shape={audio_data.shape}, dtype={audio_data.dtype}")
                    else:
                        print(f"[IndexTTS Pro] Warning: Skipping empty audio segment {idx+1} with shape: {audio_data.shape}")
                else:
                    # 打印数据类型信息以便调试
                    print(f"[IndexTTS Pro] Warning: Skipping invalid audio data of type: {type(audio_data)}")
                    if hasattr(audio_data, '__dict__'):
                        print(f"[IndexTTS Pro] Data attributes: {dir(audio_data)}")
                    print(f"[IndexTTS Pro] Data value: {str(audio_data)[:100]}...")
            except Exception as e:
                print(f"[IndexTTS Pro] Error processing segment {idx+1}: {e}")
        
        if not valid_segments:
            print("[IndexTTS Pro] Error: No valid audio segments to concatenate")
            return None
            
        # 连接所有有效的音频段落
        try:
            # 连接所有段落
            print(f"[IndexTTS Pro] Concatenating {len(valid_segments)} audio segments")
            concatenated = np.concatenate(valid_segments, axis=0)
            print(f"[IndexTTS Pro] Concatenated audio shape: {concatenated.shape}")
            
            # 确保音频数据是适当的格式
            if concatenated.ndim == 1:
                # 保持为1D格式，我们在返回前会转化为适当的维度
                print(f"[IndexTTS Pro] Audio is 1D array with {len(concatenated)} samples")
            elif concatenated.ndim > 2:
                # 如果维度过多，转为1D数组
                print(f"[IndexTTS Pro] Audio has too many dimensions: {concatenated.shape}, flattening")
                concatenated = concatenated.flatten()
                print(f"[IndexTTS Pro] Flattened to: {concatenated.shape}")
                
            return (concatenated, sample_rate)
        except Exception as e:
            print(f"[IndexTTS Pro] Error concatenating audio segments: {e}")
            import traceback
            traceback.print_exc()
            
            # 如果连接失败，返回第一个有效段落
            if valid_segments:
                print(f"[IndexTTS Pro] Falling back to first valid segment")
                first_segment = valid_segments[0]
                return (first_segment, sample_rate)
            
            print(f"[IndexTTS Pro] No valid segments found, returning None")
            return None
    
    def _seconds_to_time_format(self, seconds):
        """将秒数转换为分:秒.毫秒格式
        
        Args:
            seconds: 秒数(float)
            
        Returns:
            str: 格式化的时间字符串，如 "1:23.456"
        """
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        seconds_int = int(remaining_seconds)
        milliseconds = int((remaining_seconds - seconds_int) * 1000)
        return f"{minutes}:{seconds_int:02d}.{milliseconds:03d}"
        
    def _parse_time_format(self, time_str):
        """将时间字符串转换为秒数
        
        Args:
            time_str: 时间字符串，如 "1:23.456" 或 "1:23"
            
        Returns:
            float: 对应的秒数
        """
        # 支持带毫秒和不带毫秒的格式
        if "." in time_str:
            # 格式: mm:ss.sss
            time_part, ms_part = time_str.split(".")
            parts = time_part.split(":")
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = int(parts[1])
                milliseconds = int(ms_part[:3].ljust(3, '0'))  # 确保是3位毫秒
                return minutes * 60 + seconds + milliseconds / 1000.0
        else:
            # 格式: mm:ss (向后兼容)
            parts = time_str.split(":")
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes * 60 + seconds
        return 0.0
    
    def generate_multi_voice_speech(self, structured_text, narrator_audio, model_version="Index-TTS", 
                                   language="auto", speed=1.0, seed=0, 
                                   character1_audio=None, character2_audio=None, character3_audio=None, 
                                   character4_audio=None, character5_audio=None,
                                   temperature=1.0, top_p=0.8, top_k=30, 
                                   repetition_penalty=10.0, length_penalty=0.0, 
                                   num_beams=3, max_mel_tokens=600):
        """
        生成多角色语音的主函数
        
        参数:
            structured_text: 结构化文本，包含角色标签
            narrator_audio: 旁白/正文的参考音频
            model_version: 模型版本
            language: 语言设置
            speed: 语音速度
            seed: 随机种子
            character1_audio~character5_audio: 角色参考音频
            temperature: 温度参数
            top_p: top_p参数
            top_k: top_k参数
            repetition_penalty: 重复惩罚
            length_penalty: 长度惩罚
            num_beams: beam数量
            max_mel_tokens: 最大mel token数
        """
        try:
            print(f"[IndexTTS Pro] Starting multi-voice generation with structured_text: {structured_text[:100]}...")
            
            # 使用固定种子或随机种子
            if seed == 0:
                seed = int(time.time() * 1000) % (2**32 - 1)
            
            # 初始化模型
            self._init_model(model_version)
            
            # 解析结构化文本
            parsed_text = self._parse_structured_text(structured_text)
            print(f"[IndexTTS Pro] Parsed text segments: {len(parsed_text)}")
            
            # 构建角色音频映射
            character_audios = {}
            for i, char_audio in enumerate([character1_audio, character2_audio, character3_audio, character4_audio, character5_audio], 1):
                if char_audio is not None:
                    character_audios[f"Character{i}"] = char_audio
            
            # 生成音频片段
            audio_segments = []
            current_time = 0.0  # 当前时间位置
            subtitle_data = []  # Subtitle数据列表
            
            for role, text in parsed_text:
                print(f"[IndexTTS Pro] Processing: {role} - {text[:50]}...")
                
                # 选择参考音频
                if role == "Narrator":
                    ref_audio = narrator_audio
                elif role in character_audios:
                    ref_audio = character_audios[role]
                else:
                    # 使用旁白音频作为默认参考
                    ref_audio = narrator_audio
                    print(f"[IndexTTS Pro] Warning: No specific audio for {role}, using narrator audio")
                
                try:
                    # 生成音频
                    result = self.tts_model.infer(
                        reference_audio=self._process_audio_input(ref_audio),  # 直接传入参考音频数据
                        text=text,
                        output_path=None,  # 不保存文件，直接返回音频数据
                        language=language,
                        speed=speed,
                        verbose=False,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        length_penalty=length_penalty,
                        num_beams=num_beams,
                        max_mel_tokens=max_mel_tokens
                    )
                    
                    # 处理返回结果
                    if isinstance(result, tuple) and len(result) == 2:
                        # 返回格式: (sample_rate, audio_data)
                        sample_rate, audio_data = result
                        
                        print(f"[IndexTTS Pro] Audio data type: {type(audio_data)}")
                        if hasattr(audio_data, 'shape'):
                            print(f"[IndexTTS Pro] Audio data shape: {audio_data.shape}")
                        
                        print(f"[IndexTTS Pro] Sample rate: {sample_rate}, type: {type(sample_rate)}")
                        
                        # 计算音频长度
                        if isinstance(audio_data, np.ndarray) and audio_data.size > 0:
                            if audio_data.ndim == 1:
                                audio_length = len(audio_data) / sample_rate
                            else:
                                audio_length = audio_data.shape[-1] / sample_rate
                            
                            # 添加字幕数据
                            start_time = self._seconds_to_time_format(current_time)
                            end_time = self._seconds_to_time_format(current_time + audio_length)
                            subtitle_item = {
                                "id": role,
                                "字幕": text,
                                "start": start_time,
                                "end": end_time
                            }
                            subtitle_data.append(subtitle_item)
                            current_time += audio_length
                            
                            # 添加到段落列表
                            audio_segments.append((audio_data, sample_rate))
                        else:
                            print(f"[IndexTTS Pro] Warning: Invalid audio data")
                    else:
                        print(f"[IndexTTS Pro] Warning: Unexpected TTS model return format: {type(result)}")
                        if isinstance(result, np.ndarray):
                            print(f"[IndexTTS Pro] Assuming default sample rate for numpy array return: shape={result.shape}")
                            if result.size > 0:
                                audio_length = len(result) / 24000
                                start_time = self._seconds_to_time_format(current_time)
                                end_time = self._seconds_to_time_format(current_time + audio_length)
                                subtitle_item = {
                                    "id": role,
                                    "字幕": text,
                                    "start": start_time,
                                    "end": end_time
                                }
                                subtitle_data.append(subtitle_item)
                                current_time += audio_length
                                audio_segments.append((result, 24000))
                        elif isinstance(result, int):
                            print(f"[IndexTTS Pro] Got only sample rate without audio data")
                        else:
                            print(f"[IndexTTS Pro] Unsupported result type: {type(result)}")
                            print(f"[IndexTTS Pro] Result value: {str(result)[:100]}")
                    
                    
                except Exception as e:
                    print(f"[IndexTTS Pro] Error generating {role} voice: {e}")
                    continue
            
            # 连接所有音频片段
            final_audio = self._concatenate_audio(audio_segments)
            if final_audio is None:
                raise ValueError("Failed to generate any audio segments")
            
            # 计算音频长度（考虑可能是2D格式）
            if final_audio[0].ndim > 1:
                audio_length = final_audio[0].shape[1] / final_audio[1]
            else:
                audio_length = len(final_audio[0]) / final_audio[1]
                
            print(f"[IndexTTS Pro] Multi-voice generation complete, total length: {audio_length:.2f} seconds")
            print(f"[IndexTTS Pro] Final audio shape before processing: {final_audio[0].shape}, sample rate: {final_audio[1]}")
            
            # 转为ComfyUI格式 - 需要是3D格式: [batch, channels, samples]
            audio_numpy = final_audio[0]
            
            # 转换为PyTorch张量
            audio_tensor = torch.tensor(audio_numpy, dtype=torch.float32)
            print(f"[IndexTTS Pro] Audio tensor dimensions: {audio_tensor.dim()}")
            
            # 确保音频数据是3D张量 [batch, channels, samples]
            if audio_tensor.dim() == 1:
                # [samples] -> [1, 1, samples]
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
                print(f"[IndexTTS Pro] 1D tensor reshaped to 3D: [1, 1, {audio_tensor.shape[-1]}]")
            elif audio_tensor.dim() == 2:
                # [channels, samples] -> [1, channels, samples]
                audio_tensor = audio_tensor.unsqueeze(0)
                print(f"[IndexTTS Pro] 2D tensor reshaped to 3D: [1, {audio_tensor.shape[1]}, {audio_tensor.shape[2]}]")
            
            print(f"[IndexTTS Pro] Final tensor shape: {audio_tensor.shape}")
            
            # 生成SubtitleJSON字符串
            import json
            subtitle_json = json.dumps(subtitle_data, ensure_ascii=False, indent=2)
            print(f"[IndexTTS Pro] Generated subtitle data with {len(subtitle_data)} items")
            
            # 生成简化字幕格式 (只包含时间和处理后实际分句的文本，不包含角色名)
            simplified_subtitles = []
            
            # 在这里，由于我们没有直接获取到TTS处理后的分句，需要从模型日志中获取或使用一个模拟处理
            # 这部分需要根据tts_models.py中的实际处理逻辑调整
            
            # 我们使用带有冒号的时间格式
            process_timepoints = []
            current_pos = 0.0
            
            # 为每个角色的每句话创建时间点
            for item in subtitle_data:
                # 使用原始的带冒号时间格式
                start_time = item["start"]
                end_time = item["end"]
                text = item["字幕"]
                
                # 模拟分句处理 - 实际应该从模型中获取
                # 这里简单地按标点符号拆分
                import re
                # 将文本拆分为句子 (中文标点和英文标点)
                sentences = re.split(r'([,，.。!！?？;；])', text)
                # 过滤空字符串并重组句子和标点
                sentences = [s + next_s for s, next_s in zip(sentences[::2], sentences[1::2] + [""])] if len(sentences) > 1 else [text]
                sentences = [s for s in sentences if s.strip()]
                
                if not sentences:  # 如果没有成功分句，就使用原始文本
                    sentences = [text]
                
                # 计算每个子句的时长
                total_duration = self._parse_time_format(end_time) - self._parse_time_format(start_time)
                sentence_duration = total_duration / len(sentences) if sentences else total_duration
                
                # 为每个子句生成时间点
                for i, sentence in enumerate(sentences):
                    if not sentence.strip():  # 跳过空句
                        continue
                    
                    sub_start = self._parse_time_format(start_time) + i * sentence_duration
                    sub_end = sub_start + sentence_duration
                    
                    sub_start_formatted = self._seconds_to_time_format(sub_start)
                    sub_end_formatted = self._seconds_to_time_format(sub_end)
                    
                    time_line = f">> {sub_start_formatted}-{sub_end_formatted}"
                    text_line = f">> {sentence}"
                    
                    simplified_subtitles.append(time_line)
                    simplified_subtitles.append(text_line)
            
            # 连接为字符串
            simplified_subtitle_str = "\n".join(simplified_subtitles)
            print(f"[IndexTTS Pro] Generated simplified subtitle format with processed sentences")
            
            # 最终返回ComfyUI格式的音频数据、种子、JSON字幕和简化字幕
            return ({"waveform": audio_tensor, "sample_rate": final_audio[1]}, seed, subtitle_json, simplified_subtitle_str)
            
        except Exception as e:
            import traceback
            print(f"[IndexTTS Pro] Generation failed: {e}")
            print(traceback.format_exc())
            raise RuntimeError(f"Multi-voice generation failed: {e}")
