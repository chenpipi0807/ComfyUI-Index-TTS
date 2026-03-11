import os
import sys
import numpy as np
import torch
import random
import tempfile
import soundfile as sf
import time
from pathlib import Path

# # 确保当前目录在导入路径中
# current_dir = os.path.dirname(os.path.abspath(__file__))
# if current_dir not in sys.path:
#     sys.path.append(current_dir)

# 导入TTS模型
from .tts_models import IndexTTSModel
# 导入TTS2引擎（用于IndexTTS-2支持）
try:
    from .indextts2 import IndexTTS2Loader, IndexTTS2Engine
    HAS_TTS2 = True
except ImportError:
    HAS_TTS2 = False


# IndexTTS节点
class IndexTTSNode:
    """
    ComfyUI的IndexTTS节点，用于文本到语音合成
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "你好，这是一段测试文本。"}),
                "reference_audio": ("AUDIO", ),
                "model_version": (["Index-TTS", "IndexTTS-1.5", "IndexTTS-2"], {"default": "Index-TTS"}),
                "language": (["auto", "zh", "en"], {"default": "auto"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.5, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 30, "min": 1, "max": 100, "step": 1}),
                "repetition_penalty": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 15.0, "step": 0.5}),
                "length_penalty": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "max_mel_tokens": ("INT", {"default": 600, "min": 100, "max": 1500, "step": 50}),
                "sentence_split": (["auto", "manual"], {"default": "auto"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "INT", "STRING",)
    RETURN_NAMES = ("audio", "seed", "SimplifiedSubtitle",)
    FUNCTION = "generate_speech"
    CATEGORY = "audio"
    
    def __init__(self):
        # 根路径
        self.models_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
        # 可用模型版本
        self.model_versions = {
            "Index-TTS": os.path.join(self.models_root, "Index-TTS"),
            "IndexTTS-1.5": os.path.join(self.models_root, "IndexTTS-1.5"),
            "IndexTTS-2": os.path.join(self.models_root, "IndexTTS-2")
        }
        # TTS2 引擎（延迟初始化）
        self.tts2_loader = None
        self.tts2_engine = None
        # 默认使用 Index-TTS 版本
        self.current_version = "Index-TTS"
        self.model_dir = self.model_versions[self.current_version]
        self.tts_model = None
        
        print(f"[IndexTTS] 初始化节点，可用模型版本: {list(self.model_versions.keys())}")
        print(f"[IndexTTS] 默认模型目录: {self.model_dir}")
        
        # 检查模型目录是否存在
        for version, directory in self.model_versions.items():
            if os.path.exists(directory):
                model_files = os.listdir(directory)
                print(f"[IndexTTS] 模型 {version} 目录内容: {len(model_files)} 个文件")
            else:
                print(f"[IndexTTS] 警告: 模型 {version} 目录不存在: {directory}")
    
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
                print(f"[IndexTTS] 切换到模型版本: {model_version}, 目录: {self.model_dir}")
            else:
                print(f"[IndexTTS] 警告: 未知模型版本 {model_version}，使用默认版本 {self.current_version}")
            
            # 如果已有模型，先释放资源
            if self.tts_model is not None:
                print(f"[IndexTTS] 卸载现有模型...")
                self.tts_model = None
                # 强制垃圾回收
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            print(f"[IndexTTS] 开始加载模型版本: {self.current_version}...")
            # 检查必要的模型文件
            required_files = ["gpt.pth", "config.yaml"]
            missing_files = []
            for file in required_files:
                file_path = os.path.join(self.model_dir, file)
                if not os.path.exists(file_path):
                    missing_files.append(file)
                else:
                    file_size = os.path.getsize(file_path) / (1024*1024)  # 转换为MB
                    print(f"[IndexTTS] 找到模型文件: {file} ({file_size:.2f}MB)")
            
            if missing_files:
                error_msg = f"模型 {self.current_version} 缺少必要的文件: {', '.join(missing_files)}"
                print(f"[IndexTTS] 错误: {error_msg}")
                raise FileNotFoundError(error_msg)
                
            try:
                # 记录开始加载时间
                start_time = time.time()
                
                # 使用tts_models.py中的IndexTTSModel实现
                self.tts_model = IndexTTSModel(model_dir=self.model_dir)
                
                # 记录加载完成时间
                load_time = time.time() - start_time
                print(f"[IndexTTS] 模型 {self.current_version} 已成功加载，耗时: {load_time:.2f}秒")
                
                # 输出模型基本信息
                if hasattr(self.tts_model, 'config'):
                    print(f"[IndexTTS] 模型配置:")  
                    for key, value in vars(self.tts_model.config).items():
                        if not key.startswith('_') and not callable(value):
                            print(f"[IndexTTS]   - {key}: {value}")
                            
                # 检查模型是否有必要的组件
                components = [attr for attr in dir(self.tts_model) if not attr.startswith('_') and not callable(getattr(self.tts_model, attr))]
                print(f"[IndexTTS] 模型组件: {components}")
                
            except Exception as e:
                import traceback
                print(f"[IndexTTS] 初始化模型 {self.current_version} 失败: {e}")
                print(f"[IndexTTS] 错误详情:")
                traceback.print_exc()
                raise RuntimeError(f"初始化IndexTTS模型 {self.current_version} 失败: {e}")
    
    def _process_audio_for_tts2(self, audio):
        """处理音频格式用于TTS2引擎"""
        if isinstance(audio, dict) and "waveform" in audio and "sample_rate" in audio:
            wave = audio["waveform"]
            sr = int(audio["sample_rate"])
            if isinstance(wave, torch.Tensor):
                if wave.dim() == 3:
                    wave = wave[0, 0].detach().cpu().numpy()
                elif wave.dim() == 1:
                    wave = wave.detach().cpu().numpy()
                else:
                    wave = wave.flatten().detach().cpu().numpy()
            elif isinstance(wave, np.ndarray):
                if wave.ndim == 3:
                    wave = wave[0, 0]
                elif wave.ndim == 2:
                    wave = wave[0]
            return wave.astype(np.float32), sr
        elif isinstance(audio, tuple) and len(audio) == 2:
            wave, sr = audio
            if isinstance(wave, torch.Tensor):
                wave = wave.detach().cpu().numpy()
            return wave.astype(np.float32), int(sr)
        else:
            raise ValueError("AUDIO input must be ComfyUI dict or (wave, sr)")
    
    def generate_speech(self, text, reference_audio, model_version="Index-TTS", language="auto", speed=1.0, seed=0, temperature=1.0, top_p=0.8, top_k=30, repetition_penalty=10.0, length_penalty=0.0, num_beams=3, max_mel_tokens=600, sentence_split="auto"):
        """
        生成语音的主函数
        
        参数:
            text: 要转换为语音的文本
            reference_audio: 参考音频元组 (audio_data, sample_rate)
            language: 文本语言 (auto, zh, en)
            speed: 语速因子，1.0为正常语速
            
        返回:
            audio: 生成的音频元组 (audio_data, sample_rate)
        """
        # 如果选择了 IndexTTS-2，使用 TTS2 引擎
        if model_version == "IndexTTS-2":
            return self._generate_speech_tts2(text, reference_audio, seed, temperature, top_p, top_k, repetition_penalty, length_penalty, num_beams, max_mel_tokens)
        
        try:
            # 延迟加载模型或切换模型版本
            if self.tts_model is None or model_version != self.current_version:
                self._init_model(model_version=model_version)
            
            # 处理ComfyUI的音频格式
            processed_audio = None
            
            print(f"[IndexTTS] 接收到参考音频，类型: {type(reference_audio)}")
            
            # 如果是ComfyUI标准格式
            if isinstance(reference_audio, dict) and "waveform" in reference_audio and "sample_rate" in reference_audio:
                waveform = reference_audio["waveform"]
                sample_rate = reference_audio["sample_rate"]
                
                print(f"[IndexTTS] 参考音频格式: ComfyUI字典格式, sample_rate={sample_rate}")
                print(f"[IndexTTS] waveform类型: {type(waveform)}, 形状: {waveform.shape if hasattr(waveform, 'shape') else '未知'}")
                
                # 如果waveform是torch.Tensor，转换为numpy
                if isinstance(waveform, torch.Tensor):
                    waveform = waveform.cpu().numpy()
                    print(f"[IndexTTS] 已将waveform从tensor转换为numpy, 形状: {waveform.shape if hasattr(waveform, 'shape') else '未知'}")
                
                processed_audio = (waveform, sample_rate)
                
            # 如果已经是元组格式
            elif isinstance(reference_audio, tuple) and len(reference_audio) == 2:
                audio_data, sample_rate = reference_audio
                processed_audio = reference_audio
                print(f"[IndexTTS] 参考音频格式: 元组格式, sample_rate={sample_rate}")
                print(f"[IndexTTS] audio_data类型: {type(audio_data)}, 形状: {audio_data.shape if hasattr(audio_data, 'shape') else '未知'}")
            
            # 如果都不是，报错
            if processed_audio is None:
                print(f"[IndexTTS] 错误: 参考音频格式不正确: {type(reference_audio)}")
                if isinstance(reference_audio, dict):
                    print(f"[IndexTTS] 参考音频字典包含键: {reference_audio.keys()}")
                raise ValueError("参考音频格式不支持，应为 AUDIO 类型")
            
            # 创建临时输出文件
            temp_dir = tempfile.gettempdir()
            temp_output = os.path.join(temp_dir, f"tts_output_{int(time.time())}.wav")
            
            # 设置随机种子以确保结果可重复性
            if seed != 0:
                print(f"[IndexTTS] 设置随机种子: {seed}")
                # 保存当前随机状态
                numpy_state = np.random.get_state()
                torch_state = torch.get_rng_state()
                python_state = random.getstate()
                if torch.cuda.is_available():
                    torch_cuda_state = torch.cuda.get_rng_state()
                
                # 设置新的随机种子
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            
            print(f"[IndexTTS] 开始生成语音，使用模型: {model_version}，文本长度: {len(text)}，语言: {language}，语速: {speed}，种子: {seed}")
            print(f"[IndexTTS] 文本内容: '{text[:100]}{'...' if len(text) > 100 else ''}'")  # 只打印部分文本  # 只打印部分文本
            
            # 记录推理开始时间
            infer_start_time = time.time()
            
            # 调用TTS模型生成语音
            try:
                # 简化调用，只使用基本参数
                # 因为我们的wrapper需要要兼容原始模型接口
                result = self.tts_model.infer(
                    reference_audio=processed_audio, 
                    text=text, 
                    output_path=None,  # 不保存文件，直接返回数据
                    language=language,
                    speed=speed,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_beams=num_beams,
                    max_mel_tokens=max_mel_tokens
                )
            except Exception as e:
                print(f"[IndexTTS] 调用模型失败: {e}")
                raise
            
            # 记录推理完成时间
            infer_time = time.time() - infer_start_time
            print(f"[IndexTTS] 语音生成完成，耗时: {infer_time:.2f}秒")
            
            # 如果设置了随机种子，恢复之前的随机状态
            if seed != 0:
                # 恢复随机状态
                random.setstate(python_state)
                np.random.set_state(numpy_state)
                torch.set_rng_state(torch_state)
                if torch.cuda.is_available():
                    torch.cuda.set_rng_state(torch_cuda_state)
            
            # 处理返回结果
            print(f"[IndexTTS] 模型返回结果类型: {type(result)}")
            
            if isinstance(result, tuple) and len(result) == 2:
                # 返回格式: (sample_rate, audio_data)
                sample_rate, audio_data = result
                print(f"[IndexTTS] 生成的音频样本率: {sample_rate}Hz")
                print(f"[IndexTTS] 生成的音频数据类型: {type(audio_data)}")
                print(f"[IndexTTS] 生成的音频形状: {audio_data.shape if hasattr(audio_data, 'shape') else '未知'}")
                
                # 计算音频长度（秒）
                if hasattr(audio_data, 'shape'):
                    audio_duration = audio_data.shape[-1] / sample_rate
                    print(f"[IndexTTS] 生成的音频长度: {audio_duration:.2f}秒")
                
                # 转换为ComfyUI期望的格式
                # 如果是numpy数组，转换为torch tensor
                if isinstance(audio_data, np.ndarray):
                    print(f"[IndexTTS] 将numpy数组转换为torch tensor")
                    audio_data = torch.tensor(audio_data, dtype=torch.float32)
                    
                print(f"[IndexTTS] 转换前的张量维度: {audio_data.dim()}")
                    
                # 确保音频数据是3D张量 [batch, channels, samples]
                if audio_data.dim() == 1:
                    # [samples] -> [1, 1, samples]
                    audio_data = audio_data.unsqueeze(0).unsqueeze(0)
                    print(f"[IndexTTS] 1D张量调整为3D张量: [1, 1, {audio_data.shape[-1]}]")
                elif audio_data.dim() == 2:
                    # [batch, samples] -> [batch, 1, samples]
                    audio_data = audio_data.unsqueeze(1)
                    print(f"[IndexTTS] 2D张量调整为3D张量: [{audio_data.shape[0]}, 1, {audio_data.shape[-1]}]")
                    
                print(f"[IndexTTS] 最终张量形状: {audio_data.shape}")
                    
                # 返回字典格式，符合ComfyUI音频节点期望
                audio_dict = {
                    "waveform": audio_data,
                    "sample_rate": sample_rate
                }
                
                # 生成SimplifiedSubtitle
                try:
                    # 计算总音频长度
                    total_duration = audio_data.shape[-1] / sample_rate
                    
                    # 模拟分句处理 - 按标点符号拆分文本
                    import re
                    sentences = re.split(r'([,，.。!！?？;；])', text)
                    # 过滤空字符串并重组句子和标点
                    sentences = [s + next_s for s, next_s in zip(sentences[::2], sentences[1::2] + [""])] if len(sentences) > 1 else [text]
                    sentences = [s for s in sentences if s.strip()]
                    
                    if not sentences:  # 如果没有成功分句，就使用原始文本
                        sentences = [text]
                    
                    # 计算每个子句的时长
                    sentence_duration = total_duration / len(sentences) if sentences else total_duration
                    
                    # 生成简化字幕格式
                    simplified_subtitles = []
                    current_time = 0.0
                    
                    for i, sentence in enumerate(sentences):
                        if not sentence.strip():  # 跳过空句
                            continue
                        
                        start_time = current_time
                        end_time = current_time + sentence_duration
                        
                        start_formatted = self._seconds_to_time_format(start_time)
                        end_formatted = self._seconds_to_time_format(end_time)
                        
                        time_line = f">> {start_formatted}-{end_formatted}"
                        text_line = f">> {sentence.strip()}"
                        
                        simplified_subtitles.append(time_line)
                        simplified_subtitles.append(text_line)
                        
                        current_time = end_time
                    
                    # 连接为字符串
                    simplified_subtitle_str = "\n".join(simplified_subtitles)
                    print(f"[IndexTTS] 生成SimplifiedSubtitle，包含 {len(sentences)} 个句子")
                    
                except Exception as e:
                    print(f"[IndexTTS] 生成SimplifiedSubtitle失败: {e}")
                    # simplified_subtitle_str = f">> 0:00.000-{self._seconds_to_time_format(total_duration)}\n>> {text}"
                    raise e
                
                return (audio_dict, seed, simplified_subtitle_str)
            else:
                print(f"错误: 意外的返回格式: {type(result)}")
                raise ValueError(f"TTS模型返回了意外的格式: {type(result)}")
                
        except Exception as e:
            print(f"[IndexTTS] 生成语音失败: {e}")
            
            raise e
            # [#] 节点作为工作流的一个环节, 如果隐藏错误, 可能很难被发现, 因此不如 raise

            # # 生成一个简单的错误提示音频
            # sample_rate = 24000
            # duration = 1.0  # 1秒
            # t = np.linspace(0, duration, int(sample_rate * duration))
            # signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440Hz警告音
            # print(f"[IndexTTS] 生成警告音频作为错误处理")
            
            # # 转换为ComfyUI音频格式
            # signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
            # audio_dict = {
            #     "waveform": signal_tensor,
            #     "sample_rate": sample_rate
            # }
            
            # return (audio_dict, seed, "")
    
    def _generate_speech_tts2(self, text, reference_audio, seed, temperature, top_p, top_k, repetition_penalty, length_penalty, num_beams, max_mel_tokens):
        """使用 IndexTTS-2 引擎生成语音"""
        if not HAS_TTS2:
            raise RuntimeError("IndexTTS-2 模块未安装，无法使用 IndexTTS-2 模型")
        
        try:
            # 延迟初始化 TTS2 引擎
            if self.tts2_loader is None:
                print("[IndexTTS] 初始化 IndexTTS-2 引擎...")
                self.tts2_loader = IndexTTS2Loader()
                self.tts2_engine = IndexTTS2Engine(self.tts2_loader)
            
            # 处理参考音频
            ref = self._process_audio_for_tts2(reference_audio)
            
            print(f"[IndexTTS] 使用 IndexTTS-2 生成语音，文本长度: {len(text)}")
            
            # 调用 TTS2 引擎
            sr, wave, subtitle = self.tts2_engine.generate(
                text=text,
                reference_audio=ref,
                mode="Auto",
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                max_mel_tokens=max_mel_tokens if max_mel_tokens <= 1815 else 1815,
                seed=seed,
                return_subtitles=True,
            )
            
            # 转换为 ComfyUI 格式
            wave_tensor = torch.tensor(wave, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            audio_dict = {
                "waveform": wave_tensor,
                "sample_rate": int(sr)
            }
            
            # 生成简化字幕
            total_duration = len(wave) / sr
            import re
            sentences = re.split(r'([,，.。!！?？;；])', text)
            sentences = [s + next_s for s, next_s in zip(sentences[::2], sentences[1::2] + [""])] if len(sentences) > 1 else [text]
            sentences = [s for s in sentences if s.strip()]
            if not sentences:
                sentences = [text]
            
            sentence_duration = total_duration / len(sentences)
            simplified_subtitles = []
            current_time = 0.0
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                start_formatted = self._seconds_to_time_format(current_time)
                end_formatted = self._seconds_to_time_format(current_time + sentence_duration)
                simplified_subtitles.append(f">> {start_formatted}-{end_formatted}")
                simplified_subtitles.append(f">> {sentence.strip()}")
                current_time += sentence_duration
            
            simplified_subtitle_str = "\n".join(simplified_subtitles)
            
            print(f"[IndexTTS] IndexTTS-2 语音生成完成，长度: {total_duration:.2f}秒")
            return (audio_dict, seed, simplified_subtitle_str)
            
        except Exception as e:
            import traceback
            print(f"[IndexTTS] IndexTTS-2 生成失败: {e}")
            traceback.print_exc()
            
            # 生成错误提示音
            sample_rate = 24000
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            audio_dict = {"waveform": signal_tensor, "sample_rate": sample_rate}
            return (audio_dict, seed, "")
