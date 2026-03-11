import os
import sys
import torch
import numpy as np
import tempfile
import json
import time

# # 确保模块可被找到
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # 确保导入路径正确
# package_root = os.path.dirname(os.path.dirname(__file__))
# if package_root not in sys.path:
#     sys.path.append(package_root)

# 导入工具函数
from ..utils.audio_utils import load_audio, save_audio, get_temp_file

# 导入ComfyUI folder_paths用于获取模型目录
import folder_paths

# # 添加索引TTS路径
# INDEX_TTS_PATH = os.path.join(folder_paths.models_dir, "Index-TTS")
# sys.path.append(INDEX_TTS_PATH)

# 尝试加载IndexTTS的必要依赖
try:
    # 如果直接导入indextts包失败，我们将模拟其核心功能
    # 因为原始代码可能不会直接可用，我们在这里实现一个简单的包装器
    class IndexTTS:
        def __init__(self, model_dir=None, cfg_path=None):
            """
            初始化IndexTTS模型
            
            参数:
                model_dir: 模型目录
                cfg_path: 配置文件路径
            """
            import importlib.util
            import torch
            import os
            
            self.model_dir = model_dir if model_dir else INDEX_TTS_PATH
            self.cfg_path = cfg_path if cfg_path else os.path.join(self.model_dir, "config.yaml")
            
            # 检查模型文件是否存在
            required_files = [
                "bigvgan_discriminator.pth", "bigvgan_generator.pth", 
                "bpe.model", "dvae.pth", "gpt.pth", 
                "unigram_12000.vocab", "config.yaml"
            ]
            
            for file in required_files:
                if not os.path.exists(os.path.join(self.model_dir, file)):
                    raise FileNotFoundError(f"模型文件 {file} 未找到，请确保已下载模型文件到 {self.model_dir}")
            
            # 加载Config
            import yaml
            with open(self.cfg_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
                
            print(f"成功初始化IndexTTS模型, 模型目录: {self.model_dir}")
            
            # 尝试导入indextts模块
            try:
                from . import indextts
                self.model = indextts.infer.IndexTTS(model_dir=self.model_dir, cfg_path=self.cfg_path)
                self.use_original = True
                print("使用原始IndexTTS模块")
            except ImportError:
                # 如果无法导入，使用自定义实现
                print("无法导入原始IndexTTS模块，使用自定义实现")
                self.use_original = False
                self._init_pipeline()
                
        def _init_pipeline(self):
            """初始化语音合成管道"""
            # 这里应该加载所有必要的模型组件
            # 由于完整实现较为复杂，这里是一个简化的示例
            pass
            
        def infer(self, reference_voice, text, output_path, language="auto", speed=1.0):
            """
            使用参考声音生成语音
            
            参数:
                reference_voice: 参考声音文件路径
                text: 要合成的文本
                output_path: 输出音频文件路径
                language: 语言代码
                speed: 语速，默认1.0
            """
            if self.use_original:
                # 使用原始IndexTTS实现
                self.model.infer(reference_voice, text, output_path, language=language, speed=speed)
            else:
                # 使用自定义实现 - 这里是一个简单的占位实现
                # 在实际应用中，应该完整实现音频合成逻辑
                raise NotImplementedError("自定义实现尚未完成，请安装原始的IndexTTS模块")
            
            return output_path

except ImportError as e:
    print(f"导入IndexTTS相关模块失败: {e}")
    print("请确保已安装所有必要的依赖")


class IndexTTSNode:
    """
    ComfyUI的IndexTTS节点，用于文本到语音合成
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "你好，我是IndexTTS语音合成系统。"}),
                "reference_audio": ("AUDIO",),
                "language": (["auto", "zh", "en", "ja", "ko"], {"default": "auto"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("synthesized_audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "audio/tts"
    
    def __init__(self):
        # 获取模型目录
        self.model_dir = INDEX_TTS_PATH
        self.cfg_path = os.path.join(self.model_dir, "config.yaml")
        
        # 检查模型目录是否存在
        if not os.path.exists(self.model_dir):
            print(f"\033[91m错误: 未找到模型目录 {self.model_dir}\033[0m")
            print(f"\033[91m请确保已下载模型文件到 {self.model_dir}\033[0m")
        
        # 延迟初始化模型，直到实际需要时
        self.tts_model = None
    
    def _init_model(self):
        """初始化TTS模型（延迟加载）"""
        if self.tts_model is None:
            try:
                self.tts_model = IndexTTS(model_dir=self.model_dir, cfg_path=self.cfg_path)
                print(f"模型已成功加载，模型目录: {self.model_dir}")
            except Exception as e:
                print(f"初始化TTS模型失败: {e}")
                raise RuntimeError(f"初始化TTS模型失败: {e}")
    
    def generate_speech(self, text, reference_audio, language="auto", speed=1.0):
        """
        生成语音的主函数
        
        参数:
            text: 要合成的文本
            reference_audio: 参考音频元组 (音频数据, 采样率)
            language: 语言代码
            speed: 语速
            
        返回:
            tuple: (音频数据, 采样率)
        """
        # 初始化模型
        self._init_model()
        
        try:
            # 解析参考音频
            audio_data, sample_rate = reference_audio
            
            # 保存参考音频到临时文件
            ref_path = get_temp_file(".wav")
            save_audio(audio_data, sample_rate, ref_path)
            
            # 创建输出临时文件
            output_path = get_temp_file(".wav")
            
            # 调用TTS引擎生成语音
            self.tts_model.infer(
                ref_path, 
                text, 
                output_path,
                language=language,
                speed=speed
            )
            
            # 读取生成的音频
            result_audio, result_sr = load_audio(output_path, target_sr=sample_rate)
            
            # 清理临时文件
            try:
                os.unlink(ref_path)
                os.unlink(output_path)
            except:
                pass
            
            return ((result_audio, result_sr),)
            
        except Exception as e:
            print(f"生成语音失败: {e}")
            # 返回一个空音频（1秒静音）作为错误处理
            empty_audio = np.zeros(sample_rate, dtype=np.float32)
            return ((empty_audio, sample_rate),)
            
    @classmethod
    def IS_CHANGED(cls, text, reference_audio, language, speed):
        # 用于判断节点输入是否变化的辅助函数
        # 这里使用当前时间戳确保每次运行都会重新生成
        return time.time()
