import os
import sys
import numpy as np
import tempfile
import base64

# # 确保模块可被找到
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # 确保导入路径正确
# package_root = os.path.dirname(os.path.dirname(__file__))
# if package_root not in sys.path:
#     sys.path.append(package_root)

# 导入工具函数
from ..utils.audio_utils import save_audio

class AudioPreviewNode:
    """
    音频预览节点，用于在ComfyUI界面中预览和播放音频
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "audio"}),
                "autoplay": (["True", "False"], {"default": "True"}),
            },
            "optional": {
                "save_path": ("STRING", {"default": ""})
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "preview_audio"
    OUTPUT_NODE = True
    CATEGORY = "audio"

    def preview_audio(self, audio, filename_prefix="audio", autoplay="True", save_path=""):
        """
        处理并预览音频
        
        参数:
            audio: 音频数据元组 (音频数据, 采样率)
            filename_prefix: 输出文件名前缀
            autoplay: 是否自动播放
            save_path: 可选的保存路径
            
        返回:
            dict: UI显示字典
        """
        audio_data, sample_rate = audio
        
        # 保存音频到临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            save_audio(audio_data, sample_rate, temp_file.name)
            temp_path = temp_file.name
            
            # 如果提供了保存路径，将音频保存到指定位置
            if save_path:
                try:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    final_path = os.path.join(save_path, f"{filename_prefix}.wav")
                    save_audio(audio_data, sample_rate, final_path)
                    save_message = f"音频已保存到: {final_path}"
                except Exception as e:
                    save_message = f"保存音频失败: {e}"
            else:
                save_message = ""
                
            # 获取音频时长
            duration = len(audio_data) / sample_rate
            
            # 生成Web界面HTML代码
            autoplay_attr = "autoplay" if autoplay == "True" else ""
            
            # 获取文件的相对URL (适用于ComfyUI的文件服务)
            import urllib.parse
            
            # 使用临时文件URL路径
            filename = os.path.basename(temp_path)
            file_url = f"file/{urllib.parse.quote(filename)}"
            
            # 创建HTML音频播放器
            html_embed = f"""
            <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin: 10px 0;">
                <h3 style="margin: 0 0 10px 0;">音频预览</h3>
                <audio controls {autoplay_attr} style="width: 100%;">
                    <source src="/view?filename={file_url}" type="audio/wav">
                    您的浏览器不支持音频播放
                </audio>
                <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                    <span>采样率: {sample_rate} Hz</span>
                    <span>时长: {duration:.2f} 秒</span>
                </div>
                {f'<div style="margin-top: 5px;">{save_message}</div>' if save_message else ''}
            </div>
            """
            
        # 返回UI元素
        return {"ui": {"audio": html_embed}}
        
    @classmethod
    def IS_CHANGED(cls, audio, filename_prefix, autoplay, save_path=""):
        # 用于判断节点输入是否变化的辅助函数
        # 对于输出节点，我们总是返回True确保UI更新
        return True
