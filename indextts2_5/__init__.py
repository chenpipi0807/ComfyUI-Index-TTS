"""
IndexTTS-2.5 integration package for ComfyUI
提供 IndexTTS-2.5 的模型加载与推理封装（多语言、情感控制、语速控制）。
"""

from .model_loader import IndexTTS25Loader
from .infer import IndexTTS25Engine

__all__ = ["IndexTTS25Loader", "IndexTTS25Engine"]
