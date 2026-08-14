import json
import torch
import numpy as np
from typing import Any, Tuple

from .indextts2_5 import IndexTTS25Loader, IndexTTS25Engine

# Global shared loader/engine to avoid duplicating model weights across nodes
_GLOBAL_LOADER = IndexTTS25Loader()
_GLOBAL_ENGINE = IndexTTS25Engine(_GLOBAL_LOADER)
_EMO_VECTOR_BIAS = (0.75, 0.70, 0.80, 0.80, 0.75, 0.75, 0.55, 0.45)

# IndexTTS-2.5 官方支持语种（ZH/EN 混合为额外常用项）
LANG_CHOICES = ["ZH", "EN", "JA", "ES", "AR", "ZH/EN"]


def _normalize_emo_vector_like_demo(vec):
    tmp = np.array([max(0.0, float(x)) for x in vec], dtype=np.float32)
    tmp = tmp * np.array(_EMO_VECTOR_BIAS, dtype=np.float32)
    total = float(tmp.sum())
    if total > 0.8:
        tmp = tmp * (0.8 / total)
    return tmp.tolist()


class _IndexTTS25BaseMixin:
    @staticmethod
    def _process_audio_input(audio: Any) -> Tuple[np.ndarray, int]:
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

    @classmethod
    def _base_inputs(cls):
        return {
            "text": ("STRING", {"multiline": True, "default": "大家好，这是 IndexTTS 2.5 的语音合成演示。"}),
            "reference_audio": ("AUDIO",),
            # 2.5 新增：多语言选择（支持跨语种克隆）
            "lang": (LANG_CHOICES, {"default": "ZH"}),
            # 2.5 新增：语速控制，>1 变慢，<1 变快（0.5 - 2.0）
            "duration_factor": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
        }

    @classmethod
    def _common_optional(cls):
        return {
            # Advanced generation parameters
            "do_sample_mode": (["off", "on"], {"default": "on"}),
            "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.05}),
            "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
            "top_k": ("INT", {"default": 30, "min": 0, "max": 100, "step": 1}),
            "num_beams": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
            "repetition_penalty": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 10.0, "step": 0.1}),
            "length_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
            "max_mel_tokens": ("INT", {"default": 1500, "min": 50, "max": 1815, "step": 10}),
            "max_tokens_per_sentence": ("INT", {"default": 120, "min": 0, "max": 600, "step": 5}),
            "interval_silence_ms": ("INT", {"default": 200, "min": 0, "max": 2000, "step": 50}),
            "text_normalization": ("BOOLEAN", {"default": True}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            # External cache control dict from utility node
            "cache_control": ("DICT", {"default": None}),
        }

    def _do_generate(self, engine: IndexTTS25Engine, **kwargs):
        sr, wave, sub = engine.generate(**kwargs)
        wave_t = torch.tensor(wave, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        audio = {"waveform": wave_t, "sample_rate": int(sr)}
        return audio, kwargs.get("seed", 0), (sub or "")

    def _maybe_unload(self, cache_control):
        try:
            keep = bool(cache_control.get("keep_cached")) if isinstance(cache_control, dict) else False
            if not keep:
                self.loader.unload_tts()
        except Exception:
            pass


class IndexTTS25BaseNode(_IndexTTS25BaseMixin):
    """
    IndexTTS-2.5 基础节点：零样本音色克隆 + 多语言 + 语速控制。
    发音控制直接在 text 里写标注：<行|XING2> / <minute|M AY0 . N UW1 T> / <上手|じょうず>
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": cls._base_inputs(), "optional": cls._common_optional()}

    RETURN_TYPES = ("AUDIO", "INT", "STRING")
    RETURN_NAMES = ("audio", "seed", "subtitle")
    FUNCTION = "generate"
    CATEGORY = "audio"
    DESCRIPTION = "IndexTTS-2.5 基础合成：多语言(ZH/EN/JA/ES/AR)、语速控制、发音标注(<字|XING2>)"

    def __init__(self):
        self.loader = _GLOBAL_LOADER
        self.engine = _GLOBAL_ENGINE

    def generate(self, text, reference_audio, lang, duration_factor,
                 do_sample_mode="on", temperature=0.8, top_p=0.8, top_k=30, num_beams=3,
                 repetition_penalty=10.0, length_penalty=0.0, max_mel_tokens=1500,
                 max_tokens_per_sentence=120, interval_silence_ms=200, text_normalization=True,
                 seed=0, cache_control=None):
        ref = self._process_audio_input(reference_audio)
        out = self._do_generate(
            self.engine,
            text=text, reference_audio=ref, lang=lang, duration_factor=duration_factor,
            do_sample=(do_sample_mode == "on"), temperature=temperature, top_p=top_p, top_k=top_k, num_beams=num_beams,
            repetition_penalty=repetition_penalty, length_penalty=length_penalty,
            max_mel_tokens=max_mel_tokens, max_tokens_per_sentence=max_tokens_per_sentence,
            interval_silence=interval_silence_ms, text_normalization=text_normalization,
            emo_text=None, emo_ref_audio=None, emo_vector=None, emo_weight=1.0,
            seed=seed, return_subtitles=True,
        )
        self._maybe_unload(cache_control)
        return out


class IndexTTS25EmotionAudioNode(_IndexTTS25BaseMixin):
    """IndexTTS-2.5 情感参考音频节点：用另一段音频控制情感，emotion_weight 调节强度。"""

    @classmethod
    def INPUT_TYPES(cls):
        opt = cls._common_optional().copy()
        opt.update({
            "emo_ref_audio": ("AUDIO",),
            "emotion_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        })
        return {"required": cls._base_inputs(), "optional": opt}

    RETURN_TYPES = ("AUDIO", "INT", "STRING")
    RETURN_NAMES = ("audio", "seed", "subtitle")
    FUNCTION = "generate"
    CATEGORY = "audio"
    DESCRIPTION = "IndexTTS-2.5 情感参考音频：音色与情感解耦，可用不同人的情感音频"

    def __init__(self):
        self.loader = _GLOBAL_LOADER
        self.engine = _GLOBAL_ENGINE

    def generate(self, text, reference_audio, lang, duration_factor, emo_ref_audio,
                 emotion_weight=1.0,
                 do_sample_mode="on", temperature=0.8, top_p=0.8, top_k=30, num_beams=3,
                 repetition_penalty=10.0, length_penalty=0.0, max_mel_tokens=1500,
                 max_tokens_per_sentence=120, interval_silence_ms=200, text_normalization=True,
                 seed=0, cache_control=None):
        ref = self._process_audio_input(reference_audio)
        emo_ref = self._process_audio_input(emo_ref_audio)
        out = self._do_generate(
            self.engine,
            text=text, reference_audio=ref, lang=lang, duration_factor=duration_factor,
            do_sample=(do_sample_mode == "on"), temperature=temperature, top_p=top_p, top_k=top_k, num_beams=num_beams,
            repetition_penalty=repetition_penalty, length_penalty=length_penalty,
            max_mel_tokens=max_mel_tokens, max_tokens_per_sentence=max_tokens_per_sentence,
            interval_silence=interval_silence_ms, text_normalization=text_normalization,
            emo_text=None, emo_ref_audio=emo_ref, emo_vector=None, emo_weight=float(emotion_weight),
            seed=seed, return_subtitles=True,
        )
        self._maybe_unload(cache_control)
        return out


class IndexTTS25EmotionVectorNode(_IndexTTS25BaseMixin):
    """IndexTTS-2.5 情感向量节点：8 维情感滑条 [高兴/愤怒/悲伤/恐惧/反感/低落/惊讶/平静]。"""

    @classmethod
    def INPUT_TYPES(cls):
        opt = cls._common_optional().copy()
        # 8 sliders in-node
        opt.update({
            "Happy": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "Angry": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "Sad": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "Fear": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "Hate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "Low": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "Surprise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "Neutral": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "use_random": ("BOOLEAN", {"default": False}),
        })
        return {"required": cls._base_inputs(), "optional": opt}

    RETURN_TYPES = ("AUDIO", "INT", "STRING")
    RETURN_NAMES = ("audio", "seed", "subtitle")
    FUNCTION = "generate"
    CATEGORY = "audio"
    DESCRIPTION = "IndexTTS-2.5 情感向量：8 维情感强度自由配比"

    def __init__(self):
        self.loader = _GLOBAL_LOADER
        self.engine = _GLOBAL_ENGINE

    def generate(self, text, reference_audio, lang, duration_factor,
                 Happy=0.0, Angry=0.0, Sad=0.0, Fear=0.0, Hate=0.0, Low=0.0, Surprise=0.0, Neutral=0.0,
                 use_random=False,
                 do_sample_mode="on", temperature=0.8, top_p=0.8, top_k=30, num_beams=3,
                 repetition_penalty=10.0, length_penalty=0.0, max_mel_tokens=1500,
                 max_tokens_per_sentence=120, interval_silence_ms=200, text_normalization=True,
                 seed=0, cache_control=None, Love=None):
        ref = self._process_audio_input(reference_audio)
        low_value = Low if Love is None else Love
        vec = [Happy, Angry, Sad, Fear, Hate, low_value, Surprise, Neutral]
        emo_vec = _normalize_emo_vector_like_demo(vec)
        out = self._do_generate(
            self.engine,
            text=text, reference_audio=ref, lang=lang, duration_factor=duration_factor,
            do_sample=(do_sample_mode == "on"), temperature=temperature, top_p=top_p, top_k=top_k, num_beams=num_beams,
            repetition_penalty=repetition_penalty, length_penalty=length_penalty,
            max_mel_tokens=max_mel_tokens, max_tokens_per_sentence=max_tokens_per_sentence,
            interval_silence=interval_silence_ms, text_normalization=text_normalization,
            emo_text=None, emo_ref_audio=None, emo_vector=emo_vec, emo_weight=1.0,
            use_random=use_random,
            seed=seed, return_subtitles=True,
        )
        self._maybe_unload(cache_control)
        return out


class IndexTTS25EmotionTextNode(_IndexTTS25BaseMixin):
    """
    IndexTTS-2.5 情感文本节点：由 Qwen 小模型把情感描述转成情感向量。
    emotion_description 留空时自动分析主文本的情感。
    """

    @classmethod
    def INPUT_TYPES(cls):
        opt = cls._common_optional().copy()
        opt.update({
            "emotion_description": ("STRING", {"multiline": True, "default": ""}),
            "emotion_weight": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
        })
        return {"required": cls._base_inputs(), "optional": opt}

    RETURN_TYPES = ("AUDIO", "INT", "STRING")
    RETURN_NAMES = ("audio", "seed", "subtitle")
    FUNCTION = "generate"
    CATEGORY = "audio"
    DESCRIPTION = "IndexTTS-2.5 情感文本：情感描述留空则自动分析主文本（需加载 Qwen 情感模型）"

    def __init__(self):
        self.loader = _GLOBAL_LOADER
        self.engine = _GLOBAL_ENGINE

    def generate(self, text, reference_audio, lang, duration_factor, emotion_description="",
                 emotion_weight=0.6,
                 do_sample_mode="on", temperature=0.8, top_p=0.8, top_k=30, num_beams=3,
                 repetition_penalty=10.0, length_penalty=0.0, max_mel_tokens=1500,
                 max_tokens_per_sentence=120, interval_silence_ms=200, text_normalization=True,
                 seed=0, cache_control=None):
        ref = self._process_audio_input(reference_audio)
        emo_text = emotion_description.strip() if isinstance(emotion_description, str) else ""
        out = self._do_generate(
            self.engine,
            text=text, reference_audio=ref, lang=lang, duration_factor=duration_factor,
            do_sample=(do_sample_mode == "on"), temperature=temperature, top_p=top_p, top_k=top_k, num_beams=num_beams,
            repetition_penalty=repetition_penalty, length_penalty=length_penalty,
            max_mel_tokens=max_mel_tokens, max_tokens_per_sentence=max_tokens_per_sentence,
            interval_silence=interval_silence_ms, text_normalization=text_normalization,
            emo_text=emo_text if emo_text else None, emo_ref_audio=None, emo_vector=None,
            emo_weight=float(emotion_weight),
            # 情感文本控制必须启用 Qwen 情感分析；同时打开 verbose 日志
            use_qwen=True,
            verbose=True,
            seed=seed, return_subtitles=True,
        )
        self._maybe_unload(cache_control)
        return out


class IndexTTS25CacheControlNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Whether to keep models cached after a node call
                "keep_models_cached": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("cache_control",)
    FUNCTION = "build"
    CATEGORY = "audio"
    DESCRIPTION = "IndexTTS-2.5 显存控制：生成后是否保留模型在显存中"

    def build(self, keep_models_cached: bool = False):
        ctrl = {
            "keep_cached": bool(keep_models_cached),
        }
        return (ctrl,)
