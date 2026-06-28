import json
import torch
import numpy as np
from typing import Any, Tuple

from .indextts2 import IndexTTS2Loader, IndexTTS2Engine

# Global shared loader/engine to avoid duplicating model weights across nodes
_GLOBAL_LOADER = IndexTTS2Loader()
_GLOBAL_ENGINE = IndexTTS2Engine(_GLOBAL_LOADER)
_EMO_VECTOR_BIAS = (0.75, 0.70, 0.80, 0.80, 0.75, 0.75, 0.55, 0.45)


def _normalize_emo_vector_like_demo(vec):
    tmp = np.array([max(0.0, float(x)) for x in vec], dtype=np.float32)
    tmp = tmp * np.array(_EMO_VECTOR_BIAS, dtype=np.float32)
    total = float(tmp.sum())
    if total > 0.8:
        tmp = tmp * (0.8 / total)
    return tmp.tolist()


class _IndexTTS2BaseMixin:
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
            "text": ("STRING", {"multiline": True, "default": "Hello, this is IndexTTS2."}),
            "reference_audio": ("AUDIO",),
            "mode": (["Auto", "Duration", "Tokens"], {"default": "Auto"}),
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
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            # External cache control dict from utility node
            "cache_control": ("DICT", {"default": None}),
        }

    def _do_generate(self, engine: IndexTTS2Engine, **kwargs):
        sr, wave, sub = engine.generate(**kwargs)
        wave_t = torch.tensor(wave, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        audio = {"waveform": wave_t, "sample_rate": int(sr)}
        return audio, kwargs.get("seed", 0), (sub or "")


class IndexTTS2BaseNode(_IndexTTS2BaseMixin):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": cls._base_inputs(), "optional": cls._common_optional()}

    RETURN_TYPES = ("AUDIO", "INT", "STRING")
    RETURN_NAMES = ("audio", "seed", "subtitle")
    FUNCTION = "generate"
    CATEGORY = "audio"

    def __init__(self):
        self.loader = _GLOBAL_LOADER
        self.engine = _GLOBAL_ENGINE

    def generate(self, text, reference_audio, mode,
                 do_sample_mode="on", temperature=0.8, top_p=0.8, top_k=30, num_beams=3,
                 repetition_penalty=10.0, length_penalty=0.0, max_mel_tokens=1500,
                 max_tokens_per_sentence=120, seed=0, return_subtitles=True,
                 cache_control=None):
        ref = self._process_audio_input(reference_audio)
        out = self._do_generate(
            self.engine,
            text=text, reference_audio=ref, mode=mode,
            do_sample=(do_sample_mode == "on"), temperature=temperature, top_p=top_p, top_k=top_k, num_beams=num_beams,
            repetition_penalty=repetition_penalty, length_penalty=length_penalty,
            max_mel_tokens=max_mel_tokens, max_tokens_per_sentence=max_tokens_per_sentence,
            emo_text=None, emo_ref_audio=None, emo_vector=None, emo_weight=0.8,
            seed=seed, return_subtitles=True,
        )
        # Handle cache control: unload tts if keep_cached is False
        try:
            keep = bool(cache_control.get("keep_cached")) if isinstance(cache_control, dict) else False
            if not keep:
                self.loader.unload_tts()
        except Exception:
            pass
        return out


class IndexTTS2EmotionAudioNode(_IndexTTS2BaseMixin):
    @classmethod
    def INPUT_TYPES(cls):
        opt = cls._common_optional().copy()
        opt.update({
            "emo_ref_audio": ("AUDIO",),
            "emotion_weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
        })
        return {"required": cls._base_inputs(), "optional": opt}

    RETURN_TYPES = ("AUDIO", "INT", "STRING")
    RETURN_NAMES = ("audio", "seed", "subtitle")
    FUNCTION = "generate"
    CATEGORY = "audio"

    def __init__(self):
        self.loader = IndexTTS2Loader()
        self.engine = IndexTTS2Engine(self.loader)

    def generate(self, text, reference_audio, mode, emo_ref_audio,
                 emotion_weight=0.8,
                 
                 do_sample_mode="on", temperature=0.8, top_p=0.8, top_k=30, num_beams=3,
                 repetition_penalty=10.0, length_penalty=0.0, max_mel_tokens=1500,
                 max_tokens_per_sentence=120, seed=0, return_subtitles=True,
                 cache_control=None):
        ref = self._process_audio_input(reference_audio)
        emo_ref = self._process_audio_input(emo_ref_audio)
        out = self._do_generate(
            self.engine,
            text=text, reference_audio=ref, mode=mode,
            do_sample=(do_sample_mode == "on"), temperature=temperature, top_p=top_p, top_k=top_k, num_beams=num_beams,
            repetition_penalty=repetition_penalty, length_penalty=length_penalty,
            max_mel_tokens=max_mel_tokens, max_tokens_per_sentence=max_tokens_per_sentence,
            emo_text=None, emo_ref_audio=emo_ref, emo_vector=None, emo_weight=float(emotion_weight) * 0.8,
            seed=seed, return_subtitles=True,
        )
        try:
            keep = bool(cache_control.get("keep_cached")) if isinstance(cache_control, dict) else False
            if not keep:
                self.loader.unload_tts()
        except Exception:
            pass
        return out


class IndexTTS2EmotionVectorNode(_IndexTTS2BaseMixin):
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
        })
        return {"required": cls._base_inputs(), "optional": opt}

    RETURN_TYPES = ("AUDIO", "INT", "STRING")
    RETURN_NAMES = ("audio", "seed", "subtitle")
    FUNCTION = "generate"
    CATEGORY = "audio"

    def __init__(self):
        self.loader = IndexTTS2Loader()
        self.engine = IndexTTS2Engine(self.loader)

    def generate(self, text, reference_audio, mode,
                 Happy=0.0, Angry=0.0, Sad=0.0, Fear=0.0, Hate=0.0, Low=0.0, Surprise=0.0, Neutral=0.0,
                 
                 do_sample_mode="on", temperature=0.8, top_p=0.8, top_k=30, num_beams=3,
                 repetition_penalty=10.0, length_penalty=0.0, max_mel_tokens=1500,
                 max_tokens_per_sentence=120, seed=0, return_subtitles=True,
                 cache_control=None, Love=None):
        ref = self._process_audio_input(reference_audio)
        low_value = Low if Love is None else Love
        vec = [Happy, Angry, Sad, Fear, Hate, low_value, Surprise, Neutral]
        emo_vec = _normalize_emo_vector_like_demo(vec)
        out = self._do_generate(
            self.engine,
            text=text, reference_audio=ref, mode=mode,
            do_sample=(do_sample_mode == "on"), temperature=temperature, top_p=top_p, top_k=top_k, num_beams=num_beams,
            repetition_penalty=repetition_penalty, length_penalty=length_penalty,
            max_mel_tokens=max_mel_tokens, max_tokens_per_sentence=max_tokens_per_sentence,
            emo_text=None, emo_ref_audio=None, emo_vector=emo_vec, emo_weight=0.8,
            seed=seed, return_subtitles=True,
        )
        try:
            keep = bool(cache_control.get("keep_cached")) if isinstance(cache_control, dict) else False
            if not keep:
                self.loader.unload_tts()
        except Exception:
            pass
        return out


class IndexTTS2EmotionTextNode(_IndexTTS2BaseMixin):
    @classmethod
    def INPUT_TYPES(cls):
        opt = cls._common_optional().copy()
        opt.update({
            "emotion_description": ("STRING", {"multiline": True, "default": ""}),
        })
        return {"required": cls._base_inputs(), "optional": opt}

    RETURN_TYPES = ("AUDIO", "INT", "STRING")
    RETURN_NAMES = ("audio", "seed", "subtitle")
    FUNCTION = "generate"
    CATEGORY = "audio"

    def __init__(self):
        self.loader = IndexTTS2Loader()
        self.engine = IndexTTS2Engine(self.loader)

    def generate(self, text, reference_audio, mode, emotion_description="",
                 
                 do_sample_mode="on", temperature=0.8, top_p=0.8, top_k=30, num_beams=3,
                 repetition_penalty=10.0, length_penalty=0.0, max_mel_tokens=1500,
                 max_tokens_per_sentence=120, seed=0, return_subtitles=True,
                 cache_control=None):
        ref = self._process_audio_input(reference_audio)
        emo_text = emotion_description.strip() if isinstance(emotion_description, str) else ""
        emo_text = emo_text if emo_text else None
        out = self._do_generate(
            self.engine,
            text=text, reference_audio=ref, mode=mode,
            do_sample=(do_sample_mode == "on"), temperature=temperature, top_p=top_p, top_k=top_k, num_beams=num_beams,
            repetition_penalty=repetition_penalty, length_penalty=length_penalty,
            max_mel_tokens=max_mel_tokens, max_tokens_per_sentence=max_tokens_per_sentence,
            emo_text=emo_text, emo_ref_audio=None, emo_vector=None, emo_weight=0.8,
            # Enable Qwen emotion analysis when text is provided; also enable verbose logs
            use_qwen=bool(emo_text),
            verbose=True,
            seed=seed, return_subtitles=True,
        )
        try:
            keep = bool(cache_control.get("keep_cached")) if isinstance(cache_control, dict) else False
            if not keep:
                self.loader.unload_tts()
        except Exception:
            pass
        return out
