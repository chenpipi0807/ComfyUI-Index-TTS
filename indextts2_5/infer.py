import json
import numpy as np
import torch
from typing import Optional, Tuple

from .model_loader import IndexTTS25Loader
from .utils import save_temp_wav


class IndexTTS25Engine:
    """
    Thin wrapper calling vendored indextts/infer_v2_5.IndexTTS2.infer.
    It converts ComfyUI audio inputs to temp WAV files, forwards parameters,
    and converts returned audio back to numpy.

    IndexTTS-2.5 新能力（相对 2.0）：
    - lang: ZH / EN / JA / ES / AR（多语言 + 跨语种克隆）
    - duration_factor: 0.5 - 2.0 语速控制（>1 变慢，<1 变快）
    - 发音控制通过文本内联标注实现：<字|XING2> / <word|W ER1 D> / <詞|かな>
    """

    def __init__(self, loader: Optional[IndexTTS25Loader] = None):
        self.loader = loader or IndexTTS25Loader()

    def generate(
        self,
        text: str,
        reference_audio: Optional[Tuple[np.ndarray, int]] = None,
        lang: str = "ZH",
        # 2.5 语速控制
        duration_factor: float = 1.0,
        text_normalization: bool = True,
        # Advanced generation controls
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 30,
        num_beams: int = 3,
        repetition_penalty: float = 10.0,
        length_penalty: float = 0.0,
        max_mel_tokens: int = 1500,
        max_tokens_per_sentence: int = 120,
        interval_silence: int = 200,
        # Emotion controls
        emo_text: Optional[str] = None,
        emo_ref_audio: Optional[Tuple[np.ndarray, int]] = None,
        emo_vector: Optional[list] = None,
        emo_weight: float = 1.0,
        use_random: bool = False,
        seed: int = 0,
        use_qwen: bool = False,
        verbose: bool = False,
        return_subtitles: bool = True,
    ) -> Tuple[int, np.ndarray, Optional[str]]:
        # use_emo_text 需要 QwenEmotion 模型（use_qwen_emo=True 构造）
        need_qwen = bool(use_qwen)
        tts = self.loader.get_tts(use_qwen_emo=need_qwen)

        if reference_audio is None:
            raise ValueError("reference_audio is required for IndexTTS-2.5")
        spk_wav_path = save_temp_wav(reference_audio)

        emo_wav_path = None
        if emo_ref_audio is not None:
            emo_wav_path = save_temp_wav(emo_ref_audio)

        # Generation kwargs aligned with infer_v2_5
        gen_kwargs = dict(
            do_sample=bool(do_sample),
            top_p=float(top_p),
            top_k=int(top_k),
            temperature=float(temperature),
            length_penalty=float(length_penalty),
            num_beams=int(num_beams),
            repetition_penalty=float(repetition_penalty),
            max_mel_tokens=int(max_mel_tokens) if max_mel_tokens else 1500,
        )

        # Emotion control selection
        # 优先级：emo_ref_audio > emo_vector > emo_text(use_qwen)
        # use_qwen=True 时始终启用 use_emo_text；emo_text 留空则让上游直接用主文本自动分析情感
        use_emo_text = False
        _emo_text = None
        if emo_wav_path is None and (emo_vector is None or len(emo_vector) == 0):
            if need_qwen:
                use_emo_text = True
                _emo_text = str(emo_text).strip() if emo_text else None
                _emo_text = _emo_text if _emo_text else None

        # 语速控制参数范围 0.5 - 2.0，超出则截断
        _duration_factor = max(0.5, min(2.0, float(duration_factor)))

        # Call upstream infer; output_path=None returns (sr, wav_int16_numpy_TxC)
        result = tts.infer(
            spk_audio_prompt=spk_wav_path,
            text=text,
            output_path=None,
            lang=lang,
            emo_audio_prompt=emo_wav_path,
            emo_alpha=float(emo_weight),
            emo_vector=emo_vector if (emo_wav_path is None and emo_vector) else None,
            use_emo_text=bool(use_emo_text),
            emo_text=_emo_text,
            use_random=bool(use_random),
            interval_silence=int(interval_silence),
            verbose=bool(verbose),
            max_text_tokens_per_segment=int(max_tokens_per_sentence) if max_tokens_per_sentence else 120,
            duration_factor=_duration_factor,
            text_normalization=bool(text_normalization),
            **gen_kwargs,
        )

        if not (isinstance(result, tuple) and len(result) == 2):
            raise RuntimeError(f"Unexpected return from IndexTTS2.5.infer: {type(result)}")

        sr, wav = result
        # wav is int16 numpy with shape [T, C] (from upstream .T). Convert to mono float32
        wav = np.asarray(wav)
        if wav.ndim == 2:
            # average channels
            wav = wav.mean(axis=1)
        wav = (wav.astype(np.float32) / 32768.0).clip(-1.0, 1.0)

        subtitle = None
        if return_subtitles:
            # Minimal single-span subtitle
            duration = len(wav) / float(sr)
            subtitle = json.dumps([
                {"id": "Narrator", "字幕": text, "start": 0.0, "end": round(duration, 2)}
            ], ensure_ascii=False)

        return int(sr), wav, subtitle
