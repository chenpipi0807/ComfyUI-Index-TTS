import hashlib
import json
import os
from typing import Any, Dict

import folder_paths
import torch
import torchaudio
from aiohttp import web
from server import PromptServer

from .index_tts2_5_nodes import (
    LANG_CHOICES,
    _GLOBAL_LOADER,
    _IndexTTS25BaseMixin,
)


VOICE_TYPE = "INDEXTTS25_VOICE"
VOICE_FORMAT = "indextts25_voice_condition"
VOICE_FORMAT_VERSION = 1
VOICE_UPLOAD_SUBDIR = os.path.join("IndexTTS2.5", "voices")


def _input_dir() -> str:
    path = folder_paths.get_input_directory()
    os.makedirs(path, exist_ok=True)
    return path


def _output_dir() -> str:
    path = folder_paths.get_output_directory()
    os.makedirs(path, exist_ok=True)
    return path


def _validate_voice(voice: Dict[str, Any]) -> None:
    if not isinstance(voice, dict):
        raise ValueError("voice 必须是 IndexTTS-2.5 音色特征对象")
    if voice.get("format") != VOICE_FORMAT:
        raise ValueError(f"不支持的音色格式: {voice.get('format')!r}")
    if int(voice.get("format_version", 0)) != VOICE_FORMAT_VERSION:
        raise ValueError(f"不支持的音色格式版本: {voice.get('format_version')!r}")

    required = ("spk_cond", "style", "s2mel_prompt", "ref_mel", "emo_cond")
    missing = [key for key in required if not isinstance(voice.get(key), torch.Tensor)]
    if missing:
        raise ValueError(f"音色特征缺少 Tensor: {', '.join(missing)}")


def _voice_fingerprint(voice: Dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in ("spk_cond", "style", "s2mel_prompt", "ref_mel", "emo_cond"):
        tensor = voice[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        flat = tensor.float().reshape(-1)
        if flat.numel():
            sample = flat[: min(64, flat.numel())]
            digest.update(sample.numpy().tobytes())
    return digest.hexdigest()[:24]


def _cpu_voice(voice: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(voice)
    for key in ("spk_cond", "style", "s2mel_prompt", "ref_mel", "emo_cond"):
        result[key] = result[key].detach().cpu().contiguous()
    return result


def _load_voice_file(path: str) -> Dict[str, Any]:
    try:
        voice = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        voice = torch.load(path, map_location="cpu")
    _validate_voice(voice)
    return _cpu_voice(voice)


def _list_pt_files(base_dir: str):
    files = []
    for root, _, names in os.walk(base_dir):
        for name in names:
            if not name.lower().endswith(".pt"):
                continue
            full_path = os.path.join(root, name)
            rel_path = os.path.relpath(full_path, base_dir).replace(os.sep, "/")
            files.append(rel_path)
    return sorted(files)


def _list_voice_files():
    files = [f"input/{path}" for path in _list_pt_files(_input_dir())]
    files.extend(f"output/{path}" for path in _list_pt_files(_output_dir()))
    return sorted(files)


def _resolve_voice_path(voice_file: str) -> str:
    value = (voice_file or "").replace("\\", "/")
    if value.startswith("input/"):
        base_dir = os.path.abspath(_input_dir())
        relative = value[len("input/"):]
    elif value.startswith("output/"):
        base_dir = os.path.abspath(_output_dir())
        relative = value[len("output/"):]
    else:
        raise ValueError("voice_file 必须来自 input/ 或 output/ 目录")

    path = os.path.abspath(os.path.join(base_dir, relative))
    if os.path.commonpath([base_dir, path]) != base_dir:
        raise ValueError("voice_file 超出允许目录范围")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"音色文件不存在: {voice_file}")
    return path


def _safe_upload_filename(filename: str) -> str:
    name = os.path.basename(filename or "voice.pt")
    stem, ext = os.path.splitext(name)
    if ext.lower() != ".pt":
        raise ValueError("只允许上传 .pt 音色文件")
    safe_stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem).strip("._") or "voice"
    return safe_stem[:128] + ".pt"


def _unique_path(directory: str, filename: str):
    stem, ext = os.path.splitext(filename)
    path = os.path.join(directory, filename)
    index = 1
    while os.path.exists(path):
        path = os.path.join(directory, f"{stem}_{index:03d}{ext}")
        index += 1
    return path


@PromptServer.instance.routes.get("/indextts25/voice_files")
async def index_tts25_voice_files(_request):
    return web.json_response({"files": _list_voice_files()})


@PromptServer.instance.routes.post("/indextts25/upload_voice")
async def index_tts25_upload_voice(request):
    reader = await request.multipart()
    field = await reader.next()
    if field is None or field.name != "file" or not field.filename:
        return web.json_response({"error": "缺少上传文件"}, status=400)

    try:
        filename = _safe_upload_filename(field.filename)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)

    upload_dir = os.path.join(_input_dir(), VOICE_UPLOAD_SUBDIR)
    os.makedirs(upload_dir, exist_ok=True)
    path = _unique_path(upload_dir, filename)

    try:
        with open(path, "wb") as output:
            while True:
                chunk = await field.read_chunk(size=1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)

        _load_voice_file(path)
    except Exception as exc:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
        return web.json_response({"error": f"无效的 IndexTTS-2.5 音色文件: {exc}"}, status=400)

    relative = os.path.relpath(path, _input_dir()).replace(os.sep, "/")
    return web.json_response({"name": f"input/{relative}"})


class IndexTTS25ExtractVoiceNode(_IndexTTS25BaseMixin):
    """从参考音频提取可持久化复用的 IndexTTS-2.5 conditioning。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"reference_audio": ("AUDIO",)}}

    RETURN_TYPES = (VOICE_TYPE,)
    RETURN_NAMES = ("voice",)
    FUNCTION = "extract"
    CATEGORY = "audio/IndexTTS 2.5/voice"
    DESCRIPTION = "一次提取 IndexTTS-2.5 音色特征，后续生成无需再次处理参考 WAV。"

    def __init__(self):
        self.loader = _GLOBAL_LOADER

    @torch.no_grad()
    def extract(self, reference_audio):
        wave, sr = self._process_audio_input(reference_audio)
        tts = self.loader.get_tts(use_qwen_emo=False)

        audio = torch.as_tensor(wave, dtype=torch.float32)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        elif audio.ndim > 2:
            audio = audio.reshape(1, -1)
        elif audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        max_samples = int(sr * 15)
        audio = audio[:, :max_samples]

        audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

        inputs = tts.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(tts.device)
        attention_mask = inputs["attention_mask"].to(tts.device)
        spk_cond = tts.get_emb(input_features, attention_mask)

        ref_mel = tts.mel_fn(audio_22k.to(tts.device).float())
        ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)

        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k.to(ref_mel.device),
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000,
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        style = tts.campplus_model(feat.unsqueeze(0))

        s2mel_prompt = tts.s2mel.models["length_regulator"](
            spk_cond,
            ylens=ref_target_lengths,
            n_quantizers=3,
            f0=None,
        )[0]

        voice = {
            "format": VOICE_FORMAT,
            "format_version": VOICE_FORMAT_VERSION,
            "model": "IndexTTS-2.5",
            "model_version": getattr(tts, "model_version", None),
            "sample_rate": int(sr),
            "spk_cond": spk_cond.detach().cpu(),
            "style": style.detach().cpu(),
            "s2mel_prompt": s2mel_prompt.detach().cpu(),
            "ref_mel": ref_mel.detach().cpu(),
            "emo_cond": spk_cond.detach().cpu().clone(),
        }
        return (voice,)


class IndexTTS25SaveVoiceNode:
    """像 ComfyUI SaveImage 一样，按文件名前缀保存音色到 output。"""

    def __init__(self):
        self.output_dir = _output_dir()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voice": (VOICE_TYPE,),
                "filename_prefix": ("STRING", {"default": "IndexTTS2.5/voice"}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "audio/IndexTTS 2.5/voice"
    DESCRIPTION = "按 ComfyUI SaveImage 风格将音色 conditioning 保存到 output，支持子目录和自动计数。"

    def save(self, voice, filename_prefix):
        _validate_voice(voice)
        cpu_voice = _cpu_voice(voice)

        full_output_folder, filename, counter, _subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix,
            self.output_dir,
            0,
            0,
        )
        os.makedirs(full_output_folder, exist_ok=True)

        filename = filename.replace("%batch_num%", "0")
        file = f"{filename}_{counter:05}_.pt"
        path = os.path.join(full_output_folder, file)
        torch.save(cpu_voice, path)

        relative_path = os.path.relpath(path, self.output_dir).replace(os.sep, "/")
        return {"ui": {"text": [relative_path]}}


class IndexTTS25LoadVoiceNode:
    """从 input/output 选择或上传 .pt 音色 conditioning。"""

    @classmethod
    def INPUT_TYPES(cls):
        files = _list_voice_files()
        if not files:
            files = ["(暂无音色，可点击上传 .pt)"]
        return {"required": {"voice_file": (files,)}}

    RETURN_TYPES = (VOICE_TYPE,)
    RETURN_NAMES = ("voice",)
    FUNCTION = "load"
    CATEGORY = "audio/IndexTTS 2.5/voice"
    DESCRIPTION = "选择 input/output 中的 .pt 音色，或使用节点上的上传按钮从本地上传。"

    @classmethod
    def IS_CHANGED(cls, voice_file):
        if not voice_file or voice_file.startswith("("):
            return float("nan")
        try:
            path = _resolve_voice_path(voice_file)
            stat = os.stat(path)
            return f"{stat.st_mtime_ns}:{stat.st_size}"
        except OSError:
            return float("nan")
        except (ValueError, FileNotFoundError):
            return float("nan")

    def load(self, voice_file):
        if not voice_file or voice_file.startswith("("):
            raise FileNotFoundError("没有可加载的音色文件，请点击上传 .pt 或先使用 Save Voice 保存音色")
        return (_load_voice_file(_resolve_voice_path(voice_file)),)


class IndexTTS25VoiceBaseNode(_IndexTTS25BaseMixin):
    """使用已提取 conditioning 直接进行 IndexTTS-2.5 基础合成。"""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "text": ("STRING", {"multiline": True, "default": "大家好，这是使用已提取音色特征生成的语音。"}),
            "voice": (VOICE_TYPE,),
            "lang": (LANG_CHOICES, {"default": "ZH"}),
            "duration_factor": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
        }
        return {"required": required, "optional": cls._common_optional()}

    RETURN_TYPES = ("AUDIO", "INT", "STRING")
    RETURN_NAMES = ("audio", "seed", "subtitle")
    FUNCTION = "generate"
    CATEGORY = "audio/IndexTTS 2.5/voice"
    DESCRIPTION = "基于持久化音色 conditioning 合成，无需再次读取或编码参考音频。"

    def __init__(self):
        self.loader = _GLOBAL_LOADER

    @staticmethod
    def _prime_voice_cache(tts, voice):
        _validate_voice(voice)
        device = tts.device
        tts.cache_spk_cond = voice["spk_cond"].to(device)
        tts.cache_s2mel_style = voice["style"].to(device)
        tts.cache_s2mel_prompt = voice["s2mel_prompt"].to(device)
        tts.cache_mel = voice["ref_mel"].to(device)
        tts.cache_emo_cond = voice["emo_cond"].to(device)

        cache_key = f"voice-condition://{_voice_fingerprint(voice)}"
        tts.cache_spk_audio_prompt = cache_key
        tts.cache_emo_audio_prompt = cache_key
        return cache_key

    def generate(
        self,
        text,
        voice,
        lang,
        duration_factor,
        do_sample_mode="on",
        temperature=0.8,
        top_p=0.8,
        top_k=30,
        num_beams=3,
        repetition_penalty=10.0,
        length_penalty=0.0,
        max_mel_tokens=1500,
        max_tokens_per_sentence=120,
        interval_silence_ms=200,
        text_normalization=True,
        seed=0,
        cache_control=None,
    ):
        tts = self.loader.get_tts(use_qwen_emo=False)
        cache_key = self._prime_voice_cache(tts, voice)

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        result = tts.infer(
            spk_audio_prompt=cache_key,
            text=text,
            output_path=None,
            lang=lang,
            emo_audio_prompt=None,
            emo_alpha=1.0,
            emo_vector=None,
            use_emo_text=False,
            emo_text=None,
            use_random=False,
            interval_silence=int(interval_silence_ms),
            verbose=False,
            max_text_tokens_per_segment=int(max_tokens_per_sentence) if max_tokens_per_sentence else 120,
            duration_factor=max(0.5, min(2.0, float(duration_factor))),
            text_normalization=bool(text_normalization),
            do_sample=(do_sample_mode == "on"),
            top_p=float(top_p),
            top_k=int(top_k),
            temperature=float(temperature),
            length_penalty=float(length_penalty),
            num_beams=int(num_beams),
            repetition_penalty=float(repetition_penalty),
            max_mel_tokens=int(max_mel_tokens) if max_mel_tokens else 1500,
        )

        if not (isinstance(result, tuple) and len(result) == 2):
            raise RuntimeError(f"Unexpected return from IndexTTS2.5.infer: {type(result)}")

        sr, wav = result
        wav_t = torch.as_tensor(wav)
        if wav_t.ndim == 2:
            wav_t = wav_t.float().mean(dim=1)
        else:
            wav_t = wav_t.float().reshape(-1)
        wav_t = (wav_t / 32768.0).clamp(-1.0, 1.0).unsqueeze(0).unsqueeze(0)

        audio = {"waveform": wav_t, "sample_rate": int(sr)}
        duration = wav_t.shape[-1] / float(sr)
        subtitle = json.dumps(
            [{"id": "Narrator", "字幕": text, "start": 0.0, "end": round(duration, 2)}],
            ensure_ascii=False,
        )

        self._maybe_unload(cache_control)
        return audio, int(seed), subtitle
