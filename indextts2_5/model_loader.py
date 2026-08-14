import os
import sys
import gc
import torch
from typing import Optional, Dict, Any


class IndexTTS25Loader:
    """
    Lightweight model manager for IndexTTS-2.5.
    - Resolves model root: <ComfyUI>/models/IndexTTS-2.5
    - Validates required files
    - Lazy loads the IndexTTS2 (v2.5) instance and caches it
    - use_qwen_emo 的实例单独缓存（情感文本分析需要额外加载 Qwen 小模型）
    """

    DEFAULT_DIRNAME = "IndexTTS-2.5"

    def __init__(self, models_root: Optional[str] = None, device: Optional[str] = None, dtype: Optional[str] = None):
        self._models_root = models_root or self._default_models_root()
        self._model_dir = os.path.join(self._models_root, self.DEFAULT_DIRNAME)
        self._device = torch.device(device) if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        # IndexTTS-2.5 官方推荐 CUDA 下用 bf16（更快更省显存），CPU 下强制 fp32
        if dtype is None and self._device.type == "cuda":
            dtype = "bf16"
        self._dtype = self._resolve_dtype(dtype)
        self._cache: Dict[str, Any] = {}
        # vendored IndexTTS-2.5 source: indextts2_5/vendor/indextts
        self._vendor_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
        self._vendor_pkg_root = os.path.join(self._vendor_root, "indextts")
        if os.path.isdir(self._vendor_root):
            try:
                sys.path.remove(self._vendor_root)
            except ValueError:
                pass
            sys.path.insert(0, self._vendor_root)

    @staticmethod
    def _default_models_root() -> str:
        # <repo>/ComfyUI/models
        # This file is at: .../ComfyUI/custom_nodes/ComfyUI-Index-TTS/indextts2_5/model_loader.py
        # Go up 4 levels to reach .../ComfyUI/
        here = os.path.abspath(__file__)
        for _ in range(4):
            here = os.path.dirname(here)
        return os.path.join(here, "models")

    @staticmethod
    def _resolve_dtype(dtype: Optional[str]):
        if isinstance(dtype, torch.dtype):
            return dtype
        if dtype == "fp16":
            return torch.float16
        if dtype == "bf16":
            return torch.bfloat16
        return torch.float32

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def model_dir(self):
        return self._model_dir

    def validate(self) -> None:
        required = [
            "config.yaml",
            "feat1.pt",
            "feat2.pt",
            "gpt.pth",
            "codec.pth",
            "s2mel.pth",
            "wav2vec2bert_stats.pt",
            "multilingual_zh_ja_yue_char_del.tiktoken",
        ]
        missing = [f for f in required if not os.path.exists(os.path.join(self._model_dir, f))]
        if missing:
            raise FileNotFoundError(f"IndexTTS-2.5 missing files in {self._model_dir}: {', '.join(missing)}")

    def get_tts(self, use_qwen_emo: bool = False):
        """
        Return a cached instance of indextts.infer_v2_5.IndexTTS2 constructed with our model_dir.
        use_qwen_emo=True 时返回加载了 QwenEmotion 的实例（use_emo_text 情感文本控制需要）。
        """
        cache_key = "tts_qwen" if use_qwen_emo else "tts"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Ensure validation before heavy init for clearer error
        self.validate()

        try:
            # Purge any previously imported 'indextts' top-level modules to avoid
            # shadowing between indextts2/vendor (v2) and indextts2_5/vendor (v2.5)
            for k in list(sys.modules.keys()):
                if k == "indextts" or k.startswith("indextts."):
                    sys.modules.pop(k, None)
            if os.path.isdir(self._vendor_root):
                try:
                    sys.path.remove(self._vendor_root)
                except ValueError:
                    pass
                sys.path.insert(0, self._vendor_root)
            from indextts.infer_v2_5 import IndexTTS2  # imported from vendored 2.5 package
        except Exception as e:
            # Fallback: import by file path to avoid package name collisions
            try:
                import importlib.util
                infer_path = os.path.join(self._vendor_pkg_root, "infer_v2_5.py")
                spec = importlib.util.spec_from_file_location("indextts_infer_v2_5", infer_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"spec load failed for {infer_path}")
                mod = importlib.util.module_from_spec(spec)
                sys.modules["indextts_infer_v2_5"] = mod
                spec.loader.exec_module(mod)
                IndexTTS2 = getattr(mod, "IndexTTS2")
            except Exception as e2:
                raise ImportError(
                    f"Failed to import IndexTTS2 (v2.5) from vendored source at {self._vendor_pkg_root}. Error: {e}. Fallback failed: {e2}. "
                    "Ensure project dependencies (transformers, modelscope, librosa, tiktoken, omegaconf, torchaudio, safetensors, etc.) are installed."
                )

        cfg_path = os.path.join(self._model_dir, "config.yaml")
        tts = IndexTTS2(
            cfg_path=cfg_path,
            model_dir=self._model_dir,
            use_bf16=(self._dtype == torch.bfloat16),
            device=str(self._device),
            # Windows 下自定义 CUDA kernel 需要 JIT 编译，默认关闭，自动回退到 torch 实现
            use_cuda_kernel=False,
            use_deepspeed=False,
            use_qwen_emo=use_qwen_emo,
        )
        self._cache[cache_key] = tts
        return tts

    def unload_tts(self) -> None:
        """
        Best-effort unload of cached TTS instances and free GPU cache to reduce VRAM.
        Safe to call even if not loaded.
        """
        try:
            for key in ("tts", "tts_qwen"):
                tts = self._cache.pop(key, None)
                del tts
        except Exception:
            pass
        # Collect Python garbage and free CUDA cache
        try:
            gc.collect()
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass
