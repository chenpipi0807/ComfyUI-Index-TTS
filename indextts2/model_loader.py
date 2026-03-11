import os
import sys
import gc
import torch
from typing import Optional, Dict, Any


class IndexTTS2Loader:
    """
    Lightweight model manager for IndexTTS2.
    - Resolves model root: <ComfyUI>/models/IndexTTS-2
    - Validates required files
    - Lazy loads submodules and caches them
    """

    DEFAULT_DIRNAME = "IndexTTS-2"

    def __init__(self, models_root: Optional[str] = None, device: Optional[str] = None, dtype: Optional[str] = None):
        self._models_root = models_root or self._default_models_root()
        self._model_dir = os.path.join(self._models_root, self.DEFAULT_DIRNAME)
        self._device = torch.device(device) if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        # Default to fp16 when running on CUDA to reduce VRAM and speed up inference; stay fp32 on CPU
        if dtype is None and self._device.type == "cuda":
            dtype = "fp16"
        self._dtype = self._resolve_dtype(dtype)
        self._cache: Dict[str, Any] = {}
        # Prepare path to vendored IndexTTS2 source: indextts2/vendor/indextts
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
        # This file is at: .../ComfyUI/custom_nodes/ComfyUI-Index-TTS/indextts2/model_loader.py
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
            "bpe.model",
            "config.yaml",
            "feat1.pt",
            "feat2.pt",
            "gpt.pth",
            "s2mel.pth",
            "wav2vec2bert_stats.pt",
        ]
        missing = [f for f in required if not os.path.exists(os.path.join(self._model_dir, f))]
        if missing:
            raise FileNotFoundError(f"IndexTTS-2 missing files in {self._model_dir}: {', '.join(missing)}")

    def get_components(self) -> Dict[str, Any]:
        """
        Placeholder that returns a dict-like components container.
        Later this will actually initialize models from 原项目代码/index-tts.
        """
        if "components" not in self._cache:
            # Lazy container; real impl will import and construct modules
            self._cache["components"] = {
                "status": "placeholder",
                "model_dir": self._model_dir,
                "device": str(self._device),
                "dtype": str(self._dtype),
            }
        return self._cache["components"]

    def get_tts(self):
        """
        Return a cached instance of indextts.infer_v2.IndexTTS2 constructed with our model_dir.
        """
        if "tts" in self._cache:
            return self._cache["tts"]

        # Ensure validation before heavy init for clearer error
        self.validate()

        try:
            # Purge any previously imported legacy 'indextts' to avoid shadowing
            for k in list(sys.modules.keys()):
                if k == "indextts" or k.startswith("indextts."):
                    sys.modules.pop(k, None)
            from .vendor.indextts.infer_v2 import IndexTTS2  # imported from vendored package
        except Exception as e:
            # Fallback: import by file path to avoid package name collisions
            try:
                import importlib.util
                infer_path = os.path.join(self._vendor_pkg_root, "infer_v2.py")
                spec = importlib.util.spec_from_file_location("indextts_infer_v2", infer_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"spec load failed for {infer_path}")
                mod = importlib.util.module_from_spec(spec)
                sys.modules["indextts_infer_v2"] = mod
                spec.loader.exec_module(mod)
                IndexTTS2 = getattr(mod, "IndexTTS2")
            except Exception as e2:
                raise ImportError(
                    f"Failed to import IndexTTS2 from vendored source at {self._vendor_pkg_root}. Error: {e}. Fallback failed: {e2}. "
                    "Ensure project dependencies (transformers, modelscope, huggingface_hub, torchaudio, safetensors, omegaconf, etc.) are installed."
                )

        cfg_path = os.path.join(self._model_dir, "config.yaml")
        # Upstream IndexTTS2 expects 'is_fp16' (HF demo), not 'use_fp16'
        tts = IndexTTS2(
            cfg_path=cfg_path,
            model_dir=self._model_dir,
            is_fp16=(self._dtype == torch.float16),
            device=str(self._device),
        )
        self._cache["tts"] = tts
        return tts

    def unload_tts(self) -> None:
        """
        Best-effort unload of cached TTS instance and free GPU cache to reduce VRAM.
        Safe to call even if not loaded.
        """
        try:
            tts = self._cache.pop("tts", None)
            # Break potential cyc refs early
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
