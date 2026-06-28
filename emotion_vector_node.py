import json
import numpy as np
from typing import List, Tuple


class IndexTTSEmotionVectorNode:
    """
    ComfyUI node: Index TTS Emotion Vector
    - Outputs an emotion vector as JSON string for Index TTS 2's `emo_vector` input.
    - Matches HF demo with 8 sliders: Happy, Angry, Sad, Fear, Hate, Low, Surprise, Neutral
    - Optional random sampling (seeded) when you want quick stochastic presets.
    """

    EMO_ORDER = [
        "Happy", "Angry", "Sad", "Fear", "Hate", "Low", "Surprise", "Neutral"
    ]
    EMO_VECTOR_BIAS = np.array([0.75, 0.70, 0.80, 0.80, 0.75, 0.75, 0.55, 0.45], dtype=np.float32)

    @classmethod
    def INPUT_TYPES(cls):
        sliders = {
            "Happy": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "Angry": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "Sad": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "Fear": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "Hate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "Low": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "Surprise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "Neutral": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
        }
        return {
            "required": {
                **sliders,
            },
            "optional": {
                "random_sampling": ("BOOL", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "normalize": ("BOOL", {"default": True}),
                "top_k_random": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("emo_vector",)
    FUNCTION = "build_vector"
    CATEGORY = "audio"

    def _sample_random_vector(self, seed: int, top_k: int) -> List[float]:
        rng = np.random.default_rng(int(seed))
        # Dirichlet over 8 dims, then zero out all but top_k for sparse expressive control
        vec = rng.dirichlet(np.ones(8)).astype(np.float32)
        idx = np.argsort(vec)[::-1]
        mask = np.zeros_like(vec)
        mask[idx[: max(1, int(top_k))]] = 1.0
        vec = (vec * mask)
        s = float(vec.sum())
        if s > 0:
            vec = vec / s
        return vec.tolist()

    @classmethod
    def _normalize_like_demo(cls, vec: List[float]) -> List[float]:
        tmp = np.array([max(0.0, float(x)) for x in vec], dtype=np.float32) * cls.EMO_VECTOR_BIAS
        total = float(tmp.sum())
        if total > 0.8:
            tmp = tmp * (0.8 / total)
        return tmp.tolist()

    def build_vector(
        self,
        Happy: float,
        Angry: float,
        Sad: float,
        Fear: float,
        Hate: float,
        Low: float,
        Surprise: float,
        Neutral: float,
        random_sampling: bool = False,
        seed: int = 0,
        normalize: bool = True,
        top_k_random: int = 2,
        Love=None,
    ) -> Tuple[str]:
        if random_sampling:
            vec = self._sample_random_vector(seed, top_k_random)
        else:
            low_value = Low if Love is None else Love
            vec = [Happy, Angry, Sad, Fear, Hate, low_value, Surprise, Neutral]
            if normalize:
                vec = self._normalize_like_demo(vec)
        return (json.dumps(vec, ensure_ascii=False),)
