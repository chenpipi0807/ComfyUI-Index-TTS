import os
import tempfile
import numpy as np
import soundfile as sf
from typing import Tuple


def save_temp_wav(wave_sr: Tuple[np.ndarray, int]) -> str:
    """
    Save (wave, sr) to a temporary mono WAV file and return the path.
    Wave is expected in float32 [-1, 1] range or int16.
    """
    wave, sr = wave_sr
    if wave is None:
        raise ValueError("wave is None")
    if wave.ndim > 1:
        # force mono
        wave = wave.reshape(-1)
    # ensure float32
    if wave.dtype != np.float32:
        if wave.dtype == np.int16:
            wave = (wave.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
        else:
            wave = wave.astype(np.float32)
    tmpdir = tempfile.gettempdir()
    path = os.path.join(tmpdir, f"indextts2_5_{next(_counter)}.wav")
    sf.write(path, wave, int(sr))
    return path


def _counter_gen():
    n = 0
    while True:
        yield n
        n += 1


_counter = _counter_gen()
