# -*- coding: utf-8 -*-
"""单独补测：显式情感描述（Qwen 转换）"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import librosa
import soundfile as sf
from indextts2_5 import IndexTTS25Engine

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_outputs_2_5")
os.makedirs(OUT_DIR, exist_ok=True)

engine = IndexTTS25Engine()
ref, ref_sr = librosa.load("TimbreModel/林志玲.wav", sr=None, mono=True)

t0 = time.time()
sr, wav, sub = engine.generate(
    text="快躲起来！是他要来了！他要来抓我们了！", lang="ZH",
    reference_audio=(ref, ref_sr),
    use_qwen=True, emo_text="你吓死我了！你是鬼吗？", emo_weight=0.6, verbose=True,
)
path = os.path.join(OUT_DIR, "emo_text_explicit.wav")
sf.write(path, wav, sr)
print(f"[PASS] emo_text_explicit: {len(wav)/sr:.2f}s -> {path}, cost={time.time()-t0:.1f}s")
