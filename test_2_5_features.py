# -*- coding: utf-8 -*-
"""IndexTTS-2.5 全功能测试脚本（在 ComfyUI 嵌入式 Python 下运行）
用法: python test_2_5_features.py part1|part2
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import librosa
import soundfile as sf
from indextts2_5 import IndexTTS25Engine

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_outputs_2_5")
os.makedirs(OUT_DIR, exist_ok=True)

engine = IndexTTS25Engine()
ref, ref_sr = librosa.load("TimbreModel/林志玲.wav", sr=None, mono=True)
emo_ref, emo_sr = librosa.load("TimbreModel/哪吒 低迷.wav", sr=None, mono=True)

results = []

def run(name, **kw):
    t0 = time.time()
    try:
        sr, wav, sub = engine.generate(reference_audio=(ref, ref_sr), **kw)
        path = os.path.join(OUT_DIR, f"{name}.wav")
        sf.write(path, wav, sr)
        dur = len(wav) / sr
        results.append((name, "PASS", round(dur, 2), round(time.time() - t0, 1)))
        print(f"[PASS] {name}: {dur:.2f}s -> {path}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        results.append((name, f"FAIL: {e}", 0, round(time.time() - t0, 1)))
        print(f"[FAIL] {name}: {e}", flush=True)

part = sys.argv[1] if len(sys.argv) > 1 else "part1"

if part == "part1":
    # 1) 英语合成（多语言能力）
    run("en_basic", text="Hello everyone, this is IndexTTS two point five speaking English with a cloned voice.", lang="EN", verbose=True)
    # 2) 日语合成
    run("ja_basic", text="こんにちは、これはインデックスTTS二点五のテストです。", lang="JA", verbose=True)
    # 3) 语速控制：变慢 1.5x
    run("zh_slow_1_5", text="大家好，欢迎来到 IndexTTS 的语速控制演示。", lang="ZH", duration_factor=1.5, verbose=True)
    # 4) 语速控制：变快 0.7x
    run("zh_fast_0_7", text="大家好，欢迎来到 IndexTTS 的语速控制演示。", lang="ZH", duration_factor=0.7, verbose=True)
    # 5) 中文发音控制（拼音标注多音字）
    run("zh_pinyin", text="他在银<行|XING2>里<行|HANG2>走了半天，发现这笔业务办不<行|HANG2>。", lang="ZH", verbose=True)
    # 6) 英语发音控制（CMU 音素）
    run("en_cmu", text="He had a <minute|M IH1 . N AH0 T> to examine the <minute|M AY0 . N UW1 T> details.", lang="EN", verbose=True)
else:
    # 7) 跨语种克隆：中文音色 -> 英文文本
    run("cross_lingual_zh_en", text="This voice was cloned from a Chinese speaker, now speaking English.", lang="EN", verbose=True)
    # 8) 情感向量控制（悲伤）
    run("emo_vector_sad", text="对不起嘛，我的记性真的不太好，但是和你在一起的事情，我都会努力记住的。", lang="ZH",
        emo_vector=[0, 0, 0.8, 0, 0, 0, 0, 0], verbose=True)
    # 9) 情感参考音频 + 强度
    run("emo_audio_ref", text="酒楼丧尽天良，开始借机竞拍房间，哎，一群蠢货。", lang="ZH",
        emo_ref_audio=(emo_ref, emo_sr), emo_weight=0.9, verbose=True)
    # 10) 情感文本控制（Qwen 自动分析主文本）
    run("emo_text_auto", text="快躲起来！是他要来了！他要来抓我们了！", lang="ZH",
        use_qwen=True, emo_weight=0.6, verbose=True)
    # 11) 显式情感描述（Qwen 转换）
    run("emo_text_explicit", text="快躲起来！是他要来了！他要来抓我们了！", lang="ZH",
        use_qwen=True, emo_text="你吓死我了！你是鬼吗？", emo_weight=0.6, verbose=True)

print("\n===== TEST SUMMARY =====")
for name, status, dur, elapsed in results:
    print(f"{status:>6}  {name}  dur={dur}s  cost={elapsed}s")
fails = [r for r in results if not r[1] == "PASS"]
print(f"== {len(results) - len(fails)}/{len(results)} passed")
sys.exit(1 if fails else 0)
