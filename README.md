## 免责声明

本项目基于B站开源项目进行二次开发，由本人对项目进行了ComfyUI的实现，并进行了部分功能优化与调整与进阶功能的开发。然而，需要强调的是，本项目严禁用于任何非法目的以及与侵犯版权相关的任何行为！本项目仅用于开源社区内的交流与学习，以促进技术共享与创新，旨在为开发者提供有益的参考和学习资源。

在此郑重声明，本项目所有个人使用行为与开发者本人及本项目本身均无任何关联。开发者对于项目使用者的行为不承担任何责任，使用者应自行承担使用过程中可能产生的所有风险和法律责任。请广大使用者在遵守法律法规及相关规定的前提下，合理、合法地使用本项目，维护开源社区的良好秩序与健康发展。

感谢您的理解与支持！


# ComfyUI-Index-TTS

使用IndexTTS模型在ComfyUI中实现高质量文本到语音转换的自定义节点。已支持 IndexTTS-2.5（中/英/日/西/阿多语言、语速控制、发音标注、情感控制），可以基于参考音频复刻声音特征。

![示例截图1](https://github.com/user-attachments/assets/41960425-f739-4496-9520-8f9cae34ff51)
![示例截图2](https://github.com/user-attachments/assets/1ff0d1d0-7a04-4d91-9d53-cd119250ed67)
![微信截图_20250605215845](https://github.com/user-attachments/assets/d5eb22f6-2ca2-40cf-a619-d709746f83e3)



## 最新更新（重要）

### 🔥 2026-08-15：IndexTTS-2.5 来了！

本项目已新增对 IndexTTS-2.5 的完整支持（与 TTS2 节点互不干扰，可共存）。

**2.5 相比 2.0 的升级：**

- **多语言合成**：中文 / 英语 / 日语 / 西班牙语 / 阿拉伯语，且支持**跨语种克隆**（用中文参考音频直接说英文/日文）
- **语速控制**：`duration_factor` 0.5 – 2.0，大于 1 变慢、小于 1 变快
- **发音控制升级**：在文本里直接标注多音字/发音，2.5 的遵循能力大幅增强
  - 中文拼音：`他在银<行|XING2>里<行|HANG2>走了半天`
  - 英文音素（CMU）：`He had a <minute|M IH1 . N AH0 T> to examine the <minute|M AY0 . N UW1 T> details.`
  - 日语假名：`彼は料理が<上手|じょうず>だ`
- **保留 2.0 全部情感控制能力**（情感参考音频 / 8 维情感向量 / Qwen 情感文本分析）
- **更快**：bf16 推理，速度约为 2.0 的 1.6 倍

**新增五个节点（audio 分类下）：**

- `Index TTS 2.5 - Base (多语言/语速)`：基础合成 + 语言选择 + 语速控制 + 发音标注
- `Index TTS 2.5 - Emotion Audio`：情感参考音频（音色与情感解耦）
- `Index TTS 2.5 - Emotion Vector`：8 维情感向量滑条
- `Index TTS 2.5 - Emotion Text`：情感描述控制，留空则自动分析主文本情感（需要 Qwen 情感模型）
- `Index TTS 2.5 - Cache Control`：显存/缓存控制

基础工作流已更新，详见 `./workflow/TTS2.5.json`（直接拖进 ComfyUI 页面即可导入）。

#### 2.5 模型下载与放置（全部放到 `./ComfyUI/models/IndexTTS-2.5/`）

1) 基础模型（整仓）
   - 页面：[IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5/tree/main)
   - 国内镜像：[hf-mirror 镜像页](https://hf-mirror.com/IndexTeam/IndexTTS-2.5/tree/main)
   - 放置：`.\ComfyUI\models\IndexTTS-2.5`
   - 包含：`gpt.pth`（主模型约 3.3GB）、`codec.pth`（2.5 新统一语义编解码器）、`s2mel.pth`、`config.yaml`、`feat1.pt`、`feat2.pt`、`wav2vec2bert_stats.pt`、`multilingual_zh_ja_yue_char_del.tiktoken`（多语言分词器，替代旧版 bpe.model）

2) qwen 模型（情绪分类，仅 Emotion Text 节点需要）
   - 页面：[IndexTTS-2.5/qwen0.6bemo4-merge](https://huggingface.co/IndexTeam/IndexTTS-2.5/tree/main/qwen0.6bemo4-merge)
   - 放置：`.\ComfyUI\models\IndexTTS-2.5\qwen0.6bemo4-merge\`
   - 不用 Emotion Text 节点可以不下载

3) CampPlus 说话人嵌入
   - 页面：[https://huggingface.co/funasr/campplus](https://huggingface.co/funasr/campplus)
   - 直链：[https://huggingface.co/funasr/campplus/resolve/main/campplus_cn_common.bin](https://huggingface.co/funasr/campplus/resolve/main/campplus_cn_common.bin)
   - 放置：`.\ComfyUI\models\IndexTTS-2.5\campplus_cn_common.bin`

4) Wav2Vec2Bert 特征提取器（facebook/w2v-bert-2.0）
   - 页面：[https://huggingface.co/facebook/w2v-bert-2.0/tree/main](https://huggingface.co/facebook/w2v-bert-2.0/tree/main)
   - 放置（离线优先）：`.\ComfyUI\models\IndexTTS-2.5\w2v-bert-2.0\`（整仓文件夹）

5) BigVGAN 声码器
   - 页面：[https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x/tree/main](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x/tree/main)
   - 放置：`.\ComfyUI\models\IndexTTS-2.5\bigvgan\`（`config.json` + `bigvgan_generator.pt`）

示例目录结构（详见仓库内 `TTS2.5模型路径.txt`）：

```text
ComfyUI/models/IndexTTS-2.5/
│  codec.pth
│  config.yaml
│  feat1.pt
│  feat2.pt
│  gpt.pth
│  multilingual_zh_ja_yue_char_del.tiktoken
│  s2mel.pth
│  wav2vec2bert_stats.pt
│  campplus_cn_common.bin
│
├─bigvgan
│      bigvgan_generator.pt
│      config.json
│
├─qwen0.6bemo4-merge
│      config.json
│      model.safetensors
│      tokenizer.json
│      ...
│
└─w2v-bert-2.0
        config.json
        conformer_shaw.pt
        model.safetensors
        preprocessor_config.json
```

#### 一键下载 2.5 模型（推荐）

- 脚本位置：`ComfyUI/custom_nodes/ComfyUI-Index-TTS/TTS2_5_download.py`
- 与 TTS2 下载脚本用法一致，支持断点续传、国内镜像：

```powershell
python .\ComfyUI\custom_nodes\ComfyUI-Index-TTS\TTS2_5_download.py
```

- 运行后根据提示选择 2 使用国内镜像（默认）或 1 使用官方源。
- 若镜像设置不生效，可先在控制台设置：`$env:HF_ENDPOINT = "https://hf-mirror.com"`（PowerShell）

#### 2.5 使用方法

1. **加载参考音频**：用 `LoadAudio` 节点选择一段 3–15 秒的干净人声（也可以先用 `Timbre音频加载器` 节点选仓库自带音色）
2. **接到 `Index TTS 2.5 - Base`**：填文本、选语言（ZH/EN/JA/ES/AR 或 ZH/EN 混合）、按需调速（`duration_factor`）
3. **输出接 `SaveAudio` 或 `PreviewAudio`** 即可

发音标注直接写在文本里，例如：

```text
他在银<行|XING2>里<行|HANG2>走了半天，发现这笔业务办不<行|HANG2>。
```

情感控制三选一（与 TTS2 用法一致）：

- 有情感参考音频 → 用 `Emotion Audio` 节点，`emotion_weight` 调强度（0–1）
- 想精细配比 → 用 `Emotion Vector` 节点，8 个滑条对应 高兴/愤怒/悲伤/恐惧/反感/低落/惊讶/平静
- 只想写一句话 → 用 `Emotion Text` 节点，填情感描述（留空则自动分析主文本），`emotion_weight` 建议 0.6 左右

> 提示：Emotion Text 节点首次使用会额外加载 Qwen 情感小模型（约 1.2GB 显存）；其余节点不需要。
> 提示：2.5 与 2.0 的模型目录相互独立（`IndexTTS-2.5` vs `IndexTTS-2`），两套节点可以同时存在于一个工作流中。

---

### IndexTTS-2 支持

本项目已新增对 IndexTTS-2（简称 TTS2）的支持，并将功能拆分为四个核心节点，方便在 ComfyUI 中按需组合：
基础工作流已更新，详见./workflow/TTS2.json.
会有一些BUG，欢迎反馈。功能基本复刻了原版IndexTTS，关于功能建议欢迎交流。

- Index TTS 2 - Base（基础合成）
- Index TTS 2 - Emotion Audio（基于参考音频情绪复刻）
- Index TTS 2 - Emotion Vector（基于情绪向量复刻）
- Index TTS 2 - Emotion Text（基于情绪文本复刻）

<img width="3090" height="1389" alt="image" src="https://github.com/user-attachments/assets/b12dae62-0ae3-49a7-99a9-f153218328fa" />


TTS2 模型下载与放置位置（全部放到 `./ComfyUI/models/IndexTTS-2/`）：

1) 基础模型
   - 页面：[TTS2](https://huggingface.co/IndexTeam/IndexTTS-2/tree/main)
   - 放置：`.\ComfyUI\models\IndexTTS-2`

2) qwen 模型（情绪分类）
   - 页面：[IndexTTS-2/qwen0.6bemo4-merge](https://huggingface.co/IndexTeam/IndexTTS-2/tree/main/qwen0.6bemo4-merge)
   - 放置：`.\ComfyUI\models\IndexTTS-2\qwen0.6bemo4-merge\`

3) semantic codec（MaskGCT 语义编码器）
   - 页面：[https://huggingface.co/amphion/MaskGCT/tree/main/semantic_codec](https://huggingface.co/amphion/MaskGCT/tree/main/semantic_codec)
   - 直链：[https://huggingface.co/amphion/MaskGCT/resolve/main/semantic_codec/model.safetensors](https://huggingface.co/amphion/MaskGCT/resolve/main/semantic_codec/model.safetensors)
   - 放置：`.\ComfyUI\models\IndexTTS-2\semantic_codec\model.safetensors`

4) CampPlus 说话人嵌入
   - 页面：[https://huggingface.co/funasr/campplus](https://huggingface.co/funasr/campplus)
   - 直链：[https://huggingface.co/funasr/campplus/resolve/main/campplus_cn_common.bin](https://huggingface.co/funasr/campplus/resolve/main/campplus_cn_common.bin)
   - 放置：`.\ComfyUI\models\IndexTTS-2\campplus_cn_common.bin`

5) Wav2Vec2Bert 特征提取器（facebook/w2v-bert-2.0）
   - 页面：[https://huggingface.co/facebook/w2v-bert-2.0/tree/main](https://huggingface.co/facebook/w2v-bert-2.0/tree/main)
   - 放置（离线优先）：`.\ComfyUI\models\IndexTTS-2\w2v-bert-2.0\`（整个仓库文件夹，包含 `config.json`、`model.safetensors`、`preprocessor_config.json` 等）
   - 若未放置本地文件夹，将自动下载到 HF 缓存：`.\ComfyUI\models\IndexTTS-2\hf_cache\`

6) BigVGAN 声码器
   - 名称读取自 `config.yaml` 的 `vocoder.name`（示例：`nvidia/bigvgan_v2_22khz_80band_256x`）
   - 建议：提前将对应模型完整缓存到 `.\ComfyUI\models\IndexTTS-2\bigvgan\` 内

7) 其他本地直读文件（需与 `config.yaml` 一致）：
   - `gpt.pth`（`cfg.gpt_checkpoint`）
   - `s2mel.pth`（`cfg.s2mel_checkpoint`）
   - `bpe.model`（`cfg.dataset.bpe_model`）
   - `wav2vec2bert_stats.pt`（`cfg.w2v_stat`）
   - 语义编码配置（如 `repcodec.json`，若需要，`cfg.semantic_codec`）
   - `emo_matrix`（例如 `feat2.pt`）
   - `spk_matrix`（例如 `feat1.pt`）
   - `qwen0.6bemo4-merge\`（`cfg.qwen_emo_path` 指定目录）

示例目录结构（部分）：

```text
ComfyUI/models/IndexTTS-2/
│  .gitattributes
│  bpe.model
│  campplus_cn_common.bin
│  config.yaml
│  feat1.pt
│  feat2.pt
│  gpt.pth
│  README.md
│  s2mel.pth
│  wav2vec2bert_stats.pt
│
├─bigvgan
│  └─bigvgan_v2_22khz_80band_256x
│          .gitattributes
│          .gitignore
│          activations.py
│          bigvgan.py
│          bigvgan_discriminator_optimizer.pt
│          bigvgan_discriminator_optimizer_3msteps.pt
│          bigvgan_generator.pt
│          bigvgan_generator_3msteps.pt
│          config.json
│          env.py
│          LICENSE
│          meldataset.py
│          README.md
│          utils.py
│
├─hf_cache
├─qwen0.6bemo4-merge
│      added_tokens.json
│      chat_template.jinja
│      config.json
│      generation_config.json
│      merges.txt
│      model.safetensors
│      Modelfile
│      special_tokens_map.json
│      tokenizer.json
│      tokenizer_config.json
│      vocab.json
│
├─semantic_codec
│      model.safetensors
│
└─w2v-bert-2.0
        .gitattributes
        config.json
        conformer_shaw.pt
        model.safetensors
        preprocessor_config.json
        README.md
```

> 提示：若你只使用旧版 IndexTTS/IndexTTS-1.5，可忽略上述 TTS2 模型放置步骤。

### 一键下载脚本（推荐）

- 脚本位置：`ComfyUI/custom_nodes/ComfyUI-Index-TTS/TTS2_download.py`
- 作用：自动下载并放置上述所有 TTS2 所需模型文件，支持断点续传、国内镜像（HF_ENDPOINT=hf-mirror.com）、本地缓存（HF_HOME=./ComfyUI/models/IndexTTS-2/hf_cache）。
- 脚本使用时，可能会存在国内镜像设置不成功的问题，可直接在控制台设置环境变量：Windows Powershell `$env:HF_ENDPOINT = "https://hf-mirror.com"`,linux `export HF_ENDPOINT=https://hf-mirror.com`

```powershell
python .\ComfyUI\custom_nodes\ComfyUI-Index-TTS\TTS2_download.py
```

- 运行后根据提示选择 2 使用国内镜像（默认）或 1 使用官方源。
- 依赖：`huggingface_hub`（必须）；可选加速：`hf_transfer`、`hf_xet`。

```powershell
python -m pip install -U huggingface_hub
# 可选加速：
python -m pip install -U hf_transfer
python -m pip install -U "huggingface_hub[hf_xet]"
```

### 显存/缓存控制（新功能）

- 新增节点：`Index TTS 2 - Cache Control`
  - 输出：`cache_control`（类型：DICT），包含 `{"keep_cached": true/false}`。
  - 用法：将该输出连到以下任一/多个节点的 `cache_control` 输入上：
    - `Index TTS 2 - Base`
    - `Index TTS 2 - Emotion Audio`
    - `Index TTS 2 - Emotion Vector`
    - `Index TTS 2 - Emotion Text`

- 行为说明：
  - 关闭（默认）：本次推理结束后自动卸载 TTS2 模型并清理 CUDA 缓存，降低显存驻留峰值，适合 12GB 显卡日常使用。
  - 开启：保留已加载的权重（尽量驻留，视环境/模式），连续多次生成更快，但显存占用更高。调参批量测试时可临时打开，用完关闭。

## 功能特点

- 支持 IndexTTS-2.5 / IndexTTS-2 / IndexTTS-1.5 三代模型
- 2.5 支持中/英/日/西/阿五种语言与跨语种克隆
- 基于参考音频复刻声音特征（变声功能）
- 支持语速控制（2.5 原生 duration_factor；旧版为后处理实现效果会有一点折损）
- 支持发音标注（拼音/CMU音素/日语假名，2.5）
- 多种情感控制方式（参考音频/情感向量/情感文本）
- 多种音频合成参数控制
- Windows兼容（无需额外依赖）


## 废话两句

- 生成的很快，真的很快！而且竟然也很像！！！
- 效果很好，感谢小破站的开源哈哈哈哈哈 
- 如果你想体验一下效果 附赠道友B站的传送阵[demo](https://huggingface.co/spaces/IndexTeam/IndexTTS)
- 如果你不知道去哪找音频，那我建议你去[隔壁](https://drive.google.com/drive/folders/1AyB3egmr0hAKp0CScI0eXJaUdVccArGB)偷哈哈哈哈哈

## 演示案例

以下是一些实际使用效果演示：

| 参考音频 | 输入文本 | 推理结果 |
|---------|---------|---------|
| <video src="https://github.com/user-attachments/assets/5e8cb570-242f-4a16-8472-8a64a23183fb"></video> | 我想把钉钉的自动回复设置成"服务器繁忙，请稍后再试"，仅对老板可见。  我想把钉钉的自动回复设置成"服务器繁忙，请稍后再试"，仅对老板可见。 | <video src="https://github.com/user-attachments/assets/d8b89db3-5cf5-406f-b930-fa75d13ff0bd"></video> |
| <video src="https://github.com/user-attachments/assets/8e774223-e0f7-410b-ae4e-e46215e47e96"></video> | 我想把钉钉的自动回复设置成"服务器繁忙，请稍后再试"，仅对老板可见。 | <video src="https://github.com/user-attachments/assets/6e3e63ed-2d3d-4d5a-bc2e-b42530748fa0"></video> |

- 长文本测试：

<video src="https://github.com/user-attachments/assets/6bfa35dc-1a30-4da0-a4dc-ac3def25452b"></video>

- 多角色小说测试：

<video src="https://github.com/user-attachments/assets/6d4737f4-9d75-431e-bb11-fe3e86a4ab0e"></video>



## 更新日志

### 2026-08-15

- **新增 IndexTTS-2.5 完整支持**：
  - 新增五个 2.5 节点：`Index TTS 2.5 - Base (多语言/语速)`、`Emotion Audio`、`Emotion Vector`、`Emotion Text`、`Cache Control`
  - 支持中/英/日/西/阿多语言合成与跨语种克隆
  - 原生语速控制 `duration_factor`（0.5–2.0）
  - 发音标注控制：拼音 `<行|XING2>` / 英文 CMU 音素 / 日语假名
  - 模型目录 `ComfyUI/models/IndexTTS-2.5/`，配套 `TTS2_5_download.py` 一键下载脚本与 `TTS2.5模型路径.txt` 路径说明
  - 新增 `workflow/TTS2.5.json` 示例工作流
  - 适配 transformers 5.x（vendor 内移植兼容层，与 TTS2 节点互不干扰）

### 2025-12-18

- **修复多个社区反馈问题**：
  - 老节点 `Index TTS` 现已支持 IndexTTS-2 模型 (#121)
  - 新增 `Index TTS 2 Pro (小说多角色)` 节点，支持 TTS 2.0 多角色小说朗读 (#111)
  - 修复 tensor 尺寸不匹配随机报错问题 (#122)
  - 支持 w2v-bert-2.0 本地离线加载，无需联网 (#72/#113)
  - 适配 transformers 4.50+ 版本 API 变化 (#117)
  - 更新 safetensors 版本要求 (#123)
  - 新增 README 常见问题解答 (FAQ) 部分

### 2025-06-24

- pro节点新增了对于字幕的json输出，感谢@qy8502提供的玩法思路

![image](https://github.com/user-attachments/assets/e7f5e92a-7f76-48a1-ba01-86143d10d359)


### 2025-06-05

- 改进了小说文本解析器（Novel Text Parser）的功能
  - 增加了对预格式化文本的检测和处理
  - 优化了对话检测和角色识别算法
  - 改进了中文角色名称的识别
  - 支持引号中的对话自动识别

## 多角色小说文本解析

本项目包含一个专门用于解析小说文本的节点（Novel Text Structure Node），可以将普通小说文本解析为多角色对话结构，以便生成更加自然的多声音TTS效果。

### 使用说明

- 节点会尝试自动识别小说中的角色对话和旁白部分
- 对话部分会标记为`<CharacterX>`形式（X为数字，最多支持5个角色）
- 旁白部分会标记为`<Narrator>`
- 解析后的文本可直接用于多声音TTS生成

### 局限性

- 当前解析算法并不完美，复杂的小说结构可能导致错误的角色识别
- 对于重要文本，建议使用LLM（如GPT等）手动拆分文本为以下格式：

```
<Narrator>少女此时就站在院墙那边，她有一双杏眼，怯怯弱弱。</Narrator>
<Narrator>院门那边，有个嗓音说：</Narrator>
<Character1>"你这婢女卖不卖？"</Character1>
<Narrator>宋集薪愣了愣，循着声音转头望去，是个眉眼含笑的锦衣少年，站在院外，一张全然陌生的面孔。</Narrator>
<Narrator>锦衣少年身边站着一位身材高大的老者，面容白皙，脸色和蔼，轻轻眯眼打量着两座毗邻院落的少年少女。</Narrator>
<Narrator>老者的视线在陈平安一扫而过，并无停滞，但是在宋集薪和婢女身上，多有停留，笑意渐渐浓郁。</Narrator>
<Narrator>宋集薪斜眼道：</Narrator>
<Character2>"卖！怎么不卖！"</Character2>
<Narrator>那少年微笑道：</Narrator>
<Character1>"那你说个价。"</Character1>
<Narrator>少女瞪大眼眸，满脸匪夷所思，像一头惊慌失措的年幼麋鹿。</Narrator>
<Narrator>宋集薪翻了个白眼，伸出一根手指，晃了晃，</Narrator>
<Character2>"白银一万两！"</Character2>
<Narrator>锦衣少年脸色如常，点头道：</Narrator>
<Character1>"好。"</Character1>
<Narrator>宋集薪见那少年不像是开玩笑的样子，连忙改口道：</Narrator>
<Character2>"是黄金万两！"</Character2>
<Narrator>锦衣少年嘴角翘起，道：</Narrator>
<Character1>"逗你玩的。"</Character1>
<Narrator>宋集薪脸色阴沉。</Narrator>
```

### 示例用法

1. 将小说文本输入到 Novel Text Structure 节点
2. 连接输出到 Index TTS Pro 节点
3. 设置不同角色的语音
4. 运行工作流生成多声音小说朗读
5. 实在不会看我最新增加的工作流
6. 如果你想在comfyui中一站式完成这个，我推荐你使用各类的llm节点，比如[kimichat](https://github.com/chenpipi0807/PIP_KIMI2comfyui)
7. 我也提供了一段llm提示词模板，你可以在llm_prompt模板.txt中看到他


### 2025-05-18

- 优化了长期以来transformers库4.50+版本的API变化与原始IndexTTS模型代码不兼容导致的生成报错问题


### 2025-05-16

- 新增对**IndexTTS-1.5**模型的支持
  - 现在可以在UI中通过下拉菜单切换不同版本的模型
  - 支持原始的Index-TTS和新的IndexTTS-1.5模型
  - 切换模型时会自动加载相应版本，无需重启ComfyUI
 
  ![微信截图_20250516182957](https://github.com/user-attachments/assets/ce13f02c-9834-43b8-82e9-5567bb226280)
  

### 2025-05-11
- 增加了seed功能，现在linux也可以重复执行抽卡了
- 增加了对 Apple Silicon MPS 设备的检测（仍需测试反馈~）


### 2025-04-23

![微信截图_20250423175608](https://github.com/user-attachments/assets/f2b15d8a-3453-4c88-b609-167b372aab74)


- 新增 **Audio Cleaner** 节点，用于处理TTS输出音频中的混响和杂音问题
  - 该节点可以连接在 Index TTS 节点之后，优化生成音频的质量
  - 主要功能：去除混响、降噪、频率滤波和音频归一化
  - 适用于处理有杂音或混响问题的TTS输出

- 修复了对于transformers版本强依赖的问题

#### Audio Cleaner 参数说明

**必需参数**：：
- **audio**: 输入音频（通常为 Index TTS 节点的输出）
- **denoise_strength**: 降噪强度（0.1-1.0，默认0.5）
  - 值越大，降噪效果越强，但可能影响语音自然度
- **dereverb_strength**: 去混响强度（0.0-1.0，默认0.7）
  - 值越大，去混响效果越强，适合处理在回声环境下录制的参考音频

**可选参数**：：
- **high_pass_freq**: 高通滤波器频率（20-500Hz，默认100Hz）
  - 用于过滤低频噪音，如环境嗡嗡声
- **low_pass_freq**: 低通滤波器频率（1000-16000Hz，默认8000Hz）
  - 用于过滤高频噪音
- **normalize**: 是否归一化音频（"true"或"false"，默认"true"）
  - 开启可使音量更均衡

#### 使用建议

- 对于有明显混响的音频，将 `dereverb_strength` 设置为 0.7-0.9
- 对于有背景噪音的音频，将 `denoise_strength` 设置为 0.5-0.8
- 如果处理后音频听起来不自然，尝试减小 `dereverb_strength` 和 `denoise_strength`
- 高通和低通滤波器可以微调以获得最佳人声效果


### 2025-04-25
- 优化了阿拉伯数字的发音判断问题；可以参考这个case使用：“4 0 9 0”会发音四零九零，“4090”会发音四千零九十； 


### 2025-04-26
- 优化英文逗号导致吞字的问题；


### 2025-04-29
- 修正了语言模式切换en的时候4090依然读中文的问题，auto现在会按照中英文占比确定阿拉伯数字读法
- 新增了从列表读取音频的方法，同时新增了一些音色音频供大家玩耍；你可以将自己喜欢的音频放入 ComfyUI-Index-TTS\TimbreModel 里，当然也很鼓励你能把好玩的声音分享出来。
- 示例用法如图：

![微信截图_20250429112255](https://github.com/user-attachments/assets/a0af9a5b-7609-4c34-adf5-e14321b379a7)


## 安装

### 安装节点

1. 将此代码库克隆或下载到ComfyUI的`custom_nodes`目录：

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/chenpipi0807/ComfyUI-Index-TTS.git
   ```

2. 安装依赖： 安装依赖：

   ```bash
   cd ComfyUI-Index-TTS
   .\python_embeded\python.exe -m pip install -r requirements.txt

   git pull # 更新很频繁你可能需要
   ```

### 下载模型

#### 原始版本 (Index-TTS)

1. 从[Hugging Face](https://huggingface.co/IndexTeam/Index-TTS/tree/main)或者[魔搭](https://modelscope.cn/models/IndexTeam/Index-TTS)下载IndexTTS模型文件
2. 将模型文件放置在`ComfyUI/models/Index-TTS`目录中（如果目录不存在，请创建）
3. 模型文件夹结构：
   
   ```
   ComfyUI/models/Index-TTS/
   ├── .gitattributes
   ├── bigvgan_discriminator.pth
   ├── bigvgan_generator.pth
   ├── bpe.model
   ├── config.yaml
   ├── configuration.json
   ├── dvae.pth
   ├── gpt.pth
   ├── README.md
   └── unigram_12000.vocab
   ```
   
   确保所有文件都已完整下载，特别是较大的模型文件如`bigvgan_discriminator.pth`(1.6GB)和`gpt.pth`(696MB)。

#### 新版本 (IndexTTS-1.5)

1. 从[Hugging Face](https://huggingface.co/IndexTeam/IndexTTS-1.5/tree/main)下载IndexTTS-1.5模型文件
2. 将模型文件放置在`ComfyUI/models/IndexTTS-1.5`目录中（如果目录不存在，请创建）
3. 模型文件夹结构与Index-TTS基本相同，但文件大小和内容会有所不同：
   
   ```
   ComfyUI/models/IndexTTS-1.5/
   ├── .gitattributes
   ├── bigvgan_discriminator.pth
   ├── bigvgan_generator.pth
   ├── bpe.model
   ├── config.yaml
   ├── configuration.json
   ├── dvae.pth
   ├── gpt.pth
   ├── README.md
   └── unigram_12000.vocab
   ```

## 使用方法

1. 在ComfyUI中，找到并添加`Index TTS`节点
2. 连接参考音频输入（AUDIO类型）
3. 输入要转换为语音的文本
4. 调整参数（语言、语速等）
5. 运行工作流获取生成的语音输出

### 示例工作流

项目包含一个基础工作流示例，位于`workflow/workflow.json`，您可以在ComfyUI中通过导入此文件来快速开始使用。

## 参数说明

### 必需参数

- **text**: 要转换为语音的文本（支持中英文）
- **reference_audio**: 参考音频，模型会复刻其声音特征
- **model_version**: 模型版本选择，可选项：
  - `Index-TTS`: 原始模型版本（默认）
  - `IndexTTS-1.5`: 新版本模型
- **language**: 文本语言选择，可选项：
  - `auto`: 自动检测语言（默认）
  - `zh`: 强制使用中文模式
  - `en`: 强制使用英文模式
- **speed**: 语速因子（0.5~2.0，默认1.0）

### 可选参数

以下参数适用于高级用户，用于调整语音生成质量和特性：

- **temperature** (默认1.0): 控制生成随机性，较高的值增加多样性但可能降低稳定性
- **top_p** (默认0.8): 采样时考虑的概率质量，降低可获得更准确但可能不够自然的发音
- **top_k** (默认30): 采样时考虑的候选项数量
- **repetition_penalty** (默认10.0): 重复内容的惩罚系数
- **length_penalty** (默认0.0): 生成内容长度的调节因子
- **num_beams** (默认3): 束搜索的宽度，增加可提高质量但降低速度
- **max_mel_tokens** (默认600): 最大音频token数量
- **sentence_split** (默认auto): 句子拆分方式

## 音色优化建议

要提高音色相似度：

- 使用高质量的参考音频（清晰、无噪音）
- 尝试调整`temperature`参数（0.7-0.9范围内效果较好）
- 增加`repetition_penalty`（10.0-12.0）可以提高音色一致性
- 对于长文本，确保`max_mel_tokens`足够大

## 故障排除

### 常见问题解答 (FAQ)

#### Q: w2v-bert-2.0 加载失败 / 401 Unauthorized 错误 (#72/#113)

**问题**: 运行时提示 `401 Client Error: Unauthorized for url: https://huggingface.co/facebook/w2v-bert-2.0`

**解决方案**: 
1. 下载 w2v-bert-2.0 模型到本地：从 [HuggingFace](https://huggingface.co/facebook/w2v-bert-2.0/tree/main) 下载所有文件
2. 放置到 `ComfyUI/models/IndexTTS-2/w2v-bert-2.0/` 目录
3. 确保目录包含 `config.json`、`model.safetensors`、`preprocessor_config.json` 等文件
4. 重启 ComfyUI，插件会自动使用本地模型，无需联网

#### Q: transformers 版本不兼容 (#117)

**问题**: 使用 transformers>=4.57.1 版本后 TTS2 无法使用

**解决方案**: 
- 推荐使用 `transformers==4.52.1` 或 `transformers==4.54.1`
- 安装命令: `pip install transformers==4.52.1`
- 本插件已适配 transformers 4.50+ 版本的 API 变化

#### Q: SafeTensorFile 没有 get_slice 属性 (#123)

**问题**: `AttributeError: 'SafeTensorFile' object has no attribute 'get_slice'`

**解决方案**: 
- 升级 safetensors 到最新版本: `pip install safetensors --upgrade`
- 确保版本 >= 0.4.3

#### Q: tensor 尺寸不匹配随机报错 (#122)

**问题**: 随机出现 `RuntimeError: Sizes of tensors must match except in dimension 1`

**解决方案**: 
- 此问题已在最新版本中修复
- 请更新插件到最新版本: `git pull`

#### Q: Python 3.13 / pynini 安装失败 (#125)

**问题**: Ubuntu 24 + Python 3.13 环境下 pynini 编译失败

**解决方案**: 
- pynini 目前不支持 Python 3.13
- 建议使用 Python 3.10 或 3.11
- Windows 用户不需要 pynini，可以忽略此错误

#### Q: 老节点不支持 IndexTTS-2 模型 (#121)

**解决方案**: 
- 最新版本已支持！在 `Index TTS` 节点的 `model_version` 下拉菜单中选择 `IndexTTS-2` 即可
- 也可以使用新的 `Index TTS 2 Pro (小说多角色)` 节点进行多角色小说朗读

#### Q: TTS 2.0 读小说功能 (#111)

**解决方案**: 
- 新增了 `Index TTS 2 Pro (小说多角色)` 节点
- 支持多角色语音合成，可配合 `小说文本结构化` 节点使用
- 支持最多 5 个角色 + 旁白

### 其他常见问题

- 如果出现"模型加载失败"，检查模型文件是否完整且放置在正确目录
- 对于Windows用户，无需额外安装特殊依赖，节点已优化
- 如果显示CUDA错误，尝试重启ComfyUI或减少`num_beams`值
- 如果你是pytorch2.7运行报错，短期无法适配，请尝试降级方案(.\python_embeded\python.exe -m pip install transformers==4.48.3)



## 鸣谢

- 基于原始[IndexTTS](https://github.com/index-tts/index-tts)模型
- 感谢ComfyUI社区的支持
- 感谢使用！
- 

## 许可证

请参考原始IndexTTS项目许可证。




