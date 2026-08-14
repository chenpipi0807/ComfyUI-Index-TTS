#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IndexTTS-2.5 Model Download Script
自动下载所有 IndexTTS-2.5 所需的模型文件（基于 huggingface_hub）
支持断点续传、镜像加速（HF_ENDPOINT）、本地缓存（HF_HOME），并按项目要求放置到固定目录结构
"""

import os
import sys
from pathlib import Path
from typing import List

from huggingface_hub import snapshot_download, hf_hub_download

# huggingface_hub 1.x 移除了 local_dir_use_symlinks / resume_download 参数，做兼容
import inspect as _inspect
_sn_params = set(_inspect.signature(snapshot_download).parameters)
_hf_params = set(_inspect.signature(hf_hub_download).parameters)


class ModelDownloader:
    def __init__(self):
        # 使用相对路径，确保在不同电脑上都能正常工作
        self.script_dir = Path(__file__).parent
        self.models_dir = self.script_dir.parent.parent / "models" / "IndexTTS-2.5"
        # 创建目录
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # 默认 endpoint（可在 ask_mirror_preference 中修改）
        self.endpoint_official = "https://huggingface.co"
        self.endpoint_mirror = "https://hf-mirror.com"  # 国内镜像
        self.current_endpoint = self.endpoint_official

        # 在模型目录下设置一个 Hugging Face 缓存目录，离线优先
        self.hf_home = self.models_dir / "hf_cache"
        os.environ.setdefault("HF_HOME", str(self.hf_home))

    def ask_mirror_preference(self):
        """询问是否使用国内镜像，并设置 HF_ENDPOINT 与缓存目录"""
        print("检测到您可能在中国大陆地区访问，是否使用国内镜像加速下载？")
        print("1. 使用官方地址 (huggingface.co)")
        print("2. 使用国内镜像 (hf-mirror.com) - 推荐")

        while True:
            choice = input("请选择 (1/2，默认为2): ").strip()
            if choice == "1":
                self.current_endpoint = self.endpoint_official
                print("已选择官方地址")
                break
            elif choice == "2" or choice == "":
                self.current_endpoint = self.endpoint_mirror
                print("已选择国内镜像")
                break
            else:
                print("请输入1或2")

        # 设置 HF_ENDPOINT 与 HF_HOME（在 Windows 下同样适用）
        os.environ["HF_ENDPOINT"] = self.current_endpoint
        os.environ.setdefault("HF_HOME", str(self.hf_home))
        # 可选：启用更快的传输（若安装了 hf_transfer）
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    # 统一的下载方法集合（基于 huggingface_hub）
    def _snapshot(self, repo_id: str, allow_patterns: List[str], local_dir: Path):
        local_dir.mkdir(parents=True, exist_ok=True)
        kwargs = dict(
            repo_id=repo_id,
            revision="main",
            allow_patterns=allow_patterns,
            local_dir=str(local_dir),
        )
        if "local_dir_use_symlinks" in _sn_params:
            kwargs["local_dir_use_symlinks"] = False
        if "resume_download" in _sn_params:
            kwargs["resume_download"] = True
        snapshot_download(**kwargs)

    def _download_file(self, repo_id: str, filename: str, local_path: Path):
        local_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs = dict(
            repo_id=repo_id,
            filename=filename,
            revision="main",
            local_dir=str(local_path.parent),
        )
        if "local_dir_use_symlinks" in _hf_params:
            kwargs["local_dir_use_symlinks"] = False
        if "resume_download" in _hf_params:
            kwargs["resume_download"] = True
        try:
            hf_hub_download(**kwargs)
            return True
        except Exception as e:
            print(f"✗ 下载失败: {repo_id}:{filename} -> {e}")
            return False

    def download_all(self):
        """按固定目录结构下载所有所需模型文件"""
        print(f"\n{'='*50}")
        print("开始下载 IndexTTS-2.5 所有模型...")
        print(f"{'='*50}")

        success = True

        # 1) 基础模型文件（IndexTeam/IndexTTS-2.5 根目录下）
        print("\n[1/5] 下载基础模型 (IndexTeam/IndexTTS-2.5 根目录)...")
        base_files = [
            "config.yaml",
            "codec.pth",
            "feat1.pt",
            "feat2.pt",
            "gpt.pth",
            "s2mel.pth",
            "wav2vec2bert_stats.pt",
            "multilingual_zh_ja_yue_char_del.tiktoken",
        ]
        try:
            self._snapshot(
                repo_id="IndexTeam/IndexTTS-2.5",
                allow_patterns=base_files,
                local_dir=self.models_dir,
            )
            print("✓ 基础模型文件下载完成")
        except Exception as e:
            print(f"✗ 基础模型下载失败: {e}")
            success = False

        # 2) qwen0.6bemo4-merge 子目录（情感文本分析用，不用 Emotion Text 节点可跳过）
        print("\n[2/5] 下载 qwen0.6bemo4-merge 子目录（情感文本分析模型）...")
        try:
            self._snapshot(
                repo_id="IndexTeam/IndexTTS-2.5",
                allow_patterns=["qwen0.6bemo4-merge/*"],
                local_dir=self.models_dir,
            )
            print("✓ qwen0.6bemo4-merge 下载完成")
        except Exception as e:
            print(f"✗ qwen0.6bemo4-merge 下载失败: {e}")
            success = False

        # 3) CampPlus 说话人嵌入
        print("\n[3/5] 下载 CampPlus 说话人嵌入...")
        try:
            ok = self._download_file(
                repo_id="funasr/campplus",
                filename="campplus_cn_common.bin",
                local_path=self.models_dir / "campplus_cn_common.bin",
            )
            if ok:
                print("✓ CampPlus 下载完成")
            else:
                success = False
        except Exception as e:
            print(f"✗ CampPlus 下载失败: {e}")
            success = False

        # 4) w2v-bert-2.0 整仓（facebook/w2v-bert-2.0）
        print("\n[4/5] 下载 Wav2Vec2Bert 特征提取器 (facebook/w2v-bert-2.0)...")
        try:
            self._snapshot(
                repo_id="facebook/w2v-bert-2.0",
                allow_patterns=["config.json", "preprocessor_config.json", "conformer_shaw.pt", "model.safetensors"],
                local_dir=self.models_dir / "w2v-bert-2.0",
            )
            print("✓ w2v-bert-2.0 下载完成")
        except Exception as e:
            print(f"✗ w2v-bert-2.0 下载失败: {e}")
            success = False

        # 5) BigVGAN 声码器（nvidia/bigvgan_v2_22khz_80band_256x）
        print("\n[5/5] 下载 BigVGAN 声码器 (nvidia/bigvgan_v2_22khz_80band_256x)...")
        try:
            self._snapshot(
                repo_id="nvidia/bigvgan_v2_22khz_80band_256x",
                allow_patterns=["config.json", "bigvgan_generator.pt"],
                local_dir=self.models_dir / "bigvgan",
            )
            print("✓ BigVGAN 下载完成")
        except Exception as e:
            print(f"✗ BigVGAN 下载失败: {e}")
            success = False

        return success

    def verify_downloads(self):
        """验证下载的文件"""
        print(f"\n{'='*50}")
        print("验证下载的文件...")
        print(f"{'='*50}")

        required_files = [
            "config.yaml",
            "codec.pth",
            "gpt.pth",
            "s2mel.pth",
            "feat1.pt",
            "feat2.pt",
            "wav2vec2bert_stats.pt",
            "multilingual_zh_ja_yue_char_del.tiktoken",
            "qwen0.6bemo4-merge",
            "campplus_cn_common.bin",
            "w2v-bert-2.0",
            "bigvgan",
        ]

        missing_files = []
        for file_path in required_files:
            full_path = self.models_dir / file_path
            if not full_path.exists():
                missing_files.append(file_path)
            else:
                print(f"✓ {file_path}")

        if missing_files:
            print(f"\n缺少以下文件:")
            for file_path in missing_files:
                print(f"✗ {file_path}")
            return False
        else:
            print(f"\n✓ 所有必需文件都已下载完成!")
            return True

    def run(self):
        """运行下载脚本"""
        print("IndexTTS-2.5 模型下载脚本")
        print("=" * 50)
        print(f"模型将下载到: {self.models_dir.absolute()}")

        # 询问镜像偏好
        self.ask_mirror_preference()

        try:
            ok = self.download_all()
        except KeyboardInterrupt:
            print("\n用户中断下载")
            sys.exit(1)
        except Exception as e:
            print(f"下载过程中出错: {e}")
            ok = False

        # 验证文件
        print(f"\n{'='*50}")
        print("下载完成报告")
        print(f"{'='*50}")
        if self.verify_downloads() and ok:
            print(f"\n🎉 所有模型下载完成! 模型路径: {self.models_dir.absolute()}")
        else:
            print(f"\n⚠️  部分文件可能缺失，请重新运行脚本或检查网络/镜像设置")


if __name__ == "__main__":
    try:
        downloader = ModelDownloader()
        downloader.run()
    except KeyboardInterrupt:
        print("\n下载已取消")
        sys.exit(1)
    except Exception as e:
        print(f"脚本运行出错: {e}")
        sys.exit(1)
