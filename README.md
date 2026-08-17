# Music Source Separation

一个基于 PyTorch 的音乐源分离实验项目，包含 BS-Roformer、Mel-Band
Roformer、SCNet、BandIt v2，以及可选的说话人引导、判别器和扩散模块。

> [!IMPORTANT]
> 仓库仍处于实验整理阶段。基础推理链路和部分模型代码已提供，但训练脚本、
> 说话人模型资产及部分旧判别器代码尚未经过完整的端到端验证。使用前请检查
> 配置、数据路径和设备参数。

## 仓库结构

```text
.
├── ckpts/                 # 原项目配置和 Git LFS 权重
├── models/                # 音源分离模型
├── diffusion/             # 可选扩散模块
├── discriminator/         # 可选判别器模块
├── spk_extract/           # CAMPPlus 说话人特征代码
├── utils/                 # 数据、损失、指标和推理工具
├── mss_api/               # API/导出相关实验代码
├── inference.py           # 基础文件夹推理入口
├── inference_with_spk.py  # 说话人引导实验入口
└── train_accelerate_*.py  # Accelerate 训练入口
```

## 环境准备

推荐环境：

- Linux 或 WSL2
- Python 3.10
- NVIDIA GPU（CPU 可用于基础验证，但大型模型推理会很慢）
- FFmpeg、Git LFS

创建环境：

```bash
conda create -n mss python=3.10 -y
conda activate mss
python -m pip install --upgrade pip
```

先根据设备安装匹配的 PyTorch。下面是 CUDA 13.0 的示例；其他平台请使用
[PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)生成命令。

```bash
python -m pip install \
  torch==2.11.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130
```

安装项目的主要运行依赖：

```bash
python -m pip install -r requirements.txt
```

训练环境在此基础上安装：

```bash
python -m pip install -r requirements-train.txt
```

需要可选优化器、旧判别器、实时音频或 ONNX 导出时，再安装：

```bash
python -m pip install -r requirements-optional.txt
```

Debian/Ubuntu/WSL 先安装基础系统依赖和 Git LFS：

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg git-lfs libsndfile1
```

仅在使用实时音频的 `pyaudio` 时再安装 PortAudio 开发库：

```bash
sudo apt-get install -y portaudio19-dev
```

## 原始权重

`ckpts/` 中的权重属于原项目资产，并通过 Git LFS 跟踪。克隆后运行：

```bash
git lfs install
git lfs pull
```

可以用以下命令确认文件不是未下载的 LFS 指针：

```bash
git lfs status
stat -c '%n %s bytes' ckpts/multi_stem/*.ckpt
```

本次工程整理不修改、替换或重新生成 `ckpts/` 中的任何权重。

## 零条件推理候选（未验证）

仓库中唯一随配置记录的 checkpoint 属于 `spk_bs_roformer`。下面的
`inference.py` 不传入 speaker embedding，模型实际以 `speaker_embedding=None`
运行，因此它只适合作为零条件/blind smoke 候选，不代表完整的说话人引导推理。
将待处理音频放入单独目录，例如 `dataset/demo/`，候选命令为：

```bash
python inference.py \
  --model_type spk_bs_roformer \
  --config_path ckpts/multi_stem/config.yaml \
  --start_check_point ckpts/multi_stem/model_spk_bs_roformer_ep_5_sisdr_9.8275.ckpt \
  --input_folder dataset/demo \
  --store_dir results/demo \
  --device_ids 0
```

该模型类型来自仓库原有推理脚本，但当前工作区只含 135 字节的 LFS 指针，未下载
约 1.45 GB 原始权重，因此这条命令尚未在当前机器完成端到端验证，零条件输出质量
也没有保证。强制使用 CPU 时添加 `--force_cpu`。查看全部参数：

```bash
python inference.py --help
```

## 说话人引导推理

`inference_with_spk.py` 仍属于实验入口。它按三阶段运行：先做零条件 blind
separation，再从得到的 `vocals.wav` 调用外部脚本提取 embedding，最后使用该
embedding 再做 guided separation。它需要额外的 CAMPPlus 模型和说话人嵌入提取
脚本。通过环境变量指定外部提取环境：

```bash
export MSS_SPEAKER_PYTHON=/path/to/speaker-env/bin/python
export MSS_SPEAKER_SCRIPT=/path/to/batch_extract_embeddings.py
```

外部脚本必须为每个输入文件生成
`<store_dir>/embeddings/<输入文件名>/embedding.npy`（单数）。这与训练集每首歌曲
目录中的 `embeddings.npy`（复数）不是同一个文件约定。每次运行都会删除并重建
`<store_dir>/embeddings`，因此请为 `--store_dir` 使用专用输出目录，不要在该子目录
保存需要保留的文件。

仓库中现有的两个 CAMPPlus 目录记录是旧 gitlink，但缺少对应的
`.gitmodules` 来源信息，因此全新克隆不会自动取得该模型。请在确认原始模型
来源后自行放置资产；不要把未知或替代权重覆盖到 `ckpts/`。

## 训练

训练使用 Hugging Face Accelerate。安装训练依赖后可先完成本机配置：

```bash
accelerate config
```

在 [train_accelerate.sh](train_accelerate.sh) 顶部填写模型、配置、训练集、验证集、
输出目录和 GPU 参数后启动。数据目录格式与检查清单见
[训练数据准备说明](docs/TRAINING_DATA.md)。建议先使用单卡、小数据集验证数据读取
和一次前向/反向传播，再启动多卡训练。

## 开发检查

```bash
python -m pip install -r requirements-dev.txt
ruff check .
python -m compileall -q .
bash -n train_accelerate.sh infer_with_spk.sh
```

贡献约定见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 已知限制

- 当前仓库没有完整的端到端测试数据，CI 只执行不需要模型权重的静态检查。
- 旧判别器目录仍保留部分历史接口和脚本路径，尚未全部迁移为包内导入。
- TensorRT、SageAttention 等组件与本机 CUDA/编译器强相关，不包含在默认依赖中。
- 项目根目录目前没有明确许可证；使用或分发前请由项目作者补充授权条款。

## 致谢

部分训练与推理结构参考了
[ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)。
各源码文件中保留的第三方版权和许可证声明仍然适用。
