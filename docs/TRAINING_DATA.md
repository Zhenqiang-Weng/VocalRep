# 训练数据准备

本文描述的是本仓库 `utils/dataset.py` 与 `utils/dataset_with_spk.py` 的实际读取
行为。它与其他 MSS 项目的同名 `dataset_type` 不一定完全相同。

## 1. 先确认配置

数据文件名由配置中的 `training.instruments` 决定。例如当前
`ckpts/multi_stem/config.yaml` 使用：

```yaml
audio:
  sample_rate: 44100
  chunk_size: 529200

training:
  instruments: [vocals, backing_vocal, instrumental]
```

因此每首歌必须使用完全一致、区分大小写的 stem 名称。`chunk_size` 的单位是采样
点，不是秒；时长计算为 `chunk_size / sample_rate`。上例实际为 12 秒。

## 2. 选择 dataset type

| 类型 | 磁盘结构 | 读取行为 | 适用场景 |
| --- | --- | --- | --- |
| `1` | `root/<track>/<stem>.wav或flac` | 每个 stem 独立随机选歌曲和 offset | 非对齐随机混合 |
| `2` | `root/<stem>/*.wav或flac` | 从各 stem 素材池独立随机抽取 | 无配对素材池 |
| `3` | 单个 CSV，列为 `instrum,path` | 按 stem 从 CSV 独立随机抽取 | 文件散落在不同位置 |
| `4` | `root/<track>/<stem>.wav或flac` | 同一歌曲、同一 offset 对齐切块 | 对齐训练及说话人引导 |

当前说话人引导训练脚本使用 `MSSDatasetWithSpk`。只有 `dataset_type=4` 会加载
真实的 `embeddings.npy`；类型 1–3 会返回全零的 192 维 embedding。因此训练
`spk_*` 模型时推荐使用类型 4。

两套数据类对 `training.mix_instruments` 的处理不同：普通 `MSSDataset` 会尝试把
`mix_instruments` 中不属于 `training.instruments` 的 stem 作为仅参与 mixture、
不作为训练目标的额外来源，并为它们读取符合所选 layout 的文件。普通数据类使用
type 2 时，`mix_instruments` 还必须包含全部目标 instrument，否则 metadata 不完整。
`MSSDatasetWithSpk` 明确忽略这一扩展，mixture 只由 `training.instruments` 求和。
当前两个 `train_accelerate_bf16*.py` 入口使用的是 `MSSDatasetWithSpk`。

## 3. 推荐的 type 4 结构

```text
/absolute/path/to/train/
├── song_001/
│   ├── vocals.wav
│   ├── backing_vocal.wav
│   ├── instrumental.wav
│   └── embeddings.npy
├── song_002/
│   ├── vocals.flac
│   ├── backing_vocal.flac
│   ├── instrumental.flac
│   └── embeddings.npy
└── ...
```

同一歌曲目录中的所有 stem 应满足：

- 相同采样率，当前配置推荐 44.1 kHz；代码不会自动重采样。
- 相同起点、时长和帧数，确保 sample-level 对齐。
- 建议双声道。单声道会复制为双声道，超过双声道只保留前两路。
- 文件内容有限且非静音，不包含 NaN/Inf。
- 扩展名只能是小写 `.wav` 或 `.flac`。

缺少任一 stem 可能导致加载错误；静音素材在随机数据类型中可能被反复重抽。

### Speaker embedding

每首歌的 `embeddings.npy` 必须是数值型二维数组：

```text
shape = (N, 192), N >= 1
dtype = float32（推荐）
```

训练时会随机选择最多 20 行后求均值。验证时会对全部行求均值。embedding 应由该
歌曲目标说话人的干净人声片段生成。当前仓库不包含已验证的端到端 embedding 生成
流程；请使用可信的 CAMPPlus 提取环境，并在少量样本上人工核对身份一致性。

## 4. 其他训练布局

### Type 1：按歌曲存放、随机非对齐混合

目录与 type 4 相同，但加载每个 stem 时会独立随机选择歌曲和 offset。即使 stem
在磁盘上按歌曲对齐，训练时也不会保持这层对应关系。同一歌曲目录内的 stem 仍应
保持相同帧数：metadata 会把最短 stem 的长度用于该目录中的所有文件，短于一个
`chunk_size` 且帧数不一致的素材可能产生错误长度的切块。

### Type 2：独立 stem 素材池

```text
/absolute/path/to/train/
├── vocals/
│   ├── vocal_001.wav
│   └── vocal_002.flac
├── backing_vocal/
│   └── backing_001.wav
└── instrumental/
    └── instrumental_001.wav
```

每个配置中的 instrument 至少需要一个可读文件。

### Type 3：CSV 索引

当前实现只支持一次传入一个 CSV。CSV 至少包含 `instrum` 和 `path` 两列，这两个
列名的拼写与大小写必须完全一致；其他列会被忽略：

```csv
instrum,path
vocals,/absolute/path/to/vocal_001.wav
backing_vocal,/absolute/path/to/backing_001.wav
instrumental,/absolute/path/to/instrumental_001.wav
```

推荐使用绝对路径。相对路径按启动训练时的当前工作目录解析，而不是按 CSV 所在目录
解析。每个配置中的 instrument 至少需要一行。

## 5. 验证集结构

验证加载器固定扫描 `<valid_root>/<track>/mixture.wav`，不识别 FLAC mixture 或
更深层目录。每首验证歌曲还必须包含所有 stem 的 WAV 文件：

```text
/absolute/path/to/valid/
└── song_101/
    ├── mixture.wav
    ├── vocals.wav
    ├── backing_vocal.wav
    ├── instrumental.wav
    └── embeddings.npy
```

`mixture.wav` 应与各 stem 具有相同采样率、帧数和起点，并由训练配置中的输入 stem
求和得到。建议保存为 32-bit float WAV，以避免求和时整数 PCM 削波。若配置启用
`other_fix` 且 instrument 为 `other`，验证代码会使用 `mixture - vocals` 作为参考。

## 6. 启动前验证

安装训练依赖后运行仓库自带检查器：

```bash
python scripts/validate_training_data.py \
  --config-path ckpts/multi_stem/config.yaml \
  --dataset-type 4 \
  --data-path /absolute/path/to/train \
  --valid-path /absolute/path/to/valid \
  --require-embeddings
```

检查内容包括：目录和文件名、音频头、采样率、同曲 stem 帧数、CSV 列、验证
mixture，以及 embedding 的 `(N, 192)` 形状、数值 dtype 和 NaN/Inf。大型数据集
可以先抽查：

```bash
python scripts/validate_training_data.py \
  --config-path ckpts/multi_stem/config.yaml \
  --dataset-type 4 \
  --data-path /absolute/path/to/train \
  --valid-path /absolute/path/to/valid \
  --require-embeddings \
  --max-tracks 20
```

对 type 3，`--max-tracks 20` 表示每个 instrument 最多检查 20 行；instrument
覆盖性仍按完整 CSV 检查，因而按 instrument 分组排列的 CSV 不会被误报缺少后续 stem。

`train_accelerate.sh` 在启动训练前会自动执行完整检查。

## 7. Metadata 缓存

训练加载器会在 `results_path` 自动生成：

```text
metadata_<dataset_type>.pkl
```

不要手工编辑该文件。如果增删、替换、移动了音频，先停止训练并删除对应 metadata
缓存，再重新启动。缓存只按路径复用，不会可靠检测文件帧数或修改时间变化。

## 8. 填写训练 Bash

编辑 `train_accelerate.sh` 顶部这些参数：

```bash
TRAIN_SCRIPT="train_accelerate_bf16.py"
MODEL_TYPE="spk_bs_roformer_exportable"
CONFIG_PATH="ckpts/multi_stem/config.yaml"
RESULTS_PATH="results/multi_stem"
DATASET_TYPE=4
TRAIN_DATA_PATHS=("/absolute/path/to/train")
VALID_DATA_PATHS=("/absolute/path/to/valid")
START_CHECKPOINT=""  # 空字符串表示从头训练
GPU_IDS=(0)           # 多卡示例：(0 1 2 3)
DETERMINISTIC=false   # 需要较强可复现性时设为 true，速度可能下降
```

然后执行：

```bash
conda activate mss
python -m pip install -r requirements-train.txt
bash train_accelerate.sh
```

在线记录 W&B 时不要把 Key 写入脚本，改用环境变量：

```bash
export WANDB_API_KEY='在本机安全设置，不要提交到 Git'
export WANDB_MODE=online
bash train_accelerate.sh
```

数据集和模型权重必须具有合法来源与授权；不要将训练数据、生成的 metadata 或新权重
直接提交到普通 Git 历史。
