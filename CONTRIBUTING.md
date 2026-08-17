# Contributing

感谢参与项目维护。由于模型训练和权重体积较大，请保持每次改动小而可验证。

## 开发环境

```bash
conda activate mss
python -m pip install -r requirements-dev.txt
```

提交前至少执行：

```bash
ruff check .
python -m compileall -q .
bash -n train_accelerate.sh infer_with_spk.sh
```

涉及模型或数据管线时，还应使用小样本完成一次对应的前向或推理验证。

## 代码约定

- Python 版本以 3.10 为基线，使用 4 个空格缩进和 UTF-8/LF。
- 新增公共函数应提供清晰的类型标注和 docstring。
- 路径、GPU 编号、密钥和服务器地址通过参数或环境变量传入，不硬编码到源码。
- 不提交数据集、日志、缓存、导出文件或本地编辑器配置。
- 不在 issue、日志或提交中写入 API Key、访问令牌和其他凭据。
- 对旧模块做大规模格式化时，与功能改动分开提交，便于审阅。

## 权重与大文件

- `ckpts/` 中已有权重是原项目资产，未经明确确认不得替换或重新生成。
- 新增模型权重前先确认来源、许可证和校验值，并使用 Git LFS。
- 不运行会改写历史权重对象的 `git lfs migrate`，除非维护者明确批准历史重写。

## 提交说明

提交信息使用简洁的祈使句并说明范围，例如：

```text
docs: document CUDA installation
fix: correct discriminator dataset import
chore: remove tracked Python caches
```
