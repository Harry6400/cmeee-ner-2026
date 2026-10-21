# 中文医学实体抽取

本仓库提供中文医学命名实体识别的源码基础，支持精确字符跨度定位。实体偏移按 Unicode 码点计数，区间采用左闭右开格式 `[start, end)`。

## 主要源码模块

- `scripts/ner_core.py`：实体校验、文本窗口处理、跨度合并、词典辅助函数和严格实体指标。
- `scripts/ner_model.py`：字符级编码器与多标签 GlobalPointer 跨度分类头。
- `scripts/entity_wire.py`：将生成的实体表面形式及其出现序号映射为原文中的精确坐标。
- `scripts/ner_cli.py`：校验并输出一条实体记录。

## 运行环境

项目固定使用 Python 3.11，依赖版本由 `uv.lock` 锁定。安装依赖：

```sh
uv sync --locked
```

在仓库根目录校验一条实体记录：

```sh
uv run --locked python scripts/ner_cli.py \
  --text '出现发热与咳嗽。' \
  --entities '[{"start":2,"end":4,"type":"sym","text":"发热"}]'
```

## 数据准备

请使用经授权的 CMeEE-V2 数据压缩包，并将它保存在本仓库之外。数据准备程序会把本地文件写入已被 Git 忽略的 `data/` 目录；Git 不包含语料或生成的数据划分。使用前请阅读 `configs/data-protocol.json` 和 `THIRD_PARTY_NOTICES.md`。

```sh
uv run --locked python scripts/prepare_data.py \
  --archive /path/to/CMeEE-V2.zip \
  --output data \
  --seed 20261019 \
  --evaluation-size 500

uv run --locked python scripts/prepare_data.py --output data --verify
```

准备过程会校验字符偏移与标签，按相同文本分组后再划分数据，隔离训练集与留出集中的完全重复文本，并记录启发式近似重叠审计。该程序不会训练模型，也不会调用 API。

运行合成边界测试：

```sh
python3 -B -m unittest discover -s tests -p 'test_prepare_data.py' -v
```

## 数据与模型资源

仓库不包含数据集压缩包、预训练模型、训练检查点或实验输出。请从权威来源获取各项资源，核对相应许可，并将资源保存在版本控制之外。项目级资源范围见 `THIRD_PARTY_NOTICES.md`。
