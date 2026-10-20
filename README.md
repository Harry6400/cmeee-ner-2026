# 中文医学实体抽取

A compact source foundation for Chinese medical named entity recognition with exact character spans. Entity offsets use Unicode code points and half-open intervals: `[start, end)`.

## Source modules

- `scripts/ner_core.py`: entity validation, span windows, merging, dictionary helpers, and strict entity metrics.
- `scripts/ner_model.py`: character-level encoders and a multi-label GlobalPointer span head.
- `scripts/entity_wire.py`: converts generated surface forms and occurrence indices into exact source-text offsets.
- `scripts/ner_cli.py`: validates and prints one entity record.

## Environment

Python 3.11 is pinned in `.python-version`; dependencies are locked in `uv.lock`. Install them with:

```sh
uv sync --locked
```

Validate a record from the repository root:

```sh
uv run --locked python scripts/ner_cli.py \
  --text '出现发热与咳嗽。' \
  --entities '[{"start":2,"end":4,"type":"sym","text":"发热"}]'
```

## Data and model assets

Dataset archives, pretrained models, trained checkpoints, and experiment outputs are not included. Obtain each asset from its authoritative source, review its license, and keep it outside version control. See `THIRD_PARTY_NOTICES.md` for the project-level asset boundary.
