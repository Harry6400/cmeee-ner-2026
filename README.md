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

## Data preparation

Use an authorized CMeEE-V2 archive stored outside this repository. Preparation writes local files under the ignored `data/` directory; no corpus or generated split is included in Git. Review `configs/data-protocol.json` and `THIRD_PARTY_NOTICES.md` before use.

```sh
uv run --locked python scripts/prepare_data.py \
  --archive /path/to/CMeEE-V2.zip \
  --output data \
  --seed 20261019 \
  --evaluation-size 500

uv run --locked python scripts/prepare_data.py --output data --verify
```

The preparation checks character offsets and labels, groups identical text before splitting, quarantines exact train-to-holdout overlap, and records a heuristic near-overlap audit. It does not train a model or call an API.

Run the synthetic boundary tests with:

```sh
python3 -B -m unittest discover -s tests -p 'test_prepare_data.py' -v
```

## Data and model assets

Dataset archives, pretrained models, trained checkpoints, and experiment outputs are not included. Obtain each asset from its authoritative source, review its license, and keep it outside version control. See `THIRD_PARTY_NOTICES.md` for the project-level asset boundary.

## Evaluation audit utilities

scripts/stage4_metrics.py provides an independent, standard-library-only audit of exact entity spans, character/type metrics, invalid-output availability, paired document bootstrap intervals, and summaries across the three configured seeds. It does not fetch data, call a service, or include datasets, predictions, checkpoints, or evaluation results.

Run its synthetic-only checks with:

    python3 -B -m unittest discover -s tests -p 'test_stage4_metrics_public.py' -v
