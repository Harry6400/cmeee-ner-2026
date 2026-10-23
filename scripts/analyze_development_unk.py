"""Read existing N1 development predictions; never load a model or infer."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

TYPES = ('dis', 'sym', 'dru', 'equ', 'pro', 'bod', 'ite', 'mic', 'dep')
LENGTHS = ('<=2', '3-5', '6-10', '11-20', '>20')
VOCAB_SHA256 = '45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c'


def length_group(size):
    return next(name for upper, name in zip((2, 5, 10, 20, float('inf')), LENGTHS) if size <= upper)


def entity_set(row):
    result = set()
    for entity in row['entities']:
        start, end, kind = entity['start'], entity['end'], entity['type']
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(row['text']) or kind not in TYPES
                or row['text'][start:end] != entity['text']):
            raise ValueError('Invalid Unicode entity in ' + row['id'])
        result.add((start, end, kind))
    return result


def analyze(gold, predictions, vocabulary):
    if len(gold) != len(predictions):
        raise ValueError('Document counts differ')
    counts = Counter()
    groups = Counter()
    seen = set()
    for reference, prediction in zip(gold, predictions):
        if (reference['id'] in seen or reference['id'] != prediction['id']
                or reference['text'] != prediction['text']):
            raise ValueError('Duplicate ID or document identity/order mismatch')
        seen.add(reference['id'])
        expected, actual = entity_set(reference), entity_set(prediction)
        counts.update(gold=len(expected), predicted=len(actual), tp=len(expected & actual))
        counts.update(characters=len(reference['text']),
                      unknown_characters=sum(ch not in vocabulary for ch in reference['text']))
        for start, end, kind in expected:
            unknown = 'contains_UNK' if any(ch not in vocabulary for ch in reference['text'][start:end]) else 'no_UNK'
            hit = int((start, end, kind) in actual)
            for dimension, stratum in (('overall', 'all'), ('type', kind),
                                       ('length', length_group(end - start))):
                groups[dimension, stratum, unknown, 'gold'] += 1
                groups[dimension, stratum, unknown, 'tp'] += hit
    total_fn = counts['gold'] - counts['tp']
    rows = []
    for dimension, strata in (('overall', ('all',)), ('type', TYPES), ('length', LENGTHS)):
        for stratum in strata:
            stratum_fn = sum(groups[dimension, stratum, u, 'gold'] - groups[dimension, stratum, u, 'tp']
                             for u in ('contains_UNK', 'no_UNK'))
            for unknown in ('contains_UNK', 'no_UNK'):
                support, tp = (groups[dimension, stratum, unknown, m] for m in ('gold', 'tp'))
                fn = support - tp
                rows.append(dict(dimension=dimension, stratum=stratum, encoding_group=unknown,
                                 gold=support, tp=tp, fn=fn,
                                 recall=tp / support if support else None,
                                 share_of_all_FN=fn / total_fn if total_fn else None,
                                 share_of_stratum_FN=fn / stratum_fn if stratum_fn else None))
    denominator = counts['predicted'] + counts['gold']
    return dict(samples=len(gold), strict=dict(tp=counts['tp'], predicted=counts['predicted'], gold=counts['gold'],
                fn=total_fn, fp=counts['predicted'] - counts['tp'],
                f1=2 * counts['tp'] / denominator if denominator else 0.0),
                characters=counts['characters'], unknown_characters=counts['unknown_characters'],
                character_UNK_rate=counts['unknown_characters'] / counts['characters'] if counts['characters'] else None,
                rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True, help='A new directory, never overwritten')
    args = parser.parse_args()
    root = args.project_root.resolve()
    summary_path = root / 'runs/stage3-20261019/N1-lr-5e-05/summary.json'
    summary = json.loads(summary_path.read_text())
    paths = dict(gold=root / 'data/processed/dev.jsonl',
                 prediction=summary_path.parent / f"dev-epoch-{summary['best_epoch']:02d}.jsonl",
                 vocabulary=root / 'models/bert-base-chinese/vocab.txt', summary=summary_path)
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    if hashes['vocabulary'] != VOCAB_SHA256:
        raise ValueError('Frozen BERT vocabulary hash differs')
    read = lambda path: [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = analyze(read(paths['gold']), read(paths['prediction']), set(paths['vocabulary'].read_text().splitlines()))
    for key in ('tp', 'predicted', 'gold'):
        if result['strict'][key] != summary['metrics']['micro'][key]:
            raise ValueError('Existing development metric differs: ' + key)
    if result['samples'] != 1503 or abs(result['strict']['f1'] - summary['best_f1']) > 1e-12:
        raise ValueError('Frozen development selection differs')
    result.update(created_at=datetime.now(timezone.utc).isoformat(), status='PASS',
                  scope='Existing internal development output; no inference, training or API',
                  selected_epoch=summary['best_epoch'],
                  definition='Gold entity contains UNK iff any raw Unicode character is absent from the fixed BERT vocabulary; case-sensitive; strict unique (start,end,type). This is not unseen entity surface.',
                  interpretation='Association only; entity type, length and vocabulary mix can confound the comparison.',
                  sources={k: dict(path=str(p.relative_to(root)), sha256=hashes[k]) for k, p in paths.items()})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    with (args.output / 'strata.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['rows'][0]))
        writer.writeheader()
        writer.writerows(result['rows'])
    print(json.dumps({k: result[k] for k in ('status', 'samples', 'strict', 'character_UNK_rate')}))


if __name__ == '__main__':
    main()
