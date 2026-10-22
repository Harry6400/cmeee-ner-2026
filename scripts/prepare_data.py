"""核查官方CMeEE-V2并生成隔离划分；只用标准库，不训练、不调用API。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

TYPES = ('dis', 'sym', 'dru', 'equ', 'pro', 'bod', 'ite', 'mic', 'dep')


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    path.chmod(0o600)


def convert_row(row: dict, split: str, index: int) -> tuple[dict, list[dict]]:
    ident = f'cmeee-v2-{split}-{index:05d}'
    text = row.get('text')
    if not isinstance(text, str) or not text:
        raise ValueError(f'{ident}: text必须为非空字符串')
    converted = {'id': ident, 'source_split': split, 'source_index': index,
                 'group_id': digest(text.encode('utf-8')), 'text': text}
    if split == 'test':
        converted['label_status'] = 'official_test_not_used_for_evaluation'
        return converted, []
    raw_entities = row.get('entities')
    if not isinstance(raw_entities, list):
        raise ValueError(f'{ident}: 缺少参考entities')
    entities, warnings, seen = [], [], set()
    for n, entity in enumerate(raw_entities):
        a, b, kind = entity.get('start_idx'), entity.get('end_idx'), entity.get('type')
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(text):
            raise ValueError(f'{ident}: entity[{n}]越界或坐标非法')
        if kind not in TYPES:
            raise ValueError(f'{ident}: entity[{n}]未知类别{kind!r}')
        literal = text[a:b]
        if 'entity' in entity and entity['entity'] != literal:
            raise ValueError(f'{ident}: entity[{n}]原文与实体文本不一致；不自动修复')
        key = (a, b, kind)
        if key in seen:
            warnings.append({'id': ident, 'entity_index': n, 'kind': 'duplicate_annotation_retained'})
        seen.add(key)
        entities.append({'start': a, 'end': b, 'type': kind, 'text': literal})
    converted['entities'] = entities
    return converted, warnings


def summarize(rows: list[dict]) -> dict:
    lengths = sorted(len(r['text']) for r in rows)
    classes, entity_lengths = Counter(), []
    nested_samples = nested_entities = crossing_samples = coextensive_samples = 0
    for row in rows:
        entities = row.get('entities', [])
        classes.update(e['type'] for e in entities)
        entity_lengths.extend(e['end'] - e['start'] for e in entities)
        nested, crossing, coextensive = set(), False, False
        for i, a in enumerate(entities):
            for j, b in enumerate(entities[i + 1:], i + 1):
                if (a['start'], a['end']) == (b['start'], b['end']):
                    coextensive |= a['type'] != b['type']
                elif (a['start'] <= b['start'] and a['end'] >= b['end']) or (
                    b['start'] <= a['start'] and b['end'] >= a['end']
                ):
                    nested.update((i, j))
                elif max(a['start'], b['start']) < min(a['end'], b['end']):
                    crossing = True
        nested_samples += bool(nested)
        nested_entities += len(nested)
        crossing_samples += crossing
        coextensive_samples += coextensive
    return {
        'samples': len(rows), 'unique_text_groups': len({r['group_id'] for r in rows}),
        'characters': sum(lengths), 'characters_min': min(lengths, default=0),
        'characters_median': statistics.median(lengths) if lengths else 0,
        'characters_p95': lengths[max(0, math.ceil(.95 * len(lengths)) - 1)] if lengths else 0,
        'characters_max': max(lengths, default=0),
        'character_length_bins': {
            label: sum(low <= n <= high for n in lengths)
            for label, low, high in [('1-64', 1, 64), ('65-128', 65, 128),
                                    ('129-256', 129, 256), ('257-512', 257, 512),
                                    ('513+', 513, float('inf'))]
        },
        'entities': sum(classes.values()), 'entities_by_type': {t: classes[t] for t in TYPES},
        'max_entity_characters': max(entity_lengths, default=0),
        'nested_samples': nested_samples, 'entities_in_strict_nesting_pairs': nested_entities,
        'crossing_samples': crossing_samples, 'coextensive_different_type_samples': coextensive_samples,
        'empty_entity_samples': sum(not r.get('entities') for r in rows)
    }


def split_records(train: list[dict], holdout: list[dict], seed: int, size: int):
    holdout_groups = {r['group_id'] for r in holdout}
    quarantine = [r for r in train if r['group_id'] in holdout_groups]
    eligible = [r for r in train if r['group_id'] not in holdout_groups]
    groups = sorted({r['group_id'] for r in eligible})
    if len(groups) < 2 or len(holdout) < size:
        raise ValueError('训练文本组或最终评估样本不足，不静默缩减规模')
    random.Random(seed).shuffle(groups)
    dev_groups = set(groups[:max(1, min(len(groups) - 1, round(len(groups) * .1)))])
    internal_train = [r for r in eligible if r['group_id'] not in dev_groups]
    internal_dev = [r for r in eligible if r['group_id'] in dev_groups]
    # 同文样本可能存在多份标注；共同评估保留官方分布，不悄悄挑选容易样本。
    selected = sorted(random.Random(seed + 1).sample(range(len(holdout)), size))
    common = [holdout[i] for i in selected]
    return internal_train, internal_dev, quarantine, common


def near_overlap(train: list[dict], holdout: list[dict], seed: int) -> dict:
    """启发式8值MinHash候选，精确字符三元组Jaccard复核；不保证穷尽。"""
    rng = random.Random(seed)
    mask = (1 << 64) - 1
    coefficients = [(rng.getrandbits(64) | 1, rng.getrandbits(64)) for _ in range(8)]
    shingles, buckets = {}, defaultdict(list)

    def signature(row):
        text = ''.join(row['text'].split())
        grams = {text[i:i+3] for i in range(max(1, len(text) - 2))}
        shingles[row['id']] = grams
        hashed = [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), 'big') for s in grams]
        return [min((a * h + b) & mask for h in hashed) for a, b in coefficients]

    train_by_id = {r['id']: r for r in train}
    for row in train:
        sig = signature(row)
        for band in range(4):
            buckets[(band, *sig[2*band:2*band+2])].append(row['id'])
    pairs, truncated, checked = [], 0, 0
    for row in holdout:
        sig = signature(row)
        candidates = set()
        for band in range(4):
            candidates.update(buckets[(band, *sig[2*band:2*band+2])])
        if len(candidates) > 2000:
            truncated += 1
        for ident in sorted(candidates)[:2000]:
            if train_by_id[ident]['group_id'] == row['group_id']:
                continue
            a, b = shingles[ident], shingles[row['id']]
            checked += 1
            score = len(a & b) / len(a | b) if a | b else 1.0
            if score >= .9:
                pairs.append({'training_id': ident, 'holdout_id': row['id'], 'jaccard': score})
    return {'method': '8-value minhash;4 bands;whitespace-only candidate normalization;character trigrams',
            'exhaustive': False, 'threshold': .9, 'checked_candidate_pairs': checked,
            'candidate_limit_per_holdout': 2000, 'holdout_rows_with_truncated_candidates': truncated,
            'pairs': pairs, 'action': 'audit_and_disclose_only;raw_text_and_labels_unchanged'}


def prepare(archive: Path, output: Path, seed=20261019, evaluation_size=500) -> dict:
    if (output / 'manifests/readiness.json').exists():
        raise FileExistsError('已存在冻结清单；不覆盖已有划分')
    raw_target = output / 'raw/CMeEE-V2.zip'
    if archive.resolve() != raw_target.resolve():
        raw_target.parent.mkdir(parents=True, exist_ok=True)
        if raw_target.exists() and raw_target.read_bytes() != archive.read_bytes():
            raise FileExistsError('目标原始ZIP不同，拒绝覆盖')
        if not raw_target.exists():
            raw_target.write_bytes(archive.read_bytes())
        archive = raw_target
    inventory, source = [], {}
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('原始ZIP校验失败')
        for member in z.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in member.filename:
                raise ValueError('ZIP含不安全路径')
            inventory.append({'member': member.filename, 'bytes': member.file_size})
        for split in ('train', 'dev', 'test'):
            members = [n for n in z.namelist() if PurePosixPath(n).name == f'CMeEE-V2_{split}.json']
            if len(members) != 1:
                raise ValueError(f'要求唯一官方{split}文件，实际{members}')
            raw = z.read(members[0])
            rows = json.loads(raw.decode('utf-8-sig'))
            if not isinstance(rows, list):
                raise ValueError(f'{split}: 顶层必须是列表')
            source[split] = {'member': members[0], 'sha256': digest(raw), 'rows': rows}
    converted, warnings, fatal = {}, [], []
    for split in ('train', 'dev', 'test'):
        converted[split] = []
        for i, row in enumerate(source[split]['rows']):
            try:
                item, notes = convert_row(row, split, i)
                converted[split].append(item)
                warnings.extend(notes)
            except (ValueError, AttributeError) as exc:
                fatal.append({'source_split': split, 'source_index': i, 'error': str(exc)})
    write_json(output / 'manifests/archive-inventory.json', inventory)
    write_json(output / 'manifests/annotation-issues.json', {'fatal': fatal, 'warnings': warnings})
    if fatal:
        raise ValueError(f'{len(fatal)}条致命标注问题，停止划分；见annotation-issues.json')
    train, dev, quarantine, common = split_records(converted['train'], converted['dev'], seed, evaluation_size)
    sets = {'train': train, 'dev': dev, 'holdout': converted['dev'], 'common500': common}
    memberships = {name: [r['id'] for r in rows] for name, rows in sets.items()}
    for a, b in [('train', 'dev'), ('train', 'holdout'), ('dev', 'holdout')]:
        if {r['group_id'] for r in sets[a]} & {r['group_id'] for r in sets[b]}:
            raise AssertionError(f'划分同文交叉:{a}/{b}')
    annotation_variants = defaultdict(set)
    for row in converted['train'] + converted['dev']:
        annotation_variants[row['group_id']].add(json.dumps(sorted(
            (e['start'], e['end'], e['type']) for e in row['entities'])))
    conflict_groups = [g for g, variants in annotation_variants.items() if len(variants) > 1]
    overlaps = {'train_vs_dev': near_overlap(train, dev, seed),
                'train_and_dev_vs_holdout': near_overlap(train + dev, converted['dev'], seed)}
    write_json(output / 'manifests/near-overlap.json', overlaps)
    write_json(output / 'manifests/split-membership.json', memberships)
    write_json(output / 'manifests/quarantine.json', {
        'reason': 'official_train_text_also_in_official_dev', 'ids': [r['id'] for r in quarantine],
        'annotation_conflict_group_ids': conflict_groups})
    for name in ('train', 'dev'):
        write_rows(output / f'processed/{name}.jsonl', sets[name])
    write_rows(output / 'sealed/holdout.jsonl', sets['holdout'])
    write_rows(output / 'sealed/common500.jsonl', common)
    stats = {'official': {s: summarize(converted[s]) for s in ('train', 'dev')},
             'official_test': {'samples': len(converted['test']), 'labels_used': False},
             'project': {name: summarize(rows) for name, rows in sets.items()},
             'quarantined_training_samples': len(quarantine), 'annotation_conflict_text_groups': len(conflict_groups),
             'annotation_warnings': len(warnings),
             'near_overlap_pairs': {name: len(value['pairs']) for name, value in overlaps.items()}}
    write_json(output / 'manifests/statistics.json', stats)
    files = {p.relative_to(output).as_posix(): digest(p.read_bytes())
             for p in sorted(output.rglob('*')) if p.is_file() and 'raw' not in p.relative_to(output).parts}
    readiness = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
                 'seed': seed, 'evaluation_sampling_seed': seed + 1, 'evaluation_size': evaluation_size,
                 'dataset': 'CMeEE-V2', 'source_archive_path': 'raw/CMeEE-V2.zip',
                 'archive_sha256': digest(archive.read_bytes()),
                 'source_members': {s: {k: v for k, v in d.items() if k != 'rows'} for s, d in source.items()},
                 'coordinate_convention': 'Unicode codepoints;start inclusive,end exclusive;official CMeEE-V2 end_idx already exclusive',
                 'raw_text_modified': False, 'same_text_split_overlap': 0,
                 'near_overlap_exhaustive': False, 'heldout_content_inspected_for_tuning': False,
                 'status': 'PASS_WITH_WARNINGS' if warnings or conflict_groups or any(
                     item['pairs'] for item in overlaps.values()) or any(
                     value['crossing_samples'] or value['coextensive_different_type_samples']
                     for value in stats['official'].values()) else 'PASS',
                 'files': files}
    write_json(output / 'manifests/readiness.json', readiness)
    return stats


def verify(output: Path) -> dict:
    manifest = json.loads((output / 'manifests/readiness.json').read_text(encoding='utf-8'))
    archive = output / manifest['source_archive_path']
    if not archive.is_file() or digest(archive.read_bytes()) != manifest['archive_sha256']:
        raise ValueError('原始官方ZIP缺失或不一致')
    mismatches = [name for name, expected in manifest['files'].items()
                  if not (output / name).is_file() or digest((output / name).read_bytes()) != expected]
    if mismatches:
        raise ValueError(f'冻结数据不一致:{mismatches}')
    return {'verified_files': len(manifest['files']), 'status': manifest['status'], 'api_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--output', type=Path, default=Path('data'))
    parser.add_argument('--seed', type=int, default=20261019)
    parser.add_argument('--evaluation-size', type=int, default=500)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if not args.verify and args.archive is None:
        parser.error('准备时必须传--archive；只读核查用--verify')
    result = verify(args.output) if args.verify else prepare(args.archive, args.output, args.seed, args.evaluation_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))
