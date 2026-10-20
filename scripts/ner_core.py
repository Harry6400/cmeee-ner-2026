"""纯CPU的统一实体、窗口、词典与实体级指标。坐标始终为原文字符[start,end)。"""
from __future__ import annotations
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

TYPES = ('dis', 'sym', 'dru', 'equ', 'pro', 'bod', 'ite', 'mic', 'dep')
ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read_rows(path):
    path = Path(path)
    if 'sealed' in path.resolve().parts:
        raise ValueError('阶段二禁止读取最终留出集')
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(2**20), b''):
            h.update(part)
    return h.hexdigest()


def validate(row):
    if not isinstance(row.get('id'), str) or not isinstance(row.get('text'), str):
        raise ValueError('缺少样本id或原文')
    if not isinstance(row.get('entities'), list):
        raise ValueError('缺少entities列表，不能当作空预测')
    for e in row['entities']:
        a, b = e.get('start'), e.get('end')
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(row['text']):
            raise ValueError(f"{row['id']}: 非法边界{a},{b}")
        if e.get('type') not in TYPES or e.get('text') != row['text'][a:b]:
            raise ValueError(f"{row['id']}: 非法类别或原文位置不一致")


def keys(row):
    validate(row)
    return {(e['start'], e['end'], e['type']) for e in row['entities']}


def counts(tp, predicted, gold):
    p = tp / predicted if predicted else 0.0
    r = tp / gold if gold else 0.0
    return dict(tp=tp, predicted=predicted, gold=gold, precision=p, recall=r,
                f1=2*p*r/(p+r) if p+r else 0.0)


def score(gold, predicted):
    g = {r['id']: r for r in gold}
    p = {r['id']: r for r in predicted}
    if len(g) != len(gold) or len(p) != len(predicted) or g.keys() != p.keys():
        raise ValueError('样本缺失、重复或ID集合不一致；不能静默漏题')
    total = [0, 0, 0]
    by_type = {t: [0, 0, 0] for t in TYPES}
    nested = [0, 0, 0]
    duplicate_predictions = duplicate_gold = 0
    for ident, row in g.items():
        if row['text'] != p[ident]['text']:
            raise ValueError('预测与参考原文不同')
        a, b = keys(row), keys(p[ident])
        duplicate_gold += len(row['entities']) - len(a)
        duplicate_predictions += len(p[ident]['entities']) - len(b)
        for j, n in enumerate((len(a & b), len(b), len(a))):
            total[j] += n
        for t in TYPES:
            aa, bb = {x for x in a if x[2] == t}, {x for x in b if x[2] == t}
            for j, n in enumerate((len(aa & bb), len(bb), len(aa))):
                by_type[t][j] += n
        nesting = {x for x in a if any(x[:2] != y[:2] and
                   ((x[0] <= y[0] and y[1] <= x[1]) or
                    (y[0] <= x[0] and x[1] <= y[1])) for y in a)}
        nested[0] += len(nesting & b)
        nested[2] += len(nesting)
    return dict(samples=len(gold), micro=counts(*total),
                by_type={t: counts(*v) for t, v in by_type.items()},
                nested_gold=nested[2], nested_tp=nested[0],
                nested_recall=nested[0]/nested[2] if nested[2] else 0.0,
                duplicate_predictions=duplicate_predictions, duplicate_gold=duplicate_gold,
                empty_set_policy='zero denominator gives 0; exact duplicates count once',
                metric='strict Unicode start+exclusive end+type; case-sensitive; no normalization')


def windows(row, size=256, overlap=160):
    validate(row)
    if not 0 <= overlap < size:
        raise ValueError('窗口重叠参数非法')
    result, start = [], 0
    while start < len(row['text']):
        end = min(start + size, len(row['text']))
        ents = [dict(e, start=e['start']-start, end=e['end']-start)
                for e in row['entities'] if start <= e['start'] and e['end'] <= end]
        result.append(dict(id=row['id'], text=row['text'][start:end], start=start,
                           end=end, entities=ents))
        if end == len(row['text']):
            break
        start += size - overlap
    return result


def coverage(rows, size, overlap):
    missing = []
    n_windows = 0
    for row in rows:
        ws = windows(row, size, overlap)
        n_windows += len(ws)
        for e in row['entities']:
            if not any(w['start'] <= e['start'] and e['end'] <= w['end'] for w in ws):
                missing.append(dict(id=row['id'], entity=e))
    return dict(samples=len(rows), windows=n_windows, uncovered_entities=missing,
                uncovered_policy='retain all original reference entities in evaluation denominator')


def merge(row, predictions):
    found = set()
    for w, entities in predictions:
        local = dict(id=row['id'], text=w['text'], entities=entities)
        validate(local)
        found.update((e['start']+w['start'], e['end']+w['start'], e['type']) for e in entities)
    result = dict(id=row['id'], text=row['text'], entities=[
        dict(start=a, end=b, type=t, text=row['text'][a:b]) for a,b,t in sorted(found)])
    validate(result)
    return result


def make_subset(cfg, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    selected = {}
    for offset, name in enumerate(('train', 'dev')):
        source = ROOT / 'data/processed' / f'{name}.jsonl'
        rows = read_rows(source)
        # 仅按固定随机数抽取；长文/交叉等边界另用开发资料做映射测试，不按模型表现选样本。
        chosen = random.Random(cfg['seed']+offset).sample(rows, cfg[f'{name}_samples'])
        for r in chosen:
            validate(r)
        write_rows(directory / f'{name}.jsonl', chosen)
        selected[name] = dict(source=str(source.relative_to(ROOT)), source_sha256=digest(source),
                              ids=[r['id'] for r in chosen],
                              coverage=coverage(chosen, cfg['window_chars'], cfg['overlap_chars']))
    # 全部开发资料只做坐标覆盖检查，无模型运行，无holdout读取。
    mapping = {name: coverage(read_rows(ROOT / f'data/processed/{name}.jsonl'),
                             cfg['window_chars'], cfg['overlap_chars']) for name in ('train','dev')}
    dump(directory / 'manifest.json', dict(seed=cfg['seed'], splits=selected,
                                          full_processed_mapping_only=mapping, sealed_read=False))
    return selected


def dictionary(train):
    table = defaultdict(Counter)
    for row in train:
        validate(row)
        for e in row['entities']:
            table[e['text']][e['type']] += 1
    chosen = {word: min(freq, key=lambda t: (-freq[t], TYPES.index(t))) for word, freq in table.items()}
    return chosen, {w: dict(v) for w,v in table.items() if len(v)>1}


def dictionary_predict(row, table):
    # trie枚举所有起点和终点，不采用最长匹配覆盖嵌套实体。
    trie = {}
    for word, kind in table.items():
        node = trie
        for ch in word:
            node = node.setdefault(ch, {})
        node[''] = kind
    out = []
    for start in range(len(row['text'])):
        node = trie
        for end in range(start, len(row['text'])):
            node = node.get(row['text'][end])
            if node is None:
                break
            if '' in node:
                out.append(dict(start=start,end=end+1,type=node[''],text=row['text'][start:end+1]))
    return dict(id=row['id'], text=row['text'], entities=out)
