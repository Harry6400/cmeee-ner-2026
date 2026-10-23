"""免费核查阶段一产物；不读取密钥、不训练、不调用外部API。"""
import json
import random
from pathlib import Path

from prepare_data import digest, verify as verify_data
from historical_evidence import historical_budget, historical_isolation


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def verify(root):
    manifest = json.loads((root / 'data/manifests/readiness.json').read_text())
    data_result = verify_data(root / 'data')
    groups = {}
    samples = {}
    for name, path in [('train', 'processed/train.jsonl'), ('dev', 'processed/dev.jsonl'),
                       ('holdout', 'sealed/holdout.jsonl'), ('common500', 'sealed/common500.jsonl')]:
        samples[name] = rows(root / 'data' / path)
        groups[name] = {r['group_id'] for r in samples[name]}
        for row in samples[name]:
            if digest(row['text'].encode()) != row['group_id']:
                raise ValueError('原文分组不一致')
            for entity in row['entities']:
                if row['text'][entity['start']:entity['end']] != entity['text']:
                    raise ValueError('实体位置与原文不一致')
    for a, b in [('train', 'dev'), ('train', 'holdout'), ('dev', 'holdout')]:
        if groups[a] & groups[b]:
            raise ValueError(f'同文跨集合:{a}/{b}')
    expected = sorted(random.Random(manifest['evaluation_sampling_seed']).sample(
        range(len(samples['holdout'])), manifest['evaluation_size']))
    if samples['common500'] != [samples['holdout'][i] for i in expected]:
        raise ValueError('共同评估子集与固定随机抽样不符')
    ledger, original, budget_result = historical_budget(root, 1)
    if ledger['requests'] != original['requests'] or ledger['spent_cny'] != original['spent_cny']:
        raise ValueError('历史请求或费用发生变化')
    if ledger['limit_cny'] != '100' or ledger['api_enabled'] is not False:
        raise ValueError('阶段一预算或API状态不符')
    isolation = historical_isolation(root, 1)
    return {'status': 'PASS_WITH_WARNINGS', 'data': data_result, 'samples': {k: len(v) for k, v in samples.items()},
            'exact_text_cross_split_groups': 0, 'budget': budget_result,
            'external_isolation': isolation,
            'gpu_training_started': False, 'new_paid_api_calls': 0,
            'note': '模型效果尚未验证；数据警告详见阶段一报告'}


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parents[1]), ensure_ascii=False, indent=2))
