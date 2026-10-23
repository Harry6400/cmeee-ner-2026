"""只迁移并核查累计账本；此脚本不具备API调用能力。"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path


def verify(ledger):
    rows = ledger['requests']
    if any(r['status'] != 'settled' for r in rows):
        raise ValueError('存在未结算或未知费用；不得迁移继续')
    charges = [Decimal(r['conservative_charge_cny']) for r in rows]
    if any(not c.is_finite() or c < 0 for c in charges):
        raise ValueError('费用明细非法')
    total = sum(charges, Decimal(0))
    if total != Decimal(ledger['spent_cny']):
        raise ValueError('累计费用与明细不符')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('请求ID重复')
    if total > Decimal(ledger['limit_cny']):
        raise ValueError('已超预算')
    return {'requests': len(rows), 'spent_cny': str(total),
            'remaining_cny': str(Decimal(ledger['limit_cny']) - total)}


def migrate(source: Path, destination: Path):
    raw = source.read_bytes()
    old = json.loads(raw)
    verify(old)
    destination.mkdir(parents=True, exist_ok=True)
    ledger_path = destination / 'ledger.json'
    if ledger_path.exists():
        raise FileExistsError('新累计账本已存在，禁止覆盖或再次导入')
    snapshot = destination / 'original-ledger-20261018.json'
    with snapshot.open('xb') as f:
        f.write(raw)
    snapshot.chmod(0o400)
    new = dict(old)
    new.update(schema_version=2, limit_cny='100', api_enabled=False,
               api_disabled_reason='阶段一仅迁移；API调用器尚未实现或核价',
               migrations=[{'at': datetime.now(timezone.utc).isoformat(),
                            'kind': 'authorized_limit_extension', 'old_limit_cny': old['limit_cny'],
                            'new_limit_cny': '100', 'opening_spent_cny': old['spent_cny'],
                            'source_sha256': hashlib.sha256(raw).hexdigest(),
                            'source_snapshot': snapshot.name, 'original_request_count': len(old['requests']),
                            'authorization': '用户批准中文医学实体抽取分阶段计划；QHA无新增GPU租赁费用'}])
    with ledger_path.open('x', encoding='utf-8') as f:
        json.dump(new, f, ensure_ascii=False, indent=2)
        f.write('\n')
    ledger_path.chmod(0o600)
    if new['requests'] != old['requests']:
        raise AssertionError('历史请求被改动')
    return verify(new)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path)
    p.add_argument('--destination', type=Path, default=Path('private/budget'))
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    if a.verify:
        result = verify(json.loads((a.destination / 'ledger.json').read_text()))
    elif a.source:
        result = migrate(a.source, a.destination)
    else:
        p.error('迁移必须提供--source，核查使用--verify')
    print(json.dumps(result, ensure_ascii=False))
