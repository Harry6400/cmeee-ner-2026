"""Read authenticated historical evidence inside this project; never execute saved code."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path, PurePosixPath

REGISTRY_PATH = 'evidence/main-audit-fix-20261022/history/source-registry.json'
REGISTRY_SHA256 = 'a88753cb5548ff5feaccaba89290b382bd22abac76d9761adc2e49615f0f6900'
RECEIPTS = {
    1: ('evidence/stage1-integrity-check.json', '888bf532769c9f3ed4f2e1a0e01326e5cc8fafd72634e3cc47dafc4021a1f4de'),
    2: ('evidence/stage2/final-verification.json', '66897023ad0ce451218c65a4b6c87d438d3dc1ff2733683bc9ed54ecab59b266'),
    3: ('evidence/stage3/final-verification.json', 'b15e0647ffa6cba4970791645fce777aafe9975f679318570a1f24824db97f34'),
}
BASELINE_SHA256 = 'c8b4a2d22dcd659dd76b55adfca1224bba357a49f0235b56a13aef929e36584b'
API_PROTOCOL = 'runs/stage4-20261020/api-final/protocol-at-call.json'
API_PROTOCOL_SHA256 = '7c0a0f3d2d26ea7c95f16fea91e5d75ced5769eae73d151b7a2fe4ff3bdc2250'


def project_path(root, relative):
    root = Path(root).resolve()
    relative = PurePosixPath(str(relative))
    if relative.is_absolute() or '..' in relative.parts or '\\' in str(relative):
        raise ValueError('Historical evidence must use a project-relative path')
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError('Historical evidence resolves outside the project')
    return path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_file(root, relative, expected):
    path = project_path(root, relative)
    if not path.is_file() or digest(path) != expected:
        raise ValueError('Historical evidence missing or changed: ' + str(relative))
    return path


def historical_source(root, logical, expected):
    """The registered snapshot is mandatory; a good current file cannot mask damage."""
    registry = json.loads(checked_file(root, REGISTRY_PATH, REGISTRY_SHA256).read_text(encoding='utf-8'))
    if registry.get('schema_version') != 1:
        raise ValueError('Unsupported historical source registry')
    item = registry['entries'].get(str(logical))
    if item is None or item.get('sha256') != expected:
        raise ValueError('Historical source/version is not registered: ' + str(logical))
    return checked_file(root, item['snapshot'], expected)


def historical_budget(root, stage):
    """Select the completed stage's settled ledger; later unknown charges stay separate."""
    from migrate_budget import verify
    protocol = json.loads(checked_file(root, API_PROTOCOL, API_PROTOCOL_SHA256).read_text(encoding='utf-8'))
    opening_path = checked_file(root, protocol['opening_ledger_snapshot'], protocol['opening_ledger_sha256'])
    opening = json.loads(opening_path.read_text(encoding='utf-8'))
    migration = opening['migrations'][0]
    original_relative = 'private/budget/' + migration['source_snapshot']
    original_path = checked_file(root, original_relative, migration['source_sha256'])
    original = json.loads(original_path.read_text(encoding='utf-8'))
    if len(original['requests']) != 816 or opening['requests'][:816] != original['requests']:
        raise ValueError('Historical opening ledger changed its original 816-request prefix')
    if len(opening['requests']) != 840 or opening['spent_cny'] != '3.872064' or opening['api_enabled'] is not False or opening['limit_cny'] != '100':
        raise ValueError('Historical stage-three closing ledger differs from its frozen scope')
    verify(original)
    verify(opening)
    if stage in (1, 2):
        # In-memory migrated view only; no reconstructed file is presented as an original.
        selected = dict(original, limit_cny='100', api_enabled=False)
        source = original_relative
        source_hash = migration['source_sha256']
    elif stage == 3:
        selected = opening
        source = protocol['opening_ledger_snapshot']
        source_hash = protocol['opening_ledger_sha256']
    else:
        raise ValueError('Historical settled-ledger stage must be 1, 2 or 3')
    result = verify(selected)
    result.update(source_snapshot=source, source_sha256=source_hash,
                  scope='settled historical stage boundary; current ledger not read',
                  api_enabled=False, limit_cny='100', stage=stage)
    return selected, original, result


def historical_isolation(root, stage):
    """Verify the saved finding, without following the former external project's path."""
    relative, expected = RECEIPTS[stage]
    receipt = json.loads(checked_file(root, relative, expected).read_text(encoding='utf-8'))
    baseline = json.loads(checked_file(root, 'private/rag-baseline.json', BASELINE_SHA256).read_text(encoding='utf-8'))
    recorded = receipt['original_rag_files_unchanged']
    if recorded != len(baseline['files']):
        raise ValueError('Saved historical isolation finding differs from its baseline inventory')
    return dict(status='historical_record_only', verified_now=False,
                historical_files_recorded_unchanged=recorded, evidence=relative,
                evidence_sha256=expected, external_project_read=False,
                note='The earlier isolation check is preserved; the external project was not rechecked now')
