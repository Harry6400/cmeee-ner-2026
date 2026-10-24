"""阶段一的合成边界用例；不作为真实数据或模型实验结果。"""
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_data as data
import migrate_budget as budget


def row(text, a=0, b=1, kind='bod'):
    return {'text': text, 'entities': [{'start_idx': a, 'end_idx': b, 'type': kind}]}


class DataTests(unittest.TestCase):
    def test_unicode_codepoints_and_exclusive_boundary(self):
        result, issues = data.convert_row(row('甲😀乙', 1, 2), 'train', 0)
        self.assertEqual(result['text'], '甲😀乙')
        self.assertEqual(result['entities'][0], {'start': 1, 'end': 2, 'text': '😀', 'type': 'bod'})
        self.assertEqual(issues, [])

    def test_bad_boundaries_types_and_missing_labels_stop(self):
        for value in [row('甲', -1, 1), row('甲', 0, 2), row('甲', False, 1),
                      row('甲', 0, 0), row('甲', 0, 1, 'unknown'), {'text': '甲'}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                data.convert_row(value, 'train', 0)

    def test_duplicate_annotations_preserved_and_flagged(self):
        value = row('甲乙')
        value['entities'] *= 2
        result, issues = data.convert_row(value, 'train', 0)
        self.assertEqual(len(result['entities']), 2)
        self.assertEqual(issues[0]['kind'], 'duplicate_annotation_retained')

    def test_literal_mismatch_is_not_silently_repaired(self):
        value = row('甲乙')
        value['entities'][0]['entity'] = '乙'
        with self.assertRaises(ValueError):
            data.convert_row(value, 'train', 0)

    def test_nesting_crossing_and_coextensive(self):
        value = {'text': '甲乙丙丁', 'entities': [
            {'start_idx': 0, 'end_idx': 4, 'type': 'sym'},
            {'start_idx': 0, 'end_idx': 2, 'type': 'bod'},
            {'start_idx': 1, 'end_idx': 3, 'type': 'dis'},
            {'start_idx': 0, 'end_idx': 4, 'type': 'dis'}]}
        result, _ = data.convert_row(value, 'train', 0)
        stats = data.summarize([result])
        self.assertEqual(stats['entities'], 4)
        self.assertEqual(stats['nested_samples'], 1)
        self.assertEqual(stats['crossing_samples'], 1)
        self.assertEqual(stats['coextensive_different_type_samples'], 1)

    def test_text_groups_quarantined_and_split_together(self):
        train = [data.convert_row(row(s), 'train', i)[0]
                 for i, s in enumerate(['甲乙', '甲乙', '甲丙', '甲丁', '泄漏'])]
        holdout = [data.convert_row(row(s), 'dev', i)[0] for i, s in enumerate(['泄漏', '戊己'])]
        a, b, quarantined, common = data.split_records(train, holdout, 7, 2)
        self.assertEqual(len(quarantined), 1)
        self.assertFalse({r['group_id'] for r in a} & {r['group_id'] for r in b})
        self.assertFalse({r['group_id'] for r in a + b} & {r['group_id'] for r in holdout})
        self.assertEqual(len(common), 2)

    def test_archive_roundtrip_reproducibility_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'CMeEE-V2.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                for split, values in [('train', [row('甲乙'), row('甲丙'), row('甲丁')]),
                                      ('dev', [row('戊己'), row('庚辛')]),
                                      ('test', [{'text': '壬癸'}])]:
                    z.writestr(f'CMeEE-V2/CMeEE-V2_{split}.json', json.dumps(values))
            first = data.prepare(archive, root / 'first', evaluation_size=2)
            second = data.prepare(archive, root / 'second', evaluation_size=2)
            self.assertEqual(first, second)
            self.assertEqual((root / 'first/manifests/split-membership.json').read_bytes(),
                             (root / 'second/manifests/split-membership.json').read_bytes())
            readiness = json.loads((root / 'first/manifests/readiness.json').read_text(encoding='utf-8'))
            self.assertNotIn(chr(92), ''.join(readiness['files']))
            for name in ('processed/train.jsonl', 'processed/dev.jsonl', 'sealed/common500.jsonl', 'sealed/holdout.jsonl'):
                self.assertIn(name, readiness['files'])
            self.assertGreater(data.verify(root / 'first')['verified_files'], 0)
            with self.assertRaises(FileExistsError):
                data.prepare(archive, root / 'first', evaluation_size=2)
            (root / 'first/processed/train.jsonl').write_text('changed')
            with self.assertRaises(ValueError):
                data.verify(root / 'first')
            with (root / 'second/raw/CMeEE-V2.zip').open('ab') as f:
                f.write(b'tampered')
            with self.assertRaises(ValueError):
                data.verify(root / 'second')

    def test_zip_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.zip'
            with zipfile.ZipFile(path, 'w') as z:
                z.writestr('../outside', 'bad')
            with self.assertRaises(ValueError):
                data.prepare(path, Path(tmp) / 'output')


class BudgetTests(unittest.TestCase):
    def test_migration_preserves_requests_and_extension_does_not_reset_spend(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = {'schema_version': 1, 'limit_cny': '10', 'spent_cny': '3.792488',
                   'requests': [{'id': 'one', 'status': 'settled', 'conservative_charge_cny': '3.792488'}]}
            source = root / 'old.json'
            source.write_text(json.dumps(old))
            original = source.read_bytes()
            summary = budget.migrate(source, root / 'new')
            new = json.loads((root / 'new/ledger.json').read_text())
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(new['requests'], old['requests'])
            self.assertEqual(summary['remaining_cny'], '96.207512')
            self.assertFalse(new['api_enabled'])
            with self.assertRaises(FileExistsError):
                budget.migrate(source, root / 'new')

    def test_unknown_cost_duplicate_id_and_mismatched_sum_stop(self):
        base = {'limit_cny': '100', 'spent_cny': '1', 'requests': [
            {'id': 'one', 'status': 'settled', 'conservative_charge_cny': '1'}]}
        for mutate in [lambda d: d.update(spent_cny='0'),
                       lambda d: d['requests'][0].update(status='reserved'),
                       lambda d: d['requests'].append(dict(d['requests'][0]))]:
            changed = json.loads(json.dumps(base)); mutate(changed)
            with self.assertRaises(ValueError):
                budget.verify(changed)


if __name__ == '__main__':
    unittest.main()
