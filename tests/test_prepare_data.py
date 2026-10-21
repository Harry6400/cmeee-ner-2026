"""Synthetic boundary tests for data preparation; no real corpus samples are used."""
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_data as data


def row(text, start=0, end=1, kind='bod'):
    return {'text': text, 'entities': [{'start_idx': start, 'end_idx': end, 'type': kind}]}


def converted(text, split, index):
    return data.convert_row(row(text), split, index)[0]


class CoordinateTests(unittest.TestCase):
    def test_unicode_codepoints_and_exclusive_end(self):
        result, issues = data.convert_row(row('甲😀乙', 1, 2), 'train', 0)
        self.assertEqual(result['entities'][0], {
            'start': 1, 'end': 2, 'text': '😀', 'type': 'bod'
        })
        self.assertEqual(issues, [])

    def test_invalid_boundaries_types_and_surface_text_stop(self):
        invalid_rows = [
            row('甲', -1, 1), row('甲', 0, 2), row('甲', 0, 0),
            row('甲', False, 1), row('甲', 0, 1, 'unknown'),
        ]
        mismatch = row('甲乙')
        mismatch['entities'][0]['entity'] = '乙'
        invalid_rows.append(mismatch)
        for value in invalid_rows:
            with self.subTest(value=value), self.assertRaises(ValueError):
                data.convert_row(value, 'train', 0)

    def test_duplicate_annotations_are_retained_and_flagged(self):
        value = row('甲乙')
        value['entities'] *= 2
        result, issues = data.convert_row(value, 'train', 0)
        self.assertEqual(len(result['entities']), 2)
        self.assertEqual(issues[0]['kind'], 'duplicate_annotation_retained')


class SplitTests(unittest.TestCase):
    def test_grouped_split_is_reproducible_and_quarantines_exact_overlap(self):
        train = [converted(text, 'train', i) for i, text in enumerate(
            ['甲乙', '甲乙', '丙丁', '戊己', '重叠']
        )]
        holdout = [converted(text, 'dev', i) for i, text in enumerate(['重叠', '庚辛'])]
        first = data.split_records(train, holdout, seed=7, size=2)
        second = data.split_records(train, holdout, seed=7, size=2)
        self.assertEqual(first, second)
        internal_train, internal_dev, quarantined, common = first
        self.assertEqual(len(quarantined), 1)
        self.assertEqual(len(common), 2)
        self.assertFalse({r['group_id'] for r in internal_train} &
                         {r['group_id'] for r in internal_dev})
        self.assertFalse({r['group_id'] for r in internal_train + internal_dev} &
                         {r['group_id'] for r in holdout})
        duplicate_group = train[0]['group_id']
        self.assertEqual(sum(r['group_id'] == duplicate_group
                             for r in internal_train + internal_dev), 2)


class ArchiveTests(unittest.TestCase):
    def test_prepare_is_repeatable_and_verify_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'synthetic.zip'
            splits = {
                'train': [row('甲乙'), row('丙丁'), row('戊己')],
                'dev': [row('庚辛'), row('壬癸')],
                'test': [{'text': '子丑'}],
            }
            with zipfile.ZipFile(archive, 'w') as bundle:
                for split, values in splits.items():
                    bundle.writestr(
                        f'CMeEE-V2/CMeEE-V2_{split}.json',
                        json.dumps(values, ensure_ascii=False),
                    )
            first = data.prepare(archive, root / 'first', evaluation_size=2)
            second = data.prepare(archive, root / 'second', evaluation_size=2)
            self.assertEqual(first, second)
            self.assertEqual(
                (root / 'first/manifests/split-membership.json').read_bytes(),
                (root / 'second/manifests/split-membership.json').read_bytes(),
            )
            self.assertGreater(data.verify(root / 'first')['verified_files'], 0)
            with self.assertRaises(FileExistsError):
                data.prepare(archive, root / 'first', evaluation_size=2)
            (root / 'first/processed/train.jsonl').write_text('changed', encoding='utf-8')
            with self.assertRaises(ValueError):
                data.verify(root / 'first')
            with (root / 'second/raw/CMeEE-V2.zip').open('ab') as stream:
                stream.write(b'tampered')
            with self.assertRaises(ValueError):
                data.verify(root / 'second')

    def test_zip_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'unsafe.zip'
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('../outside', 'synthetic')
            outside = root / 'outside'
            self.assertFalse(outside.exists())
            with self.assertRaises(ValueError):
                data.prepare(archive, root / 'output', evaluation_size=1)
            self.assertFalse(outside.exists())


if __name__ == '__main__':
    unittest.main()
