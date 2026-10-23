"""Synthetic integrity cases; no real evidence, ledger or external project is modified."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import historical_evidence as history


class HistoricalEvidenceTests(unittest.TestCase):
    def fixture(self, root):
        current = root / 'scripts/example.py'
        current.parent.mkdir()
        current.write_text('# historical source\n')
        saved = root / 'saved/example.py'
        saved.parent.mkdir()
        saved.write_bytes(current.read_bytes())
        expected = history.digest(saved)
        registry = root / history.REGISTRY_PATH
        registry.parent.mkdir(parents=True)
        registry.write_text(json.dumps(dict(schema_version=1, entries={
            'scripts/example.py': dict(snapshot='saved/example.py', sha256=expected)})))
        return current, saved, expected, registry

    def test_legitimate_current_source_change_does_not_change_historical_version(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            current, saved, expected, registry = self.fixture(root)
            current.write_text('# legitimate newer implementation\n')
            with patch.object(history, 'REGISTRY_SHA256', history.digest(registry)):
                self.assertEqual(history.historical_source(root, 'scripts/example.py', expected), saved.resolve())

    def test_damaged_or_missing_saved_source_cannot_fall_back_to_good_current(self):
        for missing in (False, True):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                current, saved, expected, registry = self.fixture(root)
                if missing:
                    saved.unlink()
                else:
                    saved.write_text('# altered snapshot\n')
                with patch.object(history, 'REGISTRY_SHA256', history.digest(registry)), self.assertRaises(ValueError):
                    history.historical_source(root, 'scripts/example.py', expected)
                self.assertEqual(history.digest(current), expected)

    def test_registry_or_expected_digest_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, _, expected, registry = self.fixture(root)
            anchor = history.digest(registry)
            with patch.object(history, 'REGISTRY_SHA256', anchor):
                with self.assertRaises(ValueError):
                    history.historical_source(root, 'scripts/example.py', '0' * 64)
                registry.write_text('{}')
                with self.assertRaises(ValueError):
                    history.historical_source(root, 'scripts/example.py', expected)

    def budget_fixture(self, root):
        def write(relative, value):
            p = root / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(value))
            return p
        old_rows = [dict(id='old-'+str(i), status='settled', conservative_charge_cny='3.792488' if i == 0 else '0') for i in range(816)]
        original = write('private/budget/original-ledger-20261018.json', dict(requests=old_rows, spent_cny='3.792488', limit_cny='10'))
        new_rows = old_rows + [dict(id='dev-'+str(i), status='settled', conservative_charge_cny='0.079576' if i == 0 else '0') for i in range(24)]
        opening = write('private/budget/synthetic-opening.json', dict(requests=new_rows, spent_cny='3.872064', limit_cny='100', api_enabled=False,
            migrations=[dict(source_snapshot=original.name, source_sha256=history.digest(original))]))
        protocol = write(history.API_PROTOCOL, dict(opening_ledger_snapshot=opening.relative_to(root).as_posix(), opening_ledger_sha256=history.digest(opening)))
        return original, opening, protocol

    def test_historical_budget_ignores_later_unknown_current_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, _, protocol = self.budget_fixture(root)
            (root / 'private/budget/ledger.json').write_text('later/current ledger must not be read')
            with patch.object(history, 'API_PROTOCOL_SHA256', history.digest(protocol)):
                for stage, count in ((1, 816), (2, 816), (3, 840)):
                    _, _, result = history.historical_budget(root, stage)
                    self.assertEqual(result['requests'], count)

    def test_historical_budget_snapshot_tamper_or_loss_is_rejected(self):
        for target in ('original', 'opening', 'missing'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                original, opening, protocol = self.budget_fixture(root)
                expected = history.digest(protocol)
                if target == 'missing':
                    opening.unlink()
                else:
                    (original if target == 'original' else opening).write_text('{}')
                with patch.object(history, 'API_PROTOCOL_SHA256', expected), self.assertRaises(ValueError):
                    history.historical_budget(root, 3)

    def test_path_escape_and_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as outside:
            root = Path(d)
            external = Path(outside) / 'source.py'
            external.write_text('# outside\n')
            (root / 'escape').symlink_to(outside, target_is_directory=True)
            for relative in ('../source.py', str(external), 'escape/source.py', 'a\\source.py'):
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    history.project_path(root, relative)

    def test_isolation_only_checks_saved_record_without_following_external_root(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            baseline = root / 'private/rag-baseline.json'
            baseline.parent.mkdir()
            baseline.write_text(json.dumps(dict(source_root='/must/not/be/read', files={'one': 'synthetic'})))
            receipt = root / 'saved-isolation.json'
            receipt.write_text(json.dumps(dict(original_rag_files_unchanged=1)))
            with patch.object(history, 'BASELINE_SHA256', history.digest(baseline)), patch.dict(history.RECEIPTS, {1: ('saved-isolation.json', history.digest(receipt))}):
                result = history.historical_isolation(root, 1)
            self.assertFalse(result['verified_now'])
            self.assertFalse(result['external_project_read'])
            self.assertEqual(result['status'], 'historical_record_only')


if __name__ == '__main__':
    unittest.main()
