"""Synthetic budget/crash/output/privacy gates. Never calls a real service."""
import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

PROJECT = Path(os.environ.get("CMEEE_TEST_PROJECT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import api_final as api


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))


def rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values))


class FinalAPITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cmeee-synthetic-api-")
        self.root = Path(self.temporary.name)
        self.run = self.root / "runs/stage4-20261020/api-final"
        self.frozen = api.read_json(PROJECT / "configs/stage4-frozen-protocol.json")
        self.samples = [dict(id=f"synthetic-sample-{i:03}", text="中文检查文本", entities=[]) for i in range(500)]
        self.examples = [dict(id=ident, text="示例药物", entities=[dict(start=2, end=4, type="dru", text="药物")])
                         for ident in self.frozen["api"]["example_ids"]]
        rows(self.root / self.frozen["data"]["common_subset"], self.samples)
        rows(self.root / self.frozen["api"]["examples"], self.examples)
        write(self.root / "configs/stage4-frozen-protocol.json", self.frozen)
        self.contract = dict(stage=4, authorization="explicit_user_stage4_2026-10-20",
            explicit_final_api_authorized=True, final_holdout_authorized=True, api_budget_cny="100",
            run_root="runs/stage4-20261020", authorization_request="SYNTHETIC unit test only",
            frozen_protocol_sha256=api.sha(self.root / "configs/stage4-frozen-protocol.json"))
        self.price = dict(model="deepseek-flash", version_reported_by_official_docs="DeepSeek-V4.1-Flash-1017",
            input_peak_uncached_per_million="2", output_peak_per_million="8", context_max_tokens=1048576,
            same_official_version_as_development=True, source="https://api-docs.deepseek.com/synthetic",
            updates_source="https://api-docs.deepseek.com/synthetic", checked_at=datetime.now(timezone.utc).isoformat(),
            tokenization_bound_assumption_verified=False)
        write(self.root / "contract.json", self.contract)
        write(self.root / "price.json", self.price)
        entries = [dict(id=f"synthetic-history-{i}", status="settled", conservative_charge_cny="0") for i in range(840)]
        entries[0]["conservative_charge_cny"] = "3.792488"
        entries[839]["conservative_charge_cny"] = "0.079576"
        self.opening = dict(requests=entries, limit_cny="100", spent_cny="3.872064", status="active", api_enabled=False)
        write(self.root / "private/budget/ledger.json", self.opening)
        write(self.root / "private/budget/original-ledger-20261018.json", dict(requests=entries[:816]))
        for filename in ("ner_core.py", "api_development_grounded.py"):
            destination = self.root / "scripts" / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((PROJECT / "scripts" / filename).read_bytes())

    def tearDown(self):
        self.temporary.cleanup()

    def prepared(self):
        return api.prepare(self.root, self.run, self.root / "contract.json", self.root / "price.json")

    def privacy_pass(self, protocol):
        inventory = api.read_rows(self.run / "privacy-review-inputs.jsonl")
        write(self.run / "privacy-review.json", dict(inventory_sha256=protocol["privacy_inventory_sha256"],
            reviewed_ids=[r["id"] for r in inventory], allow_external_send=True, semantic_review_completed=True,
            reviewer_kind="SYNTHETIC fixture,not actual review", review_notes="Synthetic nonprivate text",
            unresolved_findings=[], resolved_regex_flagged_ids=[]))

    def fake_raw(self, item, ident, valid=True):
        response = dict(model="deepseek-flash", usage=dict(prompt_tokens=20, completion_tokens=10),
            choices=[dict(finish_reason="stop", message=dict(content=json.dumps(dict(entities=[] if valid else [dict(text="不存在", type="dru", occurrence=0)]))))])
        return dict(id=ident, key=item["key"], payload=item["payload"], response=response, seconds=.01,
                    received_at=api.now(), raw_body_sha256=api.sha_bytes(api.canonical(response)))

    def test_preflight_exact1000_payloads_and_context_reservations(self):
        protocol = self.prepared()
        manifest = api.read_rows(self.run / "requests-manifest.jsonl")
        self.assertEqual(len(manifest), 1000)
        self.assertEqual(manifest[0]["input_token_bound"], 1048576)
        self.assertEqual(manifest[0]["reserved_cny"], "2.12992")
        self.assertTrue(protocol["estimate_not_a_guarantee"])
        self.assertLess(float(protocol["planned_output_cap_estimate_cny"]), 80)
        self.assertEqual(len(manifest[0]["payload"]["messages"]), 2)
        self.assertEqual(len(manifest[500]["payload"]["messages"]), 8)
        self.assertEqual(manifest[0]["payload"]["messages"][-1]["content"], api.user_text(self.samples[0]))
        self.assertNotIn("entities", json.loads(manifest[0]["payload"]["messages"][-1]["content"]))
        self.assertEqual(api.read_json(self.root / "private/budget/ledger.json"), self.opening)
        with self.assertRaises(FileExistsError):
            self.prepared()

    def test_privacy_is_required_and_frozen_inputs_cannot_drift(self):
        protocol = self.prepared()
        with self.assertRaises(FileNotFoundError):
            api.validate_frozen_inputs(self.root, self.run)
        self.privacy_pass(protocol)
        api.validate_frozen_inputs(self.root, self.run)
        row = api.read_json(self.run / "privacy-review.json")
        row["reviewed_ids"] = row["reviewed_ids"][:-1]
        write(self.run / "privacy-review.json", row)
        with self.assertRaisesRegex(ValueError, "every exact"):
            api.validate_frozen_inputs(self.root, self.run)

    def test_final_calls_wait_for_training_queue(self):
        protocol = self.prepared()
        self.privacy_pass(protocol)
        write(self.root / "evidence/stage4/training-queue.json", dict(status="running"))
        with patch.object(api, "secret", side_effect=AssertionError("credential access before training completion")):
            with self.assertRaisesRegex(ValueError, "supplementary seeds"):
                api.run(self.root, self.run)
        self.assertEqual(api.read_json(self.root / "private/budget/ledger.json"), self.opening)

    def test_reservation_settlement_preserves840_and_rejects_duplicate(self):
        self.prepared()
        item = api.read_rows(self.run / "requests-manifest.jsonl")[0]
        ledger = api.Ledger(self.root, self.run, self.opening, self.price)
        ident = ledger.reserve(item)
        with self.assertRaises(ValueError):
            ledger.reserve(item)
        ledger.settle(ident, dict(prompt_tokens=500, completion_tokens=100), "deepseek-flash")
        result = ledger.read()
        self.assertEqual(result["requests"][:840], self.opening["requests"])
        self.assertEqual(result["spent_cny"], "3.873864")
        with self.assertRaises(ValueError):
            ledger.reserve(item)
        with self.assertRaises(ValueError):
            ledger.settle(ident, dict(prompt_tokens=500, completion_tokens=100), "deepseek-flash")

    def test_unknown_settlement_is_terminal_and_not_empty(self):
        protocol = self.prepared()
        item = api.read_rows(self.run / "requests-manifest.jsonl")[0]
        ledger = api.Ledger(self.root, self.run, self.opening, self.price)
        ident = ledger.reserve(item)
        ledger.unknown(ident, "SYNTHETIC transport interruption")
        with self.assertRaises(ValueError):
            ledger.read()
        summary = api.summarize(self.root, self.run, protocol, "synthetic_interruption")
        prediction = api.read_rows(self.run / "L0-predictions.jsonl")[0]
        self.assertEqual(prediction["status"], "error")
        self.assertNotIn("entities", prediction)
        self.assertIsNone(summary["routes"]["L0"]["main_strict_metrics"])

    def test_crash_after_settlement_recovers_from_raw_without_network(self):
        self.prepared()
        items = api.read_rows(self.run / "requests-manifest.jsonl")
        item = items[0]
        ledger = api.Ledger(self.root, self.run, self.opening, self.price)
        ident = ledger.reserve(item)
        raw = self.fake_raw(item, ident)
        api.create_json(self.run / "requests" / ("raw-" + ident + ".json"), raw)
        ledger.settle(ident, raw["response"]["usage"], raw["response"]["model"])
        with patch.object(api.urllib.request, "build_opener", side_effect=AssertionError("must not call network")):
            recovered = api.reconcile(self.run, ledger, items)
            self.assertEqual(len(recovered), 1)
            self.assertTrue(recovered[item["key"]]["recovered_without_network"])
            self.assertEqual(len(api.reconcile(self.run, ledger, items)), 1)
        self.assertEqual(len(api.read_rows(self.run / "calls.jsonl")), 1)

    def test_invalid_and_partial_outputs_do_not_get_a_main_f1(self):
        protocol = self.prepared()
        items = api.read_rows(self.run / "requests-manifest.jsonl")
        ledger = api.Ledger(self.root, self.run, self.opening, self.price)
        for i in range(2):
            item = items[i]
            ident = ledger.reserve(item)
            raw = self.fake_raw(item, ident, valid=i == 0)
            api.create_json(self.run / "requests" / ("raw-" + ident + ".json"), raw)
            ledger.settle(ident, raw["response"]["usage"], raw["response"]["model"])
        api.reconcile(self.run, ledger, items)
        summary = api.summarize(self.root, self.run, protocol, "synthetic_partial")
        prediction = api.read_rows(self.run / "L0-predictions.jsonl")
        self.assertEqual(len(prediction), 500)
        self.assertEqual([p["id"] for p in prediction], [p["id"] for p in self.samples])
        self.assertEqual([p["status"] for p in prediction[:3]], ["valid", "invalid", "not_sent"])
        self.assertNotIn("entities", prediction[1])
        self.assertIsNone(summary["routes"]["L0"]["main_strict_metrics"])
        self.assertEqual(summary["routes"]["L0"]["conditional_valid_only_metrics"]["samples"], 1)

    def test_unicode_overlapping_occurrences_and_exact_duplicate_dedup(self):
        row = dict(id="synthetic-unicode", text="𠮷甲甲甲", entities=[])
        entities = [dict(text="甲甲", type="bod", occurrence=0), dict(text="甲甲", type="bod", occurrence=1),
                    dict(text="甲甲", type="bod", occurrence=1), dict(text="𠮷", type="sym", occurrence=0)]
        raw = dict(choices=[dict(finish_reason="stop", message=dict(content=json.dumps(dict(entities=entities), ensure_ascii=False)))])
        prediction, duplicates = api.canonical_result(row, raw)
        self.assertEqual(prediction["status"], "valid")
        self.assertEqual(duplicates, 1)
        self.assertEqual([(e["start"], e["end"]) for e in prediction["entities"]], [(0, 1), (1, 3), (2, 4)])
        raw["choices"][0]["finish_reason"] = "length"
        failed, _ = api.canonical_result(row, raw)
        self.assertEqual(failed["status"], "invalid")
        self.assertNotIn("entities", failed)

    def test_context_usage_and_ledger_history_mutation_are_rejected(self):
        self.prepared()
        item = api.read_rows(self.run / "requests-manifest.jsonl")[0]
        ledger = api.Ledger(self.root, self.run, self.opening, self.price)
        ident = ledger.reserve(item)
        with self.assertRaisesRegex(ValueError, "bound"):
            ledger.settle(ident, dict(prompt_tokens=1048577, completion_tokens=10), "deepseek-flash")
        data = api.read_json(ledger.path)
        data["requests"][0]["id"] = "tampered-history"
        write(ledger.path, data)
        with self.assertRaisesRegex(ValueError, "prefix"):
            ledger.read()

    def test_network_uncertainty_is_one_attempt_redacted_and_not_retried(self):
        protocol = self.prepared()
        self.privacy_pass(protocol)
        write(self.root / "evidence/stage4/training-queue.json", dict(status="completed"))
        fake_key = "SYNTHETIC-KEY-DO-NOT-LOG"
        with patch.object(api, "secret", return_value=fake_key):
            with patch.object(api.urllib.request, "build_opener") as opener:
                opener.return_value.open.side_effect = api.urllib.error.URLError("synthetic uncertain " + fake_key)
                with self.assertRaises(RuntimeError):
                    api.run(self.root, self.run)
                self.assertEqual(opener.return_value.open.call_count, 1)
        ledger = api.read_json(self.root / "private/budget/ledger.json")
        self.assertEqual(len(ledger["requests"]), 841)
        self.assertEqual(ledger["requests"][-1]["status"], "unknown_cost")
        self.assertFalse(ledger["api_enabled"])
        for path in self.root.rglob("*.json"):
            self.assertNotIn(fake_key, path.read_text())
        with patch.object(api, "secret", return_value=fake_key):
            with patch.object(api.urllib.request, "build_opener", side_effect=AssertionError("retry forbidden")):
                with self.assertRaises(ValueError):
                    api.run(self.root, self.run, resume=True)

    def test_soft_subcap_stops_before_a_request_is_reserved(self):
        self.prepared()
        item = api.read_rows(self.run / "requests-manifest.jsonl")[0]
        ledger = api.Ledger(self.root, self.run, self.opening, self.price)
        data = api.read_json(ledger.path)
        data["requests"].append(dict(id="synthetic-other-stage4-settled", origin=api.ORIGIN,
            run_directory=str(self.run.relative_to(self.root)), status="settled", conservative_charge_cny="78"))
        data["spent_cny"] = "81.872064"
        write(ledger.path, data)
        with self.assertRaisesRegex(ValueError, "budget"):
            ledger.reserve(item)
        self.assertEqual(len(api.read_json(ledger.path)["requests"]), 841)


if __name__ == "__main__":
    unittest.main()
