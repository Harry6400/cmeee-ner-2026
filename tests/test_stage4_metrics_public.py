from __future__ import annotations

import unittest

from scripts import stage4_metrics as metrics


def entity(start: int, end: int, entity_type: str, surface: str) -> dict:
    return {"start": start, "end": end, "type": entity_type, "text": surface}


def row(doc_id: str, text: str, entities: list[dict]) -> dict:
    return {"id": doc_id, "text": text, "entities": entities}


class PublicStage4MetricTests(unittest.TestCase):
    def test_unicode_half_open_spans_and_malformed_entities(self):
        valid = row("d1", "药物A", [entity(1, 3, "sym", "物A")])
        self.assertEqual(metrics.entity_keys(valid), {(1, 3, "sym")})

        malformed = [
            entity(-1, 1, "sym", "药"),
            entity(2, 4, "sym", "A"),
            entity(2, 1, "sym", "物"),
            entity(0, 1, "unknown", "药"),
            entity(0, 2, "sym", "物A"),
        ]
        for item in malformed:
            with self.subTest(item=item), self.assertRaises(ValueError):
                metrics.entity_keys(row("d1", "药物A", [item]))

    def test_strict_entity_and_character_metrics_are_exact_and_deduplicated(self):
        gold = row(
            "d1",
            "甲乙丙",
            [
                entity(0, 2, "sym", "甲乙"),
                entity(1, 3, "dis", "乙丙"),
                entity(0, 2, "sym", "甲乙"),
            ],
        )
        predicted = row(
            "d1",
            "甲乙丙",
            [
                entity(0, 2, "sym", "甲乙"),
                entity(1, 3, "sym", "乙丙"),
                entity(1, 3, "sym", "乙丙"),
            ],
        )

        result = metrics.evaluate_pairs([(gold, predicted)], train_terms=set())
        self.assertEqual(result["micro"]["tp"], 1)
        self.assertEqual(result["micro"]["predicted"], 2)
        self.assertEqual(result["micro"]["gold"], 2)
        self.assertEqual(result["micro"]["f1"], 0.5)
        self.assertEqual(result["character_type_micro"]["tp"], 2)
        self.assertEqual(result["character_type_micro"]["predicted"], 3)
        self.assertEqual(result["character_type_micro"]["gold"], 4)
        self.assertEqual(result["duplicate_gold"], 1)
        self.assertEqual(result["duplicate_predictions"], 1)
        self.assertEqual(result["structural_reference_recall"]["crossing"]["gold"], 2)
        self.assertEqual(result["structural_reference_recall"]["crossing"]["tp"], 1)

    def test_invalid_output_is_reported_as_unavailable_not_empty_prediction(self):
        gold = [row("d1", "发热", [entity(0, 2, "sym", "发热")])]
        failed = [{"id": "d1", "text": "发热", "status": "error", "entities": []}]

        result = metrics.evaluate_route(gold, failed, train_terms=set(), allow_invalid=True)
        self.assertEqual(result["valid"], 0)
        self.assertEqual(result["invalid"], 1)
        self.assertIsNone(result["main_metrics"])
        self.assertTrue(result["conditional_valid_only_is_diagnostic"])
        self.assertEqual(result["conditional_valid_only_metrics"]["samples"], 0)
        self.assertTrue(result["primary_availability"].startswith("N/A"))
        self.assertEqual(result["failures"][0]["status"], "error")

    def test_prediction_ids_and_order_must_match_reference(self):
        gold = [row("d1", "药", [])]
        wrong_order = [row("different-id", "药", [])]
        with self.assertRaises(ValueError):
            metrics.validate_predictions(gold, wrong_order)

    def test_paired_bootstrap_is_reproducible_and_requires_shared_document_order(self):
        gold = [
            row("a", "甲", [entity(0, 1, "sym", "甲")]),
            row("b", "乙", [entity(0, 1, "sym", "乙")]),
        ]
        baseline_predictions = [
            {"id": "a", "text": "甲", "entities": []},
            {"id": "b", "text": "乙", "entities": []},
        ]
        model_predictions = [
            {"id": "a", "text": "甲", "entities": [entity(0, 1, "sym", "甲")]},
            {"id": "b", "text": "乙", "entities": [entity(0, 1, "sym", "乙")]},
        ]
        routes = {
            "baseline": metrics.evaluate_route(gold, baseline_predictions, train_terms=set()),
            "model": metrics.evaluate_route(gold, model_predictions, train_terms=set()),
        }

        first = metrics.paired_bootstrap(routes, replicates=32, seed=17)
        second = metrics.paired_bootstrap(routes, replicates=32, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(first["differences"]["model minus baseline"]["difference"], 1.0)
        self.assertEqual(first["replicates"], 32)

        reordered = dict(routes["model"])
        reordered["main_metrics"] = dict(routes["model"]["main_metrics"])
        reordered["main_metrics"]["per_document_counts"] = list(
            reversed(routes["model"]["main_metrics"]["per_document_counts"])
        )
        with self.assertRaisesRegex(ValueError, "identical document IDs and order"):
            metrics.paired_bootstrap(
                {"baseline": routes["baseline"], "model": reordered},
                replicates=4,
                seed=17,
            )

    def test_seed_summary_requires_all_configured_seeds(self):
        complete = {20261019: 0.5, 20261020: 0.6, 20261021: 0.7}
        result = metrics.seed_summary(complete)
        self.assertAlmostEqual(result["mean"], 0.6)
        self.assertAlmostEqual(result["sample_sd"], 0.1)
        self.assertEqual(result["seeds"], [20261019, 20261020, 20261021])
        self.assertFalse(result["best_seed_selected_on_holdout"])

        with self.assertRaisesRegex(ValueError, "Require all3 frozen seeds"):
            metrics.seed_summary({20261019: 0.5})


if __name__ == "__main__":
    unittest.main()
