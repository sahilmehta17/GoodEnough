import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts import build_cost
from src.goodenough import store

PRICE_IN = 1.0   # $1 / 1M input tokens, chosen so the arithmetic is easy to check by hand
PRICE_OUT = 2.0  # $2 / 1M output tokens


def _row(dataset, split, item_id, role, correct, parse_status="ok",
        latency_ms=100.0, input_tokens=1_000_000, output_tokens=1_000_000,
        error=None, cache_hit=False):
    return store.ResultRow(
        dataset=dataset, split=split, item_id=item_id, model_role=role,
        model_id_requested="x", model_id_returned="x",
        semantic_prompt="q", rendered_input="q", raw_response="answer: A",
        normalized_answer="A", parser_version="v0.1", parse_status=parse_status,
        correct=correct, error=error, retries=0,
        input_tokens=input_tokens, output_tokens=output_tokens,
        latency_ms_uncached=latency_ms, cache_hit=cache_hit,
        finish_reason="stop", run_date="2026-08-06T00:00:00Z", seed=42,
    )


class ItemDollarsTests(unittest.TestCase):
    def test_prices_input_and_output_separately(self):
        # 1M input tokens at PRICE_IN + 1M output tokens at PRICE_OUT.
        self.assertAlmostEqual(
            build_cost.item_dollars(1_000_000, 1_000_000, PRICE_IN, PRICE_OUT),
            PRICE_IN + PRICE_OUT,
        )

    def test_zero_tokens_is_zero_not_an_error(self):
        self.assertEqual(build_cost.item_dollars(0, 0, PRICE_IN, PRICE_OUT), 0.0)

    def test_missing_token_counts_price_as_zero(self):
        self.assertEqual(build_cost.item_dollars(None, None, PRICE_IN, PRICE_OUT), 0.0)


class SyntheticDatabaseTests(unittest.TestCase):
    """
    A small synthetic results.sqlite with known token counts and latencies,
    built the same way the real runner writes rows (store.insert_row), then
    read back read-only the way build_cost.py actually reads the real
    database, to check the cost arithmetic end to end.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmpdir.name) / "synthetic.sqlite")
        conn = store.connect(self.db_path)

        # Two router-split items, one per subject, each with a paired local +
        # hosted row. q1's local parse failed (cascade will escalate it).
        # q2's local parse succeeded (cascade keeps it local).
        rows = [
            _row("mmlu", "router", "mmlu/sub_a/1", "local", correct=1,
                parse_status="unparseable", latency_ms=1000.0,
                input_tokens=10, output_tokens=5),
            _row("mmlu", "router", "mmlu/sub_a/1", "hosted", correct=1,
                latency_ms=200.0, input_tokens=1_000_000, output_tokens=1_000_000),
            _row("mmlu", "router", "mmlu/sub_b/1", "local", correct=1,
                parse_status="ok", latency_ms=1500.0,
                input_tokens=10, output_tokens=5),
            _row("mmlu", "router", "mmlu/sub_b/1", "hosted", correct=0,
                latency_ms=300.0, input_tokens=2_000_000, output_tokens=2_000_000),
            # An error row for a third item: never paired, must not appear.
            _row("mmlu", "router", "mmlu/sub_b/2", "hosted", correct=None,
                error="boom", latency_ms=None),
            # A cache hit: must be excluded from project totals and latency.
            _row("mmlu", "dev", "mmlu/sub_a/9", "hosted", correct=1,
                latency_ms=999_999.0, input_tokens=999_999, output_tokens=999_999,
                cache_hit=True),
            # Project-wide hosted totals should also see this dev-split row.
            _row("mmlu", "dev", "mmlu/sub_a/10", "hosted", correct=1,
                latency_ms=50.0, input_tokens=100, output_tokens=100),
            _row("mmlu", "dev", "mmlu/sub_a/10", "local", correct=1,
                latency_ms=40.0, input_tokens=10, output_tokens=5),
        ]
        for r in rows:
            store.insert_row(conn, r)
        conn.close()

        self.ro_conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)

    def tearDown(self):
        self.ro_conn.close()
        self._tmpdir.cleanup()

    def test_router_items_with_metrics_pairs_only_complete_error_free_rows(self):
        items, metrics = build_cost.router_items_with_metrics(self.ro_conn)
        ids = {i["item_id"] for i in items}
        self.assertEqual(ids, {"mmlu/sub_a/1", "mmlu/sub_b/1"})
        self.assertNotIn("mmlu/sub_b/2", metrics)  # error row, never paired

    def test_local_parse_status_carried_through_for_cascade(self):
        items, _ = build_cost.router_items_with_metrics(self.ro_conn)
        by_id = {i["item_id"]: i for i in items}
        self.assertFalse(by_id["mmlu/sub_a/1"]["local_parse_ok"])
        self.assertTrue(by_id["mmlu/sub_b/1"]["local_parse_ok"])

    def test_project_hosted_totals_excludes_cache_hits_and_errors(self):
        totals = build_cost.project_hosted_totals(self.ro_conn)
        # Hosted rows counted: sub_a/1 (1e6/1e6), sub_b/1 (2e6/2e6), dev sub_a/10
        # (100/100). NOT counted: sub_b/2 (error), sub_a/9 (cache_hit).
        self.assertEqual(totals["hosted_calls"], 3)
        self.assertEqual(totals["total_input_tokens"], 1_000_000 + 2_000_000 + 100)
        self.assertEqual(totals["total_output_tokens"], 1_000_000 + 2_000_000 + 100)

    def test_project_local_occupancy_excludes_cache_hits(self):
        seconds = build_cost.project_local_occupancy_seconds(self.ro_conn)
        # local rows: sub_a/1 (1000ms, unparseable but not an error row),
        # sub_b/1 (1500ms), dev sub_a/10 (40ms). Sum = 2540ms = 2.54s.
        self.assertAlmostEqual(seconds, 2.54)

    def test_local_incremental_spend_is_computed_through_config_not_hardcoded(self):
        import unittest.mock as mock
        # With local prices at 0 (the real, current values) it must be 0.
        self.assertEqual(build_cost.project_local_incremental_spend(self.ro_conn), 0.0)
        # But if the pinned local price were ever nonzero, this function must
        # actually price it rather than always returning a hardcoded 0.0.
        with mock.patch.multiple(build_cost.config,
                                 LOCAL_PRICE_INPUT_PER_M=1.0, LOCAL_PRICE_OUTPUT_PER_M=1.0):
            spend = build_cost.project_local_incremental_spend(self.ro_conn)
        # local input+output tokens summed: sub_a/1 (10+5) + sub_b/1 (10+5) + dev (10+5) = 45
        self.assertAlmostEqual(spend, 45 / 1_000_000.0)


class CascadeDoubleChargeTests(unittest.TestCase):
    """
    The one subtlety most likely to be silently wrong: cascade runs local on
    every item, so every item pays local latency, and only the escalated
    subset additionally pays hosted latency and hosted dollars. This must
    never be conflated with map_based, which pays EITHER local OR hosted per
    item, never both.
    """

    def setUp(self):
        # q1: local parse failed -> escalated. q2: local parse ok -> not escalated.
        self.items = [
            {"item_id": "q1", "subject": "s", "local_correct": 0,
             "local_parse_ok": False, "hosted_correct": 1},
            {"item_id": "q2", "subject": "s", "local_correct": 1,
             "local_parse_ok": True, "hosted_correct": 1},
        ]
        self.metrics = {
            "q1": {"local_latency_ms": 1000.0, "hosted_latency_ms": 300.0,
                  "hosted_input_tokens": 1_000_000, "hosted_output_tokens": 1_000_000},
            "q2": {"local_latency_ms": 500.0, "hosted_latency_ms": 100.0,
                  "hosted_input_tokens": 1_000_000, "hosted_output_tokens": 1_000_000},
        }
        self.all_ids = ["q1", "q2"]
        import src.goodenough.analysis as analysis
        self.policies = analysis.evaluate_router_policies(self.items, {"s": "below_margin"})

    def _prices_are(self, price_in, price_out):
        import unittest.mock as mock
        # Patch the exact config object build_cost holds a reference to (it
        # imports via the src/-on-sys.path "goodenough" package, not
        # "src.goodenough", so patching by that string name would silently
        # miss and this test would pass for the wrong reason).
        return mock.patch.multiple(
            build_cost.config,
            HOSTED_PRICE_INPUT_PER_M=price_in, HOSTED_PRICE_OUTPUT_PER_M=price_out,
        )

    def test_cascade_pays_local_latency_for_every_item_plus_hosted_for_escalated_only(self):
        econ = build_cost.policy_economics("cascade", self.policies["cascade"],
                                          self.metrics, self.all_ids)
        # q1: 1000 (local) + 300 (hosted, escalated) = 1300ms
        # q2: 500 (local only, not escalated) = 500ms
        # total = 1800ms = 1.8s
        self.assertAlmostEqual(econ["total_seconds"], 1.8)
        self.assertAlmostEqual(econ["median_seconds_per_request"], (1.3 + 0.5) / 2)

    def test_cascade_only_charges_dollars_for_the_escalated_item(self):
        with self._prices_are(1.0, 2.0):
            econ = build_cost.policy_economics("cascade", self.policies["cascade"],
                                              self.metrics, self.all_ids)
        # Only q1 is escalated: 1M input * $1/M + 1M output * $2/M = $3. q2's
        # hosted cost must NOT be charged even though hosted_correct exists for it.
        self.assertAlmostEqual(econ["total_usd"], 3.0)

    def test_map_based_never_double_charges_latency(self):
        # Contrast case: map_based pays EITHER local OR hosted per item, never
        # both, unlike cascade.
        econ = build_cost.policy_economics("map_based", self.policies["map_based"],
                                          self.metrics, self.all_ids)
        # Whole subject "s" is below_margin -> both items routed hosted.
        # q1: 300ms, q2: 100ms -> total 400ms = 0.4s (no local latency added).
        self.assertAlmostEqual(econ["total_seconds"], 0.4)

    def test_always_local_never_touches_hosted_dollars_or_latency(self):
        econ = build_cost.policy_economics("always_local", self.policies["always_local"],
                                          self.metrics, self.all_ids)
        self.assertEqual(econ["total_usd"], 0.0)
        self.assertAlmostEqual(econ["total_seconds"], 1.5)  # 1000ms + 500ms

    def test_oracle_has_no_cost_or_latency(self):
        econ = build_cost.policy_economics("oracle", self.policies["oracle"],
                                          self.metrics, self.all_ids)
        self.assertIsNone(econ["total_usd"])
        self.assertIsNone(econ["total_seconds"])


class PartialDataTests(unittest.TestCase):
    """build_cost.py must run on partial data: no crash, no fabricated zero."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmpdir.name) / "empty.sqlite")
        store.connect(self.db_path).close()  # schema only, no rows

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_no_router_data_yet_reports_not_available_not_a_zero(self):
        report = build_cost.build(self.db_path)
        self.assertIsNone(report["router_split"])

    def test_no_hosted_data_yet_project_totals_show_zero_calls_not_a_crash(self):
        report = build_cost.build(self.db_path)
        self.assertEqual(report["project_totals"]["hosted_calls"], 0)
        self.assertEqual(report["project_totals"]["hosted_list_price_equivalent_usd"], 0.0)

    def test_no_latency_data_yet_inversion_is_none_not_a_fabricated_multiplier(self):
        report = build_cost.build(self.db_path)
        self.assertIsNone(report["latency_inversion"])

    def test_missing_database_file_does_not_crash(self):
        report = build_cost.build(str(Path(self._tmpdir.name) / "does_not_exist.sqlite"))
        self.assertIsNone(report)


if __name__ == "__main__":
    unittest.main()
