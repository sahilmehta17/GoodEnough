import math
import unittest

from src.goodenough import analysis


def _item(item_id, subject, local_correct, local_parse_ok, hosted_correct):
    return {
        "item_id": item_id,
        "subject": subject,
        "local_correct": local_correct,
        "local_parse_ok": local_parse_ok,
        "hosted_correct": hosted_correct,
    }


class RouterPolicyBackwardCompatTests(unittest.TestCase):
    """
    build_router.py calls evaluate_router_policies with items that have no
    item_id key at all (subject, local_correct, local_parse_ok, hosted_correct
    only). The cost report needs per-item attribution, so the function grew a
    new hosted_item_ids key. This must not change anything build_router.py
    already reads.
    """

    def setUp(self):
        # No "item_id" key, matching build_router.py's router_items() shape.
        self.items = [
            {"subject": "formal_logic", "local_correct": 1, "local_parse_ok": True,
             "hosted_correct": 1},
            {"subject": "formal_logic", "local_correct": 0, "local_parse_ok": True,
             "hosted_correct": 1},
            {"subject": "nutrition", "local_correct": 1, "local_parse_ok": False,
             "hosted_correct": 0},
            {"subject": "nutrition", "local_correct": 0, "local_parse_ok": True,
             "hosted_correct": 0},
        ]
        self.verdicts = {"formal_logic": "non_inferior", "nutrition": "below_margin"}
        self.result = analysis.evaluate_router_policies(self.items, self.verdicts)

    def test_existing_keys_and_values_unchanged(self):
        self.assertEqual(self.result["always_local"]["accuracy"], 0.5)
        self.assertEqual(self.result["always_local"]["hosted_calls"], 0)
        self.assertEqual(self.result["always_local"]["n"], 4)

        self.assertEqual(self.result["always_hosted"]["accuracy"], 0.5)
        self.assertEqual(self.result["always_hosted"]["hosted_calls"], 4)

        # formal_logic is non_inferior -> local both items (1 hit of 2).
        # nutrition is below_margin -> hosted both items (0 hits of 2).
        self.assertEqual(self.result["map_based"]["accuracy"], 0.25)
        self.assertEqual(self.result["map_based"]["hosted_calls"], 2)

        # Escalate only the one item with local_parse_ok=False.
        self.assertEqual(self.result["cascade"]["hosted_calls"], 1)
        self.assertEqual(self.result["cascade"]["escalation_rate"], 0.25)
        # local_correct hits: item1 (1) + item2 (0) + item4 (0) = 1; escalated
        # item3 contributes hosted_correct=0. 1 hit of 4.
        self.assertEqual(self.result["cascade"]["accuracy"], 0.25)

        self.assertEqual(self.result["oracle"]["hosted_calls"], None)
        self.assertEqual(self.result["oracle"]["accuracy"], 0.75)

    def test_all_five_policies_present_with_expected_key_set(self):
        base_keys = {"accuracy", "hosted_calls", "n", "hosted_item_ids"}
        for policy in ("always_local", "always_hosted", "map_based", "oracle"):
            self.assertTrue(base_keys.issubset(self.result[policy].keys()))
        self.assertTrue(
            {"escalation_rate"}.issubset(self.result["cascade"].keys())
        )

    def test_hosted_item_ids_is_list_of_none_when_caller_omits_item_id(self):
        # Doesn't crash, and is additive: a caller ignoring the new key sees no
        # behavior change at all.
        self.assertEqual(self.result["always_local"]["hosted_item_ids"], [])
        self.assertEqual(len(self.result["always_hosted"]["hosted_item_ids"]), 4)
        self.assertTrue(all(i is None for i in self.result["always_hosted"]["hosted_item_ids"]))


class RouterPolicyHostedItemIdsTests(unittest.TestCase):
    """With item_id supplied, hosted_item_ids attributes cost per policy."""

    def setUp(self):
        self.items = [
            _item("q1", "formal_logic", local_correct=1, local_parse_ok=True, hosted_correct=1),
            _item("q2", "formal_logic", local_correct=0, local_parse_ok=True, hosted_correct=1),
            _item("q3", "nutrition", local_correct=1, local_parse_ok=False, hosted_correct=0),
            _item("q4", "nutrition", local_correct=0, local_parse_ok=True, hosted_correct=0),
        ]
        self.verdicts = {"formal_logic": "non_inferior", "nutrition": "below_margin"}
        self.result = analysis.evaluate_router_policies(self.items, self.verdicts)

    def test_always_local_sends_nothing_hosted(self):
        self.assertEqual(self.result["always_local"]["hosted_item_ids"], [])

    def test_always_hosted_sends_everything(self):
        self.assertEqual(set(self.result["always_hosted"]["hosted_item_ids"]),
                         {"q1", "q2", "q3", "q4"})

    def test_map_based_sends_only_below_margin_subject_items(self):
        self.assertEqual(set(self.result["map_based"]["hosted_item_ids"]), {"q3", "q4"})

    def test_cascade_sends_only_escalated_items(self):
        self.assertEqual(self.result["cascade"]["hosted_item_ids"], ["q3"])

    def test_oracle_has_no_hosted_item_ids_because_it_is_not_deployable(self):
        self.assertIsNone(self.result["oracle"]["hosted_item_ids"])


class OutcomeBreakdownTests(unittest.TestCase):
    """
    The four-way split of a paired split: how often each model was alone in
    getting an item right. 'local_only' is the quantity that decides whether
    routing can beat always-hosted at all.
    """

    def test_counts_each_of_the_four_cells(self):
        local = [1, 1, 0, 0, 1]
        hosted = [1, 0, 1, 0, 1]
        got = analysis.outcome_breakdown(local, hosted)
        self.assertEqual(got["both_correct"], 2)      # items 1, 5
        self.assertEqual(got["local_only"], 1)        # item 2
        self.assertEqual(got["hosted_only"], 1)       # item 3
        self.assertEqual(got["neither_correct"], 1)   # item 4
        self.assertEqual(got["n"], 5)

    def test_four_cells_sum_to_n_so_no_item_is_dropped(self):
        local = [1, 0, 1, 0, 0, 1, 1]
        hosted = [0, 0, 1, 1, 0, 1, 0]
        got = analysis.outcome_breakdown(local, hosted)
        total = (got["both_correct"] + got["local_only"]
                 + got["hosted_only"] + got["neither_correct"])
        self.assertEqual(total, got["n"])
        self.assertEqual(got["n"], len(local))

    def test_empty_input_is_all_zeros_not_an_error(self):
        got = analysis.outcome_breakdown([], [])
        self.assertEqual(got, {"both_correct": 0, "local_only": 0, "hosted_only": 0,
                               "neither_correct": 0, "n": 0})


class PearsonTests(unittest.TestCase):
    """Pearson correlation, used across slices for the difficulty check."""

    def test_perfect_positive_relationship_is_one(self):
        self.assertAlmostEqual(analysis.pearson([1, 2, 3, 4], [2, 4, 6, 8]), 1.0, places=12)

    def test_perfect_negative_relationship_is_minus_one(self):
        self.assertAlmostEqual(analysis.pearson([1, 2, 3, 4], [8, 6, 4, 2]), -1.0, places=12)

    def test_known_value_on_a_hand_checked_sample(self):
        # Both means are 3. Centred: dx = [-2,-1,0,1,2], dy = [-1,-2,1,0,2].
        # sum(dx*dy) = 8, sum(dx^2) = sum(dy^2) = 10, so r = 8/10 = 0.8.
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [2.0, 1.0, 4.0, 3.0, 5.0]
        self.assertAlmostEqual(analysis.pearson(xs, ys), 0.8, places=12)

    def test_zero_variance_is_undefined_and_returns_none(self):
        self.assertIsNone(analysis.pearson([3, 3, 3], [1, 2, 3]))

    def test_fewer_than_two_points_returns_none(self):
        self.assertIsNone(analysis.pearson([1.0], [2.0]))


class BootstrapPearsonTests(unittest.TestCase):
    """Percentile bootstrap over the (x, y) pairs, same shape as the other CIs."""

    XS = [0.95, 0.68, 0.85, 0.91, 0.90, 0.44, 0.64, 0.92]
    YS = [-0.22, -0.33, -0.22, -0.16, -0.21, -0.18, -0.29, -0.12]

    def test_interval_is_ordered_and_inside_the_correlation_range(self):
        got = analysis.bootstrap_pearson(self.XS, self.YS, iters=500, seed=42)
        self.assertLessEqual(got["lower"], got["upper"])
        self.assertGreaterEqual(got["lower"], -1.0)
        self.assertLessEqual(got["upper"], 1.0)

    def test_same_seed_gives_the_same_interval(self):
        a = analysis.bootstrap_pearson(self.XS, self.YS, iters=500, seed=42)
        b = analysis.bootstrap_pearson(self.XS, self.YS, iters=500, seed=42)
        self.assertEqual(a, b)

    def test_perfectly_correlated_data_bootstraps_to_one(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        got = analysis.bootstrap_pearson(xs, [2 * x for x in xs], iters=300, seed=42)
        self.assertAlmostEqual(got["lower"], 1.0, places=9)
        self.assertAlmostEqual(got["upper"], 1.0, places=9)

    def test_resamples_with_no_variance_are_skipped_not_counted_as_zero(self):
        # Only two distinct points: many resamples draw the same point twice,
        # where r is undefined. Those must be dropped, leaving a finite interval.
        got = analysis.bootstrap_pearson([1.0, 2.0], [1.0, 2.0], iters=200, seed=42)
        self.assertIsNotNone(got["lower"])
        self.assertTrue(math.isfinite(got["lower"]))
        self.assertTrue(math.isfinite(got["upper"]))


if __name__ == "__main__":
    unittest.main()
