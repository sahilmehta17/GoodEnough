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


if __name__ == "__main__":
    unittest.main()
