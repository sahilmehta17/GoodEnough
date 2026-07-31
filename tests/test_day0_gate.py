import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from scripts import day0_gate


class RequestContractTests(unittest.TestCase):
    def test_semantic_prompt_does_not_contain_local_control_text(self):
        messages = day0_gate.build_messages("Question")
        self.assertNotIn("/no_think", messages[0]["content"])
        self.assertIn('"answer": "C"', messages[0]["content"])

    def test_local_payload_pins_qwen_sampling_and_disables_thinking(self):
        payload = day0_gate.build_payload("Question", "local")
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["top_p"], 0.8)
        self.assertEqual(payload["top_k"], 20)
        self.assertEqual(payload["min_p"], 0)
        self.assertEqual(payload["presence_penalty"], 1.5)
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["max_tokens"], 64)
        self.assertEqual(
            payload["chat_template_kwargs"], {"enable_thinking": False}
        )

    def test_hosted_payload_uses_only_supported_pinned_fields(self):
        payload = day0_gate.build_payload("Question", "hosted")
        self.assertEqual(payload["model"], "llama-3.3-70b-versatile")
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["top_p"], 0.8)
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["max_completion_tokens"], 64)
        for unsupported in (
            "top_k",
            "min_p",
            "presence_penalty",
            "chat_template_kwargs",
        ):
            self.assertNotIn(unsupported, payload)


class GateValidationTests(unittest.TestCase):
    def test_extracts_documented_answer_forms(self):
        self.assertEqual(day0_gate.extract_mcq_answer("answer: C"), "C")
        self.assertEqual(day0_gate.extract_mcq_answer('{"answer": "b"}'), "B")
        self.assertEqual(day0_gate.extract_mcq_answer('Answer : "D"'), "D")

    def test_rejects_empty_reasoning_only_and_unrelated_text(self):
        self.assertIsNone(day0_gate.extract_mcq_answer(""))
        self.assertIsNone(day0_gate.extract_mcq_answer("I considered every option"))

    def test_wait_for_local_survives_loading_then_returns_health(self):
        get_fn = Mock(
            side_effect=[
                RuntimeError("connection refused"),
                {"status": "loading"},
                {"status": "ok"},
            ]
        )
        result = day0_gate.wait_for_local(
            timeout=5,
            poll_interval=0,
            get_fn=get_fn,
            sleep_fn=lambda _: None,
        )
        self.assertEqual(result, {"status": "ok"})
        self.assertEqual(get_fn.call_count, 3)


class ArtifactTests(unittest.TestCase):
    def test_summary_labels_end_to_end_throughput_honestly(self):
        summary = day0_gate.summarize_calls(
            [
                {"latency_seconds": 2.0, "input_tokens": 60, "output_tokens": 8},
                {"latency_seconds": 1.0, "input_tokens": 80, "output_tokens": 10},
            ]
        )
        self.assertEqual(summary["median_e2e_latency_seconds"], 1.5)
        self.assertEqual(summary["mean_output_tokens"], 9)
        self.assertEqual(summary["e2e_output_tokens_per_second"], 6)
        self.assertNotIn("generation_tokens_per_second", summary)

    def test_atomic_writer_creates_readable_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            day0_gate.write_json_atomic(path, {"status": "pass"})
            self.assertEqual(json.loads(path.read_text()), {"status": "pass"})
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
