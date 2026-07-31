import unittest

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


if __name__ == "__main__":
    unittest.main()
