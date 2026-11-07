"""LLM client plumbing: retries, transient-error detection, provider switch."""
import os
import unittest
from unittest import mock

from src.llm import is_transient, with_retries, make_client


class TestTransient(unittest.TestCase):
    def test_overload_and_rate_limit_are_transient(self):
        self.assertTrue(is_transient(Exception("503 UNAVAILABLE. high demand")))
        self.assertTrue(is_transient(Exception("429 RESOURCE_EXHAUSTED")))

    def test_not_found_is_not_transient(self):
        self.assertFalse(is_transient(Exception("404 NOT_FOUND. no longer available")))


class TestRetries(unittest.TestCase):
    def test_retries_transient_then_succeeds(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise Exception("503 UNAVAILABLE")
            return "ok"

        self.assertEqual(with_retries(flaky, retries=4, sleep=lambda s: None), "ok")
        self.assertEqual(len(calls), 3)

    def test_non_transient_fails_fast(self):
        calls = []

        def broken():
            calls.append(1)
            raise Exception("404 NOT_FOUND")

        with self.assertRaises(Exception):
            with_retries(broken, retries=4, sleep=lambda s: None)
        self.assertEqual(len(calls), 1)


class TestProviderSwitch(unittest.TestCase):
    def test_env_var_selects_heuristic_without_any_key(self):
        cfg = {"llm": {"provider": "gemini", "model": "x"}}
        with mock.patch.dict(os.environ, {"LLM_PROVIDER": "heuristic"}):
            self.assertIsNone(make_client(cfg))


if __name__ == "__main__":
    unittest.main()
