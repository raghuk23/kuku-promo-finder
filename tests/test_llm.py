"""LLM client plumbing: retries, transient-error detection, provider switch."""
import os
import unittest
from unittest import mock

from src.llm import is_transient, is_daily_quota, with_retries, make_client


class TestTransient(unittest.TestCase):
    def test_overload_and_rate_limit_are_transient(self):
        self.assertTrue(is_transient(Exception("503 UNAVAILABLE. high demand")))
        self.assertTrue(is_transient(Exception("429 RESOURCE_EXHAUSTED")))

    def test_dropped_connections_and_timeouts_are_transient(self):
        self.assertTrue(is_transient(Exception("RemoteProtocolError: Server disconnected without sending a response")))
        self.assertTrue(is_transient(Exception("ReadTimeout: The read operation timed out")))
        self.assertTrue(is_transient(Exception("ConnectError: [Errno 54] Connection reset by peer")))

    def test_daily_quota_is_not_transient(self):
        # Real message from the first run: retrying this only burns time until reset.
        e = Exception("429 RESOURCE_EXHAUSTED. Generate content requests per day limit exceeded on "
                      "Free Tier. quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
        self.assertTrue(is_daily_quota(e))
        self.assertFalse(is_transient(e))

    def test_per_minute_rate_limit_is_transient(self):
        e = Exception("429 RESOURCE_EXHAUSTED. quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier")
        self.assertFalse(is_daily_quota(e))
        self.assertTrue(is_transient(e))

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


class TestStrictModel(unittest.TestCase):
    CFG = {"llm": {"provider": "gemini", "model": "m1", "fallback_models": ["m2"]}}

    def build(self, strict):
        cfg = {"llm": dict(self.CFG["llm"], strict_model=strict)}
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            return make_client(cfg, provider="gemini")

    def test_strict_uses_only_the_primary_and_requires_it(self):
        c = self.build(True)
        self.assertEqual(c.models, ["m1"])
        self.assertEqual(c.required_model, "m1")

    def test_explicit_api_key_needs_no_env_var(self):
        # The Streamlit app passes a pasted key straight through: never via os.environ,
        # which every visitor of a shared deployment would see.
        cfg = {"llm": dict(self.CFG["llm"])}
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch("dotenv.load_dotenv"):
            c = make_client(cfg, provider="gemini", api_key="pasted-key")
        self.assertEqual(c.models[0], "m1")

    def test_non_strict_may_fall_back(self):
        c = self.build(False)
        self.assertEqual(c.models, ["m1", "m2"])
        self.assertIsNone(c.required_model)


if __name__ == "__main__":
    unittest.main()
