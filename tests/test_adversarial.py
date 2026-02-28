"""Tests for adversarial stress tests (adversarial.py)."""

from __future__ import annotations

import string

import pytest


class TestStressTestCase:
    def test_defaults(self) -> None:
        from cipher.adversarial import StressTestCase

        c = StressTestCase()
        assert c.name == ""
        assert c.difficulty == "normal"


class TestStressTestResult:
    def test_defaults(self) -> None:
        from cipher.adversarial import StressTestResult

        r = StressTestResult()
        assert r.ser == 0.0
        assert r.success is False


class TestTextGenerators:
    def test_legal_text(self) -> None:
        from cipher.adversarial import _legal_text

        text = _legal_text()
        assert len(text) > 50
        assert all(ch in string.ascii_lowercase + " " for ch in text)

    def test_technical_text(self) -> None:
        from cipher.adversarial import _technical_text

        text = _technical_text()
        assert len(text) > 50

    def test_poetry_text(self) -> None:
        from cipher.adversarial import _poetry_text

        text = _poetry_text()
        assert len(text) > 50

    def test_repetitive_text(self) -> None:
        from cipher.adversarial import _repetitive_text

        text = _repetitive_text()
        assert len(text) > 20

    def test_short_text(self) -> None:
        from cipher.adversarial import _short_text

        text = _short_text()
        assert 30 < len(text) < 100

    def test_very_short_text(self) -> None:
        from cipher.adversarial import _very_short_text

        text = _very_short_text()
        assert len(text) < 30


class TestKeyGenerators:
    def test_uniform_key(self) -> None:
        from cipher.adversarial import _uniform_key

        key = _uniform_key()
        assert len(key) == 26
        assert sorted(key) == list(string.ascii_lowercase)

    def test_adversarial_key(self) -> None:
        from cipher.adversarial import _adversarial_key

        key = _adversarial_key()
        assert len(key) == 26
        assert sorted(key) == list(string.ascii_lowercase)


class TestGenerateStressTests:
    def test_returns_list(self) -> None:
        from cipher.adversarial import generate_stress_tests

        cases = generate_stress_tests()
        assert isinstance(cases, list)
        assert len(cases) == 8

    def test_each_case_has_fields(self) -> None:
        from cipher.adversarial import generate_stress_tests

        for case in generate_stress_tests():
            assert case.name != ""
            assert case.plaintext != ""
            assert case.ciphertext != ""
            assert case.key != ""
            assert len(case.key) == 26

    def test_ciphertext_differs_from_plaintext(self) -> None:
        from cipher.adversarial import generate_stress_tests

        for case in generate_stress_tests():
            # At least some cases should have different ct vs pt
            # (identity key would be equal, but adversarial key differs)
            if case.name not in ("near_identity_key",):
                assert (
                    case.ciphertext != case.plaintext
                    or case.key == string.ascii_lowercase
                )

    def test_cases_cover_difficulties(self) -> None:
        from cipher.adversarial import generate_stress_tests

        difficulties = {c.difficulty for c in generate_stress_tests()}
        assert len(difficulties) >= 2  # At least normal and hard


class TestRunStressTests:
    @pytest.mark.timeout(300)
    def test_runs_single_case(self, trained_model) -> None:
        """Run a single stress test to verify the framework works."""
        from cipher.adversarial import generate_stress_tests, run_stress_tests

        cases = generate_stress_tests()
        # Pick the shortest test to minimize time
        shortest = min(cases, key=lambda c: len(c.plaintext))

        results = run_stress_tests(trained_model, [shortest])
        assert len(results) == 1
        r = results[0]
        assert r.case_name == shortest.name
        assert 0 <= r.ser <= 1
        assert r.time_seconds > 0
        assert r.decrypted != ""

    @pytest.mark.timeout(120)
    def test_callback_invoked(self, trained_model) -> None:
        from cipher.adversarial import StressTestCase, run_stress_tests
        from cipher.ciphers import SimpleSubstitution

        key = SimpleSubstitution.random_key()
        case = StressTestCase(
            name="test_cb",
            plaintext="hello world test",
            ciphertext=SimpleSubstitution.encrypt("hello world test", key),
            key=key,
        )
        calls = []
        run_stress_tests(
            trained_model, [case], callback=lambda name, r: calls.append(name)
        )
        assert len(calls) == 1
