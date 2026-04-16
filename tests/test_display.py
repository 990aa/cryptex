"""Tests for display.py — Rich terminal visualisation components."""

from __future__ import annotations


from rich.panel import Panel

from cryptex.display import (
    GenericDisplay,
    HMMDisplay,
    MCMCDisplay,
    print_result_box,
)


class TestMCMCDisplay:
    def test_init(self) -> None:
        d = MCMCDisplay(num_chains=4, total_iters_per_chain=50_000, ciphertext="abc")
        assert d.num_chains == 4
        assert d.total_iters == 50_000
        assert d.ciphertext == "abc"
        assert d.current_chain == 0
        assert d.current_iter == 0
        assert d.best_score == -1e18
        assert d.score_history == []
        assert d.accept_count == 0

    def test_make_layout(self) -> None:
        d = MCMCDisplay(2, 100)
        layout = d.make_layout()
        # Layout should have named sections
        assert layout["header"] is not None
        assert layout["body"] is not None
        assert layout["footer"] is not None

    def test_render_returns_layout(self) -> None:
        d = MCMCDisplay(1, 100)
        layout = d.render()
        # Should not raise; returns a Layout object
        assert layout is not None

    def test_sparkline_empty(self) -> None:
        d = MCMCDisplay(1, 100)
        s = d._sparkline(20)
        assert isinstance(s, str)
        assert len(s) > 0  # returns "..." pattern when no history

    def test_sparkline_with_history(self) -> None:
        d = MCMCDisplay(1, 100)
        d.score_history = list(range(100))
        s = d._sparkline(50)
        assert isinstance(s, str)
        assert len(s) <= 50

    def test_render_key_mapping_no_key(self) -> None:
        d = MCMCDisplay(1, 100)
        result = d._render_key_mapping()
        assert "Waiting" in result

    def test_render_key_mapping_with_key(self) -> None:
        d = MCMCDisplay(1, 100)
        d.current_key = "zyxwvutsrqponmlkjihgfedcba"
        result = d._render_key_mapping()
        assert "z" in result
        assert len(result.splitlines()) >= 4  # 2 rows × (cipher + plain) + separator

    def test_callback_updates_fields(self) -> None:
        d = MCMCDisplay(2, 1000, ciphertext="xyz")
        d.callback(
            chain_idx=1,
            iteration=42,
            current_key="abcdefghijklmnopqrstuvwxyz",
            current_plaintext="hello world",
            current_score=-300.0,
            best_score=-250.0,
            temperature=5.0,
        )
        assert d.current_chain == 1
        assert d.current_iter == 42
        assert d.current_plaintext == "hello world"
        assert d.temperature == 5.0
        assert len(d.score_history) == 1
        assert d.score_history[0] == -300.0

    def test_callback_updates_best_plaintext(self) -> None:
        d = MCMCDisplay(1, 100)
        d.callback(0, 1, "a" * 26, "first", -100.0, -100.0, 1.0)
        assert d.best_plaintext == "first"
        # Score is at best => update
        d.callback(0, 2, "a" * 26, "second", -200.0, -100.0, 1.0)
        # -200 < -100 - 0.01 so best_plaintext should NOT update
        assert d.best_plaintext == "first"


class TestHMMDisplay:
    def test_init(self) -> None:
        d = HMMDisplay()
        assert d.iteration == 0
        assert d.plaintext == ""
        assert d.log_likelihood == -1e18
        assert d.history == []

    def test_render_returns_panel(self) -> None:
        d = HMMDisplay()
        result = d.render()
        assert isinstance(result, Panel)

    def test_callback(self) -> None:
        d = HMMDisplay()
        d.callback(5, "hello", -42.5)
        assert d.iteration == 5
        assert d.plaintext == "hello"
        assert d.log_likelihood == -42.5
        assert len(d.history) == 1

    def test_render_with_history(self) -> None:
        d = HMMDisplay()
        for i in range(10):
            d.callback(i, f"text_{i}", -100.0 + i * 5)
        panel = d.render()
        assert isinstance(panel, Panel)


class TestGenericDisplay:
    def test_init_defaults(self) -> None:
        d = GenericDisplay()
        assert d.title == "Cipher Cracker"
        assert d.messages == []
        assert d.best_score == -1e18

    def test_init_custom_title(self) -> None:
        d = GenericDisplay("My Cracker")
        assert d.title == "My Cracker"

    def test_render_returns_panel(self) -> None:
        d = GenericDisplay()
        assert isinstance(d.render(), Panel)

    def test_update(self) -> None:
        d = GenericDisplay()
        d.update("Found key=ABC", "hello world", -50.0)
        assert len(d.messages) == 1
        assert d.best_plaintext == "hello world"
        assert d.best_score == -50.0

    def test_update_only_improves(self) -> None:
        d = GenericDisplay()
        d.update("first", "aaa", -100.0)
        d.update("second", "bbb", -200.0)  # worse score
        assert d.best_plaintext == "aaa"
        assert d.best_score == -100.0

    def test_render_after_updates(self) -> None:
        d = GenericDisplay("Test")
        for i in range(15):
            d.update(f"msg {i}", f"pt {i}", float(-100 + i))
        panel = d.render()
        assert isinstance(panel, Panel)


class TestPrintResultBox:
    def test_prints_without_error(self, capsys) -> None:
        """Just ensure it runs without raising."""
        print_result_box(
            cipher_type="substitution",
            method="MCMC",
            key="abcdefghijklmnopqrstuvwxyz",
            plaintext="hello world",
            score=-123.4,
            elapsed=5.67,
        )

    def test_with_extra_info(self, capsys) -> None:
        print_result_box(
            cipher_type="vigenere",
            method="IoC",
            key="key",
            plaintext="the quick brown fox",
            score=-456.7,
            elapsed=1.23,
            extra_info={"Chains": "4", "Accuracy": "95%"},
        )
