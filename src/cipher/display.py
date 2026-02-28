"""Rich-powered real-time terminal visualisation for cipher cracking.

Makes the convergence process *visible and dramatic*.
"""

from __future__ import annotations

import time

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


console = Console(force_terminal=True)


# =====================================================================
# MCMC live display
# =====================================================================


class MCMCDisplay:
    """Rich Live display for MCMC substitution-cipher cracking."""

    def __init__(self, num_chains: int, total_iters_per_chain: int) -> None:
        self.num_chains = num_chains
        self.total_iters = total_iters_per_chain
        self.start_time = time.time()

        self.current_chain = 0
        self.current_iter = 0
        self.current_plaintext = ""
        self.current_score = 0.0
        self.best_score = -1e18
        self.best_plaintext = ""
        self.temperature = 1.0
        self.chain_best_scores: list[float] = []

    def make_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=5),
        )
        layout["body"].split_row(
            Layout(name="current", ratio=1),
            Layout(name="stats", ratio=1),
        )
        return layout

    def render(self) -> Layout:
        layout = self.make_layout()

        # Header
        elapsed = time.time() - self.start_time
        header_text = Text(
            f"  MCMC Cipher Cracker  │  Chain {self.current_chain + 1}/{self.num_chains}"
            f"  │  Iter {self.current_iter:,}  │  {elapsed:.1f}s",
            style="bold white on dark_blue",
        )
        layout["header"].update(Panel(header_text, style="dark_blue"))

        # Current decryption
        pt_display = self.current_plaintext[:500] if self.current_plaintext else "…"
        layout["current"].update(
            Panel(
                Text(
                    pt_display,
                    style="green"
                    if self.current_score > -5 * len(pt_display)
                    else "yellow",
                ),
                title="[bold cyan]Current Decryption",
                border_style="cyan",
            )
        )

        # Stats table
        stats = Table(show_header=False, box=None, padding=(0, 2))
        stats.add_column("Key", style="bold")
        stats.add_column("Value", style="bright_white")
        stats.add_row("Temperature", f"{self.temperature:.6f}")
        stats.add_row("Current Score", f"{self.current_score:,.1f}")
        stats.add_row("Best Score", f"{self.best_score:,.1f}")
        stats.add_row("Chains Done", str(len(self.chain_best_scores)))

        # Add a mini bar for temperature
        t_pct = max(0, min(100, int(self.temperature * 100)))
        t_bar = "█" * (t_pct // 2) + "░" * (50 - t_pct // 2)
        stats.add_row("Temp Bar", f"[red]{t_bar}[/red]")

        layout["stats"].update(
            Panel(stats, title="[bold magenta]Statistics", border_style="magenta")
        )

        # Footer — best decryption so far
        best_display = self.best_plaintext[:300] if self.best_plaintext else "…"
        layout["footer"].update(
            Panel(
                Text(best_display, style="bold bright_green"),
                title="[bold green]★ Best Decryption",
                border_style="green",
            )
        )

        return layout

    def callback(
        self,
        chain_idx: int,
        iteration: int,
        current_key: str,
        current_plaintext: str,
        current_score: float,
        best_score: float,
        temperature: float,
    ) -> None:
        self.current_chain = chain_idx
        self.current_iter = iteration
        self.current_plaintext = current_plaintext
        self.current_score = current_score
        self.best_score = best_score
        self.temperature = temperature
        if current_score >= best_score - 0.01:
            self.best_plaintext = current_plaintext


# =====================================================================
# HMM live display
# =====================================================================


class HMMDisplay:
    """Simple live display for HMM/EM convergence."""

    def __init__(self) -> None:
        self.start_time = time.time()
        self.iteration = 0
        self.plaintext = ""
        self.log_likelihood = -1e18
        self.history: list[float] = []

    def render(self) -> Panel:
        elapsed = time.time() - self.start_time

        lines: list[str] = []
        lines.append(f"Iteration: {self.iteration}  │  Time: {elapsed:.1f}s")
        lines.append(f"Log-likelihood: {self.log_likelihood:,.2f}")
        lines.append("")

        # Mini sparkline of log-likelihood convergence
        if len(self.history) >= 2:
            lo = min(self.history)
            hi = max(self.history)
            rng = hi - lo if hi > lo else 1.0
            bars = "▁▂▃▄▅▆▇█"
            spark = ""
            for v in self.history[-60:]:
                idx = int((v - lo) / rng * (len(bars) - 1))
                spark += bars[max(0, min(idx, len(bars) - 1))]
            lines.append(f"Convergence: {spark}")
        lines.append("")

        pt = self.plaintext[:400] if self.plaintext else "…"
        lines.append(pt)

        return Panel(
            "\n".join(lines),
            title="[bold cyan]HMM / Baum-Welch",
            border_style="cyan",
        )

    def callback(self, iteration: int, plaintext: str, log_likelihood: float) -> None:
        self.iteration = iteration
        self.plaintext = plaintext
        self.log_likelihood = log_likelihood
        self.history.append(log_likelihood)


# =====================================================================
# Generic simple display for Vigenère / Transposition / Playfair
# =====================================================================


class GenericDisplay:
    """Minimal live display for any cracker variant."""

    def __init__(self, title: str = "Cipher Cracker") -> None:
        self.title = title
        self.start_time = time.time()
        self.messages: list[str] = []
        self.best_plaintext: str = ""
        self.best_score: float = -1e18

    def render(self) -> Panel:
        elapsed = time.time() - self.start_time
        lines = [f"Elapsed: {elapsed:.1f}s  │  Best score: {self.best_score:,.1f}"]
        lines.append("")
        for msg in self.messages[-10:]:
            lines.append(msg)
        lines.append("")
        lines.append(self.best_plaintext[:400] if self.best_plaintext else "…")
        return Panel(
            "\n".join(lines),
            title=f"[bold cyan]{self.title}",
            border_style="cyan",
        )

    def update(self, msg: str, plaintext: str = "", score: float = -1e18) -> None:
        self.messages.append(msg)
        if score > self.best_score:
            self.best_score = score
            self.best_plaintext = plaintext


# =====================================================================
# Helpers to print final results
# =====================================================================


def print_result_box(
    cipher_type: str,
    method: str,
    key: str,
    plaintext: str,
    score: float,
    elapsed: float,
) -> None:
    """Print a beautiful final-result panel."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")
    table.add_row("Cipher", cipher_type)
    table.add_row("Method", method)
    table.add_row("Key", key[:80])
    table.add_row("Score", f"{score:,.2f}")
    table.add_row("Time", f"{elapsed:.2f}s")

    console.print()
    console.print(Panel(table, title="[bold green]★ Result", border_style="green"))
    console.print()
    console.print(
        Panel(
            Text(plaintext[:1000], style="bright_white"),
            title="[bold green]Decrypted Text",
            border_style="green",
        )
    )
    console.print()
