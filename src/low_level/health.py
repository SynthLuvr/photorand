"""Continuous runtime health testing for entropy-source output streams.

NIST SP 800-90B §4.4 requires continuous health tests on the generated output,
not just at startup. A source can develop a mid-run fault (stuck ADC, biased
oscillator) that startup tests cannot detect. This module runs the Repetition
Count (§4.4.1) and Adaptive Proportion (§4.4.2) tests incrementally and raises
the instant either detects a fault. Once tripped, the monitor latches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.logger import logger
from src.low_level.entropy import EntropyHealthError

# One-sided z-score for a 2^-20 false-positive rate (NIST SP 800-90B §4.4).
_Z_SCORE = 4.753


class RuntimeHealthError(EntropyHealthError):
    """Raised when a continuous runtime health test fails.

    The output stream exhibited a stuck-at run or proportion bias that is
    statistically impossible under a healthy uniform source.
    """


@dataclass(frozen=True)
class HealthStatus:
    """Snapshot of continuous health-test state at a point in time."""

    total_samples: int
    repetition_count_max_run: int
    repetition_count_cutoff: int
    adaptive_proportion_windows: int
    adaptive_proportion_max_count: int
    adaptive_proportion_cutoff: int


class HealthMonitor:
    """Stateful continuous health monitor (NIST SP 800-90B §4.4).

    Runs the Repetition Count Test (§4.4.1) and the Adaptive Proportion Test
    (§4.4.2) incrementally over a stream of symbols. Both tests use a 2^-20
    false-positive rate. A failure raises :class:`RuntimeHealthError` and
    latches the monitor permanently.

    Args:
        bits_per_symbol: Bit-width of each symbol (default 8, raw bytes).
        window: Adaptive Proportion Test window size in samples (default 1024).
    """

    def __init__(self, bits_per_symbol: int = 8, window: int = 1024) -> None:
        if not 1 <= bits_per_symbol <= 8:
            raise ValueError("bits_per_symbol must be in [1, 8]")
        if window <= 0:
            raise ValueError("window must be positive")

        self._bits_per_symbol: int = bits_per_symbol
        self._mask: int = (1 << bits_per_symbol) - 1
        self._window: int = window

        p = 2.0 ** (-bits_per_symbol)
        self._rc_cutoff: int = 1 + math.ceil(-1.0 / math.log2(1.0 - p))
        self._rc_prev: int | None = None
        self._rc_run: int = 0
        self._rc_max_run: int = 0

        mu = window * p
        sigma = math.sqrt(window * p * (1.0 - p))
        self._ap_cutoff: int = int(math.ceil(mu + _Z_SCORE * sigma))
        self._ap_buf: list[int] = []
        self._ap_windows: int = 0
        self._ap_max_count: int = 0

        self._total: int = 0
        self._failed: bool = False

    def observe(self, data: bytes) -> None:
        """Feed *data* through the continuous health tests.

        Raises :class:`RuntimeHealthError` on the first fault and latches
        so all subsequent calls also raise.
        """
        if self._failed:
            raise RuntimeHealthError(
                "Health monitor is latched after a prior failure; no further output is trusted."
            )

        for byte in data:
            sym = byte & self._mask
            self._total += 1
            self._update_repetition_count(sym)
            self._ap_buf.append(sym)
            if len(self._ap_buf) >= self._window:
                self._evaluate_adaptive_proportion()

    def _update_repetition_count(self, sym: int) -> None:
        if sym == self._rc_prev:
            self._rc_run += 1
        else:
            self._rc_prev = sym
            self._rc_run = 1

        self._rc_max_run = max(self._rc_max_run, self._rc_run)

        if self._rc_run >= self._rc_cutoff:
            self._fail(
                f"Repetition Count Test FAILED: run of {self._rc_run} identical "
                f"{self._bits_per_symbol}-bit symbols reached the cutoff of "
                f"{self._rc_cutoff} (NIST SP 800-90B §4.4.1)."
            )

    def _evaluate_adaptive_proportion(self) -> None:
        target = self._ap_buf[0]
        count = self._ap_buf.count(target)

        self._ap_max_count = max(self._ap_max_count, count)
        self._ap_windows += 1
        self._ap_buf.clear()

        if count > self._ap_cutoff:
            self._fail(
                f"Adaptive Proportion Test FAILED: symbol {target} appeared "
                f"{count} times in a {self._window}-sample window (cutoff "
                f"{self._ap_cutoff}, NIST SP 800-90B §4.4.2)."
            )

    def _fail(self, reason: str) -> None:
        self._failed = True
        logger.error("[health] %s", reason)
        raise RuntimeHealthError(reason)

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def status(self) -> HealthStatus:
        return HealthStatus(
            total_samples=self._total,
            repetition_count_max_run=self._rc_max_run,
            repetition_count_cutoff=self._rc_cutoff,
            adaptive_proportion_windows=self._ap_windows,
            adaptive_proportion_max_count=self._ap_max_count,
            adaptive_proportion_cutoff=self._ap_cutoff,
        )
