"""Continuous runtime health testing for entropy-source output streams.

NIST SP 800-90B §4.4 requires that an entropy source's health tests run not
only at startup but *continuously* on the generated output.  A source can
develop a fault mid-run (e.g. a stuck ADC, a biased oscillator) that startup
tests cannot detect.  This module provides a stateful :class:`HealthMonitor`
that feeds the Repetition Count and Adaptive Proportion tests *incrementally*
over an output stream and raises the instant either test detects a fault.

The monitor treats the conditioned output as a sequence of *n*-bit symbols
(8-bit bytes by default).  Once a fault is detected the monitor latches: every
subsequent call raises, preventing a faulty source from producing any further
output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.logger import logger
from src.low_level.entropy import EntropyHealthError

# One-sided z-score for a 2^-20 false-positive rate (NIST SP 800-90B §4.4).
# Mirrors the constant in entropy.py; defined locally to avoid importing a
# private name across module boundaries.
_Z_SCORE = 4.753


class RuntimeHealthError(EntropyHealthError):
    """Raised when a continuous runtime health test fails.

    The output stream exhibited a stuck-at run or a proportion bias that is
    statistically impossible under a healthy uniform source.  This indicates a
    catastrophic failure of the conditioning/expansion layer and the source must
    not be trusted any further.
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
    (§4.4.2) incrementally over a stream of symbols.  Both tests use a 2^-20
    false-positive rate (``C = +4.753``).  A failure raises
    :class:`RuntimeHealthError` and latches the monitor so that no further
    output can pass through.

    Args:
        bits_per_symbol: Bit-width of each symbol extracted from the stream.
            Defaults to ``8`` (raw bytes).
        window: Adaptive Proportion Test window size in samples.  Defaults to
            ``1024``.
    """

    def __init__(self, bits_per_symbol: int = 8, window: int = 1024) -> None:
        if bits_per_symbol <= 0 or bits_per_symbol > 8:
            raise ValueError("bits_per_symbol must be in [1, 8]")
        if window <= 0:
            raise ValueError("window must be positive")

        self._bits_per_symbol: int = bits_per_symbol
        self._mask: int = (1 << bits_per_symbol) - 1
        self._window: int = window

        # --- Repetition Count Test state (§4.4.1) ---
        p_rc = 2.0 ** (-bits_per_symbol)
        self._rc_cutoff: int = 1 + math.ceil(-1.0 / math.log2(1.0 - p_rc))
        self._rc_prev: int | None = None
        self._rc_run: int = 0
        self._rc_max_run: int = 0

        # --- Adaptive Proportion Test state (§4.4.2) ---
        alphabet = 1 << bits_per_symbol
        p_ap = 1.0 / alphabet
        mu = window * p_ap
        sigma = math.sqrt(window * p_ap * (1.0 - p_ap))
        self._ap_cutoff: int = int(math.ceil(mu + _Z_SCORE * sigma))
        self._ap_buf: list[int] = []
        self._ap_windows: int = 0
        self._ap_max_count: int = 0

        self._total: int = 0
        self._failed: bool = False

    def observe(self, data: bytes) -> None:
        """Feed *data* through the continuous health tests.

        Symbols are extracted as the low *bits_per_symbol* bits of each byte.
        Raises :class:`RuntimeHealthError` immediately on the first fault and
        latches so all subsequent calls also raise.

        Args:
            data: Raw output bytes to test.

        Raises:
            RuntimeHealthError: If a health test detects a fault.
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
        """Incrementally track consecutive runs and fail on an over-long run."""
        if sym == self._rc_prev:
            self._rc_run += 1
        else:
            self._rc_prev = sym
            self._rc_run = 1

        if self._rc_run > self._rc_max_run:
            self._rc_max_run = self._rc_run

        if self._rc_run >= self._rc_cutoff:
            self._fail(
                f"Repetition Count Test FAILED: run of {self._rc_run} identical "
                f"{self._bits_per_symbol}-bit symbols reached the cutoff of "
                f"{self._rc_cutoff} (NIST SP 800-90B §4.4.1)."
            )

    def _evaluate_adaptive_proportion(self) -> None:
        """Score one completed non-overlapping window of the Adaptive Proportion Test."""
        buf = self._ap_buf
        target = buf[0]
        count = sum(1 for s in buf if s == target)

        if count > self._ap_max_count:
            self._ap_max_count = count

        self._ap_windows += 1
        self._ap_buf = []

        if count > self._ap_cutoff:
            self._fail(
                f"Adaptive Proportion Test FAILED: symbol {target} appeared "
                f"{count} times in a {self._window}-sample window (cutoff "
                f"{self._ap_cutoff}, NIST SP 800-90B §4.4.2)."
            )

    def _fail(self, reason: str) -> None:
        """Latch the monitor and raise a :class:`RuntimeHealthError`."""
        self._failed = True
        logger.error("[health] %s", reason)
        raise RuntimeHealthError(reason)

    @property
    def failed(self) -> bool:
        """Whether the monitor has latched on a prior failure."""
        return self._failed

    @property
    def status(self) -> HealthStatus:
        """Return a snapshot of the current health-test counters."""
        return HealthStatus(
            total_samples=self._total,
            repetition_count_max_run=self._rc_max_run,
            repetition_count_cutoff=self._rc_cutoff,
            adaptive_proportion_windows=self._ap_windows,
            adaptive_proportion_max_count=self._ap_max_count,
            adaptive_proportion_cutoff=self._ap_cutoff,
        )
