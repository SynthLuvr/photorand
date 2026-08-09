"""Tests for src.low_level.health — continuous runtime health monitoring."""

from __future__ import annotations

import os

import pytest

from src.low_level.health import HealthMonitor, HealthStatus, RuntimeHealthError


class TestHealthMonitorHealthy:
    """A healthy uniform stream must never trip the monitor."""

    def test_random_stream_passes(self) -> None:
        """os.urandom output is well-behaved and must not raise."""
        monitor = HealthMonitor()
        monitor.observe(os.urandom(4000))
        assert monitor.failed is False

    def test_incremental_healthy_chunks(self) -> None:
        """Feeding a healthy stream byte-by-byte must not raise."""
        monitor = HealthMonitor()
        for byte in os.urandom(2000):
            monitor.observe(bytes([byte]))
        assert monitor.failed is False


class TestRepetitionCount:
    """Repetition Count Test (NIST SP 800-90B §4.4.1)."""

    def test_stuck_stream_raises(self) -> None:
        """A long run of identical bytes is a stuck-at fault and must raise."""
        monitor = HealthMonitor()
        with pytest.raises(RuntimeHealthError, match="Repetition Count"):
            monitor.observe(b"\x00" * 200)

    def test_cutoff_for_8bit(self) -> None:
        """For 8-bit symbols the cutoff is 179 (run of 179 fails)."""
        monitor = HealthMonitor(bits_per_symbol=8)
        # 178 identical bytes is the longest passing run (run < cutoff).
        monitor.observe(b"\x00" * 178)
        assert monitor.failed is False
        # One more trips it.
        with pytest.raises(RuntimeHealthError):
            monitor.observe(b"\x00")

    def test_run_detected_across_chunks(self) -> None:
        """A stuck run split across many small observe() calls is still caught."""
        monitor = HealthMonitor()
        stuck = b"\x00" * 50
        with pytest.raises(RuntimeHealthError, match="Repetition Count"):
            for _ in range(5):  # 250 identical bytes in 50-byte chunks
                monitor.observe(stuck)

    def test_brief_runs_pass(self) -> None:
        """Short runs of identical bytes are normal and pass.

        Uses uniformly distributed bytes so no single symbol is over-represented
        (which would trip the Adaptive Proportion Test), keeping the focus on the
        Repetition Count Test.
        """
        monitor = HealthMonitor()
        monitor.observe(os.urandom(2000))
        assert monitor.failed is False
        assert monitor.status.repetition_count_max_run < monitor.status.repetition_count_cutoff


class TestAdaptiveProportion:
    """Adaptive Proportion Test (NIST SP 800-90B §4.4.2)."""

    @staticmethod
    def _biased_window() -> bytes:
        """A 1024-byte window where the first symbol is over-represented.

        Symbol 0 (the window target) appears ~52 times — far above the cutoff of
        14 — yet no consecutive run reaches the repetition-count cutoff (179),
        so only the Adaptive Proportion Test fires.
        """
        window = bytearray((i % 255) + 1 for i in range(1024))
        window[0] = 0
        for pos in range(2, 1024, 20):
            window[pos] = 0
        return bytes(window)

    def test_biased_window_raises(self) -> None:
        """An over-represented symbol in a full window must raise."""
        monitor = HealthMonitor()
        with pytest.raises(RuntimeHealthError, match="Adaptive Proportion"):
            monitor.observe(self._biased_window())

    def test_partial_window_does_not_raise(self) -> None:
        """A window shorter than ``window`` samples cannot be scored yet."""
        monitor = HealthMonitor()
        # Only 500 bytes — below the 1024 default window — so no AP evaluation.
        partial = self._biased_window()[:500]
        monitor.observe(partial)
        assert monitor.failed is False

    def test_biased_detected_across_chunks(self) -> None:
        """A biased window fed in small chunks is still scored once complete."""
        monitor = HealthMonitor()
        window = self._biased_window()
        with pytest.raises(RuntimeHealthError, match="Adaptive Proportion"):
            for i in range(0, len(window), 64):
                monitor.observe(window[i : i + 64])


class TestLatching:
    """Once a fault is detected the monitor latches permanently."""

    def test_latched_after_failure(self) -> None:
        """After a failure every subsequent observe() also raises."""
        monitor = HealthMonitor()
        with pytest.raises(RuntimeHealthError):
            monitor.observe(b"\x00" * 200)
        assert monitor.failed is True
        with pytest.raises(RuntimeHealthError, match="latched"):
            monitor.observe(b"\x01")

    def test_latched_after_adaptive_failure(self) -> None:
        """Latching also applies after an Adaptive Proportion failure."""
        monitor = HealthMonitor()
        window = bytearray((i % 255) + 1 for i in range(1024))
        window[0] = 42
        for pos in range(2, 1024, 20):
            window[pos] = 42
        with pytest.raises(RuntimeHealthError, match="Adaptive Proportion"):
            monitor.observe(bytes(window))
        assert monitor.failed is True
        with pytest.raises(RuntimeHealthError, match="latched"):
            monitor.observe(b"\x01")


class TestStatus:
    """The status snapshot must reflect accumulated counters."""

    def test_initial_status(self) -> None:
        monitor = HealthMonitor()
        status = monitor.status
        assert isinstance(status, HealthStatus)
        assert status.total_samples == 0
        assert status.repetition_count_max_run == 0
        assert status.adaptive_proportion_windows == 0

    def test_status_after_healthy_stream(self) -> None:
        monitor = HealthMonitor()
        monitor.observe(os.urandom(2048))
        status = monitor.status
        assert status.total_samples == 2048
        # Two full windows of 1024 have been scored.
        assert status.adaptive_proportion_windows == 2
        # The repetition-count cutoff for 8-bit symbols.
        assert status.repetition_count_cutoff == 179
        assert status.adaptive_proportion_cutoff == 14
        assert status.repetition_count_max_run < status.repetition_count_cutoff

    def test_status_counts_max_run(self) -> None:
        """max_run tracks the longest run seen even when it does not fail."""
        monitor = HealthMonitor()
        # 50 identical bytes — under the 179 cutoff — should be recorded as max_run.
        monitor.observe(b"\x07" * 50)
        assert monitor.status.repetition_count_max_run == 50


class TestValidation:
    """Constructor parameter validation."""

    def test_invalid_bits_per_symbol(self) -> None:
        with pytest.raises(ValueError, match="bits_per_symbol"):
            HealthMonitor(bits_per_symbol=0)

    def test_invalid_bits_per_symbol_high(self) -> None:
        with pytest.raises(ValueError, match="bits_per_symbol"):
            HealthMonitor(bits_per_symbol=9)

    def test_invalid_window(self) -> None:
        with pytest.raises(ValueError, match="window"):
            HealthMonitor(window=0)


class TestExceptionHierarchy:
    """RuntimeHealthError is a HealthError so existing handlers catch it."""

    def test_runtime_health_error_is_health_error(self) -> None:
        from src.low_level.entropy import EntropyHealthError

        assert issubclass(RuntimeHealthError, EntropyHealthError)
