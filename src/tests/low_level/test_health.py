"""Tests for src.low_level.health — continuous runtime health monitoring."""

from __future__ import annotations

import os

import pytest

from src.low_level.health import HealthMonitor, HealthStatus, RuntimeHealthError


class TestHealthMonitorHealthy:
    """A healthy uniform stream must never trip the monitor."""

    def test_random_stream_passes(self) -> None:
        monitor = HealthMonitor()
        monitor.observe(os.urandom(4000))
        assert monitor.failed is False

    def test_incremental_healthy_chunks(self) -> None:
        monitor = HealthMonitor()
        for byte in os.urandom(2000):
            monitor.observe(bytes([byte]))
        assert monitor.failed is False


class TestRepetitionCount:
    """Repetition Count Test (NIST SP 800-90B §4.4.1)."""

    def test_stuck_stream_raises(self) -> None:
        monitor = HealthMonitor()
        with pytest.raises(RuntimeHealthError, match="Repetition Count"):
            monitor.observe(b"\x00" * 200)

    def test_cutoff_for_8bit(self) -> None:
        monitor = HealthMonitor(bits_per_symbol=8)
        monitor.observe(b"\x00" * 178)
        assert monitor.failed is False
        with pytest.raises(RuntimeHealthError):
            monitor.observe(b"\x00")

    def test_run_detected_across_chunks(self) -> None:
        monitor = HealthMonitor()
        stuck = b"\x00" * 50
        with pytest.raises(RuntimeHealthError, match="Repetition Count"):
            for _ in range(5):
                monitor.observe(stuck)

    def test_brief_runs_pass(self) -> None:
        # Uniform random so no single symbol over-dominates and trips the
        # Adaptive Proportion Test instead.
        monitor = HealthMonitor()
        monitor.observe(os.urandom(2000))
        assert monitor.failed is False
        assert monitor.status.repetition_count_max_run < monitor.status.repetition_count_cutoff


class TestAdaptiveProportion:
    """Adaptive Proportion Test (NIST SP 800-90B §4.4.2)."""

    @staticmethod
    def _biased_window() -> bytes:
        """A 1024-byte window where the first symbol is heavily over-represented.

        Symbol 0 appears ~52 times (cutoff 14) without a consecutive run long
        enough to trip the Repetition Count Test, so only the AP test fires.
        """
        window = bytearray((i % 255) + 1 for i in range(1024))
        window[0] = 0
        for pos in range(2, 1024, 20):
            window[pos] = 0
        return bytes(window)

    def test_biased_window_raises(self) -> None:
        monitor = HealthMonitor()
        with pytest.raises(RuntimeHealthError, match="Adaptive Proportion"):
            monitor.observe(self._biased_window())

    def test_partial_window_does_not_raise(self) -> None:
        # 500 bytes < default 1024-sample window, so no AP evaluation.
        monitor = HealthMonitor()
        monitor.observe(self._biased_window()[:500])
        assert monitor.failed is False

    def test_biased_detected_across_chunks(self) -> None:
        monitor = HealthMonitor()
        window = self._biased_window()
        with pytest.raises(RuntimeHealthError, match="Adaptive Proportion"):
            for i in range(0, len(window), 64):
                monitor.observe(window[i : i + 64])


class TestLatching:
    """Once a fault is detected the monitor latches permanently."""

    def test_latched_after_failure(self) -> None:
        monitor = HealthMonitor()
        with pytest.raises(RuntimeHealthError):
            monitor.observe(b"\x00" * 200)
        assert monitor.failed is True
        with pytest.raises(RuntimeHealthError, match="latched"):
            monitor.observe(b"\x01")

    def test_latched_after_adaptive_failure(self) -> None:
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
        assert status.adaptive_proportion_windows == 2
        assert status.repetition_count_cutoff == 179
        assert status.adaptive_proportion_cutoff == 14
        assert status.repetition_count_max_run < status.repetition_count_cutoff

    def test_status_counts_max_run(self) -> None:
        monitor = HealthMonitor()
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
