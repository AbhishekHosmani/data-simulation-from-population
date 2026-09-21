import numpy as np
import pandas as pd
import pytest

from src.simulator import (
    _sample_wrapped_hour,
    _timestamp_for_hour,
    _positive_lognormal,
)


# ============================================================
# _sample_wrapped_hour
# ============================================================

def test_sample_wrapped_hour_returns_float():
    rng = np.random.default_rng(42)

    result = _sample_wrapped_hour(
        rng,
        mean=18.0,
        sd=0.5,
    )

    assert isinstance(result, float)


def test_sample_wrapped_hour_is_within_24_hours():
    rng = np.random.default_rng(42)

    # Generate many samples to make sure wrapping always works
    samples = [
        _sample_wrapped_hour(rng, mean=23.5, sd=3.0)
        for _ in range(1000)
    ]

    assert all(0.0 <= hour < 24.0 for hour in samples)


def test_sample_wrapped_hour_is_reproducible():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    result1 = _sample_wrapped_hour(rng1, mean=18.0, sd=0.5)
    result2 = _sample_wrapped_hour(rng2, mean=18.0, sd=0.5)

    assert result1 == result2


def test_sample_wrapped_hour_wraps_after_midnight():
    """
    A value above 24 hours should wrap back into the
    valid 0-24 hour range.
    """
    rng = np.random.default_rng(42)

    result = _sample_wrapped_hour(
        rng,
        mean=25.0,
        sd=0.0,
    )

    assert result == pytest.approx(1.0)


# ============================================================
# _timestamp_for_hour
# ============================================================

def test_timestamp_for_hour():
    day = pd.Timestamp("2026-01-01")

    result = _timestamp_for_hour(
        day,
        18.5,
    )

    assert result == pd.Timestamp("2026-01-01 18:30:00")


def test_timestamp_for_hour_midnight():
    day = pd.Timestamp("2026-01-01")

    result = _timestamp_for_hour(
        day,
        0.0,
    )

    assert result == pd.Timestamp("2026-01-01 00:00:00")


def test_timestamp_for_hour_fractional_minutes():
    day = pd.Timestamp("2026-01-01")

    result = _timestamp_for_hour(
        day,
        8.25,
    )

    assert result == pd.Timestamp("2026-01-01 08:15:00")


def test_timestamp_for_hour_ignores_existing_time():
    """
    The helper should normalize the supplied date before
    constructing the timestamp.
    """
    day = pd.Timestamp("2026-01-01 15:30:00")

    result = _timestamp_for_hour(
        day,
        7.0,
    )

    assert result == pd.Timestamp("2026-01-01 07:00:00")

