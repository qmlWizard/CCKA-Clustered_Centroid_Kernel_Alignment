import pytest

from experiments.noise import (
    DEFAULT_NOISE_LEVELS,
    _parse_args,
    _validate_noise_levels,
)


def test_default_noise_sweep_has_four_unique_levels():
    assert _validate_noise_levels(DEFAULT_NOISE_LEVELS) == (
        0.01,
        0.03,
        0.05,
        0.1,
    )


def test_noise_type_defaults_to_both(monkeypatch):
    monkeypatch.setattr("sys.argv", ["noise"])
    assert _parse_args().noise_type == "both"


@pytest.mark.parametrize("noise_type", ["coherent", "depolarizing"])
def test_noise_type_can_be_selected(monkeypatch, noise_type):
    monkeypatch.setattr("sys.argv", ["noise", "--noise-type", noise_type])
    assert _parse_args().noise_type == noise_type


@pytest.mark.parametrize(
    "levels",
    [
        (0.01, 0.02, 0.03),
        (0.01, 0.02, 0.03, 0.04, 0.05),
        (0.01, 0.02, 0.03, 1.01),
        (0.01, 0.02, 0.03, 0.03),
    ],
)
def test_noise_sweep_rejects_invalid_levels(levels):
    with pytest.raises(ValueError):
        _validate_noise_levels(levels)
