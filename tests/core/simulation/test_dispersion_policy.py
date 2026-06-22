import pytest
from pydantic import ValidationError

from omega.core.simulation.dispersion import DispersionPolicy


def test_dispersion_policy_defaults():
    policy = DispersionPolicy()
    assert policy.version == "v1"
    assert policy.variance_multiplier == 1.0
    assert policy.sport_family is None
    assert policy.applied_to == []


def test_dispersion_policy_validation():
    # Valid
    policy = DispersionPolicy(variance_multiplier=1.5, sport_family="basketball")
    assert policy.variance_multiplier == 1.5
    assert policy.sport_family == "basketball"

    # Too low
    with pytest.raises(ValidationError):
        DispersionPolicy(variance_multiplier=0.0)

    # Too high
    with pytest.raises(ValidationError):
        DispersionPolicy(variance_multiplier=10.0)
