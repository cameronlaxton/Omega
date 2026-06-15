"""
Simulation Dispersion Policy configuration.

Provides an auditable schema for controlled simulation-dispersion improvements.
This allows the engine to decouple noise-scaling logic from post-hoc calibration,
resolving ECE issues at the source.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class DispersionPolicy(BaseModel):
    """
    Versioned dispersion policy defining how simulation noise is scaled.
    
    This policy is passed to simulation backends via the simulation inputs.
    Depending on the backend type (exact vs MC), it modifies specific parameters:
    - For exact Poisson models, it may scale expected lambdas.
    - For exact Markov chains, it scales pressure coefficients.
    - For Monte Carlo models, it scales standard deviation or negative binomial k.
    """
    version: str = "v1"
    
    # 1.0 means no modification to underlying dispersion.
    # Below 1.0 reduces variance (sharper distribution).
    # Above 1.0 increases variance (wider distribution).
    variance_multiplier: float = Field(default=1.0, ge=0.1, le=5.0)
    
    # Optional sport family tag to enable logging/reporting grouping.
    sport_family: Optional[str] = None
    
    # Audit trail of which components this policy actually modified.
    # Example: ["home_goals_lambda", "away_goals_lambda"] or ["nb_dispersion_k"]
    applied_to: List[str] = Field(default_factory=list)
