"""omega.core.simulation.prop_neg_binom -- Negative Binomial sampler for over-dispersed prop targets."""

from typing import Any

import numpy as np

from omega.core.simulation.backends import PropSimulationInput


class NegBinomPropBackend:
    backend_name = "prop_neg_binom"
    component_version = "prop_nb_v1"

    def run(self, request: PropSimulationInput) -> dict[str, Any]:
        """Negative Binomial sampler for over-dispersed prop targets.

        Routed to for NFL yardage and longest-play markets via
        DEFAULT_PROP_BACKEND_BY_LEAGUE_STAT. Also routed for any prop whose
        request.prior_payload includes an explicit nb_dispersion_k.

        """
        rng = np.random.default_rng(request.seed)
        mean = float(request.projection_mean)

        prior = request.prior_payload or {}
        k = prior.get("nb_dispersion_k")
        if k is None:
            k = 1.0

        p = k / (k + mean) if (k + mean) > 0 else 1.0

        samples = rng.negative_binomial(k, p, size=request.n_iter)
        market_line = request.line

        over_hits = (samples > market_line).sum()
        under_hits = (samples < market_line).sum()
        push_hits = (np.abs(samples - market_line) < 0.5).sum()
        p10, p50, p90 = np.percentile(samples, [10, 50, 90])

        return {
            "over_prob": float(over_hits / request.n_iter),
            "under_prob": float(under_hits / request.n_iter),
            "push_prob": float(push_hits / request.n_iter),
            "mean": float(samples.mean()),
            "std": float(samples.std()),
            "p10": float(p10),
            "p50": float(p50),
            "p90": float(p90),
            "distribution_type": "negative_binomial",
            "distribution_params": {
                "mu": mean,
                "k": float(k),
                "p": float(p),
            },
            "samples": samples[:100].tolist(),
        }
