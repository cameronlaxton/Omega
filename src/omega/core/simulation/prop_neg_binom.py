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
        try:
            mean = float(request.projection_mean)
        except (TypeError, ValueError) as exc:
            raise ValueError("projection_mean must be a finite non-negative number") from exc
        if not np.isfinite(mean) or mean < 0:
            raise ValueError("projection_mean must be a finite non-negative number")

        prior = request.prior_payload or {}
        k = prior.get("nb_dispersion_k")
        if k is None:
            raise ValueError("prior_payload.nb_dispersion_k is required for negative binomial props")
        try:
            k = float(k)
        except (TypeError, ValueError) as exc:
            raise ValueError("prior_payload.nb_dispersion_k must be a finite positive number") from exc
        if not np.isfinite(k) or k <= 0:
            raise ValueError("prior_payload.nb_dispersion_k must be a finite positive number")

        denom = k + mean
        if denom <= 0:
            raise ValueError("negative binomial dispersion and mean produce an invalid denominator")
        p = k / denom
        if not np.isfinite(p) or p <= 0 or p > 1:
            raise ValueError("negative binomial probability must be in (0, 1]")

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
