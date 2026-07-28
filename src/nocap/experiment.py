"""Stable identities, seeds, and manifests for reproducible experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from .scm_model import DirectedScm
    from .simulation import PairedDataArtifact, SimulationConfig


SEED_SCHEMA_VERSION = "1"
ARTIFACT_SCHEMA_VERSION = "1"


def _token(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def derive_seed(base_seed: int, *tokens: Any, schema_version: str = SEED_SCHEMA_VERSION) -> int:
    """Derive a stable uint32 seed from a root seed and typed tokens."""
    payload = "|".join([schema_version, str(int(base_seed)), *(_token(t) for t in tokens)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def canonical_condition_id(parameters: dict[str, Any]) -> str:
    """Return an order-independent identifier for a parameter condition."""
    encoded = _token(parameters)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def canonical_job_id(
    experiment_id: str,
    design_mode: str,
    scm_replicate_id: str,
    data_replicate_id: str,
    condition_id: str,
    intervention_suffix: str = "",
) -> str:
    """Identify one complete simulation job, not merely its seed pair."""
    return ":".join(
        map(
            str,
            (
                experiment_id,
                design_mode,
                scm_replicate_id,
                data_replicate_id,
                condition_id,
                intervention_suffix,
            ),
        )
    )


def artifact_id(manifest: dict[str, Any]) -> str:
    """Hash manifest identity and metadata into a content-independent artifact ID."""
    return hashlib.sha256(_token(manifest).encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class StageSeeds:
    """Deterministic seeds for each stage of experiment data generation."""

    scm_structure_seed: int
    scm_coefficient_seed: int
    data_latent_seed: int
    observation_seed: int
    missingness_score_seed: int
    sample_order_seed: int

    @classmethod
    def from_roots(cls, scm_seed: int, data_seed: int) -> StageSeeds:
        """Derive stage-specific seeds from SCM and data root seeds."""
        return cls(
            *(derive_seed(scm_seed, "scm", name) for name in ("structure", "coefficient")),
            *(
                derive_seed(data_seed, "data", name)
                for name in ("latent", "observation", "missingness_score", "sample_order")
            ),
        )


def experiment_conditions(
    n_samples: list[int],
    missing_edge_rates: list[float],
    missing_data_rates: list[float],
    mechanisms: list[str],
) -> list[dict[str, Any]]:
    """Build the canonical condition grid used by setup and execution scripts."""
    return [
        {
            "n_samples": n,
            "missing_edge_rate": edge,
            "missing_data_rate": data,
            "missing_data_mechanism": mechanism,
        }
        for n in n_samples
        for edge in missing_edge_rates
        for data in missing_data_rates
        for mechanism in mechanisms
    ]


def build_paired_artifact(
    graph: nx.DiGraph,
    nodes: list[str],
    *,
    max_samples: int,
    config: SimulationConfig,
    scm_seed: int,
    data_seed: int,
    beta_med: float,
    beta_log_sd: float,
    beta_p: float,
    beta_abs_max: float,
) -> tuple[DirectedScm, PairedDataArtifact]:
    """Build the shared base SCM and maximum-size paired data artifact."""
    from .scm_model import build_synthetic_scm
    from .simulation import generate_paired_data_artifact

    build = build_synthetic_scm(
        graph,
        nodes,
        missing_edge_rate=0.0,
        beta_med=beta_med,
        beta_log_sd=beta_log_sd,
        beta_p=beta_p,
        beta_abs_max=beta_abs_max,
        rng=np.random.default_rng(derive_seed(scm_seed, "scm", "structure")),
    )
    artifact = generate_paired_data_artifact(
        build.scm,
        config,
        max_samples=max_samples,
        scm_seed=scm_seed,
        data_seed=data_seed,
    )
    return build.scm, artifact
