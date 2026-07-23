r"""csd_estimate.py — Synthetic observational data + per-edge path coefficient estimation.

This script:

1) Accepts an input :class:`networkx.DiGraph` (via GraphML or via a small built-in demo).
2) Generates synthetic UMI data

    The latent structural causal model is

        X = B.T @ X + eps,

    where X is latent log2-expression and B[u, v] is the structural coefficient
    (beta) for the edge u -> v.

    Observed UMI counts are generated from the latent expression using

        Y_hs ~ NegativeBinomial(mu_hs, alpha_h)

    with

        mu_hs = L_s * 2**X_hs

    and

        Var(Y_hs) = mu_hs + alpha_h * mu_hs**2.

    The beta coefficients are continuous, nonzero structural log fold changes.
    For a one-unit increase in latent regulator expression, beta is the direct
    change in target log2-expression and 2**beta is the corresponding fold
    change in expected molecular abundance.

3) For **every** directed edge in the *estimation graph* calls
   :func:`nocap.cyclic_single_door.estimate_path_coefficient_for_edge`.
4) Writes a CSV with per-edge estimation results and the corresponding
   ground-truth structural coefficient (beta) under the synthetic SCM.

Notes
-----
* The estimator itself performs σ-single-door identifiability checks using only
  the graph structure.
* For cyclic graphs, the synthetic SCM is defined by the linear equations
  ``X = B^T X + eps`` solved via ``(I - B^T)^{-1}``.
* Ground-truth coefficients are the structural edge coefficients (betas)
  used to generate the synthetic SCM.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from y0.algorithm.separation.sigma_extension import sigma_extension

from nocap.cyclic_single_door import (
    classify_edge,
    estimate_path_coefficient_for_edge,
    nx_digraph_to_y0,
)

COND_NUMBER_THRESHOLD = 1000  # https://en.wikipedia.org/wiki/Condition_number


def _parse_csv_list(s: str | None, *, cast_fn):
    if s is None:
        return None
    s = s.strip()
    if not s:
        return []
    return [cast_fn(x.strip()) for x in s.split(",") if x.strip()]


def _load_graph_from_graphml(graphml_path: str) -> nx.DiGraph:
    g = nx.read_graphml(graphml_path)
    # GraphML reader can return MultiDiGraph depending on attributes.
    if not isinstance(g, nx.DiGraph):
        g = nx.DiGraph(g)

    # Force node IDs to be strings for compatibility with y0 Variable(name).
    g2 = nx.DiGraph()
    g2.add_nodes_from([str(n) for n in g.nodes()])
    g2.add_edges_from([(str(u), str(v)) for u, v in g.edges()])
    return g2


def _safe_dropna_for_cols(df: pd.DataFrame, cols: list[str], *, min_rows: int):
    sub = df.loc[:, cols].dropna()
    if len(sub) < min_rows:
        return None
    return sub


@dataclass(frozen=True)
class SyntheticScmParams:
    """Parameters for generating a synthetic UMI-count SCM."""

    n_samples: int

    # Structural SCM beta parameters.
    beta_med: float
    beta_log_sd: float
    beta_p: float
    beta_abs_max: float

    # UMI observation-model parameters.
    umi_dispersion: float
    library_size_log_mean: float
    library_size_log_sd: float
    umi_pseudocount: float
    baseline_log_sd: float

    # Missingness parameters.
    missing_edge_rate: float
    missing_data_rate: float
    missing_data_mechanism: str
    self_mask_quantile: float
    self_mask_k: float
    self_mask_direction: str

    # Optional latent confounding.
    scc_confounding_strength: float

    seed: int


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _sample_betas(
    edges: Iterable[tuple[str, str]],
    *,
    rng: np.random.Generator,
    beta_med: float = 2,
    beta_log_sd: float = 0.5,
    beta_p: float = 0.5,
    beta_abs_max: float = 5,
) -> dict[tuple[str, str], float]:
    """Sample nonzero log fold changes for regulatory edges.

    Parameters
    ----------
    edges:
        Iterable of (source_gene, target_gene) edges.

    beta_med:
        Median absolute fold change for a one-unit increase in the
        normalized regulator expression.

        For example:
            beta_med=2.0 means the median activating effect is a 2-fold
            increase, corresponding to beta = log2(2).

    beta_log_sd:
        Standard deviation of log(abs(beta)). Larger values produce
        more heterogeneous effect sizes.

    beta_p:
        Probability that an edge is activating.

        beta_p=1.0 -> all beta values are positive
        beta_p=0.0 -> all beta values are negative
        beta_p=0.5 -> equal probability of activation and inhibition

    beta_abs_max:
        Maximum allowed absolute value of beta, on the log scale.

        For example:
            beta_abs_max=np.log2(4)
        limits effects to at most 4-fold per one-unit increase.

    rng:
        NumPy random number generator.

    Returns
    -------
    dict[tuple[str, str], float]
        Mapping from edge to a nonzero signed log fold change.

    Notes
    -----
    The sampled magnitude follows approximately

        log(abs(beta)) ~ Normal(
            log(log2(beta_med)),
            beta_log_sd**2
        )

    Therefore, the median absolute beta is approximately log2(beta_med),
    and the median absolute fold change is approximately beta_med.
    """
    if beta_med <= 1:
        raise ValueError("beta_med must be greater than 1.")

    if not 0 <= beta_p <= 1:
        raise ValueError("beta_p must be between 0 and 1.")

    if beta_abs_max <= 0:
        raise ValueError("beta_abs_max must be positive.")

    if beta_log_sd <= 0:
        raise ValueError("beta_log_sd must be positive.")

    edge_list = list(edges)

    if not edge_list:
        return {}

    # Median absolute beta corresponding to the requested fold change.
    #
    # If median fold change = 2:
    #     median(abs(beta)) = log2(2)
    median_abs_beta = np.log2(beta_med)

    if median_abs_beta >= beta_abs_max:
        raise ValueError(
            "beta_abs_max must be greater than log2(beta_med). "
            f"Got beta_abs_max={beta_abs_max:.3f}, "
            f"log2(beta_med)={median_abs_beta:.3f}."
        )

    n_edges = len(edge_list)

    # Sample positive magnitudes. Rejection sampling ensures that the
    # absolute values do not exceed beta_abs_max.
    magnitudes = np.empty(n_edges, dtype=float)
    remaining = np.arange(n_edges)

    log_median_abs_beta = np.log(median_abs_beta)

    while remaining.size > 0:
        proposed = rng.lognormal(
            mean=log_median_abs_beta,
            sigma=beta_log_sd,
            size=remaining.size,
        )

        accepted = proposed < beta_abs_max

        magnitudes[remaining[accepted]] = proposed[accepted]
        remaining = remaining[~accepted]

    # Sample signs:
    # +1 = activating edge
    # -1 = inhibitory edge
    signs = np.where(
        rng.random(n_edges) < beta_p,
        1.0,
        -1.0,
    )

    betas_array = signs * magnitudes

    return {edge: float(beta) for edge, beta in zip(edge_list, betas_array)}


def _is_invertible(A: np.ndarray) -> tuple[bool, float]:
    """Checks if a matrix is invertible (not singular).

    Pulled from https://stackoverflow.com/questions/13249108/efficient-pythonic-check-for-singular-matrix
    """
    cond = np.linalg.cond(A)
    return cond < (1 / (np.finfo(A.dtype).eps)), cond


def stabilize_cyclic_beta(beta_matrix: np.ndarray, target_rho: float = 0.8) -> np.ndarray:
    """Rescale B so its spectral radius is target_rho, guaranteeing (I - B^T)
    is well-conditioned and the fixed-point solution is stable.
    """
    eigs = np.linalg.eigvals(beta_matrix)
    rho = np.max(np.abs(eigs))
    if rho >= target_rho:
        beta_matrix = beta_matrix * (target_rho / rho)
    return beta_matrix


def sync_betas_from_matrix(B, betas_by_edge, idx):
    return {(u, v): float(B[idx[u], idx[v]]) for (u, v) in betas_by_edge}


def _build_beta_matrix(
    nodes: list[str],
    edges: Iterable[tuple[str, str]],
    beta_med: float,
    beta_log_sd: float,
    beta_p: float,
    beta_abs_max: float,
    rng: np.random.Generator,
    gen_limit: int = 100,
) -> tuple[np.ndarray, dict[tuple[str, str], float]]:
    """Return B where equation is X_v = sum_{u->v} beta[u->v] X_u + eps_v.

    Will generate a B until it is not singular or gen_limit is hit.
    """
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    B = np.zeros((n, n), dtype=float)
    betas_by_edge: dict[tuple[str, str], float] = {}

    for i in range(gen_limit):
        betas_by_edge = _sample_betas(
            edges,
            beta_med=beta_med,
            beta_log_sd=beta_log_sd,
            beta_p=beta_p,
            beta_abs_max=beta_abs_max,
            rng=rng,
        )

        for (u, v), b in betas_by_edge.items():
            B[idx[u], idx[v]] = b

        B = stabilize_cyclic_beta(B)  # guarantee spectral radius <0.8
        betas_by_edge = sync_betas_from_matrix(B, betas_by_edge, idx)

        # I - B^T is guaranteed to be invertible if p(B) < 1, but still good
        # to check just in case.
        is_invertible, cond = _is_invertible(np.eye(len(nodes)) - B.T)

        if is_invertible and cond < COND_NUMBER_THRESHOLD:
            print(
                f"Beta matrix found after {i + 1} iteration(s) and k(I - B^T) = {cond}.", flush=True
            )
            break

    else:
        raise Exception("Could not find B matrix that meets invertibility criteria.")

    return B, betas_by_edge


def _solve_linear_scm(
    nodes: list[str],
    beta_matrix: np.ndarray,
    eps: np.ndarray,
) -> np.ndarray:
    """Solve X = B^T X + eps where eps shape is (n_samples, n_nodes)."""
    # X_v - sum_u beta[u,v] X_u = eps_v.
    # With our beta_matrix[u,v] = beta[u->v], the system for each sample is:
    # (I - beta_matrix^T) X = eps.
    A = np.eye(len(nodes)) - beta_matrix.T
    # Solve A X^T = eps^T for X^T.
    try:
        X_T = np.linalg.solve(A, eps.T)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            "Synthetic SCM linear system is singular / ill-conditioned. "
            "Try smaller beta magnitudes."
        ) from e
    return X_T.T


def _generate_exogenous_noises(
    nodes: list[str],
    n_samples: int,
    *,
    scc_confounding_strength: float,
    rng: np.random.Generator,
    estimation_graph_for_scc: nx.DiGraph,
) -> np.ndarray:
    """Generate eps with optional SCC-level latent confounding."""
    eps = rng.normal(0.0, 1.0, size=(n_samples, len(nodes)))
    if scc_confounding_strength <= 0:
        return eps

    # Latent factor per non-trivial SCC: eps_i += s * L_s.
    strength = float(scc_confounding_strength)
    sccs = [
        scc for scc in nx.strongly_connected_components(estimation_graph_for_scc) if len(scc) > 1
    ]
    if not sccs:
        return eps

    node_idx = {n: i for i, n in enumerate(nodes)}
    for scc in sccs:
        L = rng.normal(0.0, 1.0, size=(n_samples, 1))
        for node in scc:
            eps[:, node_idx[node]] += strength * L[:, 0]

    return eps


def _sample_library_sizes(
    n_samples: int,
    *,
    log_mean: float,
    log_sd: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate positive per-sample library sizes.

    The library sizes represent total sequencing depth: the approximate
    number of UMIs available across all genes in a sample or cell.

    For example, with ``log_mean=np.log(10_000)``, the median sample
    library size is approximately 10,000 UMIs. Gene-level expected counts
    are determined by the gene-specific baseline abundance proportions
    ``q_h``:

        mu_hs = L_s * q_h * 2**X_hs

    where ``L_s`` is the sample library size, ``q_h`` is the baseline
    proportion of UMIs assigned to gene ``h``, and ``X_hs`` is the
    latent regulatory expression effect.

    Thus, a library size of 10,000 does not imply 10,000 UMIs per gene.
    If ``q_h=0.001`` and ``X_hs=0``, the expected baseline count for
    gene ``h`` is approximately 10 UMIs.
    """
    if log_sd < 0:
        raise ValueError("library_size_log_sd must be non-negative.")

    return rng.lognormal(
        mean=log_mean,
        sigma=log_sd,
        size=n_samples,
    )


def _sample_baseline_abundances(
    n_genes: int,
    *,
    log_sd: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample gene-specific baseline abundance proportions.

    Returns q such that:

        q[h] >= 0
        sum(q) = 1

    Therefore, q[h] is the baseline fraction of the total library
    assigned to gene h when latent expression X[h] is zero.
    """
    if n_genes <= 0:
        raise ValueError("n_genes must be positive.")

    if log_sd < 0:
        raise ValueError("baseline_log_sd must be non-negative.")

    raw = rng.lognormal(
        mean=0.0,
        sigma=log_sd,
        size=n_genes,
    )

    return raw / raw.sum()


def _sample_umi_counts(
    latent_log_expression: np.ndarray,
    library_sizes: np.ndarray,
    baseline_abundances: np.ndarray,
    *,
    dispersion: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate UMI counts from latent expression.

    The model is

        mu_hs = L_s * q_h * 2**X_hs
        Y_hs ~ NB(mu_hs, alpha)

    where q_h is the gene-specific baseline abundance proportion.
    """
    if dispersion <= 0:
        raise ValueError("umi_dispersion must be positive.")

    if latent_log_expression.ndim != 2:
        raise ValueError("latent_log_expression must have shape (n_samples, n_genes).")

    n_samples, n_genes = latent_log_expression.shape

    if library_sizes.shape != (n_samples,):
        raise ValueError("library_sizes must have shape (n_samples,).")

    if baseline_abundances.shape != (n_genes,):
        raise ValueError("baseline_abundances must have shape (n_genes,).")

    if not np.isclose(baseline_abundances.sum(), 1.0):
        raise ValueError("baseline_abundances must sum to one.")

    latent_log_expression = np.clip(
        latent_log_expression,
        a_min=-30.0,
        a_max=30.0,
    )

    # Expected UMI count:
    #
    # mu_hs = L_s * q_h * 2**X_hs
    mu = library_sizes[:, None] * baseline_abundances[None, :] * np.exp2(latent_log_expression)

    # Convert mean/dispersion parameterization to NumPy's
    # negative_binomial(n, p) parameterization:
    #
    # E[Y]   = mu
    # Var[Y] = mu + dispersion * mu**2
    n = 1.0 / dispersion
    p = n / (n + mu)

    return rng.negative_binomial(
        n=n,
        p=p,
    )


def _counts_to_log_expression(
    counts: np.ndarray,
    library_sizes: np.ndarray,
    *,
    pseudocount: float,
    target_library_size: float,
) -> np.ndarray:
    """Convert UMI counts to normalized log2-expression.

    Counts are rescaled to target_library_size before applying log1p.
    """
    if pseudocount <= 0:
        raise ValueError("umi_pseudocount must be positive.")

    if target_library_size <= 0:
        raise ValueError("target_library_size must be positive.")

    normalized_counts = counts / library_sizes[:, None] * target_library_size

    return np.log2(normalized_counts + pseudocount)


def generate_synthetic_observational_data(
    estimation_graph: nx.DiGraph,
    scm_nodes: list[str],
    params: SyntheticScmParams,
):
    """Generate synthetic latent-SCM data and observed UMI expression.

    The latent SCM is

        X = B.T @ X + eps,

    where X is latent log-expression and B[u, v] is the structural beta
    for u -> v.

    UMI counts are then generated using

        mu_hs = L_s * q_h * 2**X_hs
        Y_hs ~ NB(mu_hs, alpha_h).

    Returns
    -------
    data:
        Library-size-normalized log-UMI expression. This is the noisy
        observed expression supplied to the estimator.

    scm_graph_true:
        True SCM graph, including any added edges.

    betas_by_edge_true:
        Structural beta coefficients used by the latent SCM.
    """
    rng = _rng(params.seed)

    orig_edges = list(estimation_graph.edges())
    orig_edge_set = set(orig_edges)
    edges_true = list(orig_edges)

    # "missing_edge_rate" means: how many edges are present in the *true* SCM,
    # but missing from the provided/constructed estimation graph.
    missing_edge_rate = np.clip(
        float(params.missing_edge_rate),
        0.0,
        1.0,
    )

    # Choose the number of extra edges to add (expected: rate * |E_orig|).
    n_to_add = int(
        rng.binomial(
            len(orig_edges),
            missing_edge_rate,
        )
    )

    candidates = [
        (u, v) for u in scm_nodes for v in scm_nodes if u != v and (u, v) not in orig_edge_set
    ]

    if candidates and n_to_add > 0:
        n_to_add = min(n_to_add, len(candidates))

        selected = rng.choice(
            len(candidates),
            size=n_to_add,
            replace=False,
        )

        edges_true.extend(candidates[int(i)] for i in selected)

    scm_graph_true = nx.DiGraph()
    scm_graph_true.add_nodes_from(scm_nodes)
    scm_graph_true.add_edges_from(edges_true)

    # Sample the structural coefficients used by the latent SCM.
    beta_matrix, betas_by_edge_true = _build_beta_matrix(
        scm_nodes,
        edges_true,
        beta_med=params.beta_med,
        beta_log_sd=params.beta_log_sd,
        beta_p=params.beta_p,
        beta_abs_max=params.beta_abs_max,
        rng=rng,
    )

    # Generate exogenous noise for the latent log-expression SCM.
    eps = _generate_exogenous_noises(
        scm_nodes,
        params.n_samples,
        scc_confounding_strength=params.scc_confounding_strength,
        rng=rng,
        estimation_graph_for_scc=estimation_graph,
    )

    # Generate latent log-expression:
    #
    #     X = B.T @ X + eps
    #
    latent_log_expression = _solve_linear_scm(
        scm_nodes,
        beta_matrix=beta_matrix,
        eps=eps,
    )

    # Generate sample-specific sequencing depths.
    library_sizes = _sample_library_sizes(
        params.n_samples,
        log_mean=params.library_size_log_mean,
        log_sd=params.library_size_log_sd,
        rng=rng,
    )

    # Generate gene-specific baseline abundance proportions.
    baseline_abundances = _sample_baseline_abundances(
        len(scm_nodes),
        log_sd=params.baseline_log_sd,
        rng=rng,
    )

    # Generate observed integer UMI counts.
    umi_counts = _sample_umi_counts(
        latent_log_expression,
        library_sizes,
        baseline_abundances,
        dispersion=params.umi_dispersion,
        rng=rng,
    )

    # Convert counts to normalized log-expression for estimation.
    data = pd.DataFrame(
        _counts_to_log_expression(
            umi_counts,
            library_sizes,
            pseudocount=params.umi_pseudocount,
            target_library_size=np.exp(params.library_size_log_mean),
        ),
        columns=scm_nodes,
    )

    # Apply measurement errors: instrument error is low-expression
    # self-masking, while biological error is independent absence at the
    # time of measurement.
    if params.missing_data_rate > 0:
        missing_rate = np.clip(
            float(params.missing_data_rate),
            0.0,
            1.0,
        )

        mechanism = params.missing_data_mechanism.lower()

        for j, col in enumerate(scm_nodes):
            if mechanism in {"biological_error", "biological", "bio"}:
                mask = rng.random(params.n_samples) < missing_rate

            elif mechanism in {
                "instrument_error",
                "instrument",
                "instrument-self-masking",
            }:
                # Instrument error depends on the underlying expression:
                # low expression is less likely to be detected.
                x = latent_log_expression[:, j]

                threshold = float(
                    np.quantile(
                        x,
                        float(params.self_mask_quantile),
                    )
                )

                k = float(params.self_mask_k)
                direction = params.self_mask_direction.lower()

                if direction == "low":
                    raw_probability = 1.0 / (1.0 + np.exp(-k * (threshold - x)))
                elif direction == "high":
                    raw_probability = 1.0 / (1.0 + np.exp(-k * (x - threshold)))
                else:
                    raise ValueError("--self-mask-direction must be one of {low,high}.")

                raw_mean = float(np.mean(raw_probability))

                if raw_mean <= 0:
                    probability = np.full(
                        params.n_samples,
                        missing_rate,
                    )
                else:
                    probability = missing_rate * raw_probability / raw_mean
                    probability = np.clip(
                        probability,
                        0.0,
                        1.0,
                    )

                mask = rng.random(params.n_samples) < probability

            else:
                raise ValueError(
                    f"Unknown missing-data mechanism: {params.missing_data_mechanism!r}"
                )

            data.loc[mask, col] = (
                0.0  # missingness is not "missing", but no read on gene expression value
            )

    return data, scm_graph_true, betas_by_edge_true


def _default_nodes_from_graph(g: nx.DiGraph) -> list[str]:
    # Stable node ordering for reproducible matrices.
    return sorted([str(n) for n in g.nodes()])


def _iter_experiment_grid(args) -> list[dict]:
    n_samples_list = _parse_csv_list(args.n_samples_list, cast_fn=int)
    if n_samples_list is None:
        n_samples_list = [int(args.n_samples)]
    missing_edge_rates = _parse_csv_list(args.missing_edge_rate_list, cast_fn=float)
    if missing_edge_rates is None:
        missing_edge_rates = [float(args.missing_edge_rate)]
    missing_data_rates = _parse_csv_list(args.missing_data_rate_list, cast_fn=float)
    if missing_data_rates is None:
        missing_data_rates = [float(args.missing_data_rate)]

    return [
        {
            "n_samples": n,
            "missing_edge_rate": r_edge,
            "missing_data_rate": r_data,
        }
        for n in n_samples_list
        for r_edge in missing_edge_rates
        for r_data in missing_data_rates
    ]


def _load_adjustment_sets_csv(
    path: str,
) -> dict[tuple[str, str], tuple[frozenset[str], bool, str]]:
    """Load a precomputed adjustment-set table.

    Expected CSV schema (like notebooks/.../csd_identifiable_edges.csv):
    - cause
    - effect
    - adjustment_set ("a|b|c" or empty string for empty set)
    - same_scc (boolean-ish)

    Optional columns:
    - status: if present, should be one of {identifiable, unidentifiable} (case-insensitive).
      If absent, every row is treated as identifiable (backwards compatible).
    """
    import pandas as _pd

    df = _pd.read_csv(path)
    required = {"cause", "effect", "adjustment_set"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Adjustments CSV missing columns: {sorted(missing)}")

    mapping: dict[tuple[str, str], tuple[frozenset[str], bool, str]] = {}
    for row in df.to_dict(orient="records"):
        cause = str(row["cause"])
        effect = str(row["effect"])
        adj_raw = row.get("adjustment_set")
        if adj_raw is None or (isinstance(adj_raw, float) and _pd.isna(adj_raw)):
            adj_parts: list[str] = []
        else:
            adj_str = str(adj_raw).strip()
            adj_parts = [p for p in adj_str.split("|") if p]
        adj_set = frozenset(adj_parts)

        # If status is not provided we assume the CSV is a table of identifiable edges.
        status = "identifiable"
        if "status" in df.columns:
            status_raw = row.get("status")
            if status_raw is None or (isinstance(status_raw, float) and _pd.isna(status_raw)):
                status = "unidentifiable"
            else:
                status = str(status_raw).strip().lower()
                if status not in {"identifiable", "unidentifiable"}:
                    # Be conservative: unknown statuses are treated as unidentifiable.
                    status = "unidentifiable"

        same_scc = False
        if "same_scc" in df.columns:
            same_scc = bool(row.get("same_scc"))
        mapping[(cause, effect)] = (adj_set, same_scc, status)

    return mapping


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--graphml",
        type=str,
        default=None,
        help="Path to a GraphML file describing the *estimation* nx.DiGraph.",
    )
    p.add_argument(
        "--demo",
        type=str,
        default="cycle",
        choices=[
            "chain",
            "confounded_chain",
            "cycle",
            "two_cycles_disconnected",
        ],
        help="Built-in demo graph to use when --graphml is omitted.",
    )
    p.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Write per-edge estimation results to this CSV.",
    )

    p.add_argument(
        "--adjustments-csv",
        type=str,
        default=None,
        help=(
            "Optional CSV mapping identifiable edges to precomputed adjustment sets "
            "(like notebooks/.../csd_identifiable_edges.csv)."
        ),
    )
    p.add_argument(
        "--assume-adjustments-csv-complete",
        action="store_true",
        help=(
            "If set, edges missing from --adjustments-csv are treated as unidentifiable "
            "(no oracle re-classification). Default: fall back to classifying edge via σ-single-door oracle."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed for beta/noise generation.",
    )
    p.add_argument(
        "--save-config", type=str, default=None, help="File path to save arguments as a JSON file."
    )

    # Experiment grid.
    p.add_argument("--n-samples", type=int, default=300)
    p.add_argument(
        "--n-samples-list",
        type=str,
        default=None,
        help="Comma-separated list of sample sizes (overrides --n-samples).",
    )
    p.add_argument("--missing-edge-rate", type=float, default=0.0)
    p.add_argument(
        "--missing-edge-rate-list",
        type=str,
        default=None,
        help="Comma-separated list of missing edge rates.",
    )
    p.add_argument("--missing-data-rate", type=float, default=0.0)
    p.add_argument(
        "--missing-data-rate-list",
        type=str,
        default=None,
        help="Comma-separated list of missing data rates.",
    )
    p.add_argument(
        "--missing-data-mechanism",
        type=str,
        default="biological_error",
        choices=["instrument_error", "biological_error"],
        help=(
            "Measurement-error mechanism: instrument_error models low-expression "
            "self-masking, while biological_error models genes not being expressed "
            "at measurement time."
        ),
    )
    p.add_argument(
        "--self-mask-quantile",
        type=float,
        default=0.25,
        help="Quantile used as threshold for instrument-error self-masking.",
    )
    p.add_argument(
        "--self-mask-k",
        type=float,
        default=8.0,
        help="Slope for self-masking logistic missingness.",
    )
    p.add_argument(
        "--self-mask-direction",
        type=str,
        default="low",
        choices=["low", "high"],
        help="If low: mask more when value is small. If high: mask more when value is large.",
    )

    # Linear SCM.
    p.add_argument(
        "--beta-med",
        type=float,
        default=2.0,
        help=(
            "Median absolute fold change for a one-unit increase in "
            "latent regulator log-expression. The median absolute beta "
            "is log2(beta_med)."
        ),
    )
    p.add_argument(
        "--beta-log-sd",
        type=float,
        default=0.5,
        help=("Standard deviation of log absolute structural beta values."),
    )
    p.add_argument(
        "--beta-p",
        type=float,
        default=0.5,
        help=(
            "Probability that an edge is activating. Positive beta "
            "means activation; negative beta means inhibition."
        ),
    )
    p.add_argument(
        "--beta-abs-max",
        type=float,
        default=5.0,
        help=(
            "Maximum absolute structural beta on the log2-expression "
            "scale. The corresponding maximum fold change is "
            "2**beta_abs_max."
        ),
    )
    p.add_argument(
        "--scc-confounding-strength",
        type=float,
        default=0.0,
        help="If >0, add SCC-level latent noise factor to exogenous noises (simulated confounding).",
    )

    # UMI model parameters
    p.add_argument(
        "--umi-dispersion",
        type=float,
        default=0.1,
        help=("Negative-binomial dispersion alpha. Variance is mu + alpha * mu**2."),
    )
    p.add_argument(
        "--library-size-log-mean",
        type=float,
        default=float(np.log(10_000)),
        help=(
            "Mean of log library size. With the default, the median "
            "library size is approximately 10,000 UMIs."
        ),
    )
    p.add_argument(
        "--library-size-log-sd",
        type=float,
        default=0.4,
        help="Standard deviation of log library size.",
    )
    p.add_argument(
        "--umi-pseudocount",
        type=float,
        default=1.0,
        help=("Pseudocount used when converting normalized UMI counts to log-expression."),
    )
    p.add_argument(
        "--baseline-log-sd",
        type=float,
        default=2.0,
        help=(
            "Log-scale variability of gene-specific baseline "
            "abundances. Larger values create a wider expression "
            "dynamic range."
        ),
    )

    # Regression/data cleaning.
    p.add_argument(
        "--min-rows-after-dropna",
        type=int,
        default=30,
        help="Minimum rows required after dropping NaNs for an edge regression.",
    )

    args = p.parse_args()

    # --- Graph ---
    if args.graphml is not None:
        graph = _load_graph_from_graphml(args.graphml)
    else:
        graph = nx.DiGraph()
        if args.demo == "chain":
            graph.add_edges_from([("Z", "X"), ("X", "Y")])
        elif args.demo == "confounded_chain":
            graph.add_edges_from([("Z", "X"), ("X", "Y"), ("Z", "Y")])
        elif args.demo == "cycle":
            graph.add_edges_from([("X", "Y"), ("Y", "X"), ("Y", "Z"), ("Z", "Y")])
        elif args.demo == "two_cycles_disconnected":
            graph.add_edges_from([("A", "B"), ("B", "A"), ("C", "D"), ("D", "C")])
        else:
            raise ValueError(f"Unknown demo {args.demo!r}")

    nodes = _default_nodes_from_graph(graph)
    if not graph.edges():
        raise SystemExit("Input graph has no edges; nothing to estimate.")

    # Precompute y0 and sigma-extension once.
    g_y0 = nx_digraph_to_y0(graph)
    g_sigma = sigma_extension(g_y0)

    adjustments_map: dict[tuple[str, str], tuple[frozenset[str], bool, str]] = {}
    if args.adjustments_csv is not None:
        adjustments_map = _load_adjustment_sets_csv(args.adjustments_csv)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.save_config is not None:
        with open(args.save_config, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=4)

    # --- Header ---
    fieldnames = [
        "trial",
        "n_samples",
        "missing_edge_rate",
        "missing_data_rate",
        "missing_data_mechanism",
        "seed",
        "cause",
        "effect",
        "same_scc",
        "status",
        "adjustment_set",
        "n_rows_used",
        "estimated_path_coefficient",
        "stderr",
        "residual_variance",
        "t_value",
        "ground_truth_beta",
        "scm_true_missing_edges_count",
        "error",
    ]

    # Prepare to write rows.
    write_header = not out_csv.exists() or out_csv.stat().st_size == 0
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        # --- Experiment grid ---
        grid = _iter_experiment_grid(args)
        for trial_idx, cell in enumerate(grid):
            n_samples = int(cell["n_samples"])
            missing_edge_rate = float(cell["missing_edge_rate"])
            missing_data_rate = float(cell["missing_data_rate"])

            params = SyntheticScmParams(
                n_samples=n_samples,
                # Structural SCM parameters.
                beta_med=float(args.beta_med),
                beta_log_sd=float(args.beta_log_sd),
                beta_abs_max=float(args.beta_abs_max),
                beta_p=float(args.beta_p),
                # UMI observation-model parameters.
                umi_dispersion=float(args.umi_dispersion),
                library_size_log_mean=float(args.library_size_log_mean),
                library_size_log_sd=float(args.library_size_log_sd),
                umi_pseudocount=float(args.umi_pseudocount),
                baseline_log_sd=float(args.baseline_log_sd),
                # Missingness parameters.
                missing_edge_rate=missing_edge_rate,
                missing_data_rate=missing_data_rate,
                missing_data_mechanism=args.missing_data_mechanism,
                self_mask_quantile=float(args.self_mask_quantile),
                self_mask_k=float(args.self_mask_k),
                self_mask_direction=args.self_mask_direction,
                # Confounding and reproducibility.
                scc_confounding_strength=float(args.scc_confounding_strength),
                seed=int(args.seed),
            )

            data, scm_graph_true, betas_true = generate_synthetic_observational_data(
                graph, nodes, params
            )

            n_true_edges = scm_graph_true.number_of_edges()
            n_orig_edges = graph.number_of_edges()
            n_missing_edges = n_true_edges - n_orig_edges

            for cause, effect in graph.edges():
                same = (
                    cause in graph
                    and effect in graph
                    and (
                        nx.has_path(graph, cause, effect) and nx.has_path(graph, effect, cause)
                        if cause != effect
                        else True
                    )
                )

                # Determine adjustment set first so we can safely drop NaNs.
                ce_key = (str(cause), str(effect))
                if ce_key in adjustments_map:
                    adj_set, same_scc_pre, status_from_ident = adjustments_map[ce_key]
                    adj_info_same_scc = same_scc_pre
                else:
                    if args.assume_adjustments_csv_complete:
                        continue
                    else:
                        adj_info = classify_edge(
                            graph,
                            cause=str(cause),
                            effect=str(effect),
                            precomputed_extension=g_sigma,
                            precomputed_y0=g_y0,
                        )
                        adj_set = adj_info["adjustment_set"]
                        status_from_ident = adj_info["status"]
                        adj_info_same_scc = bool(adj_info.get("same_scc", same))

                cols = [str(cause), str(effect)]
                if adj_set is not None:
                    cols.extend([str(z) for z in sorted(adj_set)])

                cleaned = _safe_dropna_for_cols(
                    data,
                    cols,
                    min_rows=int(args.min_rows_after_dropna),
                )

                row = {
                    "trial": trial_idx,
                    "n_samples": params.n_samples,
                    "missing_edge_rate": params.missing_edge_rate,
                    "missing_data_rate": params.missing_data_rate,
                    "missing_data_mechanism": params.missing_data_mechanism,
                    "seed": params.seed,
                    "cause": cause,
                    "effect": effect,
                    "same_scc": bool(adj_info_same_scc),
                    "status": status_from_ident,
                    "adjustment_set": (
                        "|".join(sorted(adj_set)) if adj_set is not None else "null"
                    ),
                    "n_rows_used": len(cleaned) if cleaned is not None else 0,
                    "estimated_path_coefficient": "",
                    "stderr": "",
                    "residual_variance": "",
                    "t_value": "",
                    "ground_truth_beta": float(betas_true.get((str(cause), str(effect)), 0.0)),
                    "scm_true_missing_edges_count": n_missing_edges,
                    "error": "",
                }

                if cleaned is None:
                    row["status"] = (
                        "insufficient_data"
                        if status_from_ident == "identifiable"
                        else status_from_ident
                    )
                    writer.writerow(row)
                    continue

                # Only run the regression estimator when the edge is identifiable.
                # Unidentifiable edges would otherwise waste time fitting a model
                # that we already know cannot be used for identification.
                if status_from_ident != "identifiable":
                    row["status"] = "unidentifiable"
                    writer.writerow(row)
                    continue

                # Always call the estimator for every edge.
                try:
                    est = estimate_path_coefficient_for_edge(
                        graph,
                        str(cause),
                        str(effect),
                        cleaned,
                        adj_set=adj_set,
                        precomputed_extension=g_sigma,
                        precomputed_y0=g_y0,
                    )
                    if est is None:
                        row["status"] = "unidentifiable"
                    else:
                        path_coef, stderr, residual_var, t_val = est
                        row["estimated_path_coefficient"] = float(path_coef)
                        row["stderr"] = float(stderr)
                        row["residual_variance"] = float(residual_var)
                        row["t_value"] = float(t_val)
                        row["status"] = "identifiable"
                except Exception as exc:
                    row["status"] = "estimation_error"
                    row["error"] = repr(exc)

                writer.writerow(row)

    print(f"Wrote: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
