r"""csd_estimate.py — Synthetic observational data + per-edge path coefficient estimation.

This script:

1) Accepts an input :class:`networkx.DiGraph` (via GraphML or via a small built-in demo).
2) Generates linear-Gaussian observational data from that graph (with knobs for
   missing edges and missing data / MNAR self-masking).
3) For **every** directed edge in the *estimation graph* calls
   :func:`nocap.cyclic_single_door.estimate_path_coefficient_for_edge`.
4) Writes a CSV with per-edge estimation results and the corresponding
   ground-truth structural coefficient (beta) under the synthetic SCM.

Notes
-----
* The estimator itself performs σ-single-door identifiability checks using only
  the graph structure.
* Missing data is handled by dropping rows with NaNs in the regression
  variables for each edge.
* For cyclic graphs, the synthetic SCM is defined by the linear equations
  ``X = B^T X + eps`` solved via ``(I - B^T)^{-1}``.
* Ground-truth coefficients are the structural edge coefficients (betas)
  used to generate the synthetic SCM.
"""

from __future__ import annotations

import argparse
import csv
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
    """Parameters for generating a synthetic SCM."""

    n_samples: int
    beta_mean: float
    beta_std: float
    beta_abs_max: float
    missing_edge_rate: float
    missing_edge_seed: int
    missing_data_rate: float
    missing_data_mechanism: str
    self_mask_quantile: float
    self_mask_k: float
    self_mask_direction: str
    scc_confounding_strength: float
    seed: int


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _sample_betas(
    edges: Iterable[tuple[str, str]],
    *,
    beta_mean: float,
    beta_std: float,
    beta_abs_max: float,
    rng: np.random.Generator,
) -> dict[tuple[str, str], float]:
    betas: dict[tuple[str, str], float] = {}
    for e in edges:
        if beta_std == 0:
            b = beta_mean
        else:
            # Truncated normal via rejection.
            for _ in range(10_000):
                b = float(rng.normal(beta_mean, beta_std))
                if abs(b) <= beta_abs_max:
                    break
            else:
                raise RuntimeError("Failed to sample truncated betas")
        betas[e] = b
    return betas


def _is_invertible(A: np.ndarray) -> bool:
    """Checks if a matrix is invertible (not singular).

    Pulled from https://stackoverflow.com/questions/13249108/efficient-pythonic-check-for-singular-matrix
    """
    return np.linalg.cond(A) < 1 / (np.finfo(A.dtype).eps)


def _build_beta_matrix(
    nodes: list[str],
    edges: Iterable[tuple[str, str]],
    beta_mean: float,
    beta_std: float,
    beta_abs_max: float,
    rng: np.random.Generator,
    gen_limit: int = 10000,
) -> tuple[np.ndarray, dict[tuple[str, str], float]]:
    """Return B where equation is X_v = sum_{u->v} beta[u->v] X_u + eps_v.

    Will generate a B until it is not singular or gen_limit is hit.
    """
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    B = np.zeros((n, n), dtype=float)
    betas_by_edge: dict[tuple[str, str], float] = {}

    for _ in range(gen_limit):
        betas_by_edge = _sample_betas(
            edges,
            beta_mean=beta_mean,
            beta_std=beta_std,
            beta_abs_max=beta_abs_max,
            rng=rng,
        )

        for (u, v), b in betas_by_edge.items():
            B[idx[u], idx[v]] = b

        if _is_invertible(np.eye(len(nodes)) - B.T):
            break

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


def generate_synthetic_observational_data(
    estimation_graph: nx.DiGraph,
    scm_nodes: list[str],
    params: SyntheticScmParams,
):
    """Generate a linear-Gaussian observational dataset and metadata.

    Returns
    -------
    data : pd.DataFrame
    scm_graph_true : nx.DiGraph
    betas_by_edge_true : dict[(u,v)] -> beta
    """
    # "missing_edge_rate" means: how many edges are present in the *true* SCM,
    # but missing from the provided/constructed estimation graph.
    rng = _rng(params.seed)

    orig_edges = list(estimation_graph.edges())
    orig_edge_set = set(orig_edges)
    edges_true = list(orig_edges)

    rate = float(params.missing_edge_rate)
    rate = min(max(rate, 0.0), 1.0)

    # Choose the number of extra edges to add (expected: rate * |E_orig|).
    n_to_add = int(rng.binomial(len(orig_edges), rate))

    if n_to_add <= 0:
        scm_graph_true = nx.DiGraph()
        scm_graph_true.add_nodes_from(scm_nodes)
        scm_graph_true.add_edges_from(edges_true)
    else:
        # Candidate edges are all directed pairs of distinct nodes that are not
        # already present in the estimation graph.
        candidates: list[tuple[str, str]] = []
        for u in scm_nodes:
            for v in scm_nodes:
                if u == v:
                    continue
                if (u, v) in orig_edge_set:
                    continue
                candidates.append((u, v))

        if not candidates:
            n_to_add = 0
            scm_graph_true = nx.DiGraph()
            scm_graph_true.add_nodes_from(scm_nodes)
            scm_graph_true.add_edges_from(edges_true)
        else:
            n_to_add = min(n_to_add, len(candidates))
            added_edges = rng.choice(len(candidates), size=n_to_add, replace=False)
            edges_true.extend(candidates[int(i)] for i in added_edges)

            scm_graph_true = nx.DiGraph()
            scm_graph_true.add_nodes_from(scm_nodes)
            scm_graph_true.add_edges_from(edges_true)

    beta_matrix, betas_by_edge_true = _build_beta_matrix(
        scm_nodes,
        edges_true,
        beta_mean=params.beta_mean,
        beta_std=params.beta_std,
        beta_abs_max=params.beta_abs_max,
        rng=rng,
    )
    eps = _generate_exogenous_noises(
        scm_nodes,
        params.n_samples,
        scc_confounding_strength=params.scc_confounding_strength,
        rng=rng,
        estimation_graph_for_scc=estimation_graph,
    )

    X = _solve_linear_scm(scm_nodes, beta_matrix=beta_matrix, eps=eps)
    data = pd.DataFrame(X, columns=scm_nodes)

    # Apply missing data (entrywise missingness, MNAR or MCAR).
    if params.missing_data_rate > 0:
        rate = float(params.missing_data_rate)
        rate = min(max(rate, 0.0), 1.0)

        mech = params.missing_data_mechanism.lower()
        for col in scm_nodes:
            y = data[col].to_numpy()
            if mech in ("mcar", "mc ar", "mc"):
                mask = rng.random(params.n_samples) < rate
            elif mech in ("mnar_self_mask", "self_mask", "self-masking", "mnar"):
                # Use log(|y| + tiny) so the mechanism works with negative values.
                tiny = 1e-6
                y_log = np.log(np.abs(y) + tiny)
                q = float(params.self_mask_quantile)
                thr = float(np.quantile(y_log, q))
                k = float(params.self_mask_k)

                direction = params.self_mask_direction.lower()
                if direction == "low":
                    # Missing more likely when y is small (y_log below threshold).
                    raw = 1.0 / (1.0 + np.exp(-k * (thr - y_log)))
                elif direction == "high":
                    # Missing more likely when y is large (y_log above threshold).
                    raw = 1.0 / (1.0 + np.exp(-k * (y_log - thr)))
                else:
                    raise ValueError("--self-mask-direction must be one of {low,high}.")

                # Rescale raw probabilities so the *expected* missingness is ~ rate.
                raw_mean = float(np.mean(raw))
                if raw_mean <= 0:
                    mask = rng.random(params.n_samples) < rate
                else:
                    prob = rate * raw / raw_mean
                    prob = np.clip(prob, 0.0, 1.0)
                    mask = rng.random(params.n_samples) < prob
            else:
                raise ValueError(
                    f"Unknown missing-data mechanism: {params.missing_data_mechanism!r}"
                )

            data.loc[mask, col] = np.nan

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
        default="MCAR",
        choices=["MCAR", "MNAR_self_mask"],
        help="Mechanism for missingness (MCAR or MNAR self-masking).",
    )
    p.add_argument(
        "--self-mask-quantile",
        type=float,
        default=0.25,
        help="Quantile used as threshold for self-masking (MNAR).",
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
        "--beta-mean",
        type=float,
        default=1.0,
        help="Mean for edge coefficients (used if beta-std != 0).",
    )
    p.add_argument(
        "--beta-std",
        type=float,
        default=0.2,
        help="Std for edge coefficients (truncated). Set to 0 for fixed betas.",
    )
    p.add_argument(
        "--beta-abs-max",
        type=float,
        default=0.9,
        help="Truncate sampled betas to |beta| <= this.",
    )
    p.add_argument(
        "--scc-confounding-strength",
        type=float,
        default=0.0,
        help="If >0, add SCC-level latent noise factor to exogenous noises (simulated confounding).",
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
                beta_mean=float(args.beta_mean),
                beta_std=float(args.beta_std),
                beta_abs_max=float(args.beta_abs_max),
                missing_edge_rate=missing_edge_rate,
                missing_data_rate=missing_data_rate,
                missing_data_mechanism=args.missing_data_mechanism,
                self_mask_quantile=float(args.self_mask_quantile),
                self_mask_k=float(args.self_mask_k),
                self_mask_direction=args.self_mask_direction,
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

    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
