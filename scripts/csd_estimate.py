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

Seed contract
-------------
``--seed`` is the legacy shorthand and sets both phases to the same seed.
For replicate designs, pass both ``--scm-seed`` and ``--data-seed``. The SCM
seed controls structural beta draws and the realized true missing-edge
pattern. The data seed controls exogenous noise, UMI/count generation,
library sizes, and missingness. Separate RNG instances ensure that changing
the data seed does not change the SCM, while changing the SCM seed does.
Output rows include both explicit seeds; the compatibility ``seed`` column is
the data seed when explicit seed pairs are used.

Examples
--------
Legacy reproducible run::

    uv run python scripts/csd_estimate.py --demo cycle --output-csv out.csv --seed 7

Fixed-SCM data replicate::

    uv run python scripts/csd_estimate.py --demo cycle --output-csv out.csv \
        --scm-seed 101 --data-seed 202 --n-samples-list 500
"""

from __future__ import annotations

import argparse
import csv
import json
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
from nocap.scm_model import build_synthetic_scm
from nocap.simulation import SimulationConfig, generate_from_scm


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


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_synthetic_observational_data(
    estimation_graph: nx.DiGraph,
    scm_nodes: list[str],
    config: SimulationConfig,
    *,
    missing_edge_rate: float,
    beta_med: float,
    beta_log_sd: float,
    beta_p: float,
    beta_abs_max: float,
    seed: int | None = None,
    scm_seed: int | None = None,
    data_seed: int | None = None,
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
    if seed is not None:
        scm_seed = seed if scm_seed is None else scm_seed
        data_seed = seed if data_seed is None else data_seed
    if scm_seed is None or data_seed is None:
        raise ValueError("scm_seed and data_seed must both be provided")

    build = build_synthetic_scm(
        estimation_graph,
        scm_nodes,
        missing_edge_rate=missing_edge_rate,
        beta_med=beta_med,
        beta_log_sd=beta_log_sd,
        beta_p=beta_p,
        beta_abs_max=beta_abs_max,
        rng=_rng(scm_seed),
    )
    scm = build.scm
    betas_by_edge_true = scm.betas
    result = generate_from_scm(scm, config, rng=_rng(data_seed))
    print(f"Generated results from SCM with cond number: {build.condition_number}")
    return result.observed_data, scm.graph, betas_by_edge_true


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
    p.add_argument("--seed", type=int, default=None,
                   help="Legacy shorthand: sets both --scm-seed and --data-seed.")
    p.add_argument("--scm-seed", type=int, default=None,
                   help="RNG seed for structural betas and true missing edges.")
    p.add_argument("--data-seed", type=int, default=None,
                   help="RNG seed for exogenous noise, counts, libraries, and missingness.")
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

    if args.seed is not None:
        if args.scm_seed is not None or args.data_seed is not None:
            raise SystemExit("--seed cannot be combined with --scm-seed or --data-seed")
        args.scm_seed = args.seed
        args.data_seed = args.seed
    elif args.scm_seed is None and args.data_seed is None:
        args.seed = args.scm_seed = args.data_seed = 0
    elif args.scm_seed is None or args.data_seed is None:
        raise SystemExit("provide --seed, or provide both --scm-seed and --data-seed")

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
        "scm_seed",
        "data_seed",
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

            config = SimulationConfig(
                n_samples=n_samples,
                umi_dispersion=float(args.umi_dispersion),
                library_size_log_mean=float(args.library_size_log_mean),
                library_size_log_sd=float(args.library_size_log_sd),
                umi_pseudocount=float(args.umi_pseudocount),
                baseline_log_sd=float(args.baseline_log_sd),
                missing_data_rate=missing_data_rate,
                missing_data_mechanism=args.missing_data_mechanism,
                self_mask_quantile=float(args.self_mask_quantile),
                self_mask_k=float(args.self_mask_k),
                self_mask_direction=args.self_mask_direction,
                scc_confounding_strength=float(args.scc_confounding_strength),
                estimation_graph=graph,
            )

            data, scm_graph_true, betas_true = generate_synthetic_observational_data(
                graph,
                nodes,
                config,
                missing_edge_rate=missing_edge_rate,
                beta_med=float(args.beta_med),
                beta_log_sd=float(args.beta_log_sd),
                beta_p=float(args.beta_p),
                beta_abs_max=float(args.beta_abs_max),
                scm_seed=int(args.scm_seed),
                data_seed=int(args.data_seed),
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
                    "n_samples": config.n_samples,
                    "missing_edge_rate": missing_edge_rate,
                    "missing_data_rate": config.missing_data_rate,
                    "missing_data_mechanism": config.missing_data_mechanism,
                    "seed": int(args.seed) if args.seed is not None else int(args.data_seed),
                    "scm_seed": int(args.scm_seed),
                    "data_seed": int(args.data_seed),
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
