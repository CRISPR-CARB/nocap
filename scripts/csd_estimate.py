r"""Generate synthetic latent-SCM data and estimate directed path coefficients.

This script combines a latent linear structural causal model (SCM), a
DESeq2-style negative-binomial observation model, and the existing regression-
based cyclic single-door estimator. It supports independent generation and the
``paired_hierarchical`` design. In paired mode, one maximum-size observation
artifact is generated for a nested replicate unit and each parameter condition
is evaluated through a deterministic view of that artifact.

The simulation uses log2-fold change (relative to a geometric control baseline)
as its latent scale. In the equations below, ``B[target, parent]`` is the
mathematical target-parent matrix. The repository stores its transpose internally
as ``beta_matrix[source, target]`` for the edge ``source -> target``.

The equilibrium model is:

    X = B @ X + epsilon

The solver therefore evaluates ``(I - beta_matrix.T) X = epsilon``. Structural
betas are continuous signed log2 effects. For target ``g`` and parent ``k``:

    X_g = sum_k beta_gk * X_k + epsilon_g
    beta_gk = d X_g / d X_k

Thus ``beta_gk`` is the direct change in target log2-fold change for a
one-unit increase in the parent's log2-fold change. Since one unit on the
log2 scale is a doubling, ``2**beta_gk`` is the direct target-expression
multiplier for a doubling of the parent. For a general parent change
``delta_X_k``, the direct multiplier is ``2**(beta_gk * delta_X_k)``.

Observed counts use positive gene-specific baselines ``q0_g``, geometrically
centered sample size factors ``s_i``, and scalar or gene-specific dispersions
``alpha_g``:

    q_ig = q0_g * 2**X_ig
    mu_ig = s_i * q_ig
    Y_ig ~ NB(mu_ig, alpha_g)
    Var(Y_ig) = mu_ig + alpha_g * mu_ig**2

Here ``X_ig = log2(q_ig / q0_g)`` and ``2**X_ig`` is the cell-specific
normalized-expression multiplier relative to baseline. ``q0_g`` is a positive
geometric control baseline.

The estimator receives ``X-hat`` rather than raw counts. The pipeline also
computes normalized expression ``q-hat``:

    q_hat_ig = (Y_ig + pseudocount) / s_i
    X_hat_ig = log2((Y_ig + pseudocount) / (s_i * q0_g))

Missingness is applied to the estimator-facing ``X-hat`` DataFrame after
observation. Instrument-error missingness uses latent ``X`` so it is not
driven by observation noise or an already-masked representation.

The script:

1) Accepts an input :class:`networkx.DiGraph` (via GraphML or via a small built-in demo).
2) For **every** directed edge in the *estimation graph*, calls
   :func:`nocap.cyclic_single_door.estimate_path_coefficient_for_edge`.
3) Writes a CSV with per-edge estimation results and the corresponding
    ground-truth structural coefficient (beta) under the synthetic SCM.

Notes
-----
* The estimator itself performs σ-single-door identifiability checks using only
  the graph structure.
* For cyclic graphs, the coefficient matrix is stabilized and checked for
  conditioning before the equilibrium solve.
* Ground-truth coefficients are the structural edge coefficients (betas)
  used to generate the synthetic SCM.

Design modes
------------
``legacy`` is the default and preserves the existing one-condition generation
behavior. ``paired_hierarchical`` adds explicit experiment and nested
replicate identifiers and evaluates the full condition grid against a shared
maximum-size artifact. Use ``--max-samples`` to set the artifact size when it
should exceed the largest value in ``--n-samples-list``.

Seed contract
-------------
``--seed`` is the legacy shorthand and sets both phases to the same seed.
For replicate designs, pass both ``--scm-seed`` and ``--data-seed``. The SCM
seed controls structural beta draws and the realized true missing-edge
pattern. The data seed controls exogenous noise, size factors, q0, count
generation, and missingness. Separate RNG instances ensure that changing
the data seed does not change the SCM, while changing the SCM seed does.
Output rows include both explicit seeds; the compatibility ``seed`` column is
the data seed when explicit seed pairs are used. Paired mode intentionally
reuses the replicate seed pair across conditions and distinguishes jobs using
the experiment, replicate, and parameter-condition identifiers.

Examples
--------
Legacy reproducible run::

    uv run python scripts/csd_estimate.py --demo cycle --output-csv out.csv --seed 7

Fixed-SCM data replicate::

    uv run python scripts/csd_estimate.py --demo cycle --output-csv out.csv \
         --scm-seed 101 --data-seed 202 --n-samples-list 500

Paired hierarchical replicate::

    uv run python scripts/csd_estimate.py --demo cycle --output-csv out.csv \
        --design-mode paired_hierarchical --experiment-id demo-v1 \
        --scm-replicate-id scm-000 --data-replicate-id data-000 \
        --scm-seed 101 --data-seed 202 --n-samples-list 250,500 \
        --missing-data-rate-list 0,0.1
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
import pandas as pd
from y0.algorithm.separation.sigma_extension import sigma_extension

from nocap.cyclic_single_door import (
    classify_edge,
    estimate_path_coefficient_for_edge,
    nx_digraph_to_y0,
)
from nocap.experiment import canonical_condition_id
from nocap.scm_model import build_synthetic_scm
from nocap.simulation import SimulationConfig, generate_from_scm, generate_paired_data_artifact


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
    g2.add_edges_from(
        (str(u), str(v), dict(data)) for u, v, data in g.edges(data=True)
    )
    return g2


def _edge_signs(graph: nx.DiGraph) -> dict[tuple[str, str], float]:
    """Extract explicit +/- GraphML edge signs, ignoring missing/unknown values."""
    signs = {}
    for source, target, data in graph.edges(data=True):
        value = data.get("polarity", data.get("d0", data.get("sign")))
        if isinstance(value, str):
            value = value.strip()
        if value in ("+", "1", 1, 1.0):
            signs[(str(source), str(target))] = 1.0
        elif value in ("-", "-1", -1, -1.0):
            signs[(str(source), str(target))] = -1.0
    return signs


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
    forbidden_edges=(),
):
    """Generate synthetic latent-SCM data and observed expression.

    The latent SCM is

        X = B.T @ X + eps,

    where X is latent log-expression and B[u, v] is the structural beta
    for u -> v.

    UMI counts are then generated using

        mu_ig = s_i * q0_g * 2**X_ig
        Y_ig ~ NB(mu_ig, alpha_g).

    Returns
    -------
    data:
        X-hat expression DataFrame supplied to the estimator.

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
        forbidden_edges=forbidden_edges,
        signs=_edge_signs(estimation_graph),
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


def _task_latent_expression_hat(value) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError("Task field 'use_latent_expression_hat' must be a JSON boolean")


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
            "small_network",
            "frontdoor_cycle"
        ],
        help="Built-in demo graph to use when --graphml is omitted.",
    )
    p.add_argument(
        "--output-csv",
        type=str,
        required=False,
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
        default=None,
        help="Legacy shorthand: sets both --scm-seed and --data-seed.",
    )
    p.add_argument(
        "--scm-seed",
        type=int,
        default=None,
        help="RNG seed for structural betas and true missing edges.",
    )
    p.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="RNG seed for exogenous noise, counts, libraries, and missingness.",
    )
    p.add_argument("--design-mode", choices=["legacy", "paired_hierarchical"], default="legacy")
    p.add_argument("--experiment-id", default="legacy")
    p.add_argument("--scm-replicate-id", default="0")
    p.add_argument("--data-replicate-id", default="0")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--task-json",
        type=str,
        default=None,
        help="JSON task record generated by setup_csd_experiment.py.",
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
        choices=["instrument_error", "biological_error", "biological_error+instrument_error"],
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
        default=0.5,
        help=(
            "Median increase in log2fc of a genes expression over its baseline"
            "expression value given an increase in its parents log2fc. The median "
            "absolute beta is log(beta_med)."
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

    # Negative-binomial observation model parameters.
    p.add_argument(
        "--dispersion",
        type=float,
        default=None,
        help="Scalar negative-binomial dispersion alpha. Variance is mu + alpha * mu**2.",
    )
    p.add_argument(
        "--dispersions",
        type=str,
        default=None,
        help="Comma-separated gene-specific negative-binomial dispersions.",
    )
    p.add_argument(
        "--size-factor-log-sd",
        type=float,
        default=0.4,
        help="Standard deviation of the natural log of raw size factors.",
    )
    p.add_argument(
        "--umi-pseudocount",
        type=float,
        default=1.0,
        help=("Pseudocount used when converting counts to q-hat and X-hat."),
    )
    p.add_argument(
        "--no-latent-expression-hat",
        action="store_true",
        help="Use true latent expression as estimator input instead of count-derived expression.",
    )
    p.add_argument(
        "--baseline-expression-mean",
        type=float,
        default=1.0,
        help="Mean of the positive gene-specific q0 baselines.",
    )
    p.add_argument(
        "--baseline-expression-dispersion",
        type=float,
        default=2.25,
        help="Negative-binomial dispersion of gene-specific q0 baselines.",
    )

    # Regression/data cleaning.
    p.add_argument(
        "--min-rows-after-dropna",
        type=int,
        default=30,
        help="Minimum rows required after dropping NaNs for an edge regression.",
    )

    args = p.parse_args()

    if args.task_json is not None:
        with open(args.task_json, encoding="utf-8") as handle:
            task = json.load(handle)
        for name in (
            "experiment_id",
            "design_mode",
            "scm_replicate_id",
            "data_replicate_id",
            "scm_seed",
            "data_seed",
            "output_csv",
            "graphml",
            "intervention_id",
            "intervention_set_index",
            "intervention_genes",
            "intervention_semantics",
            "forbidden_edges",
            "fixed_intervention_values",
        ):
            if name in task:
                setattr(args, name, task[name])
        if "use_latent_expression_hat" in task:
            args.no_latent_expression_hat = not _task_latent_expression_hat(
                task["use_latent_expression_hat"]
            )
        args.n_samples_list = str(task["n_samples"])
        args.missing_edge_rate = float(task["missing_edge_rate"])
        args.missing_data_rate = float(task["missing_data_rate"])
        args.missing_data_mechanism = task["missing_data_mechanism"]
        # Intervention graphs can contain edges absent from the observational
        # adjustment table. Always classify those edges from the supplied graph.
        if task.get("intervention_id"):
            args.assume_adjustments_csv_complete = False

    if args.output_csv is None:
        raise SystemExit("--output-csv is required unless the task JSON supplies output_csv")

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
        elif args.demo == "small_network":
            graph.add_edges_from([("TF1", "G1"), ("TF1", "G2"), ("G2", "G3"), ("G1", "TF2"), ("G3", "TF2"), ("TF2", "G4"), ("G3", "TF3"), ("TF3", "G2"), ("TF3", "TF1")])
        elif args.demo == "frontdoor_cycle":
            graph.add_edges_from([("TF1", "G1"), ("G1", "TF2"), ("G1", "G2"), ("TF2", "G2"), ("TF2", "TF1"), ("TF1", "G2")])
        else:
            raise ValueError(f"Unknown demo {args.demo!r}")

    nodes = _default_nodes_from_graph(graph)
    if args.dispersion is not None and args.dispersions is not None:
        raise SystemExit("--dispersion and --dispersions cannot be used together")
    if args.dispersions is not None:
        dispersion: float | list[float] = cast(
            list[float], _parse_csv_list(args.dispersions, cast_fn=float)
        )
    else:
        dispersion = 0.1 if args.dispersion is None else float(args.dispersion)
    if isinstance(dispersion, list) and len(dispersion) != len(nodes):
        raise SystemExit("--dispersions must contain one value per graph node")
    fixed_intervention_values = getattr(args, "fixed_intervention_values", {}) or {}
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
        "experiment_id",
        "design_mode",
        "seed_schema_version",
        "scm_replicate_id",
        "data_replicate_id",
        "parameter_condition_id",
        "target_effect_id",
        "pairing_scope",
        "sample_parent_id",
        "sample_selection_rule",
        "run_status",
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
        "intervention_id",
        "intervention_set_index",
        "intervention_genes",
        "intervention_n_genes",
        "intervention_semantics",
        "target_removed",
        "target_present_in_intervened_graph",
        "recovery_status",
    ]

    # Prepare to write rows.
    write_header = not out_csv.exists() or out_csv.stat().st_size == 0
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        # --- Experiment grid ---
        grid = _iter_experiment_grid(args)
        paired_artifacts = None
        if args.design_mode == "paired_hierarchical":
            max_samples = args.max_samples or max(int(cell["n_samples"]) for cell in grid)
            artifact_config = SimulationConfig(
                n_samples=max_samples,
                estimation_graph=graph,
                fixed_intervention_values=fixed_intervention_values,
                dispersion=dispersion,
                size_factor_log_sd=float(args.size_factor_log_sd),
                umi_pseudocount=float(args.umi_pseudocount),
                baseline_expression_mean=float(args.baseline_expression_mean),
                baseline_expression_dispersion=float(args.baseline_expression_dispersion),
                use_latent_expression_hat=not args.no_latent_expression_hat,
            )
            paired_artifacts = {}
            artifact_key_prefix = getattr(args, "intervention_id", "observational")
            for edge_rate in sorted({float(cell["missing_edge_rate"]) for cell in grid}):
                variant_build = build_synthetic_scm(
                    graph,
                    nodes,
                    missing_edge_rate=edge_rate,
                    beta_med=float(args.beta_med),
                    beta_log_sd=float(args.beta_log_sd),
                    beta_p=float(args.beta_p),
                    beta_abs_max=float(args.beta_abs_max),
                    rng=_rng(int(args.scm_seed)),
                    forbidden_edges=getattr(args, "forbidden_edges", []),
                    signs=_edge_signs(graph),
                )
                paired_artifacts[(artifact_key_prefix, edge_rate)] = generate_paired_data_artifact(
                    variant_build.scm,
                    artifact_config,
                    max_samples=max_samples,
                    scm_seed=int(args.scm_seed),
                    data_seed=int(args.data_seed),
                )
        for trial_idx, cell in enumerate(grid):
            n_samples = int(cell["n_samples"])
            missing_edge_rate = float(cell["missing_edge_rate"])
            missing_data_rate = float(cell["missing_data_rate"])

            config = SimulationConfig(
                n_samples=n_samples,
                dispersion=dispersion,
                size_factor_log_sd=float(args.size_factor_log_sd),
                umi_pseudocount=float(args.umi_pseudocount),
                baseline_expression_mean=float(args.baseline_expression_mean),
                baseline_expression_dispersion=float(args.baseline_expression_dispersion),
                missing_data_rate=missing_data_rate,
                missing_data_mechanism=args.missing_data_mechanism,
                self_mask_quantile=float(args.self_mask_quantile),
                self_mask_k=float(args.self_mask_k),
                self_mask_direction=args.self_mask_direction,
                scc_confounding_strength=float(args.scc_confounding_strength),
                estimation_graph=graph,
                use_latent_expression_hat=not args.no_latent_expression_hat,
            )

            if paired_artifacts is not None:
                paired_artifact = paired_artifacts[
                    (getattr(args, "intervention_id", "observational"), missing_edge_rate)
                ]
                data = paired_artifact.view(config)
                scm_graph_true = paired_artifact.scm.graph
                betas_true = paired_artifact.scm.betas
            else:
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
                    forbidden_edges=getattr(args, "forbidden_edges", []),
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

                if not isinstance(data, pd.DataFrame):
                    raise TypeError("synthetic data must be a pandas DataFrame")
                cleaned = _safe_dropna_for_cols(
                    data,
                    cols,
                    min_rows=int(args.min_rows_after_dropna),
                )

                row = {
                    "experiment_id": args.experiment_id,
                    "design_mode": args.design_mode,
                    "seed_schema_version": "1",
                    "scm_replicate_id": args.scm_replicate_id,
                    "data_replicate_id": args.data_replicate_id,
                    "parameter_condition_id": canonical_condition_id(cell),
                    "target_effect_id": f"{cause}->{effect}",
                    "pairing_scope": "shared_complete_observation_shared_scores"
                    if paired_artifacts is not None
                    else "none",
                    "sample_parent_id": "maximum" if paired_artifacts is not None else "",
                    "sample_selection_rule": "prefix" if paired_artifacts is not None else "",
                    "run_status": "complete",
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
                    "intervention_id": getattr(args, "intervention_id", ""),
                    "intervention_set_index": getattr(args, "intervention_set_index", ""),
                    "intervention_genes": json.dumps(getattr(args, "intervention_genes", [])),
                    "intervention_n_genes": len(getattr(args, "intervention_genes", [])),
                    "intervention_semantics": getattr(args, "intervention_semantics", ""),
                    "target_removed": False,
                    "target_present_in_intervened_graph": True,
                    "recovery_status": "",
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
                except Exception as exc:  # noqa: BLE001
                    row["status"] = "estimation_error"
                    row["error"] = repr(exc)

                writer.writerow(row)

    print(f"Wrote: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
