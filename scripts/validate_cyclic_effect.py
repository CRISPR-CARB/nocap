"""Validate identified continuous effects against simulated cyclic SCM data."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import networkx as nx
import numpy as np

from nocap.estimation import (
    EstimationProfiler,
    KDEpyKDE,
    estimate_ate,
    estimated_scm_ate,
    evaluate_probability_expression,
    identify_query_status,
)
from nocap.scm_model import build_synthetic_scm
from nocap.simulation import (
    SimulationConfig,
    generate_from_scm,
    simulate_intervention_levels,
    true_scm_ate,
)


def _graph(args: argparse.Namespace) -> nx.DiGraph:
    """Load a graph or build the selected demo."""
    if args.graphml:
        graph = nx.read_graphml(args.graphml)
        result = nx.DiGraph()
        result.add_nodes_from(str(node) for node in graph.nodes())
        result.add_edges_from((str(u), str(v)) for u, v in graph.edges())
        return result
    demos = {
        "direct": [("X", "Y")],
        "chain": [("Z", "X"), ("X", "Y")],
        "cycle": [("X", "Y"), ("Y", "X"), ("Y", "Z"), ("Z", "Y")],
        "small_network": [
            ("TF1", "G1"),
            ("TF1", "G2"),
            ("G2", "G3"),
            ("G1", "TF2"),
            ("G3", "TF2"),
            ("TF2", "G4"),
            ("G3", "TF3"),
            ("TF3", "G2"),
            ("TF3", "TF1"),
        ],
        "frontdoor_cycle": [
            ("TF1", "TF2"),
            ("TF2", "TF3"),
            ("TF3", "TF1"),
            ("TF1", "G1"),
            ("TF2", "G1"),
            ("TF3", "G1"),
        ],
        "frontdoor": [("X", "Z"), ("Z", "Y"), ("U", "X"), ("U", "Y")],
        "backdoor": [("X", "Y"), ("Z", "X"), ("Z", "Y")],
    }
    return nx.DiGraph(demos[args.demo])


def main() -> None:
    """Run the end-to-end validation command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphml")
    parser.add_argument(
        "--demo",
        choices=(
            "direct",
            "chain",
            "cycle",
            "small_network",
            "frontdoor_cycle",
            "frontdoor",
            "backdoor",
        ),
        default="cycle",
    )
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--outcome", required=True)
    parser.add_argument("--treatment-levels", default="0,1")
    parser.add_argument(
        "--outcome-bounds",
        default="-inf,inf",
        help="lower,upper integration bounds; use -inf or inf for unbounded tails",
    )
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta-med", type=float, default=0.5)
    parser.add_argument("--beta-log-sd", type=float, default=1)
    parser.add_argument("--beta-p", type=float, default=0.5)
    parser.add_argument("--beta-abs-max", type=float, default=5.0)
    parser.add_argument("--missing-edge-rate", type=float, default=0.0)
    parser.add_argument("--min-rows-after-dropna", type=int, default=30)
    parser.add_argument("--allow-partial-estimate", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="collect and include estimation timing statistics in the output",
    )
    parser.add_argument(
        "--profile-log",
        help="append per-operation timing records and flush them immediately to this file",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    graph = _graph(args)
    levels = tuple(float(value) for value in args.treatment_levels.split(","))
    bounds = tuple(float(value) for value in args.outcome_bounds.split(","))
    if len(bounds) != 2:
        raise ValueError("Bounds must be two values")
    query = identify_query_status(
        graph,
        args.treatment,
        args.outcome,
        frozenset(node for node in graph.nodes if "U" in node or "u_" in node or "U_" in node),
    )
    print(query)
    if not query.identifiable or query.expression is None:
        raise SystemExit(f"query is unidentifiable: {query.error}")
    nodes = sorted(graph.nodes())
    build = build_synthetic_scm(
        graph,
        nodes,
        missing_edge_rate=args.missing_edge_rate,
        beta_med=args.beta_med,
        beta_log_sd=args.beta_log_sd,
        beta_p=args.beta_p,
        beta_abs_max=args.beta_abs_max,
        rng=np.random.default_rng(args.seed),
    )
    config = SimulationConfig(
        n_samples=args.n_samples,
        estimation_graph=graph,
        use_latent_expression_hat=False,
        use_umi_counts_as_observed_data=False,
        size_factor_log_sd=0.0,
        missing_data_rate=0.0,
    )
    observational = generate_from_scm(build.scm, config, seed=args.seed + 1)
    interventions = simulate_intervention_levels(
        build.scm, config, args.treatment, levels, seed=args.seed + 2
    )
    if observational.latent_log_expression is None:
        raise RuntimeError("observational simulation produced no data")
    observational_data = observational.observed_data
    if observational_data is None:
        raise RuntimeError("latent observational data was not materialized")
    profiler = EstimationProfiler(args.profile_log) if args.profile or args.profile_log else None
    density_ate = estimate_ate(
        query.expression,
        observational_data,
        args.treatment,
        args.outcome,
        (levels[0], levels[1]),
        mode="continuous",
        backend=KDEpyKDE(),
        outcome_bounds=bounds,
        profiler=profiler,
        marginalization="kde",
    )
    density_bounds = {
        column.name: (
            float(observational_data[column.name].min()),
            float(observational_data[column.name].max()),
        )
        for column in query.expression.get_variables()
        if column.name in observational_data
    }
    density_bounds[args.outcome] = bounds
    density = evaluate_probability_expression(
        query.expression,
        observational_data,
        mode="continuous",
        backend=KDEpyKDE(),
        bounds=density_bounds,
        profiler=profiler,
        marginalization="kde",
    )
    if args.outcome not in density.variables:
        raise ValueError("identified expression does not contain the requested outcome")
    grid_lower = (
        bounds[0] if np.isfinite(bounds[0]) else float(observational_data[args.outcome].min())
    )
    grid_upper = (
        bounds[1] if np.isfinite(bounds[1]) else float(observational_data[args.outcome].max())
    )
    if not grid_lower < grid_upper:
        raise ValueError("outcome bounds must span a non-empty density grid")
    outcome_grid = np.linspace(grid_lower, grid_upper, 201)
    density_values = []
    with (
        profiler.measure("script.density_grid", levels=len(levels), points=len(outcome_grid))
        if profiler
        else nullcontext()
    ):
        for level in levels:
            values = {args.outcome: outcome_grid}
            if args.treatment in density.variables:
                values[args.treatment] = float(level)
            missing = set(density.variables) - set(values)
            if missing:
                raise ValueError(
                    "cannot save density evaluation; provide values for free variables: "
                    f"{sorted(missing)}"
                )
            density_values.append(
                {"treatment_level": level, "density": np.asarray(density(**values)).tolist()}
            )
    true_ate = true_scm_ate(
        build.scm, args.treatment, args.outcome, (levels[0], levels[1]), seed=args.seed + 3
    )
    with profiler.measure("script.estimated_scm_ate") if profiler else nullcontext():
        estimated_ate_value, edge_estimates, estimated_scm = estimated_scm_ate(
            graph,
            observational_data,
            args.treatment,
            args.outcome,
            (levels[0], levels[1]),
            min_rows=args.min_rows_after_dropna,
            n_samples=args.n_samples,
            seed=args.seed + 4,
            allow_partial=args.allow_partial_estimate,
        )
    result = {
        "treatment": args.treatment,
        "outcome": args.outcome,
        "data_scale": "latent_log_expression",
        "treatment_levels": levels,
        "outcome_bounds": [str(bound) for bound in bounds],
        "n_samples": args.n_samples,
        "identified_expression": str(query.expression),
        "true_ate": true_ate,
        "identified_distribution_ate": density_ate,
        "identified_ate_error": density_ate - true_ate,
        "estimated_scm_ate": estimated_ate_value,
        "estimated_scm_ate_error": estimated_ate_value - true_ate,
        "estimated_scm_edges": list(estimated_scm.graph.edges()),
        "added_hidden_edges": build.added_edges,
        "intervention_treatment_values": {
            str(level): float(state.latent_log_expression[0, state.scm.nodes.index(args.treatment)])
            for level, state in interventions.items()
        },
    }
    if profiler is not None:
        result["estimation_profile"] = profiler.summary()
        profiler.close()
    output.with_suffix(".edges.json").write_text(
        json.dumps(
            [
                {
                    **estimate.__dict__,
                    "true_coefficient": build.scm.betas.get((estimate.cause, estimate.effect)),
                }
                for estimate in edge_estimates
            ],
            indent=2,
            default=list,
        ),
        encoding="utf-8",
    )
    # output.with_suffix(".distribution.json").write_text(
    #     json.dumps(
    #         {
    #             "expression": str(query.expression),
    #             "variables": list(density.variables),
    #             "outcome": args.outcome,
    #             "outcome_grid": outcome_grid.tolist(),
    #             "evaluations": density_values,
    #         },
    #         indent=2,
    #     ),
    #     encoding="utf-8",
    # )
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    # print(json.dumps({"summary": result, "distribution_file": str(output.with_suffix('.distribution.json'))}, indent=2))


if __name__ == "__main__":
    main()
