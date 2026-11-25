"""Estimate a discrete ATE from synthetic and binarized expression data.

Run the notebook's discrete ATE example on the built-in feedback graph:

    uv run python scripts/validate_cyclic_effect_discrete.py

Choose another built-in demo graph:

    uv run python scripts/validate_cyclic_effect_discrete.py --demo chain
    uv run python scripts/validate_cyclic_effect_discrete.py --demo fork
    uv run python scripts/validate_cyclic_effect_discrete.py --demo diamond

Or provide a directed GraphML graph and choose its treatment and outcome:

    uv run python scripts/validate_cyclic_effect_discrete.py --graphml network.graphml \
        --treatment TF4 --outcome G1

The graph must encode an identifiable treatment-to-outcome query. Install the
``boolean`` optional dependencies to use scBoolSeq binarization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from nocap.cyclic_id import identify_causal_query
from nocap.estimation import (
    DistributionEstimationError,
    estimate_ate,
    evaluate_probability_expression,
)
from nocap.scm_model import build_synthetic_scm
from nocap.simulation import (
    SimulationConfig,
    generate_paired_data_artifact,
    simulate_expression_treatment,
)


def _demo_graph(name: str) -> tuple[nx.DiGraph, str, str]:
    """Create a built-in graph and its default treatment/outcome query."""
    demo_edges = {
        "small_network": (
            [
                ("TF1", "TF2"),
                ("TF2", "TF3"),
                ("TF3", "TF1"),
                ("TF1", "TF4"),
                ("TF2", "TF4"),
                ("TF3", "TF4"),
                ("TF4", "G1"),
            ],
            "TF4",
            "G1",
        ),
        "chain": ([("A", "B"), ("B", "C")], "A", "C"),
        "fork": ([("A", "B"), ("A", "C")], "A", "C"),
        "diamond": (
            [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
            "A",
            "D",
        ),
    }
    if name not in demo_edges:
        raise ValueError(f"unknown demo graph: {name}")
    edges, treatment, outcome = demo_edges[name]
    return nx.DiGraph(edges), treatment, outcome


def _load_graph(graphml: Path | None, demo: str) -> tuple[nx.DiGraph, str | None, str | None]:
    """Load and normalize a user graph or return a built-in graph and default query."""
    if graphml is None:
        graph, treatment, outcome = _demo_graph(demo)
        return graph, treatment, outcome
    source = nx.read_graphml(graphml)
    if not source.is_directed():
        raise ValueError("the input GraphML graph must be directed")
    graph = nx.DiGraph(source)
    graph = nx.relabel_nodes(graph, {node: str(node) for node in graph.nodes})
    if not graph.nodes:
        raise ValueError("the input graph must contain at least one node")
    return graph, None, None


def _log_normalized_counts(counts: np.ndarray, size_factors: np.ndarray, nodes) -> pd.DataFrame:
    """Convert UMI counts to the log-normalized matrix expected by scBoolSeq."""
    return pd.DataFrame(np.log1p(counts / size_factors[:, None]), columns=nodes)


def _binarize(expression: pd.DataFrame) -> pd.DataFrame:
    """Binarize expression with the workflow's zero-count and unknown policies."""
    try:
        from nocap.binarization import binarize_expression
    except ImportError as error:
        raise SystemExit(
            "scBoolSeq is required for this workflow; install NOCAP with its boolean extras "
            "(for example: uv sync --extra boolean)."
        ) from error

    result = binarize_expression(
        expression,
        unknown="off",
        binarizer_kwargs={"zeroes_are": 0.0},
    )
    if not isinstance(result, pd.DataFrame):
        raise TypeError("DataFrame binarization must return a labeled DataFrame")
    return result


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    graph_source = parser.add_mutually_exclusive_group()
    graph_source.add_argument("--graphml", type=Path, help="Directed input graph in GraphML format")
    graph_source.add_argument(
        "--demo",
        choices=["small_network", "chain", "fork", "diamond"],
        default="small_network",
        help="Built-in demo graph (default: small_network)",
    )
    parser.add_argument("--treatment", help="Treatment node (defaults to the demo's query)")
    parser.add_argument("--outcome", help="Outcome node (defaults to the demo's query)")
    parser.add_argument("--n-samples", type=int, default=10_000, help="Observational samples")
    parser.add_argument(
        "--treatment-samples", type=int, default=20_000, help="Samples per A/K intervention"
    )
    parser.add_argument("--seed", type=int, default=42, help="Root random seed")
    parser.add_argument("--beta-med", type=float, default=0.5, help="Median edge strength")
    parser.add_argument("--beta-log-sd", type=float, default=1.0, help="Log-SD of edge strengths")
    parser.add_argument("--beta-p", type=float, default=0.5, help="Probability an edge activates")
    parser.add_argument("--beta-abs-max", type=float, default=5.0, help="Maximum edge strength")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("discrete_ate_workflow.json"),
        help="JSON summary path; related result tables are written beside it",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run synthetic data generation, binary recovery, and discrete ATE estimation."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.n_samples < 1 or args.treatment_samples < 1:
        parser.error("--n-samples and --treatment-samples must be positive")

    graph, default_treatment, default_outcome = _load_graph(args.graphml, args.demo)
    treatment = args.treatment or default_treatment
    outcome = args.outcome or default_outcome
    if treatment is None or outcome is None:
        parser.error("--treatment and --outcome are required with --graphml")
    if treatment not in graph or outcome not in graph:
        parser.error(f"treatment {treatment!r} and outcome {outcome!r} must both be graph nodes")
    if treatment == outcome:
        parser.error("treatment and outcome must be different nodes")
    try:
        import scboolseq  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "scBoolSeq is required for this workflow; install NOCAP with its boolean extras "
            "(for example: uv sync --extra boolean)."
        ) from error

    nodes = tuple(sorted(str(node) for node in graph.nodes))
    scm = build_synthetic_scm(
        graph,
        nodes,
        missing_edge_rate=0.0,
        beta_med=args.beta_med,
        beta_log_sd=args.beta_log_sd,
        beta_p=args.beta_p,
        beta_abs_max=args.beta_abs_max,
        rng=np.random.default_rng(args.seed),
    ).scm
    config = SimulationConfig(
        n_samples=args.n_samples,
        estimation_graph=graph,
        use_latent_expression_hat=False,
        use_umi_counts_as_observed_data=False,
        use_binary_expression_state=True,
    )
    artifact = generate_paired_data_artifact(
        scm,
        config,
        max_samples=args.n_samples,
        scm_seed=args.seed,
        data_seed=args.seed + 1,
        binarize=True,
        unknown="off",
        binarizer_kwargs={"zeroes_are": 0.0},
    )

    binary_data = artifact.binarized_view(config)
    true_binary_state = artifact.latent_expression_state_view(config)
    outcome_index = scm.nodes.index(outcome)
    treatment_config = SimulationConfig(**{**config.__dict__, "n_samples": args.treatment_samples})
    treatment_states = {
        treatment_type: simulate_expression_treatment(
            scm,
            treatment_config,
            target=treatment,
            treatment=treatment_type,
            seed=args.seed + 10 + index,
        )
        for index, treatment_type in enumerate(("A", "K"))
    }

    treated_latent_fractions: dict[str, float] = {}
    treated_binarized_fractions: dict[str, float] = {}
    treated_pooled_fractions: dict[str, float] = {}
    agreement_by_gene: dict[str, np.ndarray] = {}
    pooled_agreement_by_treatment: dict[str, np.ndarray] = {}
    treated_log_expression: dict[str, pd.DataFrame] = {}

    for treatment_type, state in treatment_states.items():
        if state.umi_counts is None or state.size_factors is None:
            raise RuntimeError("expression treatment did not produce UMI observations")
        if state.latent_expression_state is None:
            raise RuntimeError("expression treatment did not produce binary latent truth")

        treated_log_expression[treatment_type] = _log_normalized_counts(
            state.umi_counts, state.size_factors, scm.nodes
        )
        binarized = _binarize(treated_log_expression[treatment_type]).to_numpy().astype(bool)
        latent = state.latent_expression_state
        treated_latent_fractions[treatment_type] = float(latent[:, outcome_index].mean())
        treated_binarized_fractions[treatment_type] = float(binarized[:, outcome_index].mean())
        agreement_by_gene[treatment_type] = (binarized == latent).mean(axis=0)

    pooled_log_expression = pd.concat(treated_log_expression, names=["treatment", "sample"])
    pooled_binary = _binarize(pooled_log_expression)
    for treatment_type, state in treatment_states.items():
        if state.latent_expression_state is None:
            raise RuntimeError("expression treatment did not produce binary latent truth")
        pooled_calls = pooled_binary.loc[treatment_type].to_numpy().astype(bool)
        treated_pooled_fractions[treatment_type] = float(pooled_calls[:, outcome_index].mean())
        pooled_agreement_by_treatment[treatment_type] = (
            pooled_calls == state.latent_expression_state
        ).mean(axis=0)

    treated_comparison = pd.DataFrame(
        {
            "True latent outcome ON fraction": treated_latent_fractions,
            "Separate-fit outcome ON fraction": treated_binarized_fractions,
            "Pooled-fit outcome ON fraction": treated_pooled_fractions,
        }
    )
    treated_comparison["Separate-fit outcome agreement"] = [
        agreement_by_gene[name][outcome_index] for name in treated_comparison.index
    ]
    treated_comparison["Pooled-fit outcome agreement"] = [
        pooled_agreement_by_treatment[name][outcome_index] for name in treated_comparison.index
    ]
    latent_ate = treated_latent_fractions["A"] - treated_latent_fractions["K"]
    separate_fit_ate = treated_binarized_fractions["A"] - treated_binarized_fractions["K"]
    pooled_fit_ate = treated_pooled_fractions["A"] - treated_pooled_fractions["K"]
    treated_comparison.loc["ATE (A - K)"] = [
        latent_ate,
        separate_fit_ate,
        pooled_fit_ate,
        np.nan,
        np.nan,
    ]

    expression = identify_causal_query(graph, interventions={treatment}, outcomes={outcome})
    print(f"CyclicID Expression: {expression}")
    try:
        binary_ate = estimate_ate(
            expression,
            binary_data,
            treatment=treatment,
            outcome=outcome,
            treatment_levels=(0, 1),
        )
        latent_binary_ate = estimate_ate(
            expression,
            true_binary_state,
            treatment=treatment,
            outcome=outcome,
            treatment_levels=(0, 1),
        )
    except DistributionEstimationError as error:
        raise SystemExit(
            "The identified expression could not be estimated from the sampled data, "
            "usually because a required conditional stratum has no observations. "
            "Increase --n-samples or choose a smaller/less sparse graph. "
            f"Estimator details: {error}"
        ) from error
    observational_comparison = pd.DataFrame(
        {"ATE": [binary_ate, latent_binary_ate, latent_ate]},
        index=[
            "Observational data: binarized counts",
            "Observational data: latent binary states",
            "Simulator intervention benchmark: A - K",
        ],
    )
    observational_comparison["Absolute error vs intervention benchmark"] = (
        observational_comparison["ATE"] - latent_ate
    ).abs()
    observational_comparison["Relative error vs latent observational estimate"] = [
        (binary_ate - latent_binary_ate) / latent_binary_ate if latent_binary_ate != 0 else np.nan,
        np.nan,
        np.nan,
    ]

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_stem = output_path.with_suffix("")
    treatment_csv = output_stem.with_name(f"{output_stem.name}.treatment_diagnostics.csv")
    ate_csv = output_stem.with_name(f"{output_stem.name}.ate_comparison.csv")
    binary_distribution_csv = output_stem.with_name(
        f"{output_stem.name}.binarized_distribution.csv"
    )
    latent_distribution_csv = output_stem.with_name(f"{output_stem.name}.latent_distribution.csv")
    treated_comparison.to_csv(treatment_csv, index_label="condition")
    observational_comparison.to_csv(ate_csv, index_label="estimate")
    binary_distribution = evaluate_probability_expression(expression, binary_data)
    latent_distribution = evaluate_probability_expression(expression, true_binary_state)
    assert isinstance(binary_distribution, pd.DataFrame) and isinstance(
        latent_distribution, pd.DataFrame
    )
    binary_distribution.to_csv(binary_distribution_csv, index=False)
    latent_distribution.to_csv(latent_distribution_csv, index=False)

    result = {
        "demo": args.demo if args.graphml is None else None,
        "graphml": str(args.graphml) if args.graphml is not None else None,
        "graph": {
            "nodes": list(nodes),
            "edges": [[str(source), str(target)] for source, target in graph.edges],
        },
        "query": {"treatment": treatment, "outcome": outcome},
        "seed": args.seed,
        "n_samples": args.n_samples,
        "treatment_samples": args.treatment_samples,
        "scm_betas": {
            f"{source} -> {target}": coefficient
            for (source, target), coefficient in scm.betas.items()
        },
        "ate": {
            "binarized_observational": float(binary_ate),
            "latent_binary_observational": float(latent_binary_ate),
            "activation_minus_knockout_benchmark": float(latent_ate),
        },
        "identified_expression": str(expression),
        "outputs": {
            "treatment_diagnostics": str(treatment_csv),
            "ate_comparison": str(ate_csv),
            "binarized_distribution": str(binary_distribution_csv),
            "latent_distribution": str(latent_distribution_csv),
        },
    }
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")

    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Query: {treatment} -> {outcome}")
    print("\nA/K treatment binarization diagnostics:")
    print(treated_comparison.to_string(float_format=lambda value: f"{value:.4f}"))
    print("\nObservational ATE comparison:")
    print(observational_comparison.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nWrote JSON summary and result tables beside {output_path}")


if __name__ == "__main__":
    main()
