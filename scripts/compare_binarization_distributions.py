"""Compare recovered expression states with a thresholded latent-SCM diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from nocap.scm_model import build_synthetic_scm
from nocap.simulation import (
    SimulationConfig,
    generate_paired_data_artifact,
)


def _demo_graph(name: str) -> nx.DiGraph:
    """Create one of the built-in demonstration SCM graphs."""
    graph = nx.DiGraph()
    if name == "chain":
        graph.add_edges_from([("A", "B"), ("B", "C")])
    elif name == "cycle":
        graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    else:
        graph.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    graph.add_nodes_from(sorted(graph.nodes()))
    return graph


def _per_gene_metrics(
    truth: np.ndarray, binary_expression: np.ndarray, genes: list[str]
) -> pd.DataFrame:
    """Compute known-state metrics and ON/unknown fractions by gene."""
    rows = []
    for index, gene in enumerate(genes):
        actual = truth[:, index].astype(bool)
        recovered = binary_expression[:, index]
        known = np.isfinite(recovered)
        predicted = recovered[known].astype(bool)
        observed = actual[known]
        tp = int(np.sum(observed & predicted))
        tn = int(np.sum(~observed & ~predicted))
        fp = int(np.sum(~observed & predicted))
        fn = int(np.sum(observed & ~predicted))
        n_known = int(known.sum())
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        specificity = tn / (tn + fp) if tn + fp else np.nan
        rows.append(
            {
                "gene": gene,
                "true_on_fraction": float(actual.mean()),
                "recovered_on_fraction": float(predicted.mean()) if n_known else np.nan,
                "unknown_fraction": float((~known).mean()),
                "known_entry_accuracy": (tp + tn) / n_known if n_known else np.nan,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": 2 * precision * recall / (precision + recall)
                if np.isfinite(precision + recall) and precision + recall
                else np.nan,
                "known_entries": n_known,
            }
        )
    return pd.DataFrame(rows)


def _write_outputs(metrics: pd.DataFrame, prefix: Path) -> tuple[Path, Path]:
    """Write per-gene metrics and an ON-fraction distribution figure."""
    import matplotlib.pyplot as plt

    csv_path = prefix.with_suffix(".csv")
    figure_path = prefix.with_name(f"{prefix.name}_on_fractions.png")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(csv_path, index=False)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    values = [
        metrics["true_on_fraction"].dropna(),
        metrics["recovered_on_fraction"].dropna(),
    ]
    axis.boxplot(values, tick_labels=["Latent gene expressed", "scBoolSeq known entries"])
    axis.set_ylabel("Across-gene ON fraction")
    axis.set_title("Latent threshold diagnostic vs. recovered expression")
    axis.set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    return csv_path, figure_path


def main(argv: list[str] | None = None) -> None:
    """Run paired simulation, binary recovery, and distribution comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--demo", choices=["chain", "cycle", "small_network"], default="chain")
    parser.add_argument("--graphml", type=Path)
    parser.add_argument("--output-prefix", type=Path, default=Path("binarization_comparison"))
    parser.add_argument("--missing-data-rate", type=float, default=0.0)
    parser.add_argument(
        "--missing-data-mechanism",
        choices=["biological_error", "instrument_error", "biological_error+instrument_error"],
        default="biological_error",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--covariance-type", choices=["full", "tied", "diag", "spherical"], default="full"
    )
    parser.add_argument(
        "--zero-count-state",
        choices=["off", "on", "unknown"],
        default="off",
        help="Binary state assigned by scBoolSeq to zero UMI counts (default: off).",
    )
    args = parser.parse_args(argv)
    if args.n_samples < 1:
        parser.error("--n-samples must be positive")

    graph = nx.DiGraph(nx.read_graphml(args.graphml)) if args.graphml else _demo_graph(args.demo)
    graph = nx.relabel_nodes(graph, {node: str(node) for node in graph.nodes})
    nodes = tuple(sorted(graph.nodes))
    scm = build_synthetic_scm(
        graph,
        nodes,
        missing_edge_rate=0.0,
        beta_med=0.25,
        beta_log_sd=0.2,
        beta_p=0.5,
        beta_abs_max=0.75,
        rng=np.random.default_rng(args.seed),
    ).scm
    config = SimulationConfig(
        n_samples=args.n_samples,
        missing_data_rate=args.missing_data_rate,
        missing_data_mechanism=args.missing_data_mechanism,
        use_binary_expression_state=True,
    )
    artifact = generate_paired_data_artifact(
        scm,
        config,
        max_samples=args.n_samples,
        scm_seed=args.seed,
        data_seed=args.seed + 1,
        binarize=True,
        unknown="keep",
        binarizer_kwargs={
            "confidence": args.confidence,
            "covariance_type": args.covariance_type,
            "zeroes_are": {
                "off": 0.0,
                "on": 1.0,
                "unknown": np.nan,
            }[args.zero_count_state],
        },
    )
    binary = artifact.binarized_view(config)
    if artifact.latent_expression_state is not None:
        truth = (artifact.latent_expression_state_view(config) > 0).to_numpy()
    else:
        raise ValueError("No latent expression state in data artifact.")
    metrics = _per_gene_metrics(truth, binary.to_numpy(), list(scm.nodes))
    csv_path, figure_path = _write_outputs(metrics, args.output_prefix)
    print(
        metrics[
            [
                "true_on_fraction",
                "recovered_on_fraction",
                "unknown_fraction",
                "known_entry_accuracy",
                "precision",
                "recall",
                "f1",
            ]
        ]
        .mean(numeric_only=True)
        .to_string()
    )
    print(f"Wrote {csv_path} and {figure_path}")


if __name__ == "__main__":
    main()
