#!/usr/bin/env python3
"""Convert a GraphML file to a semopy model description string.

Each directed edge  source → target  becomes a regression term:
    target ~ {polarity}_{src}_{target}*source

Polarity mapping (configurable via pos_values / neg_values):
    '+', 'activation', 'positive', '1'  → prefix 'pos'  → BOUND(0, inf)
    '-', 'repression', 'negative', '-1' → prefix 'neg'  → BOUND(-inf, 0)
    anything else (incl. '+/-', None)   → prefix 'reg'  → free (no bound)

Self-loops are dropped with a warning.
Cycles are preserved by default (non-recursive SEM fit); a warning is emitted.
Pass ``break_cycles=True`` to remove a minimal set of feedback edges (DFS
back-edges) so the result is a DAG — recommended for standard SEM.

Usage (CLI)
----------
    python graphml_to_semopy.py network.graphml
    python graphml_to_semopy.py network.graphml -o model.txt
    python graphml_to_semopy.py network.graphml --no-polarity
    python graphml_to_semopy.py network.graphml --break-cycles
    python graphml_to_semopy.py network.graphml --validate
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import networkx as nx

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_POS_VALUES: frozenset[str] = frozenset({"+", "activation", "positive", "1"})
_DEFAULT_NEG_VALUES: frozenset[str] = frozenset({"-", "repression", "negative", "-1"})
_DEFAULT_INF: float = 1e6
_POLARITY_ATTR: str = "polarity"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    """Replace non-identifier characters with underscores."""
    sanitized = re.sub(r"\W", "_", name)
    if sanitized != name:
        warnings.warn(
            f"Node/edge name {name!r} contains non-identifier characters; "
            f"sanitized to {sanitized!r}.",
            UserWarning,
            stacklevel=3,
        )
    return sanitized


def _polarity_prefix(
    raw: str | None,
    pos_values: frozenset[str],
    neg_values: frozenset[str],
) -> str:
    """Return 'pos', 'neg', or 'reg' for a raw polarity string."""
    if raw is None:
        return "reg"
    v = raw.strip()
    if v in pos_values:
        return "pos"
    if v in neg_values:
        return "neg"
    return "reg"


def _break_cycles(G: nx.DiGraph) -> tuple[nx.DiGraph, list[tuple]]:
    """Remove a minimal set of feedback edges to make *G* a DAG.

    Uses a DFS-based back-edge detection (iterative, O(V+E)).  A back edge
    is any edge that points from a node to one of its DFS ancestors; removing
    all back edges yields a DAG.  The set is minimal in the sense that each
    removed edge is the *last* edge that closes a cycle discovered during the
    DFS traversal.

    Parameters
    ----------
    G:
        A directed graph (will not be modified in place).

    Returns
    -------
    dag:
        A copy of *G* with the feedback edges removed.
    removed:
        List of ``(src, tgt, data)`` tuples for every removed edge.
    """
    # States for iterative DFS colouring
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in G.nodes()}
    removed: list[tuple] = []
    feedback_set: set[tuple] = set()

    for start in G.nodes():
        if color[start] != WHITE:
            continue
        # Stack entries: (node, iterator-over-successors)
        stack = [(start, iter(G.successors(start)))]
        color[start] = GRAY
        while stack:
            node, children = stack[-1]
            try:
                child = next(children)
                if color[child] == GRAY:
                    # Back edge → feedback arc
                    feedback_set.add((node, child))
                elif color[child] == WHITE:
                    color[child] = GRAY
                    stack.append((child, iter(G.successors(child))))
            except StopIteration:
                color[node] = BLACK
                stack.pop()

    dag = G.copy()
    for src, tgt in feedback_set:
        data = dict(G[src][tgt])
        removed.append((src, tgt, data))
        dag.remove_edge(src, tgt)

    return dag, removed


# ---------------------------------------------------------------------------
# Main conversion function
# ---------------------------------------------------------------------------

def graphml_to_semopy(
    path_or_graph: str | Path | nx.DiGraph,
    use_polarity: bool = True,
    pos_values: Iterable[str] = _DEFAULT_POS_VALUES,
    neg_values: Iterable[str] = _DEFAULT_NEG_VALUES,
    inf: float = _DEFAULT_INF,
    polarity_attr: str = _POLARITY_ATTR,
    break_cycles: bool = False,
) -> str:
    """Convert a GraphML file (or an existing DiGraph) to a semopy description.

    Parameters
    ----------
    path_or_graph:
        Path to a ``.graphml`` file, or an already-loaded ``nx.DiGraph``.
    use_polarity:
        If True, read the ``polarity_attr`` edge attribute and emit
        ``BOUND`` constraints for positive/negative edges.
    pos_values:
        Edge polarity strings that map to a non-negative (≥ 0) bound.
    neg_values:
        Edge polarity strings that map to a non-positive (≤ 0) bound.
    inf:
        Finite substitute for ±∞ in ``BOUND`` statements (semopy requires
        finite floats).
    polarity_attr:
        Name of the GraphML edge attribute that stores polarity information.
    break_cycles:
        If True, detect cycles and remove a minimal set of DFS back-edges to
        produce a DAG before generating the description.  A warning is emitted
        listing how many edges were removed.  If False (default), cycles are
        preserved and a warning is emitted instead.

    Returns
    -------
    str
        A semopy model description string ready to pass to ``semopy.Model``.
    """
    pos_values = frozenset(pos_values)
    neg_values = frozenset(neg_values)

    # ------------------------------------------------------------------
    # Load graph
    # ------------------------------------------------------------------
    if isinstance(path_or_graph, (str, Path)):
        G = nx.read_graphml(str(path_or_graph))
    else:
        G = path_or_graph

    if not isinstance(G, nx.DiGraph):
        raise TypeError(
            f"Expected a directed graph (DiGraph), got {type(G).__name__}. "
            "Make sure the GraphML file declares edgedefault='directed'."
        )

    # ------------------------------------------------------------------
    # Drop self-loops
    # ------------------------------------------------------------------
    self_loops = list(nx.selfloop_edges(G))
    if self_loops:
        warnings.warn(
            f"Dropping {len(self_loops)} self-loop(s) "
            f"(e.g. {self_loops[0][0]!r} → {self_loops[0][1]!r}). "
            "Self-loops are not valid in SEM.",
            UserWarning,
            stacklevel=2,
        )
        G = G.copy()
        G.remove_edges_from(self_loops)

    # ------------------------------------------------------------------
    # Handle cycles
    # ------------------------------------------------------------------
    if break_cycles:
        G, removed_edges = _break_cycles(G)
        if removed_edges:
            examples = ", ".join(
                f"{s!r}→{t!r}" for s, t, _ in removed_edges[:3]
            )
            suffix = f" (e.g. {examples})" if removed_edges else ""
            warnings.warn(
                f"Removed {len(removed_edges)} feedback edge(s) to break cycles{suffix}. "
                "The resulting graph is a DAG.",
                UserWarning,
                stacklevel=2,
            )
    else:
        try:
            cycles = list(nx.simple_cycles(G))
            if cycles:
                warnings.warn(
                    f"Graph contains {len(cycles)} cycle(s). "
                    "semopy will attempt a non-recursive (cyclic) fit. "
                    "Identification is not guaranteed.",
                    UserWarning,
                    stacklevel=2,
                )
        except Exception:
            pass  # very large graphs may time out; skip cycle check

    # ------------------------------------------------------------------
    # Build regression lines and collect bound groups
    # ------------------------------------------------------------------
    # target -> list of "paramname*source" terms
    regressions: dict[str, list[str]] = defaultdict(list)
    pos_params: list[str] = []
    neg_params: list[str] = []

    for src, tgt, data in sorted(G.edges(data=True)):
        src_s = _sanitize(src)
        tgt_s = _sanitize(tgt)

        if use_polarity:
            raw_pol = data.get(polarity_attr, None)
            prefix = _polarity_prefix(raw_pol, pos_values, neg_values)
        else:
            prefix = "reg"

        param = f"{prefix}_{src_s}_{tgt_s}"
        regressions[tgt_s].append(f"{param}*{src_s}")

        if use_polarity:
            if prefix == "pos":
                pos_params.append(param)
            elif prefix == "neg":
                neg_params.append(param)

    # ------------------------------------------------------------------
    # Assemble description string
    # ------------------------------------------------------------------
    lines: list[str] = ["# Structural part"]
    for tgt in sorted(regressions):
        terms = " + ".join(regressions[tgt])
        lines.append(f"{tgt} ~ {terms}")

    if use_polarity and (pos_params or neg_params):
        lines.append("")
        lines.append("# Polarity constraints")
        # semopy BOUND syntax: BOUND(lower, upper) param1 param2 ...
        # Emit in chunks to avoid extremely long lines
        _chunk = 10
        if pos_params:
            for i in range(0, len(pos_params), _chunk):
                chunk = " ".join(pos_params[i : i + _chunk])
                lines.append(f"BOUND(0, {inf:g}) {chunk}")
        if neg_params:
            for i in range(0, len(neg_params), _chunk):
                chunk = " ".join(neg_params[i : i + _chunk])
                lines.append(f"BOUND({-inf:g}, 0) {chunk}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert a GraphML file to a semopy model description.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="Path to the input .graphml file.")
    p.add_argument(
        "-o", "--output",
        default=None,
        help="Write description to this file instead of stdout.",
    )
    p.add_argument(
        "--no-polarity",
        action="store_true",
        default=False,
        help="Ignore polarity edge attributes; all parameters are free.",
    )
    p.add_argument(
        "--break-cycles",
        action="store_true",
        default=False,
        help=(
            "Detect cycles and remove a minimal set of DFS back-edges to "
            "produce a DAG (recommended for standard SEM). "
            "Default: leave cycles in and emit a warning."
        ),
    )
    p.add_argument(
        "--inf",
        type=float,
        default=_DEFAULT_INF,
        help="Finite substitute for ±∞ in BOUND statements.",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="Try to instantiate semopy.Model with the result to check syntax.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    """Entry point for CLI usage."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    desc = graphml_to_semopy(
        args.input,
        use_polarity=not args.no_polarity,
        inf=args.inf,
        break_cycles=args.break_cycles,
    )

    if args.validate:
        try:
            import semopy  # noqa: F401
            semopy.Model(desc)
            print("✓ semopy.Model parsed the description successfully.", file=sys.stderr)
        except ImportError:
            print("semopy not installed; skipping validation.", file=sys.stderr)
        except Exception as exc:
            print(f"✗ semopy.Model raised an error: {exc}", file=sys.stderr)

    if args.output:
        Path(args.output).write_text(desc, encoding="utf-8")
        print(f"Description written to {args.output}", file=sys.stderr)
    else:
        print(desc)


if __name__ == "__main__":
    main()
