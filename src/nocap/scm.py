"""Structural causal models."""

import os
import re
from functools import partial
from statistics import fmean

import networkx as nx
import pandas as pd
import pydot
import sympy as sy
from networkx.drawing.nx_pydot import from_pydot
from pgmpy.models import BayesianNetwork
from sklearn.linear_model import LinearRegression
from y0.dsl import Variable
from y0.graph import NxMixedGraph
from y0.simulation import LinearSCM


def dagitty_to_dot(daggity_string: str | None) -> str:
    """Convert from dagitty format to DOT format.

    Modified from dowhy: https://www.pywhy.org/dowhy/v0.11.1/_modules/dowhy/utils/graph_operations.html#daggity_to_dot
    Converts the input daggity_string to valid DOT graph format.
    """  # noqa: DAR101 DAR201 DAR401
    if type(daggity_string) == str:
        graph = re.sub(r"\n", "; ", daggity_string)
        graph = re.sub(r"^dag ", "digraph ", graph)
        graph = re.sub("{;", "{", graph)
        graph = re.sub("};", "}", graph)
        graph = re.sub("outcome,*,", "", graph)
        graph = re.sub("adjusted,*", "", graph)
        graph = re.sub("exposure,*", "", graph)
        graph = re.sub("latent,*", 'observed="no",', graph)
        graph = re.sub(",]", "]", graph)
        graph = re.sub(
            r'bb="[\d.,]+";?', "", graph
        )  # Remove bb line with four numbers and optional trailing semicolon.
        graph = re.sub(r"\s+", " ", graph).strip()  # Trim all extra spaces
        return graph
    else:
        raise ValueError("Input was not a string.")


def read_dag_file(file_path: str) -> str | None:
    """Read the contents of a .dag file and returns it as a multiline string."""
    try:
        with open(file_path, "r") as file:
            return file.read()
    except IOError:
        # print(f"Error reading file: {e}")
        return None


def dagitty_to_mixed_graph(dagitty_input: str, str_var_name: bool = False) -> NxMixedGraph:
    """Convert a string in dagitty (.dag) to NxMixedGraph."""
    # Check if the input is a file path
    if os.path.isfile(dagitty_input):
        dagitty_graph_str = read_dag_file(dagitty_input)
    else:
        dagitty_graph_str = dagitty_input

    # .dag string -> DOT -> networkx -> NxMixedGraph
    dot_graph_string = dagitty_to_dot(dagitty_graph_str)
    dot_graph = pydot.graph_from_dot_data(dot_graph_string)[0]
    nx_graph = from_pydot(dot_graph)
    if str_var_name:
        mixed_graph = NxMixedGraph.from_edges(
            directed=[(u, v) for u, v, d in nx_graph.edges]
        )  # convert from str to variable
    else:
        mixed_graph = NxMixedGraph.from_str_edges(
            directed=[(u, v) for u, v, d in nx_graph.edges]
        )  # convert from str to variable
    assert isinstance(mixed_graph, NxMixedGraph)  # noqa: S101
    return mixed_graph


def dagitty_to_digraph(dagitty_input: str) -> nx.DiGraph:
    """Convert a string in dagitty (.dag) to NX DiGraph."""
    # Check if the input is a file path
    if os.path.isfile(dagitty_input):
        dagitty_graph_str = read_dag_file(dagitty_input)
    else:
        dagitty_graph_str = dagitty_input

    # .dag string -> DOT -> networkx -> NxMixedGraph
    dot_graph_string = dagitty_to_dot(dagitty_graph_str)
    dot_graph = pydot.graph_from_dot_data(dot_graph_string)[0]
    nx_graph = from_pydot(dot_graph)
    assert isinstance(nx_graph, nx.DiGraph)  # noqa: S101
    return nx_graph


def generate_lscm_from_dag(graph: nx.DiGraph) -> dict[sy.Symbol, sy.Expr]:  # noqa: N802
    """Generate a linear structural causal model from a directed acycle graph (networkx DAG)."""
    assert nx.is_directed_acyclic_graph(graph), "Not a DAG"  # noqa: S101 # check input DAG
    equations = {}
    sorted_nodes = list(nx.topological_sort(graph))
    for node in sorted_nodes:
        node_sym = sy.Symbol(node)
        expression_terms = []
        parents = list(graph.predecessors(node))
        for parent in parents:
            beta_sym = sy.Symbol(f"beta_{parent}_->{node}")
            parent_sym = sy.Symbol(f"{parent}")
            expression_terms.append(beta_sym * parent_sym)
        epsilon_sym = sy.Symbol(f"epsilon_{node}")
        expression_terms.append(epsilon_sym)
        expression = sum(expression_terms)
        equations[node_sym] = expression
    return equations


def generate_lscm_from_mixed_graph(graph: NxMixedGraph) -> dict[sy.Symbol, sy.Expr]:  # noqa: N802
    """Generate a linear structural causal model from a mixed directed and bidirected graph (y0 NxMixedGraph)."""
    assert nx.is_directed_acyclic_graph(graph.directed), "Not a DAG"  # noqa: S101 # check input DAG
    equations = {}
    sorted_nodes = list(nx.topological_sort(graph.directed))
    for node in sorted_nodes:
        node_sym = sy.Symbol(node.name)
        expression_terms = []
        parents = list(graph.directed.predecessors(node))
        for parent in parents:
            beta_sym = sy.Symbol(f"beta_{parent.name}_->{node.name}")
            parent_sym = sy.Symbol(f"{parent.name}")
            expression_terms.append(beta_sym * parent_sym)
        epsilon_sym = sy.Symbol(f"epsilon_{node.name}")
        expression_terms.append(epsilon_sym)
        # get bidirected edges
        for u, v in graph.undirected.edges(node):
            u, v = sorted([u, v])
            temp_gamma_sym = sy.Symbol(f"gamma_{u}_<->{v}")
            expression_terms.append(temp_gamma_sym)
        expression = sum(expression_terms)
        equations[node_sym] = expression
    return equations


def get_symbols_from_bi_edges(graph: NxMixedGraph) -> dict[tuple[Variable, Variable], sy.Symbol]:
    """Get symbols from bidirectional edges in graph."""
    symbol_dict = {}
    for u, v in graph.undirected.edges():
        u, v = sorted([str(u), str(v)])
        symbol_dict[(u, v)] = sy.Symbol(f"gamma_{u}_<->{v}")
    return symbol_dict


def get_symbols_from_di_edges(graph: NxMixedGraph) -> dict[tuple[Variable, Variable], sy.Symbol]:
    """Get symbols from directional edges in graph."""
    # for u,v in G.directed.edges():
    #     sy.Symbol(f"beta_{u.name}_->{v.name}")
    return {
        (str(u), str(v)): sy.Symbol(f"beta_{u.name}_->{v.name}") for u, v in graph.directed.edges()
    }


def get_symbols_from_nodes(graph: NxMixedGraph) -> dict[Variable, sy.Symbol]:
    """Get symbols from nodes in graph."""
    return {str(node): sy.Symbol(f"epsilon_{node.name}") for node in graph.nodes()}


def evaluate_lscm(
    lscm: dict[sy.Symbol, sy.Expr], params: dict[sy.Symbol, float]
) -> dict[sy.Symbol, sy.core.numbers.Float]:
    """Given an lscm, assign values to the parameters and return variable assignments dictionary."""
    # solve set of simulateous linear equations in sympy
    eqns = [sy.Eq(lhs.subs(params), rhs.subs(params)) for lhs, rhs in lscm.items()]
    return sy.solve(eqns, list(lscm), rational=False)


# todo: lscm to graph


def convert_to_latex(equations_dict: dict[sy.Symbol, sy.Expr]) -> str:
    """Convert equations into latex equations."""
    latex_equations = []
    for lhs, rhs in equations_dict.items():
        equation_latex = r"$$" + sy.latex(lhs) + " = " + sy.latex(rhs) + r"$$"
        latex_equations.append(equation_latex)
    eqn_array = "\n ".join(latex_equations)
    return eqn_array


def convert_to_eqn_array_latex(equations_dict: dict[sy.Symbol, sy.Expr]) -> str:
    """Convert equations into latex equation array."""
    latex_equations = []
    for lhs, rhs in equations_dict.items():
        equation_latex = sy.latex(lhs) + " &=& " + sy.latex(rhs)
        latex_equations.append(equation_latex)
    eqn_array = r"$$ \begin{array}{rcl}" + r"\\ ".join(latex_equations) + r"\end{array}$$"
    return eqn_array


def mixed_graph_to_pgmpy(graph: NxMixedGraph) -> BayesianNetwork:
    """Convert a mixed graph to an equivalent :class:`pgmpy.BayesianNetwork`."""
    # Get all directed edges with node names
    edges = [(u.name, v.name) for u, v in graph.directed.edges() if u != v]

    # Initialize set for latent variables
    latents = set()

    # Handle undirected edges by introducing latent variables
    for u, v in graph.undirected.edges():
        if u != v:  # no self loops
            latent = f"U_{u.name}_{v.name}"
            latents.add(latent)
            edges.append((latent, u.name))
            edges.append((latent, v.name))

    # Initialize the BayesianNetwork with edges
    model = BayesianNetwork(ebunch=edges, latents=latents)

    # Add all original nodes to the BayesianNetwork
    for node in graph.directed.nodes():
        if node.name not in model.nodes():
            model.add_node(node.name)

    return model


def simulate_lscm(
    graph: NxMixedGraph,
    node_generators: dict[Variable, partial],
    edge_weights: dict[tuple[Variable, Variable], float],
    n_samples: int,
) -> pd.DataFrame:
    """Generate N lscm samples based on node generators and edge weights."""
    assert nx.is_directed_acyclic_graph(graph.directed)  # noqa: S101

    linear_scm = LinearSCM(graph, generators=node_generators, weights=edge_weights)
    results = {
        trial: {variable.name: values for variable, values in linear_scm.trial().items()}
        for trial in range(n_samples)
    }
    rv = pd.DataFrame(results).T
    return rv


def calibrate_lscm(
    graph: NxMixedGraph, data: pd.DataFrame
) -> dict[tuple[Variable, Variable], float]:
    """Estimate parameter values for a linear SCM using single door criterion."""
    rv = {}

    assert nx.is_directed_acyclic_graph(graph.directed)  # noqa: S101

    for source, target in graph.directed.edges():
        temp_graph = graph.copy()
        temp_graph.directed.remove_edge(source, target)  # remove edge from source to target
        pgmpy_graph = mixed_graph_to_pgmpy(temp_graph)
        try:
            raw_adjustment_sets = pgmpy_graph.minimal_dseparator(source.name, target.name)
            if raw_adjustment_sets is None:
                # There are no valid adjustment sets.
                continue
            # Ensure we have a set of frozensets, with each frozenset containing variable names
            adjustment_sets = {
                (
                    frozenset([adjustment_set])
                    if isinstance(adjustment_set, str)
                    else frozenset(adjustment_set)
                )
                for adjustment_set in raw_adjustment_sets
            }
        except ValueError:
            # There are no valid adjustment sets.
            continue

        if not adjustment_sets:
            # There is a valid adjustment set, and it is the empty set, so just regress the target on the source.
            adjustment_sets = {frozenset()}

        coefficients = []
        for adjustment_set in adjustment_sets:
            # Ensure adjustment_set is a set before performing the union operation.
            variables = sorted(set(adjustment_set) | {source.name})
            idx = variables.index(source.name)
            model = LinearRegression()
            model.fit(data[variables], data[target.name])
            coefficients.append(model.coef_[idx])

        rv[source, target] = fmean(coefficients)

    return rv


def intervene_on_lscm(
    original_graph: NxMixedGraph,
    intervention_node: tuple[Variable, float],
    original_node_generators: dict[Variable, partial],
    original_edge_weights: dict[tuple[Variable, Variable], float],
) -> LinearSCM:
    """Intervenes on lscm graph by removing incoming edges to the intervention node."""
    assert nx.is_directed_acyclic_graph(original_graph.directed)  # noqa: S101

    # copy graph
    intervened_graph = original_graph.copy()

    # create intervened graph
    incoming_edges = list(intervened_graph.directed.in_edges(intervention_node[0]))
    intervened_graph.directed.remove_edges_from(incoming_edges)

    # adjust node generators ()
    intervened_node_generators = original_node_generators.copy()
    intervened_node_generators[intervention_node[0]] = partial(
        lambda *args, **kwargs: intervention_node[1]
    )  # return fixed value

    # update edge_weights to remove edges
    intervened_edge_weights = original_edge_weights.copy()
    for edge in incoming_edges:
        if edge in intervened_edge_weights:
            del intervened_edge_weights[edge]

    # create lscm
    intervened_lscm = LinearSCM(
        intervened_graph, generators=intervened_node_generators, weights=intervened_edge_weights
    )

    return intervened_lscm


def compute_average_treatment_effect(
    treatment1: pd.DataFrame, treatment2: pd.DataFrame, outcome_variables: list[Variable]
) -> dict[Variable, float]:
    """Compute the average treatment effect (i.e., E[Y1-Y0])."""
    avg_treatment_effect_dict = {}
    for v in outcome_variables:
        avg_untreated = treatment1.loc[:, str(v)].mean()
        avg_treated = treatment2.loc[:, str(v)].mean()
        avg_treatment_effect = avg_treated - avg_untreated
        avg_treatment_effect_dict[v] = avg_treatment_effect
    return avg_treatment_effect_dict
