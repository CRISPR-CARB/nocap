"""Structural causal models."""

import os
import re

import networkx as nx
import numpy as np
import pgmpy.inference.CausalInference as CausalInference
import plotly.graph_objects as go
import pydot
import sympy as sy
from networkx.drawing.nx_pydot import from_pydot
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import DiscreteBayesianNetwork, LinearGaussianBayesianNetwork
from y0.dsl import Variable
from y0.graph import NxMixedGraph


def dagitty_to_dot(daggity_string: str | None) -> str:
    """Convert from dagitty format to DOT format.

    Modified from dowhy: https://www.pywhy.org/dowhy/v0.11.1/_modules/dowhy/utils/graph_operations.html#daggity_to_dot
    Converts the input daggity_string to valid DOT graph format.
    """  # noqa: DAR101 DAR201 DAR401
    if type(daggity_string) is str:
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


def dagitty_to_mixed_graph(
    dagitty_input: str, str_var_name: bool = False
) -> NxMixedGraph:
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


def get_symbols_from_bi_edges(
    graph: NxMixedGraph,
) -> dict[tuple[Variable, Variable], sy.Symbol]:
    """Get symbols from bidirectional edges in graph."""
    symbol_dict = {}
    for u, v in graph.undirected.edges():
        u, v = sorted([str(u), str(v)])
        symbol_dict[(u, v)] = sy.Symbol(f"gamma_{u}_<->{v}")
    return symbol_dict


def get_symbols_from_di_edges(
    graph: NxMixedGraph,
) -> dict[tuple[Variable, Variable], sy.Symbol]:
    """Get symbols from directional edges in graph."""
    return {
        (str(u), str(v)): sy.Symbol(f"beta_{u.name}_->{v.name}")
        for u, v in graph.directed.edges()
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
    eqn_array = (
        r"$$ \begin{array}{rcl}" + r"\\ ".join(latex_equations) + r"\end{array}$$"
    )
    return eqn_array


def plot_interactive_lscm_graph(lscm: dict[sy.Symbol, sy.Expr]):
    """Create an interactive Plotly graph where hovering over nodes reveals full LaTeX equations."""
    # Create graph from LSCM
    graph = nx.DiGraph()

    for node_sym in lscm.keys():
        node_name = str(node_sym)
        graph.add_node(node_name)

        # Parse expression to identify relationships
        expression = lscm[node_sym]
        for term in expression.as_ordered_terms():
            if term.has(sy.Symbol):
                for sym in term.atoms(sy.Symbol):
                    parent_name = str(sym)
                    if parent_name.startswith("beta_"):
                        parent_node = parent_name.split("_->")[0].replace("beta_", "")
                        graph.add_edge(parent_node, node_name)

    # Generate node positions using spring layout
    pos = nx.spring_layout(graph, k=2, seed=42)

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        customdata=[],
        mode="markers+text",
        hoverinfo="text",
        marker=dict(size=20, color="lightblue"),
        textposition="bottom center",
    )

    edge_trace = go.Scatter(x=[], y=[], mode="lines", line=dict(width=2, color="grey"))

    # Populate node_trace and edge_trace
    for node_name in graph.nodes():
        x, y = pos[node_name]
        node_trace["x"] += (x,)
        node_trace["y"] += (y,)
        node_trace["text"] += (node_name,)  # Show plain text as node label
        node_expr = lscm[sy.Symbol(node_name)]
        node_trace["customdata"] += (
            f"${sy.latex(node_expr)}$",
        )  # Format equation with LaTeX for hover data

    for edge in graph.edges():
        start_pos = pos[edge[0]]
        end_pos = pos[edge[1]]
        edge_trace["x"] += (start_pos[0], end_pos[0], None)
        edge_trace["y"] += (start_pos[1], end_pos[1], None)

    # Create plot layout
    layout = go.Layout(
        title="Interactive LSCM Graph",
        hovermode="closest",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    # Plot figure
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)

    # Customize hovertemplate for displaying rendered LaTeX equations
    fig.update_traces(
        hovertemplate="<b>%{text}</b><br>%{customdata}"
    )  # Render customdata as LaTeX

    fig.show()


def create_lgbn_from_dag(dag):
    """Create a Linear Gaussian Bayesian Network from a directed acyclic graph (DAG)."""
    # Todo: add test
    model = LinearGaussianBayesianNetwork(dag)

    for node in dag.nodes():
        parents = list(dag.predecessors(node))
        num_parents = len(parents)
        if num_parents == 0:
            # If the node has no parents, it is an independent variable
            cpd = LinearGaussianCPD(variable=node, beta=np.array([0]), std=1)
        else:
            # If the node has parents, create a CPD with random coefficients
            beta = np.random.rand(num_parents + 1)  # Including intercept
            cpd = LinearGaussianCPD(variable=node, beta=beta, std=1, evidence=parents)

        model.add_cpds(cpd)

    if model.check_model() is not True:
        raise ValueError("The model is not valid. Please check the CPDs.")
    return model


def simulate_data_with_outliers(
    nocap_model,
    backend="pgmpy",
    num_samples=1000,
    outlier_fraction=0.01,
    outlier_magnitude=10,
    seed=42,
):
    """Simulate data from a structural causal model with outliers."""
    # Todo: add test

    np.random.seed(seed)
    if backend == "pgmpy":
        assert type(nocap_model) is nx.DiGraph, (
            "Model must be a networkx DiGraph for pgmpy backend"
        )
        model = create_lgbn_from_dag(nocap_model)
        simulated_data = model.simulate(n=num_samples, seed=seed)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # Apply non-negative constraint
    simulated_data[simulated_data < 0] = 0

    # Outliers introduction
    num_outliers = int(outlier_fraction * num_samples)
    outlier_indices = np.random.choice(
        simulated_data.index, num_outliers, replace=False
    )

    # Assume outlier adds an arbitrary large value or multiplies by a high factor
    simulated_data.loc[outlier_indices] *= outlier_magnitude

    return simulated_data


def fit_model(
    nocap_model,
    data,
    backend="pgmpy",
    method="mle",
):
    """Fit a model to the data using the specified backend."""
    # Todo: add test
    if backend == "pgmpy":
        assert type(nocap_model) is nx.DiGraph, (
            "Model must be a networkx DiGraph for pgmpy backend"
        )
        model = create_lgbn_from_dag(nocap_model)
        model.fit(data, method=method)
        return model
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def estimate_ate(nocap_model, data, X, Y, backend="pgmpy"):
    """Estimate the Average Treatment Effect (ATE) using the specified backend."""
    # Todo: add test
    if backend == "pgmpy":
        assert type(nocap_model) is nx.DiGraph, (
            "Model must be a networkx DiGraph for pgmpy backend"
        )
        model = DiscreteBayesianNetwork(nocap_model)
        inference = CausalInference(model)
        ate = inference.estimate_ate(X, Y, data)

    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return ate
