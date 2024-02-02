"""Structural causal models."""
import re
import os
import pydot
import networkx as nx
import sympy as sy
from networkx.drawing.nx_pydot import from_pydot


from y0.dsl import Variable
from y0.graph import NxMixedGraph

def dagitty_to_dot(daggity_string: str) -> str:
    """Convert from dagitty format to DOT format.

    Modified from dowhy: https://www.pywhy.org/dowhy/v0.11.1/_modules/dowhy/utils/graph_operations.html#daggity_to_dot
    Converts the input daggity_string to valid DOT graph format.

    :param daggity_string: Output graph from Daggity site
    :returns: DOT string
    """
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
    )  # Remove bb line with four numbers and optional trailing semicolon
    graph = re.sub(r"\s+", " ", graph).strip()  # Trim all extra spaces
    return graph


def read_dag_file(file_path:str) -> str | None:
    """Reads the contents of a .dag file and returns it as a multiline string."""
    try:
        with open(file_path, "r") as file:
            return file.read()
    except IOError as e:
        print(f"Error reading file: {e}")
        return None
    
def dagitty_to_mixed_graph(dagitty_input:str, str_var_name:bool=False) -> NxMixedGraph:
    """Converts a string in dagitty (.dag) to NxMixedGraph."""

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
    assert isinstance(mixed_graph, NxMixedGraph)
    return mixed_graph


def dagitty_to_digraph(dagitty_input: str) -> nx.DiGraph:
    """Converts a string in dagitty (.dag) to NX DiGraph."""

    # Check if the input is a file path
    if os.path.isfile(dagitty_input):
        dagitty_graph_str = read_dag_file(dagitty_input)
    else:
        dagitty_graph_str = dagitty_input

    # .dag string -> DOT -> networkx -> NxMixedGraph
    dot_graph_string = dagitty_to_dot(dagitty_graph_str)
    dot_graph = pydot.graph_from_dot_data(dot_graph_string)[0]
    nx_graph = from_pydot(dot_graph)
    assert isinstance(nx_graph, nx.DiGraph)
    return nx_graph


def generate_LSCM_from_DAG(G: nx.DiGraph) -> dict[sy.Symbol, sy.Expr]:
    """Generates a linear structural causal model from a directed acycle graph (networkx DAG)."""
    assert nx.is_directed_acyclic_graph(G), "Not a DAG"  # check input DAG
    equations = {}
    sorted_nodes = list(nx.topological_sort(G))
    for node in sorted_nodes:
        node_sym = sy.Symbol(node)
        expression_terms = []
        parents = list(G.predecessors(node))
        for parent in parents:
            beta_sym = sy.Symbol(f"beta_{parent}_->{node}")
            parent_sym = sy.Symbol(f"{parent}")
            expression_terms.append(beta_sym * parent_sym)
        epsilon_sym = sy.Symbol(f"epsilon_{node}")
        expression_terms.append(epsilon_sym)
        expression = sum(expression_terms)
        equations[node_sym] = expression
    return equations


def generate_LSCM_from_mixed_graph(G: NxMixedGraph) -> dict[sy.Symbol, sy.Expr]:
    """Generates a linear structural causal model from a mixed directed and bidirected graph (y0 NxMixedGraph)."""
    assert nx.is_directed_acyclic_graph(G.directed), "Not a DAG"  # check input DAG
    equations = {}
    sorted_nodes = list(nx.topological_sort(G.directed))
    for node in sorted_nodes:
        node_sym = sy.Symbol(node.name)
        expression_terms = []
        parents = list(G.directed.predecessors(node))
        for parent in parents:
            beta_sym = sy.Symbol(f"beta_{parent.name}_->{node.name}")
            parent_sym = sy.Symbol(f"{parent.name}")
            expression_terms.append(beta_sym * parent_sym)
        epsilon_sym = sy.Symbol(f"epsilon_{node.name}")
        expression_terms.append(epsilon_sym)
        # get bidirected edges
        for u, v in G.undirected.edges(node):
            u, v = sorted([u, v])
            temp_gamma_sym = sy.Symbol(f"gamma_{u}_<->{v}")
            expression_terms.append(temp_gamma_sym)
        expression = sum(expression_terms)
        equations[node_sym] = expression
    return equations


def get_symbols_from_bi_edges(G:NxMixedGraph) -> dict[tuple[Variable, Variable], sy.Symbol]:
    """Gets symbols from bidirectional edges in graph."""
    symbol_dict = {}
    for u, v in G.undirected.edges():
        u, v = sorted([str(u), str(v)]) 
        symbol_dict[(u, v)] = sy.Symbol(f"gamma_{u}_<->{v}")
    return symbol_dict

    # return {(u,v):sy.Symbol(f'gamma_{u}_<->{v}') if str(u)>str(v) else sy.Symbol(f'gamma_{u}_<->{v}') for u,v in G.undirected.edges()}


def get_symbols_from_di_edges(G:NxMixedGraph) -> dict[tuple[Variable, Variable], sy.Symbol]:
    """Gets symbols from directional edges in graph."""
    # for u,v in G.directed.edges():
    #     sy.Symbol(f"beta_{u.name}_->{v.name}")
    return {(u, v): sy.Symbol(f"beta_{u.name}_->{v.name}") for u, v in G.directed.edges()}


def get_symbols_from_nodes(G:NxMixedGraph) -> dict[Variable, sy.Symbol]:
    """Gets symbols from nodes in graph."""
    return {node: sy.Symbol(f"epsilon_{node.name}") for node in G.nodes()}


def evaluate_LSCM(
    LSCM: dict[sy.Symbol, sy.Expr], params: dict[sy.Symbol, float]
) -> dict[sy.Symbol, sy.core.numbers.Rational]:
    """Given an LSCM, assign values to the parameters (i.e. beta, epsilon, gamma terms), and return variable assignments dictionary."""
    # solve set of simulateous linear equations in sympy
    eqns = [sy.Eq(lhs.subs(params), rhs.subs(params)) for lhs, rhs in LSCM.items()]
    print(eqns)
    return sy.solve(eqns, list(LSCM))


# todo: LSCM to graph


def convert_to_latex(equations_dict: dict[sy.Symbol, sy.Expr]) -> str:
    """Converts equations into latex equations."""
    latex_equations = []
    for lhs, rhs in equations_dict.items():
        equation_latex = r"$$" + sy.latex(lhs) + " = " + sy.latex(rhs) + r"$$"
        latex_equations.append(equation_latex)
    eqn_array = "\n ".join(latex_equations)
    return eqn_array


def convert_to_eqn_array_latex(equations_dict: dict[sy.Symbol, sy.Expr]) -> str:
    """Converts equations into latex equation array."""
    latex_equations = []
    for lhs, rhs in equations_dict.items():
        equation_latex = sy.latex(lhs) + " &=& " + sy.latex(rhs)
        latex_equations.append(equation_latex)
    eqn_array = r"$$ \begin{array}{rcl}" + r"\\ ".join(latex_equations) + r"\end{array}$$"
    return eqn_array


def generate_synthetic_data_from_LSCM():
    """"""
    raise NotImplementedError


def regress_LSCM():
    """"""
    raise NotImplementedError