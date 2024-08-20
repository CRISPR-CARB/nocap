"""Structural causal models."""

import re
import warnings
from typing import Dict, List, Tuple, Union

# import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import sympy as sy

# from causalgraphicalmodels import CausalGraphicalModel


def parse_regulation_file(file_path: str) -> nx.DiGraph:
    """Parse a biocyc network regulation text file and converts it into a directed graph."""
    # Regulation text file should have the following format:
    # '#' for comments.
    # All regulators have a "*" after their name.
    # Top level (master) regulators are not indented.
    # descendants of the regulator are indented and spaced on the subsequent line.
    # The regulatees are prefixed with a '+', '-', or '+/-' for polarity.

    # Create a directed graph to represent the network
    graph = nx.DiGraph()

    # Open and read the file line by line
    with open(file_path, "r") as file:
        current_regulator = None  # Initialize the current regulator variable
        for line in file:
            # Ignore lines starting with '#' or that are empty
            if line.startswith("#") or not line.strip():
                continue

            # Remove trailing whitespace from the line
            line = line.rstrip()

            # Check if a line does not start with an indent and also ends with an asterisk
            if not line.startswith("  ") and line.endswith("*"):
                # Remove the asterisk and set that as the current_regulator
                current_regulator = line[:-1].strip()
                # Make first letter lowercase (expected format for genes)
                current_regulator = current_regulator[0].lower() + current_regulator[1:]
                # If the current regulator is not already in the graph, add it
                if current_regulator not in graph:
                    graph.add_node(current_regulator)
            elif current_regulator and line.startswith("  "):
                # Process the regulatees
                regulatees = line.strip().split()
                for regulatee in regulatees:
                    # Determine the polarity and adjust slicing accordingly
                    if regulatee.startswith("+/-"):
                        polarity = "+/-"
                        gene = regulatee[3:]
                    elif regulatee.startswith("+") or regulatee.startswith("-"):
                        polarity = regulatee[0]
                        gene = regulatee[1:]
                    else:
                        warnings.warn(f"regulatee: {regulatee} has no polarity", stacklevel=2)
                        polarity = ""
                        gene = regulatee

                    # Remove the trailing asterisk if present
                    gene = gene.rstrip("*")
                    # Make first letter lowercase
                    gene = gene[0].lower() + gene[1:]

                    # If the regulatee is not in the graph, add it
                    if gene not in graph:
                        graph.add_node(gene)
                    # Add an edge between the current_regulator and the regulatee with the polarity as an attribute
                    graph.add_edge(current_regulator, gene, polarity=polarity)
    return graph


# def convert_to_acyclic_graph(graph):
#     """Converts a (cyclic) directed graph into an acyclic directed graph by removing edges and disconnected nodes."""
#     G = graph.copy()

#     # Remove self-loops
#     self_loops = list(nx.selfloop_edges(G))
#     G.remove_edges_from(self_loops)

#     # Remove nodes that became disconnected after removing self-loops
#     for u, v in self_loops:
#         if G.degree(u) == 0:
#             G.remove_node(u)

#     # If the graph is still cyclic, find and break the cycles
#     if not nx.is_directed_acyclic_graph(G):
#         cycles = list(nx.simple_cycles(G))
#         for cycle in cycles:
#             # Remove one edge from the cycle
#             G.remove_edge(cycle[-1], cycle[0])
#             # Remove nodes that became disconnected after removing the edge
#             if G.degree(cycle[-1]) == 0:
#                 G.remove_node(cycle[-1])
#             if G.degree(cycle[0]) == 0:
#                 G.remove_node(cycle[0])

#     return G


def convert_to_acyclic_graph(graph: nx.DiGraph, target_node: str) -> nx.DiGraph:
    """Convert a (cyclic) directed graph into an acyclic directed graph by removing edges and disconnected nodes."""
    graph = graph.copy()

    # Get all descendants of the target node
    descendants = nx.descendants(graph, target_node)

    # Find all edges in shortest paths from the target node to its descendants
    edges_in_paths = set()
    for descendant in descendants:
        path = nx.shortest_path(graph, target_node, descendant)
        edges_in_path = list(zip(path, path[1:]))
        edges_in_paths.update(edges_in_path)

    # Remove self-loops
    self_loops = list(nx.selfloop_edges(graph))
    graph.remove_edges_from(self_loops)

    # Remove nodes that became disconnected after removing self-loops
    for u, _v in self_loops:
        if graph.degree(u) == 0:
            graph.remove_node(u)

    # While the graph is cyclic, find and break the cycles
    while not nx.is_directed_acyclic_graph(graph):
        try:
            cycle_edges = nx.find_cycle(graph, orientation="original")
        except nx.exception.NetworkXNoCycle:
            break  # No cycle found.

        # Remove an edge from the cycle that's not part of a path from the target node to a descendant
        for u, v, _ in cycle_edges:
            if (u, v) not in edges_in_paths:
                graph.remove_edge(u, v)

                # Remove nodes that became disconnected after removing the edge
                if graph.degree(u) == 0:
                    graph.remove_node(u)
                if graph.degree(v) == 0:
                    graph.remove_node(v)
                break  # Break the cycle processing loop once an edge has been removed

    return graph


# def draw_network(G):
#     """Draw network ."""
#     pos = nx.spring_layout(G)  # Position nodes using Fruchterman-Reingold force-directed algorithm
#     nx.draw(G, pos, with_labels=True, node_size=2, node_color="lightblue", edge_color="gray")
#     edge_labels = nx.get_edge_attributes(G, "polarity")
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
#     plt.show()


def get_subgraph_from_nodes(
    graph: nx.DiGraph, node_list: List[str], descendants_only: bool = False
) -> nx.DiGraph:
    """Get subgraph of a graph from a node list."""
    # Initialize a set for all the nodes in the subgraph
    subgraph_nodes = set()

    # For each node in the list, find all descendants
    for node in node_list:
        if node in graph:
            subgraph_nodes.add(node)
            descendants = nx.descendants(graph, node)
            subgraph_nodes.update(descendants)
            if not descendants_only:
                # For each descendant, add its ancestors
                for desc in descendants:
                    ancestors = nx.ancestors(graph, desc)
                    subgraph_nodes.update(ancestors)
        else:
            warnings.warn(f"{node} not found in graph.", stacklevel=2)

    # Create the subgraph from the full graph using the nodes in the set
    subgraph = graph.subgraph(subgraph_nodes)
    return subgraph


# def generate_subnetwork_with_backdoor_adjustment(graph, target_gene):
#     """Generates a subnetwork with backdoor adjustment."""
#     # Create a new directed graph for the subnetwork
#     subnetwork = nx.DiGraph()

#     # Step 1: Identify the target gene and its descendants
#     descendants = nx.descendants(graph, target_gene)
#     # Include the target gene itself in the set
#     descendants.add(target_gene)

#     # Step 2: Determine the backdoor adjustment set
#     # Convert the networkx graph to a CausalGraphicalModel
#     cgm = CausalGraphicalModel(nodes=list(graph.nodes), edges=[(u, v) for u, v in graph.edges])
#     # Get the backdoor adjustment set for the target gene
#     backdoor_adjustment_set = set()
#     for node in graph.nodes:
#         if node != target_gene:
#             for adjustment_set in cgm.get_all_backdoor_adjustment_sets(node, target_gene):
#                 backdoor_adjustment_set.update(adjustment_set)

#     # Step 3: Construct the subnetwork
#     # Add nodes for the target gene, its descendants, and the backdoor adjustment set
#     subnetwork.add_nodes_from(descendants.union(backdoor_adjustment_set))

#     # Add edges between the nodes in the subnetwork that exist in the original graph
#     for u, v in graph.edges():
#         if u in subnetwork.nodes() and v in subnetwork.nodes():
#             subnetwork.add_edge(u, v, polarity=graph[u][v]["polarity"])
#     return subnetwork


def generate_hill_equations(dag: nx.DiGraph, activation_probability: float = 0.5, seed: int=0) -> Tuple[
    Dict[str, sy.Basic],
    Dict[str, sy.Symbol],
    Dict[Tuple[str, str], sy.Symbol],
    Dict[Tuple[str, str], sy.Symbol],
    Dict[Tuple[str, str], sy.Symbol],
    Dict[str, sy.Symbol],
]:
    """
    Generate symbolic Hill-like reaction dynamics equations for a gene regulatory network.

    :param dag: The directed acyclic graph representing the gene regulatory network.
    :type dag: nx.DiGraph
    :param activation_probability: Probability of activation for edges with '+/-' polarity. Defaults to 0.5.
    :type activation_probability: float
    :returns: A tuple containing:
        - dict: Symbolic equations for the production rates of genes.
        - dict: Symbols for basal rates.
        - dict: Symbols for maximum contributions.
        - dict: Symbols for Hill coefficients.
        - dict: Symbols for half responses.
        - dict: Symbols for gene expressions.
    :rtype: tuple
    :raises ValueError: If an unknown polarity is encountered.

    :seealso: `https://doi.org/10.1016/j.cels.2020.08.003`

    .. note::
        The basal rate = 0 for non-master regulators,
        and the production rate is equal to the basal rate for master regulators.
    """
    np.random.seed(seed)
    # Initialize a dictionary to hold the equations
    equations = {}

    # Define symbols for basal rates, maximum contributions, Hill coefficients, and half responses
    basal_rates = {gene: sy.Symbol(f"b_{gene}") for gene in dag.nodes()}
    max_contributions = {
        (regulator, gene): sy.Symbol(f"K_{regulator}_{gene}") for regulator, gene in dag.edges()
    }
    hill_coefficients = {
        (regulator, gene): sy.Symbol(f"n_{regulator}_{gene}") for regulator, gene in dag.edges()
    }
    half_responses = {
        (regulator, gene): sy.Symbol(f"h_{regulator}_{gene}") for regulator, gene in dag.edges()
    }

    # Define gene expression symbols
    gene_expressions = {gene: sy.Symbol(f"x_{gene}") for gene in dag.nodes()}

    # Construct the equations for each gene
    for gene in dag.nodes():
        # If gene is a master regulator, its production rate is its basal rate (SERGIO assumption)
        if dag.in_degree(gene) == 0:
            equations[gene] = basal_rates[gene]
            continue

        # Sum contributions from each regulator using Hill functions
        production_rate = basal_rates[gene]  # Start with the basal rate
        for regulator in dag.predecessors(gene):
            x = gene_expressions[regulator]
            k = max_contributions[(regulator, gene)]
            n = hill_coefficients[(regulator, gene)]
            h = half_responses[(regulator, gene)]
            polarity = dag[regulator][gene]["polarity"]

            # double check these eqns
            # k should be exported as negative if repressor
            if polarity == "+":  # Activation
                pij = k * x**n / (h**n + x**n)
            elif polarity == "-":  # Repression
                pij = k * (1 - x**n) / (h**n + x**n)
            elif polarity == "+/-":  # Randomly assigned polarity based on user-defined probability
                if np.random.rand() < activation_probability:
                    pij = k * x**n / (h**n + x**n)  # Activation
                else:
                    pij = k * (1 - x**n) / (h**n + x**n)  # Repression
            else:
                raise ValueError(f"Unknown polarity '{polarity}' for edge {regulator} -> {gene}")
            production_rate += pij

        equations[gene] = production_rate

    return (
        equations,
        basal_rates,
        max_contributions,
        hill_coefficients,
        half_responses,
        gene_expressions,
    )


def generate_hill_equations_for_n_replicas(
    dag: nx.DiGraph, activation_probability: float, n: int, seed: list[int],
) -> Dict[
    int,
    Tuple[
        Dict[str, sy.Basic],
        Dict[str, sy.Symbol],
        Dict[Tuple[str, str], sy.Symbol],
        Dict[Tuple[str, str], sy.Symbol],
        Dict[Tuple[str, str], sy.Symbol],
        Dict[str, sy.Symbol],
    ],
]:
    """
    Generate multiple replicas of Hill-like reaction dynamics equations for a gene regulatory network.

    :param dag: The directed acyclic graph representing the gene regulatory network.
    :type dag: nx.DiGraph
    :param activation_probability: Probability of activation for edges with '+/-' polarity. Defaults to 0.5.
    :type activation_probability: float
    :param n: Number of replicas to generate. Defaults to 1.
    :type n: int
    :returns: A dictionary of N sets of symbolic equations for the production rates of genes and their parameters.
    :rtype: dict

    .. warning::
        Ensure that the replica ID is 0-indexed when using the replicas in the range(N).

    .. seealso::
        `generate_hill_equations` function for generating a single set of Hill-like reaction dynamics equations.

    .. note::
        The basal rate = 0 for non-master regulators,
        and the production rate is equal to the basal rate for master regulators.
    """
    
    warnings.warn("using replica in range(N) - ensure replica id is 0 index.", stacklevel=2)
    all_replica_equations = {}

    # Generate N replicas of equations
    for replica in range(n):
        # Call the original generate_hill_equations function
        (
            equations,
            basal_rates,
            max_contributions,
            hill_coefficients,
            half_responses,
            gene_expressions,
        ) = generate_hill_equations(dag, activation_probability, seed=seed[replica])

        # Store the equations and parameters for this replica
        all_replica_equations[replica] = (
            equations,
            basal_rates,
            max_contributions,
            hill_coefficients,
            half_responses,
            gene_expressions,
        )

    return all_replica_equations


def assign_random_parameter_values(
    basal_rates: Dict[str, sy.Symbol],
    max_contributions: Dict[Tuple[str, str], sy.Symbol],
    hill_coefficients: Dict[Tuple[str, str], sy.Symbol],
    half_responses: Dict[Tuple[str, str], sy.Symbol],
    param_distributions: Dict[str, Dict[str, float]],
    seed: int = 0,
) -> Dict[sy.Symbol, float]:
    """
    Assign values to the parameters of the Hill equations based on specified distributions.

    :param basal_rates: Dictionary of symbols for basal rates.
    :type basal_rates: dict
    :param max_contributions: Dictionary of symbols for maximum contributions.
    :type max_contributions: dict
    :param hill_coefficients: Dictionary of symbols for Hill coefficients.
    :type hill_coefficients: dict
    :param half_responses: Dictionary of symbols for half responses.
    :type half_responses: dict
    :param param_distributions: Dictionary with distribution settings for each parameter.
    :type param_distributions: dict
    :returns: Dictionary with symbols as keys and assigned values as values.
    :rtype: dict
    """
    np.random.seed(seed)
    parameter_values = {}

    # Assign values to basal rates
    for gene, br in basal_rates.items():
        if gene in param_distributions["master_regulators"]:
            a, b = param_distributions["b"]["a"], param_distributions["b"]["b"]
            parameter_values[br] = np.random.beta(a, b) * param_distributions["b"]["scale"]
            print(f'{br} {parameter_values[br]}')
        else:
            parameter_values[br] = 0

    # Assign values to maximum contributions
    min_val, max_val = param_distributions["K"]["min"], param_distributions["K"]["max"]
    for mc in max_contributions.values():
        parameter_values[mc] = np.random.uniform(min_val, max_val)

    # Assign random integer values to Hill coefficients within the specified range
    n_min, n_max = param_distributions["n"]["min"], param_distributions["n"]["max"]
    for hc in hill_coefficients.values():
        parameter_values[hc] = np.random.randint(
            n_min, n_max + 1
        )  # +1 because randint is exclusive on the high end

    # Assign values to half responses
    min_val, max_val = param_distributions["h"]["min"], param_distributions["h"]["max"]
    for hr in half_responses.values():
        parameter_values[hr] = np.random.uniform(min_val, max_val)

    return parameter_values


def assign_random_values_for_n_replicas(
    all_replica_equations: Dict[
        int,
        Tuple[
            Dict[str, sy.Basic],
            Dict[str, sy.Symbol],
            Dict[Tuple[str, str], sy.Symbol],
            Dict[Tuple[str, str], sy.Symbol],
            Dict[Tuple[str, str], sy.Symbol],
            Dict[str, sy.Symbol],
        ],
    ],
    param_distributions_by_replica: Dict[int, Dict[str, Dict[str, float]]],
    seed: List[int]
) -> Dict[int, Dict[sy.Symbol, float]]:
    """
    Assign random values to the parameters of Hill equations for all replicas based on specified distributions.

    :param all_replica_equations: Dictionary containing N sets of symbolic equations for the production rates of genes.
    :type all_replica_equations: dict
    :param param_distributions_by_replica: Dictionary where each key is a replica ID and the value is the associated
                                           param_distributions dictionary for that replica.
    :type param_distributions_by_replica: dict
    :returns: A dictionary containing N sets of assigned parameter values for the Hill equations.
    :rtype: dict
    """
    all_replica_parameter_values = {}

    # Iterate over each set of equations in all replicas
    for i, (replica, equations_info) in enumerate(all_replica_equations.items()):
        # Extract the parameter dictionaries from the equations_info tuple
        (
            equations,
            basal_rates,
            max_contributions,
            hill_coefficients,
            half_responses,
            gene_expressions,
        ) = equations_info

        # Retrieve the specific param_distributions for this replica
        param_distributions = param_distributions_by_replica[replica]

        # Assign parameter values for this set of equations
        parameter_values = assign_random_parameter_values(
            basal_rates, max_contributions, hill_coefficients, half_responses, param_distributions, seed[i]
        )
        # Store the assigned parameter values for this replica
        all_replica_parameter_values[replica] = parameter_values

    return all_replica_parameter_values


def create_node_to_idx_mapping(graph: nx.DiGraph) -> Dict[str, int]:
    """
    Create a mapping from gene names to indices for a given graph.

    :param graph: The directed acyclic graph representing the gene regulatory network.
    :type graph: nx.DiGraph
    :returns: Mapping from gene names to indices starting at 0.
    :rtype: dict
    """
    return {gene: idx for idx, gene in enumerate(graph.nodes())}


# def create_dataframes_for_sergio_inputs_from_one_replica(
#     dag: nx.DiGraph, parameter_values: Dict[sy.Symbol, float], node_to_idx: Dict[str, int]
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Create pandas DataFrames for SERGIO inputs from one replica of a gene regulatory network.

#     :param dag: The directed acyclic graph representing the gene regulatory network.
#     :type dag: nx.DiGraph
#     :param parameter_values: Dictionary (k:symbol,v:parameter values) for the Hill equations for one replica.
#     :type parameter_values: dict
#     :param node_to_idx: Mapping from gene names to indices.
#     :type node_to_idx: dict
#     :returns: Data frame for the targets and data frame for the master regulators.
#     :rtype: tuple
#     :raises KeyError: If a gene is not found in the node_to_idx mapping
#         or an expected key is not found in parameter_values.
#     """
#     # Determine the maximum number of regulators for any gene
#     max_regs = max(dag.in_degree(gene) for gene in dag.nodes())

#     # Initialize empty lists to store data
#     regs_data = []
#     targets_data = []

#     # Iterate over genes in the DAG
#     for gene in dag.nodes():
#         gene_idx = node_to_idx.get(gene)
#         if gene_idx is None:
#             raise KeyError(f"Gene {gene} not found in node_to_idx mapping")

#         # Check for master regulators
#         if dag.in_degree(gene) == 0:
#             b_key = sy.Symbol("b_" + gene)
#             if b_key in parameter_values:
#                 b_value = parameter_values[b_key]
#                 regs_data.append([gene_idx, b_value])
#             else:
#                 raise KeyError(f"Expected key {b_key} not found in parameter_values")

#         else:
#             # Target genes
#             regulators = list(dag.predecessors(gene))
#             row = [gene_idx, float(len(regulators))] + [None] * (3 * max_regs)
#             # row = [gene_idx, len(regulators)] + [None] * (3 * max_regs)
#             for i, reg in enumerate(regulators):
#                 reg_idx = node_to_idx[reg]
#                 k_key = sy.Symbol("K_" + reg + "_" + gene)
#                 n_key = sy.Symbol("n_" + reg + "_" + gene)

#                 k_value = parameter_values[k_key]
#                 coop_state = parameter_values[n_key]

#                 # If the regulator is a repressor, use -K_value
#                 if dag[reg][gene]["polarity"] == "-" and k_value > 0:
#                     k_value *= -1

#                 # row[2 + int(i)] = reg_idx
#                 # row[2 + int(max_regs) + int(i)] = k_value
#                 # row[2 + 2 * int(max_regs) + int(i)] = coop_state
#                 row[2 + int(i)] = float(reg_idx)
#                 row[2 + int(max_regs) + int(i)] = float(k_value)
#                 row[2 + 2 * int(max_regs) + int(i)] = float(coop_state)
#             targets_data.append(row)

#     # Create data frames
#     targets_df = pd.DataFrame(
#         targets_data,
#         columns=["Target Idx", "#regulators"]
#         + ["regIdx" + str(i + 1) for i in range(max_regs)]
#         + ["K" + str(i + 1) for i in range(max_regs)]
#         + ["coop_state" + str(i + 1) for i in range(max_regs)],
#     )
#     regs_df = pd.DataFrame(regs_data, columns=["Master regulator Idx", "production_rate"])

#     return targets_df, regs_df


# Define the type for elements in the row, which can be either int or float
RowElement = Union[int, float, None]


def create_dataframes_for_sergio_inputs_from_one_replica(
    dag: nx.DiGraph, parameter_values: Dict[sy.Symbol, float], node_to_idx: Dict[str, int]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create pandas DataFrames for SERGIO inputs from one replica of a gene regulatory network.

    :param dag: The directed acyclic graph representing the gene regulatory network.
    :type dag: nx.DiGraph
    :param parameter_values: Dictionary (k:symbol,v:parameter values) for the Hill equations for one replica.
    :type parameter_values: dict
    :param node_to_idx: Mapping from gene names to indices.
    :type node_to_idx: dict
    :returns: Data frame for the targets and data frame for the master regulators.
    :rtype: tuple
    :raises KeyError: If a gene is not found in the node_to_idx mapping or an expected key is not found in parameter_values.
    """
    # Determine the maximum number of regulators for any gene
    max_regs = max(dag.in_degree(gene) for gene in dag.nodes())

    # Initialize empty lists to store data
    regs_data: List[List[Union[int, float]]] = []
    targets_data: List[List[RowElement]] = []

    # Iterate over genes in the DAG
    for gene in dag.nodes():
        gene_idx = node_to_idx.get(gene)
        if gene_idx is None:
            raise KeyError(f"Gene {gene} not found in node_to_idx mapping")

        # Check for master regulators
        if dag.in_degree(gene) == 0:
            b_key = sy.Symbol("b_" + gene)
            if b_key in parameter_values:
                b_value = parameter_values[b_key]
                regs_data.append([gene_idx, b_value])
            else:
                raise KeyError(f"Expected key {b_key} not found in parameter_values")

        else:
            # Target genes
            regulators = list(dag.predecessors(gene))
            row: List[RowElement] = [gene_idx, len(regulators)] + [None] * (3 * max_regs)
            for i, reg in enumerate(regulators):
                reg_idx = node_to_idx[reg]
                k_key = sy.Symbol("K_" + reg + "_" + gene)
                n_key = sy.Symbol("n_" + reg + "_" + gene)

                k_value = parameter_values[k_key]
                coop_state = parameter_values[n_key]

                # If the regulator is a repressor, use -K_value
                if dag[reg][gene]["polarity"] == "-" and k_value > 0:
                    k_value *= -1

                row[2 + i] = reg_idx
                row[2 + max_regs + i] = k_value
                row[2 + 2 * max_regs + i] = coop_state
            targets_data.append(row)

    # Create data frames
    targets_df = pd.DataFrame(
        targets_data,
        columns=["Target Idx", "#regulators"]
        + ["regIdx" + str(i + 1) for i in range(max_regs)]
        + ["K" + str(i + 1) for i in range(max_regs)]
        + ["coop_state" + str(i + 1) for i in range(max_regs)],
    )
    regs_df = pd.DataFrame(regs_data, columns=["Master regulator Idx", "production_rate"])

    return targets_df, regs_df


def create_dataframes_for_sergio_inputs_from_n_replicas(
    dag: nx.DiGraph,
    all_parameter_values: Dict[int, Dict[sy.Symbol, float]],
    node_to_idx: Dict[str, int],
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Create lists of pandas DataFrames for SERGIO inputs from multiple replicas.

    :param dag: The directed acyclic graph representing the gene regulatory network.
    :type dag: nx.DiGraph
    :param all_parameter_values: List of parameter values dictionaries for the Hill equations for each replica.
    :type all_parameter_values: dict
    :param node_to_idx: Mapping from gene names to indices.
    :type node_to_idx: dict
    :returns: List of dataframes for the targets for each replica
        and list of data frames for the master regulators for each replica.
    :rtype: tuple
    """
    assert isinstance(dag, nx.DiGraph)  # noqa: S101
    assert isinstance(all_parameter_values, dict)  # noqa: S101
    assert isinstance(node_to_idx, dict)  # noqa: S101

    all_targets_dfs = []
    all_regs_dfs = []

    # Iterate over the parameter values for each replica
    for _, parameter_values in all_parameter_values.items():
        # Convert parameter keys to sympy Symbols if they are not already
        parameter_values_symbols = {
            sy.Symbol(k) if isinstance(k, str) else k: v for k, v in parameter_values.items()
        }
        targets_df, regs_df = create_dataframes_for_sergio_inputs_from_one_replica(
            dag, parameter_values_symbols, node_to_idx
        )
        all_targets_dfs.append(targets_df)
        all_regs_dfs.append(regs_df)

    return all_targets_dfs, all_regs_dfs


def merge_n_replica_dataframes_for_sergio_inputs(
    all_targets_dfs: List[pd.DataFrame],
    all_regs_dfs: List[pd.DataFrame],
    merge_type: str = "first_only",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge lists of pandas DataFrames for SERGIO inputs from multiple replicas based on the specified merge type.

    :param all_targets_dfs: List of data frames for the targets for each replica.
    :type all_targets_dfs: list
    :param all_regs_dfs: List of data frames for the master regulators for each replica.
    :type all_regs_dfs: list
    :param merge_type: Merge type with default "first_only".
        If "first_only", return the first elements of the target df and concatenates regulator production rates.
    :type merge_type: str
    :returns: Merged dataframe for the targets and
        merged dataframe for the master regulators with additional production rate columns.
    :rtype: tuple
    :raises NotImplementedError: If the merge type is not implemented.
    """
    if merge_type == "first_only":
        # Make a copy of the first master regulators DataFrame
        merged_regs_df = all_regs_dfs[0].copy()

        # Add additional columns for production rates from other replicas
        for idx, regs_df in enumerate(all_regs_dfs[1:], start=2):
            merged_regs_df[f"production_rate{idx}"] = regs_df["production_rate"]

        # Return the first DataFrame for targets and the merged DataFrame for master regulators
        return all_targets_dfs[0].copy(), merged_regs_df
    else:
        # Placeholder for other merge types, which can be implemented later
        raise NotImplementedError("Merge type not implemented.")


def write_input_files_for_sergio(
    targets_df: pd.DataFrame, regs_df: pd.DataFrame, filename_targets: str, filename_regs: str
) -> None:
    """
    Save the targets and regulators data frames to text files without the index.

    :param targets_df: Data frame for the targets.
    :type targets_df: pd.DataFrame
    :param regs_df: Data frame for the master regulators.
    :type regs_df: pd.DataFrame
    :param filename_targets: Filename for the targets text file.
    :type filename_targets: str
    :param filename_regs: Filename for the master regulators text file.
    :type filename_regs: str
    """
    # Convert entire dataframes to float
    targets_df = targets_df.astype(float)
    regs_df = regs_df.astype(float)

    # Convert NaNs to empty strings and save to CSV format
    targets_csv = targets_df.to_csv(index=False, header=False, na_rep="")
    regs_csv = regs_df.to_csv(index=False, header=False, na_rep="")

    # Remove unnecessary commas from CSV strings
    def _clean_csv_string(csv_string):
        """Remove trailing commas and multiple commas."""
        # Remove trailing commas at the end of each line and any repeating commas
        csv_string = re.sub(r",+\n", "\n", csv_string)  # Remove trailing commas at the end of lines
        csv_string = re.sub(r",+", ",", csv_string)  # Replace multiple commas with a single one
        return csv_string

    # Write the cleaned CSV strings to text files
    with open(filename_targets, "w") as f:
        f.write(_clean_csv_string(targets_csv))
    with open(filename_regs, "w") as f:
        f.write(_clean_csv_string(regs_csv))


def write_equation_info_to_file(
    hill_equations: Dict[str, sy.Basic],
    parameter_values: Dict[sy.Symbol, float],
    data_filename: str,
) -> None:
    """Write equations and parameter values as strings in a textfile."""
    # Convert Sympy expressions to strings for serialization
    hill_equations_str = {str(k): str(v) for k, v in hill_equations.items()}

    # Open a text file for writing
    with open(data_filename, "w") as text_file:
        # Write the hill equations
        text_file.write("Hill Equations:\n")
        for key, equation in hill_equations_str.items():
            text_file.write(f"{key}: {equation}\n")

        # Write a separator
        text_file.write("\nParameter Values:\n")

        # Write the parameter values
        for key, value in parameter_values.items():
            text_file.write(f"{key}: {value}\n")
