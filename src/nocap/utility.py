import networkx as nx
import matplotlib.pyplot as plt
from causalgraphicalmodels import CausalGraphicalModel
import sympy as sy
import numpy as np
import warnings
import pandas as pd
import re

def parse_regulation_file(file_path):
    "Parses a biocyc network regulation text file and converts it into a directed graph."
    # Regulation text file should have the following format:
    # '#' for comments.
    # All regulators have a "*" after their name.
    # Top level (master) regulators are not indented.
    # descendants of the regulator are indented and spaced on the subsequent line.
    # The regulatees are prefixed with a '+', '-', or '+/-' for polarity.

    # Create a directed graph to represent the network
    G = nx.DiGraph()

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
                if current_regulator not in G:
                    G.add_node(current_regulator)
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
                        warnings.warn(f"regulatee: {regulatee} has no polarity")
                        polarity = ""
                        gene = regulatee

                    # Remove the trailing asterisk if present
                    gene = gene.rstrip("*")
                    # Make first letter lowercase
                    gene = gene[0].lower() + gene[1:]

                    # If the regulatee is not in the graph, add it
                    if gene not in G:
                        G.add_node(gene)
                    # Add an edge between the current_regulator and the regulatee with the polarity as an attribute
                    G.add_edge(current_regulator, gene, polarity=polarity)
    return G


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


def convert_to_acyclic_graph_fancy(graph, target_node):
    """Converts a (cyclic) directed graph into an acyclic directed graph by removing edges and disconnected nodes."""
    G = graph.copy()

    # Get all descendants of the target node
    descendants = nx.descendants(G, target_node)

    # Find all edges in shortest paths from the target node to its descendants
    edges_in_paths = set()
    for descendant in descendants:
        path = nx.shortest_path(G, target_node, descendant)
        edges_in_path = list(zip(path, path[1:]))
        edges_in_paths.update(edges_in_path)

    # Remove self-loops
    self_loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(self_loops)

    # Remove nodes that became disconnected after removing self-loops
    for u, v in self_loops:
        if G.degree(u) == 0:
            G.remove_node(u)

    # While the graph is cyclic, find and break the cycles
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle_edges = nx.find_cycle(G, orientation="original")
        except nx.exception.NetworkXNoCycle:
            break  # No cycle found.

        # Remove an edge from the cycle that's not part of a path from the target node to a descendant
        for u, v, _ in cycle_edges:
            if (u, v) not in edges_in_paths:
                G.remove_edge(u, v)

                # Remove nodes that became disconnected after removing the edge
                if G.degree(u) == 0:
                    G.remove_node(u)
                if G.degree(v) == 0:
                    G.remove_node(v)
                break  # Break the cycle processing loop once an edge has been removed

    return G


def draw_network(G):
    """Placeholder function to draw a network."""
    pos = nx.spring_layout(G)  # Position nodes using Fruchterman-Reingold force-directed algorithm
    nx.draw(G, pos, with_labels=True, node_size=2, node_color="lightblue", edge_color="gray")
    edge_labels = nx.get_edge_attributes(G, "polarity")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()


def get_subgraph_from_nodes(G, node_list, descendants_only=False):
    """Get subgraph of a graph from a node list."""
    # Initialize a set for all the nodes in the subgraph
    subgraph_nodes = set()

    # For each node in the list, find all descendants
    for node in node_list:
        if node in G:
            subgraph_nodes.add(node)
            descendants = nx.descendants(G, node)
            subgraph_nodes.update(descendants)
            if not descendants_only:
                # For each descendant, add its ancestors
                for desc in descendants:
                    ancestors = nx.ancestors(G, desc)
                    subgraph_nodes.update(ancestors)
        else:
            warnings.warn(f"{node} not found in graph.")

    # Create the subgraph from the full graph using the nodes in the set
    subgraph = G.subgraph(subgraph_nodes)
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


def generate_hill_equations(dag, activation_probability=0.5):
    """
    Generate symbolic Hill-like reaction dynamics equations for a gene regulatory network.
    See:https://doi.org/10.1016/j.cels.2020.08.003
    Note that basal rate = 0 for non master regulators, and production rate = basal rate for master regulators.

    Args:
    dag (nx.DiGraph): The directed acyclic graph representing the gene regulatory network.

    Returns:
    dict: Symbolic equations for the production rates of genes.
    """
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
            K = max_contributions[(regulator, gene)]
            n = hill_coefficients[(regulator, gene)]
            h = half_responses[(regulator, gene)]
            polarity = dag[regulator][gene]["polarity"]

            # double check these eqns
            # k should be exported as negative if repressor
            if polarity == "+":  # Activation
                pij = K * x**n / (h**n + x**n)
            elif polarity == "-":  # Repression
                pij = K * (1 - x**n) / (h**n + x**n)
            elif polarity == "+/-":  # Randomly assigned polarity based on user-defined probability
                if np.random.rand() < activation_probability:
                    pij = K * x**n / (h**n + x**n)  # Activation
                else:
                    pij = K * (1 - x**n) / (h**n + x**n)  # Repression
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


def generate_hill_equations_for_n_replicas(dag, activation_probability=0.5, N=1):
    """
    Generate multiple replicas of Hill-like reaction dynamics equations for a gene regulatory network.

    Args:
    dag (nx.DiGraph): The directed acyclic graph representing the gene regulatory network.
    activation_probability (float): Probability of activation for edges with '+/-' polarity.
    N (int): Number of replicas to generate.

    Returns:
    dict: A dictionary containing N sets of symbolic equations for the production rates of genes and their parameters.
    """

    warnings.warn("using replica in range(N) - ensure replica id is 0 index.")
    all_replica_equations = {}

    # Generate N replicas of equations
    for replica in range(N):
        # Call the original generate_hill_equations function
        (
            equations,
            basal_rates,
            max_contributions,
            hill_coefficients,
            half_responses,
            gene_expressions,
        ) = generate_hill_equations(dag, activation_probability)

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
    basal_rates, max_contributions, hill_coefficients, half_responses, param_distributions
):
    """
    Assign values to the parameters of the Hill equations based on specified distributions.

    Args:
    basal_rates (dict): Dictionary of symbols for basal rates.
    max_contributions (dict): Dictionary of symbols for maximum contributions.
    hill_coefficients (dict): Dictionary of symbols for Hill coefficients.
    half_responses (dict): Dictionary of symbols for half responses.
    param_distributions (dict): Dictionary with distribution settings for each parameter.

    Returns:
    dict: Dictionary with symbols as keys and assigned values as values.
    """
    parameter_values = {}

    # Assign values to basal rates
    for gene, br in basal_rates.items():
        if gene in param_distributions["master_regulators"]:
            a, b = param_distributions["b"]["a"], param_distributions["b"]["b"]
            parameter_values[br] = np.random.beta(a, b) * param_distributions["b"]["scale"]
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


def assign_random_values_for_n_replicas(all_replica_equations, param_distributions_by_replica):
    """
    Assign random values to the parameters of Hill equations for all replicas based on specified distributions.

    Args:
    all_replica_equations (dict): Dictionary containing N sets of symbolic equations for the production rates of genes.
    param_distributions_by_replica (dict): Dictionary where each key is a replica ID and the value is the associated
                                           param_distributions dictionary for that replica.

    Returns:
    dict: A dictionary containing N sets of assigned parameter values for the Hill equations.
    """
    all_replica_parameter_values = {}

    # Iterate over each set of equations in all replicas
    for replica, equations_info in all_replica_equations.items():
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
            basal_rates, max_contributions, hill_coefficients, half_responses, param_distributions
        )
        # Store the assigned parameter values for this replica
        all_replica_parameter_values[replica] = parameter_values

    return all_replica_parameter_values


def create_node_to_idx_mapping(graph):
    """
    Create a mapping from gene names to indices for a given graph.

    Args:
    dag (nx.DiGraph): The directed acyclic graph representing the gene regulatory network.

    Returns:
    dict: Mapping from gene names to indices starting at 0.
    """
    return {gene: idx for idx, gene in enumerate(graph.nodes())}


def create_dataframes_for_sergio_inputs_from_one_replica(dag, parameter_values, node_to_idx):
    """
    Create pandas DataFrames for SERGIO inputs from one replica of a gene regulatory network.

    Args:
    dag (nx.DiGraph): The directed acyclic graph representing the gene regulatory network.
    parameter_values (dict): Dictionary (k:symbol,v:parameter values) for the Hill equations for one replica.
    node_to_idx (dict): Mapping from gene names to indices.

    Returns:
    pd.DataFrame: Data frame for the targets.
    pd.DataFrame: Data frame for the master regulators.
    """

    # Determine the maximum number of regulators for any gene
    max_regs = max(dag.in_degree(gene) for gene in dag.nodes())

    # Initialize empty lists to store data
    regs_data = []
    targets_data = []

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
            row = [gene_idx, len(regulators)] + [None] * (3 * max_regs)
            for i, reg in enumerate(regulators):
                reg_idx = node_to_idx[reg]
                K_key = sy.Symbol("K_" + reg + "_" + gene)
                n_key = sy.Symbol("n_" + reg + "_" + gene)

                K_value = parameter_values[K_key]
                coop_state = parameter_values[n_key]

                # If the regulator is a repressor, use -K_value
                if dag[reg][gene]["polarity"] == "-" and K_value > 0:
                    K_value *= -1

                row[2 + i] = reg_idx
                row[2 + max_regs + i] = K_value
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


def create_dataframes_for_sergio_inputs_from_n_replicas(dag, all_parameter_values, node_to_idx):
    """
    Create lists of pandas DataFrames for SERGIO inputs from multiple replicas.

    Args:
    dag (nx.DiGraph): The directed acyclic graph representing the gene regulatory network.
    all_parameter_values (list of dicts): List of parameter values dictionaries for the Hill equations for each replica.
    node_to_idx (dict): Mapping from gene names to indices.

    Returns:
    list of pd.DataFrame: List of data frames for the targets for each replica.
    list of pd.DataFrame: List of data frames for the master regulators for each replica.
    """
    assert isinstance(dag, nx.DiGraph)
    assert isinstance(all_parameter_values, dict)
    assert isinstance(node_to_idx, dict)

    all_targets_dfs = []
    all_regs_dfs = []

    # Iterate over the parameter values for each replica
    for replica, parameter_values in all_parameter_values.items():
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
    all_targets_dfs, all_regs_dfs, merge_type="first_only"
):
    """
    Merge lists of pandas DataFrames for SERGIO inputs from multiple replicas based on the specified merge type.

    Args:
    all_targets_dfs (list of pd.DataFrame): List of data frames for the targets for each replica.
    all_regs_dfs (list of pd.DataFrame): List of data frames for the master regulators for each replica.
    merge_type (str): Merge type with default "first_only". If "first_only", return the first elements of the target df and concatenates reg production rates.

    Returns:
    pd.DataFrame: Merged data frame for the targets.
    pd.DataFrame: Merged data frame for the master regulators with additional production rate columns.
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
    

def write_input_files_for_sergio(targets_df, regs_df, filename_targets, filename_regs):
    """
    Save the targets and regulators data frames to text files without the index.

    Args:
    targets_df (pd.DataFrame): Data frame for the targets.
    regs_df (pd.DataFrame): Data frame for the master regulators.
    filename_targets (str): Filename for the targets text file.
    filename_regs (str): Filename for the master regulators text file.
    """
    # Convert entire dataframes to float
    targets_df = targets_df.astype(float)
    regs_df = regs_df.astype(float)

    # Convert NaNs to empty strings and save to CSV format
    targets_csv = targets_df.to_csv(index=False, header=False, na_rep="")
    regs_csv = regs_df.to_csv(index=False, header=False, na_rep="")

    # Remove unnecessary commas from CSV strings
    def clean_csv_string(csv_string):
        # Remove trailing commas at the end of each line and any repeating commas
        csv_string = re.sub(r",+\n", "\n", csv_string)  # Remove trailing commas at the end of lines
        csv_string = re.sub(r",+", ",", csv_string)  # Replace multiple commas with a single one
        return csv_string

    # Write the cleaned CSV strings to text files
    with open(filename_targets, "w") as f:
        f.write(clean_csv_string(targets_csv))
    with open(filename_regs, "w") as f:
        f.write(clean_csv_string(regs_csv))    


def write_equation_info_to_file(hill_equations, parameter_values, data_filename):
    """Writes equations and parameter values as strings in a textfile. TODO: save to .json"""
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


