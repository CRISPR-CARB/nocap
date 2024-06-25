"""Test functions for the utility module."""

import os

import networkx as nx
import numpy as np
import pandas as pd
import sympy as sy

import nocap.utility as utility


def test_parse_regulation_file_simple():
    """Check that the expected and actual graphs are identical after parsing."""
    # Path to the file with the regulation data
    file_path = "./tests/simple_regulation_network.txt"

    # Parse the file using the function
    G = utility.parse_regulation_file(file_path)

    # Define the expected graph structure
    expected_G = nx.DiGraph()
    expected_G.add_edges_from(
        [
            ("geneA", "geneA", {"polarity": "+"}),
            ("geneA", "geneB", {"polarity": "+"}),
            ("geneA", "geneC", {"polarity": "+"}),
            ("geneB", "geneB", {"polarity": "-"}),
            ("geneB", "geneC", {"polarity": ""}),
            ("geneC", "geneD", {"polarity": "+/-"}),
        ]
    )
    assert G.nodes == expected_G.nodes, "Nodes do not match"
    assert nx.is_isomorphic(
        G, expected_G, edge_match=nx.algorithms.isomorphism.categorical_edge_match("polarity", None)
    ), "The graphs are not isomorphic or the edge polarities do not match."


# def test_convert_to_acyclic_graph():
#     """Test if a graph is correctly converted into an acyclic graph."""
#     # Create a directed graph representing a simple gene regulatory network
#     cyclic_graph = nx.DiGraph()
#     cyclic_graph.add_edges_from(
#         [
#             ("Gene1", "Gene2"),  # Gene1 activates Gene2
#             ("Gene2", "Gene3"),  # Gene2 activates Gene3
#             ("Gene3", "Gene1"),  # Gene3 activates Gene1, forming a cycle
#             ("Gene3", "Gene3"),  # Gene3 regulates itself, a self-loop
#             ("Gene4", "Gene2"),  # Gene4 activates Gene2, not part of a cycle
#         ]
#     )

#     # Convert the cyclic graph to an acyclic one
#     acyclic_graph = utility.convert_to_acyclic_graph(cyclic_graph)

#     # Test if the resulting graph is acyclic
#     assert nx.is_directed_acyclic_graph(acyclic_graph), "The graph is not acyclic."

#     # Test if the self-loop has been removed
#     assert not acyclic_graph.has_edge("Gene3", "Gene3"), "The self-loop has not been removed."

#     # Test if the larger cycle has been broken by checking for cycles
#     assert len(list(nx.simple_cycles(acyclic_graph))) == 0, "The cycle has not been broken."

#     # Test if non-cycle edges are preserved
#     assert acyclic_graph.has_edge(
#         "Gene4", "Gene2"
#     ), "The edge not part of a cycle was incorrectly removed."


def test_convert_to_acyclic_graph():
    """Test if a graph is correctly converted into an acyclic graph."""
    # Create a directed graph representing a simple gene regulatory network
    cyclic_graph = nx.DiGraph()
    cyclic_graph.add_edges_from(
        [
            ("gene1", "gene2"),  # Gene1 activates Gene2
            ("gene2", "gene3"),  # Gene2 activates Gene3
            ("gene3", "gene1"),  # Gene3 activates Gene1, forming a cycle
            ("gene3", "gene3"),  # Gene3 regulates itself, a self-loop
            ("gene4", "gene2"),  # Gene4 activates Gene2, not part of a cycle
        ]
    )

    # Convert the cyclic graph to an acyclic one
    acyclic_graph = utility.convert_to_acyclic_graph(cyclic_graph, "gene1")

    # Test if the resulting graph is acyclic
    assert nx.is_directed_acyclic_graph(acyclic_graph), "The graph is not acyclic."

    # Test if the self-loop has been removed
    assert not acyclic_graph.has_edge("gene3", "gene3"), "The self-loop has not been removed."

    # Test if the larger cycle has been broken by checking for cycles
    assert len(list(nx.simple_cycles(acyclic_graph))) == 0, "The cycle has not been broken."

    # Test if non-cycle edges are preserved
    assert acyclic_graph.has_edge(
        "gene4", "gene2"
    ), "The edge not part of a cycle was incorrectly removed."

    # Test if edges of the target node and its descendants are preserved
    assert acyclic_graph.has_edge(
        "gene1", "gene2"
    ), "The edge of the target node was incorrectly removed."
    assert acyclic_graph.has_edge(
        "gene2", "gene3"
    ), "The edge of a descendant of the target node was incorrectly removed."


def test_get_subgraph_from_nodes_include_ancestors_of_descendants():
    """Test subgraph from DAG, including descendants and their ancestors."""
    # Create a sample directed acyclic graph (DAG)
    G = nx.DiGraph()
    G.add_edges_from(
        [
            ("A", "B"),  # A is an ancestor of B
            ("B", "C"),  # B is an ancestor of C and a descendant of A
            ("C", "D"),  # C is an ancestor of D
            ("D", "E"),  # D is an ancestor of E
        ]
    )

    # Node list containing a single node
    node_list = ["B"]

    # Call the function to get a subgraph including descendants and their ancestors
    subgraph = utility.get_subgraph_from_nodes(G, node_list, descendants_only=False)

    # Define the expected nodes in the subgraph, which should include the ancestors of descendants
    expected_nodes = {
        "A",
        "B",
        "C",
        "D",
        "E",
    }  # A is an ancestor of B; B, C, D, and E are descendants

    # Check if the actual nodes match the expected nodes
    assert (
        set(subgraph.nodes()) == expected_nodes
    ), "The subgraph does not contain the correct nodes."


def test_get_subgraph_from_nodes_complex_structure():
    """Test subgraph from a complex graph structure."""
    # Create a sample directed acyclic graph (DAG) with the specified structure
    G = nx.DiGraph()
    G.add_edges_from(
        [
            ("B", "C"),  # B is an ancestor of C
            ("B", "D"),  # B is an ancestor of D
            ("D", "E"),  # D is an ancestor of E
            ("A", "E"),  # A is an ancestor of E
        ]
    )

    # Node list containing a single node
    node_list = ["B"]

    # Call the function to get a subgraph including descendants and their ancestors
    subgraph = utility.get_subgraph_from_nodes(G, node_list, descendants_only=False)

    # Define the expected nodes in the subgraph
    # Since B is the starting node, we expect B, C, and D as descendants,
    # and A as the ancestor of E (which is a descendant of D)
    expected_nodes = {"B", "C", "D", "E", "A"}

    # Check if the actual nodes match the expected nodes
    assert (
        set(subgraph.nodes()) == expected_nodes
    ), "The subgraph does not contain the correct nodes."


def test_get_subgraph_from_nodes():
    """Test subgraph from different node configurations."""
    # Create a simple DAG for testing
    dag = nx.DiGraph()
    dag.add_edge("Gene1", "Gene2")
    dag.add_edge("Gene1", "Gene3")
    dag.add_edge("Gene2", "Gene4")
    dag.add_edge("Gene3", "Gene4")
    # Add an edge to create a single descendant scenario
    dag.add_edge("Gene4", "Gene5")  # Gene4 has a single descendant Gene5

    # Test getting the subgraph for a single node
    subgraph = utility.get_subgraph_from_nodes(dag, ["Gene1"])
    assert set(subgraph.nodes) == {
        "Gene1",
        "Gene2",
        "Gene3",
        "Gene4",
        "Gene5",
    }, "Subgraph should contain all descendants of Gene1."

    # Test getting the subgraph for a single node with descendants only
    subgraph = utility.get_subgraph_from_nodes(dag, ["Gene2"], descendants_only=True)
    assert set(subgraph.nodes) == {
        "Gene2",
        "Gene4",
        "Gene5",
    }, "Subgraph should contain all descendants of Gene2."

    # Test getting the subgraph for a single node with a single descendant
    subgraph = utility.get_subgraph_from_nodes(dag, ["Gene4"], descendants_only=True)
    assert set(subgraph.nodes) == {
        "Gene4",
        "Gene5",
    }, "Subgraph should contain the single descendant of Gene4."

    # Test getting the subgraph for a node not in the graph
    subgraph = utility.get_subgraph_from_nodes(dag, ["Gene6"])
    assert (
        set(subgraph.nodes) == set()
    ), "Subgraph should be empty when the node is not in the graph."

    # Test getting the subgraph for multiple nodes
    subgraph = utility.get_subgraph_from_nodes(dag, ["Gene2", "Gene3"])
    assert set(subgraph.nodes) == {
        "Gene1",
        "Gene2",
        "Gene3",
        "Gene4",
        "Gene5",
    }, "Subgraph should contain all descendants and ancestors of Gene2 and Gene3."


# def test_generate_subnetwork_with_backdoor_adjustment():
#     # Test case 1: Simple graph with 1 target gene and 2 descendants, no backdoor paths
#     graph1 = nx.DiGraph()
#     graph1.add_edge("GeneM", "GeneT1", polarity="+")
#     graph1.add_edge("GeneM", "GeneT2", polarity="-")
#     subnetwork1 = utility.generate_subnetwork_with_backdoor_adjustment(graph1, "GeneM")
#     assert set(subnetwork1.nodes()) == {"GeneM", "GeneT1", "GeneT2"}
#     assert set(subnetwork1.edges()) == {("GeneM", "GeneT1"), ("GeneM", "GeneT2")}

#     # Test case 2: More complex graph with 1 target gene, 2 descendants, and 1 backdoor path
#     graph2 = nx.DiGraph()
#     graph2.add_edge("GeneM", "GeneT1", polarity="+")
#     graph2.add_edge("GeneM", "GeneT2", polarity="-")
#     graph2.add_edge("GeneT1", "GeneT2", polarity="+")  # Backdoor path
#     subnetwork2 = utility.generate_subnetwork_with_backdoor_adjustment(graph2, "GeneM")
#     assert set(subnetwork2.nodes()) == {"GeneM", "GeneT1", "GeneT2"}
#     assert set(subnetwork2.edges()) == {
#         ("GeneM", "GeneT1"),
#         ("GeneM", "GeneT2"),
#         ("GeneT1", "GeneT2"),
#     }

#     # Test case 3: Graph with 1 target gene, 3 descendants, and multiple backdoor paths
#     graph3 = nx.DiGraph()
#     graph3.add_edge("GeneM", "GeneT1", polarity="+")
#     graph3.add_edge("GeneM", "GeneT2", polarity="-")
#     graph3.add_edge("GeneT1", "GeneT3", polarity="+")  # Backdoor path
#     graph3.add_edge("GeneT2", "GeneT3", polarity="+")  # Another backdoor path
#     subnetwork3 = utility.generate_subnetwork_with_backdoor_adjustment(graph3, "GeneM")
#     assert set(subnetwork3.nodes()) == {"GeneM", "GeneT1", "GeneT2", "GeneT3"}
#     assert set(subnetwork3.edges()) == {
#         ("GeneM", "GeneT1"),
#         ("GeneM", "GeneT2"),
#         ("GeneT1", "GeneT3"),
#         ("GeneT2", "GeneT3"),
#     }

#     # Test case 4: Single node graph
#     graph4 = nx.DiGraph()
#     graph4.add_node("GeneM")
#     subnetwork4 = utility.generate_subnetwork_with_backdoor_adjustment(graph4, "GeneM")
#     assert set(subnetwork4.nodes()) == {"GeneM"}
#     assert set(subnetwork4.edges()) == set()

#     # Test case 5: Linear graph
#     graph5 = nx.DiGraph()
#     graph5.add_edge("GeneM", "GeneT1", polarity="+")
#     graph5.add_edge("GeneT1", "GeneT2", polarity="+")
#     subnetwork5 = utility.generate_subnetwork_with_backdoor_adjustment(graph5, "GeneM")
#     assert set(subnetwork5.nodes()) == {"GeneM", "GeneT1", "GeneT2"}
#     assert set(subnetwork5.edges()) == {("GeneM", "GeneT1"), ("GeneT1", "GeneT2")}

#     # Test case 6: Diverging graph
#     graph6 = nx.DiGraph()
#     graph6.add_edge("GeneM", "GeneT1", polarity="+")
#     graph6.add_edge("GeneM", "GeneT2", polarity="+")
#     subnetwork6 = utility.generate_subnetwork_with_backdoor_adjustment(graph6, "GeneM")
#     assert set(subnetwork6.nodes()) == {"GeneM", "GeneT1", "GeneT2"}
#     assert set(subnetwork6.edges()) == {("GeneM", "GeneT1"), ("GeneM", "GeneT2")}

#     # Test case 7: Converging graph
#     graph7 = nx.DiGraph()
#     graph7.add_edge("GeneT1", "GeneM", polarity="+")
#     graph7.add_edge("GeneT2", "GeneM", polarity="+")
#     subnetwork7 = utility.generate_subnetwork_with_backdoor_adjustment(graph7, "GeneM")
#     assert set(subnetwork7.nodes()) == {"GeneM", "GeneT1", "GeneT2"}
#     assert set(subnetwork7.edges()) == {("GeneT1", "GeneM"), ("GeneT2", "GeneM")}


def test_generate_hill_equations():
    """Test Hill equation generation for a gene regulatory network."""
    # Define the activation probability for the test
    activation_probability = 0.5

    # Create a DAG for the test
    test_dag = nx.DiGraph()
    test_dag.add_edge("GeneA", "GeneB", polarity="+")
    test_dag.add_edge("GeneC", "GeneB", polarity="-")
    test_dag.add_edge("GeneD", "GeneB", polarity="+/-")  # Ambiguous polarity

    # Set a fixed seed for numpy's random number generator for reproducibility
    np.random.seed(42)

    # Generate the Hill equations
    (
        hill_equations,
        basal_rates,
        max_contributions,
        hill_coefficients,
        half_responses,
        gene_expressions,
    ) = utility.generate_hill_equations(test_dag, activation_probability=activation_probability)

    np.random.seed(42)

    # Define the expected equations explicitly
    x_GeneA = gene_expressions["GeneA"]
    x_GeneC = gene_expressions["GeneC"]
    x_GeneD = gene_expressions["GeneD"]
    b_GeneB = basal_rates["GeneB"]
    K_GeneA_GeneB = max_contributions[("GeneA", "GeneB")]
    K_GeneC_GeneB = max_contributions[("GeneC", "GeneB")]
    K_GeneD_GeneB = max_contributions[("GeneD", "GeneB")]
    n_GeneA_GeneB = hill_coefficients[("GeneA", "GeneB")]
    n_GeneC_GeneB = hill_coefficients[("GeneC", "GeneB")]
    n_GeneD_GeneB = hill_coefficients[("GeneD", "GeneB")]
    h_GeneA_GeneB = half_responses[("GeneA", "GeneB")]
    h_GeneC_GeneB = half_responses[("GeneC", "GeneB")]
    h_GeneD_GeneB = half_responses[("GeneD", "GeneB")]

    # Expected equation for GeneB, considering the polarities
    expected_equation_GeneB = b_GeneB
    expected_equation_GeneB += (
        K_GeneA_GeneB
        * x_GeneA**n_GeneA_GeneB
        / (h_GeneA_GeneB**n_GeneA_GeneB + x_GeneA**n_GeneA_GeneB)
    )  # Activation from GeneA
    expected_equation_GeneB += (
        K_GeneC_GeneB
        * (1 - x_GeneC**n_GeneC_GeneB)
        / (h_GeneC_GeneB**n_GeneC_GeneB + x_GeneC**n_GeneC_GeneB)
    )  # Repression from GeneC
    # For GeneD, we use the predetermined outcome of the random polarity based on the fixed seed
    if np.random.rand() < activation_probability:
        expected_equation_GeneB += (
            K_GeneD_GeneB
            * x_GeneD**n_GeneD_GeneB
            / (h_GeneD_GeneB**n_GeneD_GeneB + x_GeneD**n_GeneD_GeneB)
        )  # Activation from GeneD
    else:
        expected_equation_GeneB += (
            K_GeneD_GeneB
            * (1 - x_GeneD**n_GeneD_GeneB)
            / (h_GeneD_GeneB**n_GeneD_GeneB + x_GeneD**n_GeneD_GeneB)
        )  # Repression from GeneD

    # Compare the actual and expected equations
    assert (
        sy.simplify(hill_equations["GeneB"] - expected_equation_GeneB) == 0
    ), "Equation for GeneB does not match the expected equation."


def test_laci_hill_equations():
    """Test Hill equation generation for the LacI network."""
    # Create a DAG for the LacI network
    laci_dag = nx.DiGraph()
    laci_dag.add_edge("LacI", "LacA", polarity="-")
    laci_dag.add_edge("LacI", "LacY", polarity="-")
    laci_dag.add_edge("LacI", "LacZ", polarity="-")

    # Generate the Hill equations for the LacI network
    (
        hill_equations,
        basal_rates,
        max_contributions,
        hill_coefficients,
        half_responses,
        gene_expressions,
    ) = utility.generate_hill_equations(laci_dag)

    # Define the expected equations explicitly
    x_LacI = gene_expressions["LacI"]
    b_LacI = basal_rates["LacI"]  # Basal production rate for LacI
    b_LacA = basal_rates["LacA"]
    b_LacY = basal_rates["LacY"]
    b_LacZ = basal_rates["LacZ"]
    K_LacI_LacA = max_contributions[("LacI", "LacA")]
    K_LacI_LacY = max_contributions[("LacI", "LacY")]
    K_LacI_LacZ = max_contributions[("LacI", "LacZ")]
    n_LacI_LacA = hill_coefficients[("LacI", "LacA")]
    n_LacI_LacY = hill_coefficients[("LacI", "LacY")]
    n_LacI_LacZ = hill_coefficients[("LacI", "LacZ")]
    h_LacI_LacA = half_responses[("LacI", "LacA")]
    h_LacI_LacY = half_responses[("LacI", "LacY")]
    h_LacI_LacZ = half_responses[("LacI", "LacZ")]

    # Expected equation for LacI (master regulator with no regulators)
    expected_equation_LacI = b_LacI

    # Expected equations for LacA, LacY, and LacZ, considering repression by LacI
    expected_equation_LacA = b_LacA + K_LacI_LacA * (1 - x_LacI**n_LacI_LacA) / (
        h_LacI_LacA**n_LacI_LacA + x_LacI**n_LacI_LacA
    )
    expected_equation_LacY = b_LacY + K_LacI_LacY * (1 - x_LacI**n_LacI_LacY) / (
        h_LacI_LacY**n_LacI_LacY + x_LacI**n_LacI_LacY
    )
    expected_equation_LacZ = b_LacZ + K_LacI_LacZ * (1 - x_LacI**n_LacI_LacZ) / (
        h_LacI_LacZ**n_LacI_LacZ + x_LacI**n_LacI_LacZ
    )

    # Compare the actual and expected equations
    assert (
        sy.simplify(hill_equations["LacI"] - expected_equation_LacI) == 0
    ), "Equation for LacI does not match the expected equation."
    assert (
        sy.simplify(hill_equations["LacA"] - expected_equation_LacA) == 0
    ), "Equation for LacA does not match the expected equation."
    assert (
        sy.simplify(hill_equations["LacY"] - expected_equation_LacY) == 0
    ), "Equation for LacY does not match the expected equation."
    assert (
        sy.simplify(hill_equations["LacZ"] - expected_equation_LacZ) == 0
    ), "Equation for LacZ does not match the expected equation."


def test_generate_hill_equations_multiple_masters():
    """Test Hill equation generation for a network with multiple master regulators."""
    # Define the activation probability for the test
    activation_probability = 0.5

    # Create a DAG for the test with multiple master regulators
    test_dag = nx.DiGraph()
    test_dag.add_node("Master1")  # Master regulator 1 with no outgoing edges
    test_dag.add_node("Master2")  # Master regulator 2 with no outgoing edges
    test_dag.add_edge("Master1", "GeneA", polarity="+")
    test_dag.add_edge("Master2", "GeneA", polarity="-")
    test_dag.add_edge("GeneB", "GeneA", polarity="+/-")  # Ambiguous polarity for GeneB -> GeneA

    # Set a fixed seed for numpy's random number generator for reproducibility
    np.random.seed(42)

    # Generate the Hill equations
    (
        hill_equations,
        basal_rates,
        max_contributions,
        hill_coefficients,
        half_responses,
        gene_expressions,
    ) = utility.generate_hill_equations(test_dag, activation_probability=activation_probability)

    # Reset seed for reproducibility
    np.random.seed(42)

    # Define the expected equations explicitly
    x_Master1 = gene_expressions["Master1"]
    x_Master2 = gene_expressions["Master2"]
    x_GeneB = gene_expressions["GeneB"]
    b_Master1 = basal_rates["Master1"]
    b_Master2 = basal_rates["Master2"]
    b_GeneA = basal_rates["GeneA"]
    K_Master1_GeneA = max_contributions[("Master1", "GeneA")]
    K_Master2_GeneA = max_contributions[("Master2", "GeneA")]
    K_GeneB_GeneA = max_contributions[("GeneB", "GeneA")]
    n_Master1_GeneA = hill_coefficients[("Master1", "GeneA")]
    n_Master2_GeneA = hill_coefficients[("Master2", "GeneA")]
    n_GeneB_GeneA = hill_coefficients[("GeneB", "GeneA")]
    h_Master1_GeneA = half_responses[("Master1", "GeneA")]
    h_Master2_GeneA = half_responses[("Master2", "GeneA")]
    h_GeneB_GeneA = half_responses[("GeneB", "GeneA")]

    # Expected equations for master regulators
    expected_equation_Master1 = b_Master1
    expected_equation_Master2 = b_Master2

    # Expected equation for GeneA, considering the polarities
    expected_equation_GeneA = b_GeneA
    expected_equation_GeneA += (
        K_Master1_GeneA
        * x_Master1**n_Master1_GeneA
        / (h_Master1_GeneA**n_Master1_GeneA + x_Master1**n_Master1_GeneA)
    )  # Activation from Master1
    expected_equation_GeneA += (
        K_Master2_GeneA
        * (1 - x_Master2**n_Master2_GeneA)
        / (h_Master2_GeneA**n_Master2_GeneA + x_Master2**n_Master2_GeneA)
    )  # Repression from Master2
    # For GeneB, we use the predetermined outcome of the random polarity based on the fixed seed
    if np.random.rand() < activation_probability:
        expected_equation_GeneA += (
            K_GeneB_GeneA
            * x_GeneB**n_GeneB_GeneA
            / (h_GeneB_GeneA**n_GeneB_GeneA + x_GeneB**n_GeneB_GeneA)
        )  # Activation from GeneB
    else:
        expected_equation_GeneA += (
            K_GeneB_GeneA
            * (1 - x_GeneB**n_GeneB_GeneA)
            / (h_GeneB_GeneA**n_GeneB_GeneA + x_GeneB**n_GeneB_GeneA)
        )  # Repression from GeneB

    # Compare the actual and expected equations for master regulators
    assert (
        sy.simplify(hill_equations["Master1"] - expected_equation_Master1) == 0
    ), "Equation for Master1 does not match the expected equation."
    assert (
        sy.simplify(hill_equations["Master2"] - expected_equation_Master2) == 0
    ), "Equation for Master2 does not match the expected equation."

    # Compare the actual and expected equations for GeneA
    assert (
        sy.simplify(hill_equations["GeneA"] - expected_equation_GeneA) == 0
    ), "Equation for GeneA does not match the expected equation."


def test_generate_hill_equations_for_n_replicas_shape():
    """Test the shape of multiple Hill equation replicas."""
    # Create a simple DAG for testing
    test_dag = nx.DiGraph()
    test_dag.add_edge("Gene1", "Gene2", polarity="+")

    # Generate multiple replicas of Hill equations
    N = 3
    replicas = utility.generate_hill_equations_for_n_replicas(test_dag, n=N)

    # Check that the dictionary contains N sets
    assert len(replicas) == N, "Number of replicas returned should be equal to N."

    # Check that each set contains the expected tuple structure
    for replica in range(N):
        assert isinstance(replicas[replica], tuple), "Each replica should be a tuple."
        assert len(replicas[replica]) == 6, "Each replica tuple should contain 6 elements."
        (
            equations,
            basal_rates,
            max_contributions,
            hill_coefficients,
            half_responses,
            gene_expressions,
        ) = replicas[replica]
        assert isinstance(equations, dict), "Equations should be a dictionary."
        assert isinstance(basal_rates, dict), "Basal rates should be a dictionary."
        assert isinstance(max_contributions, dict), "Max contributions should be a dictionary."
        assert isinstance(hill_coefficients, dict), "Hill coefficients should be a dictionary."
        assert isinstance(half_responses, dict), "Half responses should be a dictionary."
        assert isinstance(gene_expressions, dict), "Gene expressions should be a dictionary."


def test_generate_hill_equations_for_n_replicas():
    """Test Hill equation generation for multiple replicas."""
    # Create a simple DAG for testing with ambiguous polarity
    test_dag = nx.DiGraph()
    test_dag.add_edge("Gene1", "Gene2", polarity="+/-")

    # Test with N=1 to ensure the function returns a single set of equations
    np.random.seed(42)
    single_replica = utility.generate_hill_equations_for_n_replicas(test_dag, n=1)
    assert (
        len(single_replica) == 1
    ), "The function should return a single set of equations when N=1."

    # Test with N=3 to ensure the function returns three sets of equations
    np.random.seed(42)
    multiple_replicas = utility.generate_hill_equations_for_n_replicas(test_dag, n=3)
    assert (
        len(multiple_replicas) == 3
    ), "The function should return three sets of equations when N=3."

    # Check if the equations for 'Gene2' are different between the first and last replicas
    gene2_first_replica = multiple_replicas[0][0]["Gene2"]
    gene2_last_replica = multiple_replicas[2][0]["Gene2"]
    assert (
        gene2_first_replica != gene2_last_replica
    ), "Equations for 'Gene2' should be different between replicas due to the '+/-' polarity."

    # Check reproducibility
    # Use a fixed seed here for reproducibility
    np.random.seed(42)
    first_replica_set = utility.generate_hill_equations_for_n_replicas(test_dag, n=3)
    np.random.seed(42)
    second_replica_set = utility.generate_hill_equations_for_n_replicas(test_dag, n=3)

    # Check if the equations for 'Gene2' are identical across replicas generated with the same seed
    for i in range(3):
        gene2_first_replica = first_replica_set[i][0]["Gene2"]
        gene2_second_replica = second_replica_set[i][0]["Gene2"]
        assert (
            gene2_first_replica == gene2_second_replica
        ), f"Equations for 'Gene2' in replica {i} should be identical due to the fixed seed."


def test_assign_random_parameter_values():
    """Test random parameter value assignment for Hill equations."""
    # Define a small set of symbols for testing
    basal_rates_test = {"GeneA": sy.Symbol("b_GeneA"), "GeneB": sy.Symbol("b_GeneB")}
    max_contributions_test = {("GeneA", "GeneB"): sy.Symbol("K_GeneA_GeneB")}
    hill_coefficients_test = {("GeneA", "GeneB"): sy.Symbol("n_GeneA_GeneB")}
    half_responses_test = {("GeneA", "GeneB"): sy.Symbol("h_GeneA_GeneB")}

    # Define distribution settings for the parameters
    param_distributions_test = {
        "K": {"min": 1, "max": 5},
        "n": {"min": 1, "max": 3},  # Hill coefficient range is now 1 to 3
        "h": {"min": 1, "max": 5},
        "b": {"a": 2, "b": 5, "scale": 4},
        "master_regulators": ["GeneA"],  # GeneA is a master regulator in this test
    }

    # Set a fixed seed for numpy's random number generator for reproducibility
    np.random.seed(42)

    # Assign values to the parameters
    parameter_values_test = utility.assign_random_parameter_values(
        basal_rates_test,
        max_contributions_test,
        hill_coefficients_test,
        half_responses_test,
        param_distributions_test,
    )

    # Check that the dictionary has the correct keys and that the values are within the expected ranges
    assert (
        parameter_values_test[basal_rates_test["GeneA"]] >= 0
        and parameter_values_test[basal_rates_test["GeneA"]]
        <= param_distributions_test["b"]["scale"]
    ), "Basal rate for master regulator is out of range."
    assert (
        parameter_values_test[basal_rates_test["GeneB"]] == 0
    ), "Basal rate for non-master regulator should be zero."
    assert (
        parameter_values_test[max_contributions_test[("GeneA", "GeneB")]]
        >= param_distributions_test["K"]["min"]
        and parameter_values_test[max_contributions_test[("GeneA", "GeneB")]]
        <= param_distributions_test["K"]["max"]
    ), "Max contribution is out of range."

    # Check that Hill coefficients are within the specified range
    n_min, n_max = param_distributions_test["n"]["min"], param_distributions_test["n"]["max"]
    for hc in hill_coefficients_test.values():
        assert (
            n_min <= parameter_values_test[hc] <= n_max
        ), f"Hill coefficient {hc} is out of range."

    assert (
        parameter_values_test[half_responses_test[("GeneA", "GeneB")]]
        >= param_distributions_test["h"]["min"]
        and parameter_values_test[half_responses_test[("GeneA", "GeneB")]]
        <= param_distributions_test["h"]["max"]
    ), "Half response is out of range."


def test_laci_parameter_values():
    """Test parameter value assignment for the LacI network."""
    np.random.seed(42)
    # Create a DAG for the LacI network
    laci_dag = nx.DiGraph()
    laci_dag.add_edge("LacI", "LacA", polarity="-")
    laci_dag.add_edge("LacI", "LacY", polarity="-")
    laci_dag.add_edge("LacI", "LacZ", polarity="-")

    # Define symbols for the LacI network
    basal_rates_laci = {gene: sy.Symbol(f"b_{gene}") for gene in laci_dag.nodes()}
    max_contributions_laci = {
        (regulator, gene): sy.Symbol(f"K_{regulator}_{gene}")
        for regulator, gene in laci_dag.edges()
    }
    hill_coefficients_laci = {
        (regulator, gene): sy.Symbol(f"n_{regulator}_{gene}")
        for regulator, gene in laci_dag.edges()
    }
    half_responses_laci = {
        (regulator, gene): sy.Symbol(f"h_{regulator}_{gene}")
        for regulator, gene in laci_dag.edges()
    }

    # Define distribution settings for the parameters
    param_distributions_laci = {
        "K": {"min": 1, "max": 5},
        "n": {"min": 1, "max": 3},  # Updated range for Hill coefficients
        "h": {"min": 1, "max": 5},
        "b": {"a": 2, "b": 5, "scale": 4},
        "master_regulators": ["LacI"],  # LacI is a master regulator
    }

    # Assign values to the parameters
    parameter_values_laci = utility.assign_random_parameter_values(
        basal_rates_laci,
        max_contributions_laci,
        hill_coefficients_laci,
        half_responses_laci,
        param_distributions_laci,
    )

    # Perform checks to ensure the values are within the expected ranges
    # Check basal rates for master regulator
    assert (
        parameter_values_laci[basal_rates_laci["LacI"]] >= 0
        and parameter_values_laci[basal_rates_laci["LacI"]]
        <= param_distributions_laci["b"]["scale"]
    ), "Basal rate for LacI is out of range."

    # Check max contributions and half responses for uniform distribution
    for gene in ["LacA", "LacY", "LacZ"]:
        for param_type in ["K", "h"]:
            symbol = (
                max_contributions_laci[("LacI", gene)]
                if param_type == "K"
                else half_responses_laci[("LacI", gene)]
            )
            min_val, max_val = (
                param_distributions_laci[param_type]["min"],
                param_distributions_laci[param_type]["max"],
            )
            assert (
                parameter_values_laci[symbol] >= min_val
                and parameter_values_laci[symbol] <= max_val
            ), f"{param_type} value for {gene} is out of range."

    # Check Hill coefficients for integer range
    n_min, n_max = param_distributions_laci["n"]["min"], param_distributions_laci["n"]["max"]
    for gene in ["LacA", "LacY", "LacZ"]:
        symbol = hill_coefficients_laci[("LacI", gene)]
        assert (
            n_min <= parameter_values_laci[symbol] <= n_max
        ), f"Hill coefficient value for {gene} is out of range."


def test_assign_random_parameter_values_multiple_masters():
    """Test random parameter value assignment for multiple master regulators."""
    np.random.seed(42)
    # Define a set of symbols for two master regulators and one target gene
    basal_rates_test = {
        "Master1": sy.Symbol("b_Master1"),
        "Master2": sy.Symbol("b_Master2"),
        "GeneA": sy.Symbol("b_GeneA"),
    }
    max_contributions_test = {
        ("Master1", "GeneA"): sy.Symbol("K_Master1_GeneA"),
        ("Master2", "GeneA"): sy.Symbol("K_Master2_GeneA"),
    }
    hill_coefficients_test = {
        ("Master1", "GeneA"): sy.Symbol("n_Master1_GeneA"),
        ("Master2", "GeneA"): sy.Symbol("n_Master2_GeneA"),
    }
    half_responses_test = {
        ("Master1", "GeneA"): sy.Symbol("h_Master1_GeneA"),
        ("Master2", "GeneA"): sy.Symbol("h_Master2_GeneA"),
    }

    # Define distribution settings for the parameters
    param_distributions_test = {
        "K": {"min": 1, "max": 5},
        "n": {"min": 1, "max": 3},  # Updated range for Hill coefficients
        "h": {"min": 1, "max": 5},
        "b": {"a": 2, "b": 5, "scale": 4},
        "master_regulators": [
            "Master1",
            "Master2",
        ],  # Both Master1 and Master2 are master regulators
    }

    # Assign values to the parameters
    parameter_values_test = utility.assign_random_parameter_values(
        basal_rates_test,
        max_contributions_test,
        hill_coefficients_test,
        half_responses_test,
        param_distributions_test,
    )

    # Check that the dictionary has the correct keys and that the values are within the expected ranges
    # Check basal rates for master regulators
    for master in param_distributions_test["master_regulators"]:
        br = basal_rates_test[master]
        assert (
            parameter_values_test[br] >= 0
            and parameter_values_test[br] <= param_distributions_test["b"]["scale"]
        ), f"Basal rate for master regulator {master} is out of range."

    # Check basal rate for non-master regulator
    assert (
        parameter_values_test[basal_rates_test["GeneA"]] == 0
    ), "Basal rate for non-master regulator should be zero."

    # Check max contributions for uniform distribution
    for (regulator, gene), mc in max_contributions_test.items():
        assert (
            parameter_values_test[mc] >= param_distributions_test["K"]["min"]
            and parameter_values_test[mc] <= param_distributions_test["K"]["max"]
        ), f"Max contribution for edge {regulator}->{gene} is out of range."

    # Check Hill coefficients for integer range
    n_min, n_max = param_distributions_test["n"]["min"], param_distributions_test["n"]["max"]
    for (regulator, gene), hc in hill_coefficients_test.items():
        assert (
            n_min <= parameter_values_test[hc] <= n_max
        ), f"Hill coefficient for edge {regulator}->{gene} is out of range."

    # Check half responses for uniform distribution
    for (regulator, gene), hr in half_responses_test.items():
        assert (
            parameter_values_test[hr] >= param_distributions_test["h"]["min"]
            and parameter_values_test[hr] <= param_distributions_test["h"]["max"]
        ), f"Half response for edge {regulator}->{gene} is out of range."


def test_basal_rate_scaling():
    """Test scaling of basal rates in parameter value assignment."""
    # Define a small set of symbols for testing
    basal_rates_test = {"GeneA": sy.Symbol("b_GeneA"), "GeneB": sy.Symbol("b_GeneB")}
    max_contributions_test = {("GeneA", "GeneB"): sy.Symbol("K_GeneA_GeneB")}
    hill_coefficients_test = {("GeneA", "GeneB"): sy.Symbol("n_GeneA_GeneB")}
    half_responses_test = {("GeneA", "GeneB"): sy.Symbol("h_GeneA_GeneB")}

    # Define distribution settings for the parameters
    param_distributions_test = {
        "K": {"min": 1, "max": 5},
        "n": {"min": 1, "max": 3},  # Hill coefficient range is now 1 to 3
        "h": {"min": 1, "max": 5},
        "b": {"a": 2, "b": 5, "scale": 4},
        "master_regulators": ["GeneA"],  # GeneA is a master regulator in this test
    }

    # Set a fixed seed for reproducibility
    np.random.seed(42)

    # Get the parameter values with the original scale
    original_values = utility.assign_random_parameter_values(
        basal_rates_test,
        max_contributions_test,
        hill_coefficients_test,
        half_responses_test,
        param_distributions_test,
    )
    original_basal_rate = original_values[basal_rates_test["GeneA"]]

    # Update the scale by 10x
    param_distributions_test["b"]["scale"] *= 10
    np.random.seed(42)
    # Get the parameter values with the updated scale
    scaled_values = utility.assign_random_parameter_values(
        basal_rates_test,
        max_contributions_test,
        hill_coefficients_test,
        half_responses_test,
        param_distributions_test,
    )
    scaled_basal_rate = scaled_values[basal_rates_test["GeneA"]]

    # Check if the scaled basal rate is approximately 10 times the original basal rate
    assert np.isclose(
        scaled_basal_rate, 10 * original_basal_rate
    ), "The scaled basal rate is not approximately 10 times the original basal rate."


def test_assign_values_for_n_replicas():
    """Tests random parameter value assignment for multiple replicas."""
    np.random.seed(42)
    # Create a simple DAG for testing
    test_dag = nx.DiGraph()
    test_dag.add_edge("Gene1", "Gene2", polarity="+")

    # Define distribution settings for the parameters for each replica
    param_distributions_test = {
        0: {
            "K": {"min": 1, "max": 5},
            "n": {"min": 1, "max": 5},
            "h": {"min": 1, "max": 5},
            "b": {"a": 2, "b": 5, "scale": 4},
            "master_regulators": ["Gene1"],
        },
        1: {
            "K": {"min": 1, "max": 5},
            "n": {"min": 1, "max": 5},
            "h": {"min": 1, "max": 5},
            "b": {"a": 2, "b": 5, "scale": 40},  # Scale is 10x replica 0
            "master_regulators": ["Gene1"],
        },
    }

    # Generate two replicas of Hill equations
    replicas = utility.generate_hill_equations_for_n_replicas(test_dag, n=2)

    # Assign values to the parameters for all replicas
    all_values = utility.assign_random_values_for_n_replicas(replicas, param_distributions_test)

    # Check that the parameter values for the replicas are not identical
    for symbol in all_values[0]:
        if (
            "b_" not in str(symbol)
            or str(symbol)[2:] in param_distributions_test[0]["master_regulators"]
        ):
            assert (
                all_values[0][symbol] != all_values[1][symbol]
            ), f"Parameter values for {symbol} should not be identical across replicas."
        else:
            assert (
                all_values[0][symbol] == all_values[1][symbol]
            ), "Basal rate for non-master regulator should be zero across replicas."

    # Check that values are assigned for all replicas and within the expected ranges
    for replica, values in all_values.items():
        for symbol, value in values.items():
            param_dist = param_distributions_test[replica]
            if "b_" in str(symbol):  # Basal rates check
                if str(symbol)[2:] in param_dist["master_regulators"]:
                    assert (
                        0 <= value <= param_dist["b"]["scale"]
                    ), "Basal rate for master regulator is out of range."
                else:
                    assert value == 0, "Basal rate for non-master regulator should be zero."
            elif "K_" in str(symbol):  # Max contributions check
                assert (
                    param_dist["K"]["min"] <= value <= param_dist["K"]["max"]
                ), "Max contribution is out of range."
            elif "n_" in str(symbol):  # Hill coefficients check
                n_min, n_max = (
                    param_dist["n"]["min"],
                    param_dist["n"]["max"],
                )
                assert (
                    n_min <= value <= n_max
                ), "Hill coefficient should be within the specified range."
            elif "h_" in str(symbol):  # Half responses check
                assert (
                    param_dist["h"]["min"] <= value <= param_dist["h"]["max"]
                ), "Half response is out of range."

    # Check if the basal rate for the master regulator is larger for replica 2 as expected
    master_regulator_symbol = sy.Symbol("b_Gene1")
    basal_rate_replica_0 = all_values[0][master_regulator_symbol]
    basal_rate_replica_1 = all_values[1][master_regulator_symbol]
    assert (
        basal_rate_replica_1 > basal_rate_replica_0
    ), "Basal rate for master regulator in replica 1 should be larger than in replica 0."

    # Check for reproducibility with the same seed
    np.random.seed(42)
    replica1_with_seed = utility.generate_hill_equations_for_n_replicas(test_dag, n=1)
    values_replica1_with_seed = utility.assign_random_values_for_n_replicas(
        replica1_with_seed, {0: param_distributions_test[0]}  # Use the distribution for replica 0
    )

    np.random.seed(42)
    replica2_with_seed = utility.generate_hill_equations_for_n_replicas(test_dag, n=1)
    values_replica2_with_seed = utility.assign_random_values_for_n_replicas(
        replica2_with_seed, {0: param_distributions_test[0]}  # Use the distribution for replica 0
    )

    # Check that all the values are identical between replicas generated with the same seed
    for symbol in values_replica1_with_seed[0]:
        assert (
            values_replica1_with_seed[0][symbol] == values_replica2_with_seed[0][symbol]
        ), "Parameter values should be identical when generated with the same seed."


def test_create_dataframes_for_sergio_inputs_from_one_replica_laci():
    """Test DataFrame creation for SERGIO inputs from one LacI replica."""
    # Create the LacI DAG
    laci_dag = nx.DiGraph()
    laci_dag.add_edge("LacI", "LacA", polarity="-")
    laci_dag.add_edge("LacI", "LacY", polarity="-")
    laci_dag.add_edge("LacI", "LacZ", polarity="-")

    # Define a mapping from gene names to indices
    node_to_idx_laci = utility.create_node_to_idx_mapping(laci_dag)

    # Assign parameter values for the test using Sympy Symbols
    parameter_values_laci = {
        sy.Symbol("b_LacI"): 3.0,
        sy.Symbol("K_LacI_LacA"): 2.0,
        sy.Symbol("K_LacI_LacY"): 2.0,
        sy.Symbol("K_LacI_LacZ"): 2.0,
        sy.Symbol("n_LacI_LacA"): 2,
        sy.Symbol("n_LacI_LacY"): 2,
        sy.Symbol("n_LacI_LacZ"): 2,
    }

    # Create input files for the build_graph function
    targets_df_laci, regs_df_laci = utility.create_dataframes_for_sergio_inputs_from_one_replica(
        laci_dag, parameter_values_laci, node_to_idx_laci
    )

    # Assertions to check the correctness of the data frames
    # Assertions for test_create_dataframes_for_sergio_inputs_from_one_replica_laci
    assert regs_df_laci.iloc[0]["Master regulator Idx"] == node_to_idx_laci["LacI"]
    assert regs_df_laci.iloc[0]["production_rate"] == parameter_values_laci[sy.Symbol("b_LacI")]

    for i, gene in enumerate(["LacA", "LacY", "LacZ"]):
        target_row = targets_df_laci.iloc[i]
        assert target_row["Target Idx"] == node_to_idx_laci[gene]
        assert target_row["#regulators"] == 1
        assert target_row["regIdx1"] == node_to_idx_laci["LacI"]
        assert target_row["K1"] == -1 * parameter_values_laci[sy.Symbol("K_LacI_" + gene)]
        assert target_row["coop_state1"] == parameter_values_laci[sy.Symbol("n_LacI_" + gene)]


def test_create_dataframes_for_sergio_inputs_from_one_replica_hypothetical():
    """Test DataFrame creation for SERGIO inputs from one hypothetical replica."""
    # Create a hypothetical DAG for the gene regulatory network
    hypothetical_dag = nx.DiGraph()
    hypothetical_dag.add_edge("GeneM", "GeneT1", polarity="+")
    hypothetical_dag.add_edge("GeneM", "GeneT2", polarity="-")

    # Define a mapping from gene names to indices
    node_to_idx_hypothetical = utility.create_node_to_idx_mapping(hypothetical_dag)

    # Assign parameter values for the test
    parameter_values_hypothetical = {
        sy.Symbol("b_GeneM"): 1.5,
        sy.Symbol("K_GeneM_GeneT1"): 1.0,
        sy.Symbol("K_GeneM_GeneT2"): 1.0,
        sy.Symbol("n_GeneM_GeneT1"): 2,
        sy.Symbol("n_GeneM_GeneT2"): 2,
    }

    # Create input files for the build_graph function
    targets_df_hypothetical, regs_df_hypothetical = (
        utility.create_dataframes_for_sergio_inputs_from_one_replica(
            hypothetical_dag, parameter_values_hypothetical, node_to_idx_hypothetical
        )
    )

    # Assertions for test_create_dataframes_for_sergio_inputs_from_one_replica_hypothetical
    assert regs_df_hypothetical.iloc[0]["Master regulator Idx"] == node_to_idx_hypothetical["GeneM"]
    assert (
        regs_df_hypothetical.iloc[0]["production_rate"]
        == parameter_values_hypothetical[sy.Symbol("b_GeneM")]
    )

    for i, gene in enumerate(["GeneT1", "GeneT2"]):
        target_row = targets_df_hypothetical.iloc[i]
        assert target_row["Target Idx"] == node_to_idx_hypothetical[gene]
        assert target_row["#regulators"] == 1
        assert target_row["regIdx1"] == node_to_idx_hypothetical["GeneM"]

        # For repressors, the K value in the DataFrame should be negative, so we take the absolute value for comparison
        expected_k_value = (
            -1 * parameter_values_hypothetical[sy.Symbol("K_GeneM_" + gene)]
            if hypothetical_dag["GeneM"][gene]["polarity"] == "-"
            else parameter_values_hypothetical[sy.Symbol("K_GeneM_" + gene)]
        )
        assert target_row["K1"] == expected_k_value
        assert (
            target_row["coop_state1"] == parameter_values_hypothetical[sy.Symbol("n_GeneM_" + gene)]
        )


def test_create_node_to_idx_mapping():
    """Test creation of node-to-index mapping for a graph."""
    # Create a simple DAG for testing
    test_dag = nx.DiGraph()
    test_dag.add_edge("GeneA", "GeneB")
    test_dag.add_edge("GeneA", "GeneC")
    test_dag.add_edge("GeneB", "GeneC")

    # Call the function to test
    node_to_idx = utility.create_node_to_idx_mapping(test_dag)

    # Check that the mapping contains all nodes
    assert set(node_to_idx.keys()) == set(test_dag.nodes()), "Mapping should contain all nodes."

    # Check that the indices start at 0 and are consecutive
    expected_indices = list(range(len(test_dag.nodes())))
    assert (
        sorted(node_to_idx.values()) == expected_indices
    ), "Indices should start at 0 and be consecutive."

    # Check that the mapping is correct
    expected_mapping = {"GeneA": 0, "GeneB": 1, "GeneC": 2}
    # We cannot directly compare the mappings because the order of nodes() is not guaranteed
    for gene, idx in expected_mapping.items():
        assert node_to_idx[gene] == idx, f"Index for {gene} should be {idx}."


def test_create_dataframes_for_sergio_inputs_from_n_replicas():
    """Test DataFrame creation for SERGIO inputs from multiple replicas."""
    test_dag = nx.DiGraph()
    test_dag.add_edge("Gene1", "Gene2", polarity="+")
    node_to_idx = utility.create_node_to_idx_mapping(test_dag)

    # Update parameter values to use Sympy Symbols for each replica
    all_parameter_values = {
        1: {
            sy.Symbol("b_Gene1"): 1.0,
            sy.Symbol("K_Gene1_Gene2"): 2.0,
            sy.Symbol("n_Gene1_Gene2"): 3,
            sy.Symbol("b_Gene2"): 0.0,  # Gene2 is not a master regulator
        },
        2: {
            sy.Symbol("b_Gene1"): 1.5,
            sy.Symbol("K_Gene1_Gene2"): 2.5,
            sy.Symbol("n_Gene1_Gene2"): 4,
            sy.Symbol("b_Gene2"): 0.0,  # Gene2 is not a master regulator
        },
    }

    all_targets_dfs, all_regs_dfs = utility.create_dataframes_for_sergio_inputs_from_n_replicas(
        test_dag, all_parameter_values, node_to_idx
    )

    assert len(all_targets_dfs) == len(
        all_parameter_values
    ), "Incorrect number of target DataFrames."
    assert len(all_regs_dfs) == len(
        all_parameter_values
    ), "Incorrect number of master regulator DataFrames."

    expected_first_targets_df = pd.DataFrame(
        {"Target Idx": [1], "#regulators": [1], "regIdx1": [0], "K1": [2.0], "coop_state1": [3]}
    )
    pd.testing.assert_frame_equal(all_targets_dfs[0], expected_first_targets_df)

    expected_first_regs_df = pd.DataFrame({"Master regulator Idx": [0], "production_rate": [1.0]})
    pd.testing.assert_frame_equal(all_regs_dfs[0], expected_first_regs_df)


def test_merge_n_replica_dataframes_for_sergio_inputs():
    """Test merging of DataFrames for SERGIO inputs from multiple replicas."""
    # Create simple master regulators DataFrames for testing with two master regulators
    master_regulators_df1 = pd.DataFrame(
        {"Master regulator Idx": [0, 1], "production_rate": [1.0, 0.5]}
    )
    master_regulators_df2 = pd.DataFrame(
        {"Master regulator Idx": [0, 1], "production_rate": [1.5, 0.7]}
    )
    master_regulators_df3 = pd.DataFrame(
        {"Master regulator Idx": [0, 1], "production_rate": [2.0, 0.9]}
    )

    # Create a simple targets DataFrame for testing
    targets_df1 = pd.DataFrame(
        {"Target Idx": [1], "#regulators": [1], "regIdx1": [0], "K1": [2.0], "coop_state1": [3]}
    )
    targets_df2 = pd.DataFrame(
        {"Target Idx": [1], "#regulators": [1], "regIdx1": [0], "K1": [3.0], "coop_state1": [2]}
    )
    targets_df3 = pd.DataFrame(
        {"Target Idx": [1], "#regulators": [1], "regIdx1": [0], "K1": [1.0], "coop_state1": [2]}
    )

    # Call the function to test
    merged_targets_df, merged_regs_df = utility.merge_n_replica_dataframes_for_sergio_inputs(
        [targets_df1, targets_df2, targets_df3],  # Different targets DataFrame for all replicas
        [
            master_regulators_df1,
            master_regulators_df2,
            master_regulators_df3,
        ],  # Different master regulators DataFrames
        merge_type="first_only",
    )

    # Check that the merged master regulators DataFrame has the correct columns
    expected_columns = [
        "Master regulator Idx",
        "production_rate",
        "production_rate2",
        "production_rate3",
    ]
    assert (
        list(merged_regs_df.columns) == expected_columns
    ), "Merged master regulators DataFrame should have the correct columns."

    # Check the content of the merged master regulators DataFrame
    expected_merged_regs_df = pd.DataFrame(
        {
            "Master regulator Idx": [0, 1],
            "production_rate": [1.0, 0.5],
            "production_rate2": [1.5, 0.7],
            "production_rate3": [2.0, 0.9],
        }
    )
    pd.testing.assert_frame_equal(merged_regs_df, expected_merged_regs_df)
    pd.testing.assert_frame_equal(merged_targets_df, targets_df1)


def test_write_input_files_for_sergio():
    """Test writing of input files for SERGIO."""
    # Sample data frames with both floats and integers
    targets_df = pd.DataFrame(
        {
            "Target Idx": [0, 1, 2],
            "#regulators": [1, 2, 3],
            "regIdx1": [78, 4, 110],
            "regIdx2": [None, 6.0, None],
            "regIdx3": [None, 15.0, 58.0],
            "K1": [None, 91.0, None],
            "K2": [None, 114.0, None],
            "coop_state1": [2.789, -2.497, 4.972],
            "coop_state2": [None, 3.347, -3.222],
            "coop_state3": [None, None, 1.234],
        }
    )
    regs_df = pd.DataFrame(
        {
            "Master regulator Idx": [17, 24, 26],
            "production_rate1": [1.850344, 1.447644, 1.551863],
            "production_rate2": [None, None, 2.345],
        }
    )

    # Save the data frames to text files
    utility.write_input_files_for_sergio(targets_df, regs_df, "test_targets.txt", "test_regs.txt")

    # Check if the text files exist
    assert os.path.exists("test_targets.txt"), "The targets text file was not created."
    assert os.path.exists("test_regs.txt"), "The regulators text file was not created."

    # Check if the text files have the correct contents
    with open("test_targets.txt", "r") as f:
        targets_contents = f.read().strip()
        expected_contents_targets = (
            "0.0,1.0,78.0,2.789\n"
            "1.0,2.0,4.0,6.0,15.0,91.0,114.0,-2.497,3.347\n"  # Reflects the default precision
            "2.0,3.0,110.0,58.0,4.972,-3.222,1.234"  # No trailing newline for the last line
        )
        assert (
            targets_contents == expected_contents_targets
        ), "The targets text file has the wrong contents."

    with open("test_regs.txt", "r") as f:
        regs_contents = f.read().strip()
        expected_contents_regs = (
            "17.0,1.850344\n"
            "24.0,1.447644\n"
            "26.0,1.551863,2.345"  # Reflects the default precision and includes production_rate2 for index 26
        )
        assert (
            regs_contents == expected_contents_regs
        ), "The regulators text file has the wrong contents."

    # Clean up
    os.remove("test_targets.txt")
    os.remove("test_regs.txt")


def test_write_equation_info_to_file():
    """Test writing of equation information to a file."""
    # Mock data for hill_equations and parameter_values
    hill_equations = {"eq1": "Gene1 * Gene2", "eq2": "Gene3 + Gene4"}
    parameter_values = {"param1": 1.23, "param2": 4.56}

    # Test filename
    test_filename = "test_equation_info.txt"

    # Write the mock data to the test file
    utility.write_equation_info_to_file(hill_equations, parameter_values, test_filename)

    # Read the contents of the file
    with open(test_filename, "r") as file:
        contents = file.read()

    # Define the expected contents of the file
    expected_contents = (
        "Hill Equations:\n"
        "eq1: Gene1 * Gene2\n"
        "eq2: Gene3 + Gene4\n"
        "\nParameter Values:\n"
        "param1: 1.23\n"
        "param2: 4.56\n"
    )

    # Check if the contents match the expected output
    assert contents == expected_contents, "File contents do not match expected output."

    # Clean up and remove the test file
    os.remove(test_filename)
