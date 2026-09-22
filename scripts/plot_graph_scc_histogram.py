from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx


def _load_graph_from_graphml(graphml_path: str) -> nx.DiGraph:
    g = nx.read_graphml(graphml_path)
    # GraphML reader can return MultiDiGraph depending on attributes.
    if not isinstance(g, nx.DiGraph):
        g = nx.DiGraph(g)

    # Force node IDs to be strings for compatibility with y0 Variable(name).
    g2 = nx.DiGraph()
    g2.add_nodes_from([str(n) for n in g.nodes()])
    g2.add_edges_from([(str(u), str(v)) for u, v in g.edges()])
    return g2


def scc_size_histogram(G: nx.DiGraph, plot=True, ax=None):
    """
    Given a NetworkX DiGraph, compute a histogram of strongly connected
    component sizes. Excludes SCCs of size 1.

    Returns
    -------
    hist : dict
        Mapping from SCC size to number of SCCs of that size.
        Example: {1: 10, 3: 2} means 10 SCCs of size 1 and 2 SCCs of size 3.
    """
    if not isinstance(G, nx.DiGraph):
        raise TypeError("Input must be a networkx.DiGraph")

    sccs = list(nx.strongly_connected_components(G))
    max_scc = max(sccs, key=lambda x: len(x))
    print(max_scc, len(max_scc))
    scc_sizes = [len(scc) for scc in sccs if len(scc) != 1]
    hist = dict(Counter(scc_sizes))

    if plot:
        if ax is None:
            fig, ax = plt.subplots()

        sizes = sorted(hist)
        counts = [hist[size] for size in sizes]

        ax.bar(sizes, counts, width=0.8, edgecolor="black")
        ax.set_xlabel("SCC size")
        ax.set_ylabel("Number of SCCs")
        ax.set_title("Histogram of SCC Sizes")

        plt.savefig("scc_hist.png")

    return hist


G = _load_graph_from_graphml(
    "notebooks/Ecoli_Analysis_Notebooks/ecoli_full_network_no_small_rna.graphml"
)
hist = scc_size_histogram(G)
print(hist)
