import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, Tuple, Optional


def create_airport_graph(n: int, k: int, seed: Optional[int] = 42) -> nx.Graph:
    """
    Function to create and represent an airport graph
    """
    # Check if a k-regular graph is possible
    if (n * k) % 2 != 0:
        raise ValueError(f"Cannot create a {k}-regular graph with {n} vertices (n*k must be even)")

    if n <= k:
        raise ValueError(f"Number of vertices ({n}) must be greater than degree ({k})")

    try:
        G = nx.random_regular_graph(k, n, seed=seed)
        return G
    except Exception as e:
        raise ValueError(f"Cannot create {k}-regular graph with {n} vertices: {str(e)}")


def analyze_airport_graph(G: nx.Graph, verbose: bool = True) -> Dict:
    """
    Analyze properties of an airport graph
    """
    properties = {}

    # Basic properties
    properties['num_vertices'] = G.number_of_nodes()
    properties['num_edges'] = G.number_of_edges()
    properties['degrees'] = dict(G.degree())
    properties['total_degree'] = sum(properties['degrees'].values())
    properties['mean_degree'] = np.mean(list(properties['degrees'].values()))

    # Connectivity
    properties['is_connected'] = nx.is_connected(G)

    if properties['is_connected']:
        properties['diameter'] = nx.diameter(G)
        properties['radius'] = nx.radius(G)
        properties['avg_path_length'] = nx.average_shortest_path_length(G)
    else:
        properties['diameter'] = float('inf')
        properties['radius'] = float('inf')
        properties['num_components'] = nx.number_connected_components(G)

    # Planarity and faces
    properties['is_planar'], _ = nx.check_planarity(G)
    if properties['is_planar']:
        # Euler's formula: V - E + F = 2
        properties['num_faces'] = 2 - properties['num_vertices'] + properties['num_edges']
    else:
        properties['num_faces'] = None

    # Density
    properties['density'] = nx.density(G)

    # Clustering
    properties['avg_clustering'] = nx.average_clustering(G)

    # Matrices
    properties['adjacency_matrix'] = nx.adjacency_matrix(G).todense()
    properties['incidence_matrix'] = nx.incidence_matrix(G).todense()
    properties['laplacian_matrix'] = nx.laplacian_matrix(G).todense()

    # Check airport problem constraint (max 1 stopover)
    if properties['is_connected']:
        properties['satisfies_constraint'] = properties['diameter'] <= 2
        properties['max_stopovers'] = properties['diameter'] - 1
    else:
        properties['satisfies_constraint'] = False
        properties['max_stopovers'] = float('inf')

    if verbose:
        print_graph_properties(properties)

    return properties


def print_graph_properties(properties: Dict):
    """Print formatted graph properties"""
    print("\n" + "=" * 70)
    print("GRAPH PROPERTIES ANALYSIS")
    print("=" * 70)

    print(f"\nBASIC PROPERTIES:")
    print(f"  Number of vertices: {properties['num_vertices']}")
    print(f"  Number of edges: {properties['num_edges']}")
    print(f"  Total degree: {properties['total_degree']}")
    print(f"  Average degree: {properties['mean_degree']:.2f}")

    print(f"\nCONNECTIVITY:")
    print(f"  Connected: {'YES' if properties['is_connected'] else 'NO'}")
    if properties['is_connected']:
        print(f"  Diameter: {properties['diameter']}")
        print(f"  Radius: {properties['radius']}")
        print(f"  Average path length: {properties['avg_path_length']:.4f}")
    else:
        print(f"  Number of components: {properties['num_components']}")

    print(f"\nPLANARITY:")
    print(f"  Planar: {'YES' if properties['is_planar'] else 'NO'}")
    if properties['is_planar']:
        print(f"  Number of faces: {properties['num_faces']}")

    print(f"\nDENSITY & CLUSTERING:")
    print(f"  Density: {properties['density']:.4f} ({properties['density'] * 100:.2f}% of complete graph)")
    print(f"  Average clustering coefficient: {properties['avg_clustering']:.4f}")

    print(f"\nAIRPORT PROBLEM CONSTRAINT (max 1 stopover):")
    if properties['satisfies_constraint']:
        print(f"   SATISFIED: All cities reachable with at most {properties['max_stopovers']} stopover(s)")
    else:
        if properties['is_connected']:
            print(f"   NOT SATISFIED: Some cities require {properties['max_stopovers']} stopover(s)")
        else:
            print(f"   NOT SATISFIED: Graph is not connected")


def visualize_airport_graph(G: nx.Graph, properties: Dict, save_path: str = None):
    """
    Create visualization of the airport graph
    """
    fig = plt.figure(figsize=(16, 12))

    # 1. Graph visualization
    ax1 = plt.subplot(2, 3, 1)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=800, font_size=12, font_weight='bold',
            edge_color='gray', width=2, ax=ax1)
    ax1.set_title("Airport Network Graph", fontsize=14, fontweight='bold')

    # 2. Degree distribution
    ax2 = plt.subplot(2, 3, 2)
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degree_count = {}
    for degree in degree_sequence:
        degree_count[degree] = degree_count.get(degree, 0) + 1

    ax2.bar(degree_count.keys(), degree_count.values(), color='steelblue', edgecolor='black')
    ax2.set_xlabel("Degree", fontsize=12)
    ax2.set_ylabel("Number of vertices", fontsize=12)
    ax2.set_title("Degree Distribution", fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # 3. Adjacency matrix
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(properties['adjacency_matrix'], cmap='Blues', interpolation='nearest')
    plt.colorbar(im3, ax=ax3, label='Connection')
    ax3.set_title("Adjacency Matrix", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Vertex")
    ax3.set_ylabel("Vertex")

    # 4. Laplacian matrix
    ax4 = plt.subplot(2, 3, 4)
    im4 = ax4.imshow(properties['laplacian_matrix'], cmap='RdBu', interpolation='nearest')
    plt.colorbar(im4, ax=ax4, label='Value')
    ax4.set_title("Laplacian Matrix", fontsize=14, fontweight='bold')
    ax4.set_xlabel("Vertex")
    ax4.set_ylabel("Vertex")

    # 5. Properties summary
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    summary_text = f"""
    GRAPH PROPERTIES

    Vertices (n): {properties['num_vertices']}
    Edges (m): {properties['num_edges']}
    Degree: {properties['mean_degree']:.0f} (regular)

    Diameter: {properties['diameter']}
    Avg Path Length: {properties['avg_path_length']:.2f}

    Density: {properties['density']:.3f}
    Clustering: {properties['avg_clustering']:.3f}

    Connected: {'Yes' if properties['is_connected'] else 'No'}
    Planar: {'Yes' if properties['is_planar'] else 'No'}

    Airport Constraint:
    {'Satisfied' if properties['satisfies_constraint'] else ' Not Satisfied'}
    (Max {properties['max_stopovers']} stopover(s))
    """
    ax5.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # 6. Shortest path lengths histogram
    ax6 = plt.subplot(2, 3, 6)
    if properties['is_connected']:
        path_lengths = []
        for source in G.nodes():
            lengths = nx.single_source_shortest_path_length(G, source)
            path_lengths.extend(lengths.values())

        ax6.hist(path_lengths, bins=range(0, max(path_lengths) + 2),
                 color='green', alpha=0.7, edgecolor='black')
        ax6.set_xlabel("Path Length (number of flights)", fontsize=12)
        ax6.set_ylabel("Frequency", fontsize=12)
        ax6.set_title("Distribution of Shortest Path Lengths", fontsize=14, fontweight='bold')
        ax6.grid(axis='y', alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Graph is not connected',
                 ha='center', va='center', fontsize=12)
        ax6.set_title("Path Length Distribution", fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n Visualization saved: {save_path}")

    return fig


def compare_airport_configurations(n_values: list, k_value: int):
    """
    Compare different airport network configurations
    """
    print("\n" + "=" * 70)
    print(f"COMPARING AIRPORT NETWORKS WITH k={k_value} FLIGHTS PER AIRPORT")
    print("=" * 70)

    results = []

    for n in n_values:
        try:
            G = create_airport_graph(n, k_value)
            props = analyze_airport_graph(G, verbose=False)
            results.append({
                'n': n,
                'diameter': props['diameter'],
                'avg_path': props['avg_path_length'],
                'density': props['density'],
                'satisfies': props['satisfies_constraint']
            })
        except ValueError as e:
            print(f"\nSkipping n={n}: {e}")
            continue

    # Print comparison table
    print(f"\n{'n':<6} {'Edges':<8} {'Diameter':<10} {'Avg Path':<12} {'Density':<10} {'Constraint'}")
    print("-" * 70)

    for r in results:
        edges = r['n'] * k_value // 2
        print(f"{r['n']:<6} {edges:<8} {r['diameter']:<10} {r['avg_path']:<12.2f} "
              f"{r['density']:<10.3f} {'✓' if r['satisfies'] else '✗'}")


if __name__ == "__main__":
    print("=" * 70)
    print("AIRPORT PROBLEM")
    print("=" * 70)

    # Problem parameters
    n = 6  # number of cities (vertices)
    k = 3  # number of flights per airport (degree of each vertex)

    print(f"\nProblem Parameters:")
    print(f"  Number of cities (n): {n}")
    print(f"  Flights per airport (k): {k}")

    # Question 1: Create the graph using NetworkX
    print("\n" + "=" * 70)
    print("1. GRAPH CREATION")
    print("=" * 70)
    G = create_airport_graph(n, k)
    print(f" Created {k}-regular graph with {n} vertices")

    # Question 2: Visualize using Matplotlib (will be done later with all plots)
    print("\n" + "=" * 70)
    print("2. VISUALIZATION")
    print("=" * 70)
    print(" Graph visualization prepared (see output plot)")

    # Questions 3-9: Analyze all properties
    print("\n" + "=" * 70)
    print("3-9. GRAPH ANALYSIS")
    print("=" * 70)
    properties = analyze_airport_graph(G, verbose=True)

    # Print matrices
    print("\n" + "=" * 70)
    print("8. MATRICES")
    print("=" * 70)
    print("\na) Adjacency Matrix:")
    print(properties['adjacency_matrix'])

    print("\nb) Incidence Matrix (first 10 edges):")
    inc_matrix = properties['incidence_matrix']
    print(inc_matrix[:, :min(10, inc_matrix.shape[1])])

    print("\nc) Laplacian Matrix:")
    print(properties['laplacian_matrix'])

    # Question 10: Function to create and analyze airport graphs
    print("\n" + "=" * 70)
    print("10. PARAMETERIZED FUNCTION FOR AIRPORT GRAPH")
    print("=" * 70)
    print(" Function create_airport_graph(n, k) created")
    print(" Function analyze_airport_graph(G) created")
    print("\nTesting with different configurations:")

    # Test different configurations
    test_configs = [
        (4, 3),
        (6, 3),
        (8, 3),
        (10, 4),
        (12, 5)
    ]

    print("\nTesting various (n, k) configurations:")
    print(f"{'n':<4} {'k':<4} {'Edges':<8} {'Diameter':<10} {'Avg Path':<12} {'Satisfies Constraint'}")
    print("-" * 70)

    for n_test, k_test in test_configs:
        try:
            G_test = create_airport_graph(n_test, k_test)
            props_test = analyze_airport_graph(G_test, verbose=False)
            edges = n_test * k_test // 2
            print(f"{n_test:<4} {k_test:<4} {edges:<8} {props_test['diameter']:<10} "
                  f"{props_test['avg_path_length']:<12.2f} "
                  f"{'✓ Yes' if props_test['satisfies_constraint'] else '✗ No'}")
        except ValueError as e:
            print(f"{n_test:<4} {k_test:<4} Error: {str(e)}")

    # Compare networks with same k but different n
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS")
    print("=" * 70)
    compare_airport_configurations([6, 8, 10, 12, 14, 16], k_value=3)

    # Create comprehensive visualization
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    visualize_airport_graph(G, properties, 'airport_graph_complete_analysis.png')

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\n Summary:")
    print(f"  - Created and analyzed {k}-regular graph with {n} vertices")
    print(
        f"  - Airport constraint (max 1 stopover): {' SATISFIED' if properties['satisfies_constraint'] else ' NOT SATISFIED'}")
    print(f"  - Average flight path: {properties['avg_path_length']:.2f} flights")
    print(f"  - Network density: {properties['density']:.1%}")
    print(f"\n Outputs saved:")
    print(f"  - airport_graph_complete_analysis.png")