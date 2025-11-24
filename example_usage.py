"""
Example usage of the create_airport_graph() function (Question 10)

This script demonstrates how we can use the parametric function to create
and analyze airport graphs with different configurations.
"""

import matplotlib.pyplot as plt
import networkx as nx
import sys

from tp1 import create_airport_graph, analyze_airport_graph, visualize_airport_graph

print("=" * 70)
print("QUESTION 10 - AIRPORT GRAPH EXAMPLES")
print("=" * 70)

# Example 1: Small network with 4 cities
print("\n" + "=" * 70)
print("EXAMPLE 1: Small Network (n=4, k=3)")
print("=" * 70)
try:
    G1 = create_airport_graph(n=4, k=3)
    props1 = analyze_airport_graph(G1, verbose=True)
    print(f"\n With 4 cities and 3 flights each:")
    print(f"  - Diameter: {props1['diameter']}")
    print(f"  - Max stopovers: {props1['max_stopovers']}")
    print(f"  - Constraint satisfied: {props1['satisfies_constraint']}")
except ValueError as e:
    print(f"✗ Error: {e}")

# Example 2: Medium network with 10 cities
print("\n" + "=" * 70)
print("EXAMPLE 2: Medium Network (n=10, k=4)")
print("=" * 70)
try:
    G2 = create_airport_graph(n=10, k=4)
    props2 = analyze_airport_graph(G2, verbose=True)
    print(f"\n With 10 cities and 4 flights each:")
    print(f"  - Diameter: {props2['diameter']}")
    print(f"  - Max stopovers: {props2['max_stopovers']}")
    print(f"  - Constraint satisfied: {props2['satisfies_constraint']}")
except ValueError as e:
    print(f"✗ Error: {e}")

# Example 3: Large network with 20 cities
print("\n" + "=" * 70)
print("EXAMPLE 3: Large Network (n=20, k=5)")
print("=" * 70)
try:
    G3 = create_airport_graph(n=20, k=5)
    props3 = analyze_airport_graph(G3, verbose=True)
    print(f"\n With 20 cities and 5 flights each:")
    print(f"  - Diameter: {props3['diameter']}")
    print(f"  - Max stopovers: {props3['max_stopovers']}")
    print(f"  - Constraint satisfied: {props3['satisfies_constraint']}")

    # Create visualization for this larger network
    visualize_airport_graph(G3, props3, 'airport_example_n20_k5.png')
except ValueError as e:
    print(f" Error: {e}")

# Example 4: Invalid configuration (demonstrates error handling)
print("\n" + "=" * 70)
print("EXAMPLE 4: Invalid Configuration (n=5, k=3)")
print("=" * 70)
try:
    G4 = create_airport_graph(n=5, k=3)  # This should fail: 5*3=15 is odd
    props4 = analyze_airport_graph(G4)
except ValueError as e:
    print(f"✓ Correctly caught error: {e}")

# Comparative analysis: Find optimal k for different city counts
print("\n" + "=" * 70)
print("FINDING OPTIMAL CONFIGURATIONS")
print("=" * 70)
print("\nFor different city counts, which k value satisfies the constraint?")
print(f"\n{'Cities (n)':<12} {'Flights (k)':<14} {'Diameter':<12} {'Satisfies?'}")
print("-" * 70)

city_counts = [6, 8, 10, 12, 15, 20]
for n in city_counts:
    best_k = None
    for k in range(2, min(n, 10)):
        if (n * k) % 2 == 0:  # Check if valid
            try:
                G = create_airport_graph(n, k)
                props = analyze_airport_graph(G, verbose=False)
                status = "✓" if props['satisfies_constraint'] else "✗"
                print(f"{n:<12} {k:<14} {props['diameter']:<12} {status}")
                if props['satisfies_constraint'] and best_k is None:
                    best_k = k
            except:
                continue
    if best_k:
        print(f"  → Minimum k for n={n}: k={best_k}")
    print()

