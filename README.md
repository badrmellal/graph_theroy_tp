# Airport Graph Theory 

## Overview
This solution addresses the airport problem where:
- **n cities** each have an airport
- Each airport can handle **k flights**
- Goal: Allow travel between any two cities with **at most 1 stopover**

### `tp1.py` - MAIN FILE
Addressing all 10 questions:

1. **Graph Creation**: Using NetworkX to create k-regular graph
2. **Visualization**: Using Matplotlib to display the graph
3. **Diameter Calculation**: Maximum distance between any two vertices
4. **Vertex/Edge/Face Count**: Using Euler's formula for planar graphs
5. **Degree Analysis**: Total and average degree calculation
6. **Degree Distribution**: Histogram of vertex degrees
7. **Connectivity Check**: Determine if graph is connected
8. **Matrix Generation**: Adjacency, Incidence, and Laplacian matrices
9. **Density Calculation**: Graph density and network efficiency
10. **Parametric Function**: `create_airport_graph(n, k)` with full analysis

## Results Summary

### For n=6, k=3:
 **CONSTRAINT SATISFIED**
- Diameter: 2 (max 1 stopover)
- Edges: 9 flights
- Average path: 1.40 flights
- Density: 60%

## Quick Start
```bash
# Run complete analysis
python tp1.py

# See examples
python example_usage.py
```
