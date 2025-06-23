import os
from collections import defaultdict
from typing import Dict, List, Set, Any

import matplotlib.pyplot as plt
import networkx as nx


def convert_to_binary_tree(
    graph: Dict[Any, List[Any]], root: Any
) -> Dict[Any, List[Any]]:
    """
    Convert a general graph to a binary tree starting from the given root.

    Args:
        graph (Dict[Any, List[Any]]): The input graph represented as an adjacency list.
        root (Any): The root node to start the conversion.

    Returns:
        Dict[Any, List[Any]]: The resulting binary tree as an adjacency list.
    """
    binary_tree: Dict[Any, List[Any]] = defaultdict(list)
    visited: Set[Any] = set()

    def dfs(current: Any) -> None:
        visited.add(current)
        children = [n for n in graph[current] if n not in visited]

        if not children:
            return

        leftmost = children[0]
        binary_tree[current].append(leftmost)
        dfs(leftmost)

        # Siblings become right child chain
        prev = leftmost
        for sibling in children[1:]:
            binary_tree[prev].append(sibling)
            dfs(sibling)
            prev = sibling

    dfs(root)
    return binary_tree


def binary_tree_to_list(binary_tree: Dict[Any, List[Any]], root: Any) -> List[Any]:
    """
    Convert a binary tree represented as an adjacency list to a list using pre-order traversal.

    Args:
        binary_tree (Dict[Any, List[Any]]): The binary tree as an adjacency list.
        root (Any): The root node of the binary tree.

    Returns:
        List[Any]: The list of nodes in pre-order traversal.
    """
    result: List[Any] = []

    def dfs(node: Any) -> None:
        if node is None:
            return
        result.append(node)
        children = binary_tree.get(node, [])
        left = children[0] if len(children) > 0 else None
        right = children[1] if len(children) > 1 else None
        dfs(left)
        dfs(right)

    dfs(root)
    return result


def visualize_graph(graph: Dict[Any, List[Any]]) -> None:
    """
    Visualize a graph using NetworkX and Matplotlib.

    Args:
        graph (Dict[Any, List[Any]]): The input graph represented as an adjacency list.

    Returns:
        None
    """
    G = nx.Graph()

    for node, neighbors in graph.items():
        node_name = os.path.basename(node)
        G.add_node(node_name)
        for neighbor in neighbors:
            neighbor_name = os.path.basename(neighbor)
            G.add_edge(node_name, neighbor_name)

    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=500)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title("🔗 Graph of images")
    plt.axis("off")
    plt.show()
