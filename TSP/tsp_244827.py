from dataclasses import dataclass
from typing import List
import numpy as np
import random
import matplotlib.pyplot as plt

random.seed()

SIZE = 11
cities = []
total = np.math.factorial(SIZE - 1) / 2
directional = True
broken = False
distances = np.zeros((SIZE, SIZE))


@dataclass(order=True)
class Node:
    path: List[int]
    cost: float
    remaining: List[int] = None

    def __post_init__(self):
        self.remaining = [i for i in range(SIZE) if i not in self.path]


def rand_coords():
    return np.array((random.randrange(-100, 100), random.randrange(-100, 100), random.randrange(0, 50)))


def cost_between(a, b, asym):
    cost = np.linalg.norm(a - b)
    if asym and a[2] > b[2]:
        cost *= 0.9
    elif asym and a[2] < b[2]:
        cost *= 1.1
    return cost


def depth_generate_tree_leaves(node: Node, leaves: List[Node]):
    if not directional and len(leaves) >= total:
        return

    if len(node.remaining) == 0:
        node.path.append(node.path[0])
        node.cost += distances[node.path[-2], node.path[-1]]
        leaves.append(node)
        return

    while len(node.remaining) > 0:
        new_path = node.path.copy()
        new_path.append(node.remaining[0])
        child = Node(new_path, node.cost)
        node.remaining.pop(0)
        child.cost += distances[child.path[-2], child.path[-1]]
        depth_generate_tree_leaves(child, leaves)


def breadth_generate_tree_leaves(nodes: List[Node], leaves: List[Node]):
    if not directional and len(leaves) >= total:
        return

    next_level_nodes = []
    for node in nodes:
        if len(node.remaining) == 0:
            node.path.append(node.path[0])
            node.cost += distances[node.path[-2], node.path[-1]]
            leaves.append(node)

        for i in node.remaining:
            new_path = node.path.copy()
            new_path.append(i)
            child = Node(new_path, node.cost)
            child.cost += distances[child.path[-2], child.path[-1]]
            next_level_nodes.append(child)

    if len(next_level_nodes) == 0:
        return
    breadth_generate_tree_leaves(next_level_nodes, leaves)


def calculate_costs(dsts):
    for from_counter, from_node in enumerate(cities):
        for to_counter, to_node in enumerate(cities):
            if from_counter == to_counter:
                dsts[from_counter][to_counter] = 0
            else:
                dsts[from_counter][to_counter] = np.linalg.norm(from_node - to_node)
                if directional and from_node[2] > to_node[2]:
                    dsts[from_counter][to_counter] *= 0.9
                elif directional and from_node[2] < to_node[2]:
                    dsts[from_counter][to_counter] *= 1.1
            if broken and random.random() > 0.8:
                dsts[from_counter][to_counter] = float("inf")

    return dsts


def plot_results(answer):
    x = [i[0] for i in answer]
    y = [i[1] for i in answer]
    z = [i[2] for i in answer]

    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(x, y, z, c='red', s=100)
    ax.plot(x, y, z, color='black')
    plt.show()


def main():
    global cities
    cities = [rand_coords() for i in range(SIZE)]
    calculate_costs(distances)

    depth_leaves_full_asym: List[Node] = []
    breadth_leaves_full_asym: List[Node] = []

    root = Node([0], 0)
    depth_generate_tree_leaves(root, depth_leaves_full_asym)

    root = Node([0], 0)
    breadth_generate_tree_leaves([root], breadth_leaves_full_asym)

    shortest = depth_leaves_full_asym[0]
    for leaf in depth_leaves_full_asym:
        if leaf.cost < shortest.cost:
            shortest = leaf
    print(repr(shortest))

    cost = 0
    for i in range(len(shortest.path) - 1):
        cost += cost_between(cities[shortest.path[i]], cities[shortest.path[i+1]], True)
    print(f"True cost = {cost}")

    shortest = breadth_leaves_full_asym[0]
    for leaf in breadth_leaves_full_asym:
        if leaf.cost < shortest.cost:
            shortest = leaf
    print(repr(shortest))
    cost = 0
    for i in range(len(shortest.path) - 1):
        cost += cost_between(cities[shortest.path[i]], cities[shortest.path[i + 1]], True)
    print(f"True cost = {cost}")

    plot_results([cities[i] for i in shortest.path])

if __name__ == '__main__':
    main()
