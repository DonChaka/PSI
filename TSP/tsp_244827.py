import time
from dataclasses import dataclass
from typing import List, Callable, NoReturn
import numpy as np
import random
import matplotlib.pyplot as plt

random.seed()

SIZE = 5
cities = []
total = np.math.factorial(SIZE - 1) / 2
directional = False
broken = False
distances = np.zeros((SIZE, SIZE))


@dataclass
class HeuristicNode:
    path: List[int]
    wip_path: List[List[int]]

    def make_path(self):
        self.path.append(self.wip_path[0][0])
        self.path.append(self.wip_path[0][1])
        self.wip_path.pop(0)
        while len(self.wip_path):
            for node in self.wip_path:
                if node[1] == self.path[-1]:
                    self.path.append(node[1])
                    self.wip_path.remove(node)
                    break

    def allowed_route(self, point):
        for x in self.wip_path:
            if x[0] == point[1] or x[1] == point[0]:
                test = point
                for y in self.wip_path:
                    if y != x and y[1] == point[2]:
                        return False


def greedy_heuristic():
    dists = distances.copy()
    route = HeuristicNode([], [])

    while len(route.wip_path) < SIZE:
        pos = np.argwhere(dists == np.min(dists))
        route.allowed_route(pos[0])
        route.wip_path.append(pos[0])
        dists[pos[0, 0], :] = float('inf')
        dists[:, pos[0, 1]] = float('inf')
        dists[pos[0, 1], pos[0, 0]] = float('inf')

    route.make_path()
    print(repr(route))


@dataclass(order=True)
class Node:
    path: List[int]
    cost: float
    remaining: List[int] = None
    c: int = 0

    def __post_init__(self):
        self.remaining = [i for i in range(SIZE) if i not in self.path]

    def __str__(self):
        return f'Sciezka: {self.path}, koszt={self.cost}'


def calculate_heuristic(node: Node, func: Callable) -> NoReturn:
    if len(node.remaining) == 0:
        node.c = 0
    else:
        node.c = func(
            [distances[node.path[-1], i] for i in node.remaining if distances[node.path[-1], i] != float('inf')])

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


def nearest_neighbour(node: Node):
    if len(node.remaining) == 0:
        node.path.append(node.path[0])
        node.cost += distances[node.path[-2], node.path[-1]]
        return node

    new_path = node.path.copy()
    new_path.append(node.remaining[0])
    nearest = Node(new_path, node.cost)
    nearest.cost += distances[nearest.path[-2], nearest.path[-1]]
    for i in node.remaining[1:]:
        new_path = node.path.copy()
        new_path.append(i)
        temp = Node(new_path, node.cost)
        temp.cost += distances[temp.path[-2], temp.path[-1]]
        if temp.cost < nearest.cost:
            nearest = temp

    return nearest_neighbour(nearest)


def a_star(node: Node, heuristic: Callable) -> Node:
    nodes = [node]
    calculate_heuristic(node, heuristic)

    while True:
        current = min(nodes, key=lambda x: x.c)
        if len(current.remaining) == 0:
            current.path.append(current.path[0])
            current.cost += distances[current.path[-2], current.path[-1]]
            return current

        left = current.remaining.copy()
        while len(left):
            new_temp_path = current.path.copy()
            new_temp_path.append(left[0])
            child = Node(new_temp_path, current.cost)
            left.pop(0)
            child.cost += distances[child.path[-2], child.path[-1]]
            calculate_heuristic(child, heuristic)
            nodes.append(child)

        nodes.remove(current)


def calculate_costs(dsts):
    for from_counter, from_node in enumerate(cities):
        for to_counter, to_node in enumerate(cities):
            if from_counter == to_counter:
                dsts[from_counter][to_counter] = float("inf")
            else:
                dsts[from_counter][to_counter] = np.linalg.norm(from_node - to_node)
                if directional and from_node[2] > to_node[2]:
                    dsts[from_counter][to_counter] *= 0.9
                elif directional and from_node[2] < to_node[2]:
                    dsts[from_counter][to_counter] *= 1.1
            if broken and random.random() > 0.8:
                dsts[from_counter][to_counter] = float("inf")

    return dsts


def plot_results(answer, title=''):
    x = [i[0] for i in answer]
    y = [i[1] for i in answer]
    z = [i[2] for i in answer]

    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(x, y, z, c='red', s=100)
    ax.plot(x, y, z, color='black')
    plt.title(title)
    plt.show()


def run_dfs():
    print('Algorytm DFS')
    dfs_start = time.time()
    depth_leaves_full_asym: List[Node] = []

    root = Node([0], 0)
    depth_generate_tree_leaves(root, depth_leaves_full_asym)

    shortest = depth_leaves_full_asym[0]
    for leaf in depth_leaves_full_asym:
        if leaf.cost < shortest.cost:
            shortest = leaf
    _time = round((time.time() - dfs_start), 5)
    print(shortest)
    print(f'Czas dzialania: {_time}', end='\n\n')
    plot_results([cities[i] for i in shortest.path], title='Trasa wyznaczona przez DFS')


def run_bfs():
    print('Algorytm BFS')
    bfs_start = time.time()
    root = Node([0], 0)
    breadth_leaves_full_asym: List[Node] = []
    breadth_generate_tree_leaves([root], breadth_leaves_full_asym)
    shortest = breadth_leaves_full_asym[0]
    for leaf in breadth_leaves_full_asym:
        if leaf.cost < shortest.cost:
            shortest = leaf
    _time = round((time.time() - bfs_start), 5)
    print(shortest)
    print(f'Czas dzialania: {_time}', end='\n\n')
    plot_results([cities[i] for i in shortest.path], title='Trasa wyznaczona przez algorytm bfs')


def run_nearest_neighbour():
    print('Algorytm najblizszego sasiada')
    nn_start = time.time()
    root = Node([0], 0)
    nearest_route = nearest_neighbour(root)
    print(nearest_route)
    _time = round((time.time() - nn_start), 5)
    print(f'Czas dzialania: {_time}', end='\n\n')
    plot_results([cities[i] for i in nearest_route.path], title='Trasa wyznaczona przez nearest neighboiur')


def run_a_star(heuristic: Callable):
    print(f'Algorytm A* korzystajacy z heurystyki {heuristic.__name__}')
    a_start = time.time()
    root = Node([0], 0)
    a_star_ans = a_star(root, heuristic)
    print(a_star_ans)
    _time = round((time.time() - a_start), 5)
    print(f'Czas dzialania: {_time}', end='\n\n')
    plot_results([cities[i] for i in a_star_ans.path], title=f'Trasa wyznaczona przez a* z heurystyką {heuristic.__name__}')

def main():
    global cities
    cities = [rand_coords() for i in range(SIZE)]
    calculate_costs(distances)

    run_a_star(np.min)

    run_a_star(np.mean)

if __name__ == '__main__':
    main()
