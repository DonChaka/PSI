import time
from dataclasses import dataclass
from typing import List, Callable, NoReturn
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

random.seed()

SIZE = 25
cities = []
total = np.math.factorial(SIZE - 1) / 2
directional = True
broken = True
distances = np.zeros((SIZE, SIZE))


def timeit(method):
    def timed(*args, **kwargs):
        start = time.time()
        res = method(*args, **kwargs)
        _time = time.time() - start
        return res, _time

    return timed


def softmax(vector):
    vector = np.array(vector)
    z = vector - max(vector)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    return list(numerator / denominator)


@dataclass(order=True, eq=True)
class Node:
    path: List[int]
    cost: float
    remaining: List[int] = None
    heuristic_value: float = 0

    def __post_init__(self):
        self.remaining = [i for i in range(SIZE) if i not in self.path]

    def __str__(self):
        return f'Sciezka: {self.path}, koszt={self.cost}'
        # return f'Sciezka: {self.path}, koszt={self.cost}, heurystyka={self.heuristic_value}'

    def __repr__(self):
        return f'Sciezka: {self.path}, koszt={self.cost}, heurystyka={self.heuristic_value}'


def check_cut(dists: np.ndarray, i: int, j: int) -> bool:
    if i == j:
        return False
    visited = np.zeros(SIZE, dtype=bool)
    queue = [i]
    while len(queue) > 0:
        node = queue.pop(0)
        visited[node] = True
        for k in range(SIZE):
            if dists[node][k] != float('inf') and not visited[k]:
                queue.append(k)
    return visited[j]


def prune_branches(dists: np.ndarray, chance: float = 0.2) -> np.ndarray:
    for i in range(SIZE):
        for j in range(SIZE):
            if random.random() < chance:
                if check_cut(dists, i, j):
                    dists[i][j] = float('inf')
    return dists


def rand_coords():
    return np.array((random.randrange(-100, 100), random.randrange(-100, 100), random.randrange(0, 50)))


def cost_between(a, b, asym):
    cost = np.linalg.norm(a - b)
    if asym and a[2] > b[2]:
        cost *= 0.9
    elif asym and a[2] < b[2]:
        cost *= 1.1
    return cost


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

    return dsts


def plot_results(answer, title=''):
    x = [i[0] for i in answer]
    y = [i[1] for i in answer]
    z = [i[2] for i in answer]

    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(x[0], y[0], z[0], color=(0, 0, 1, 0.5), s=100)
    ax.scatter(x[1:], y[1:], z[1:], color=(1, 0, 0, 0.5), s=100)
    ax.plot(x, y, z, color='black')
    plt.title(title)
    plt.show()


def calculate_heuristic(node: Node, func: Callable) -> NoReturn:
    if len(node.remaining) == 0:
        node.heuristic_value = 0
    else:
        temp = [distances[node.path[-1], i] for i in node.remaining if distances[node.path[-1], i] != float('inf')]
        if len(temp):
            node.heuristic_value = len(node.remaining) * func(temp)
        else:
            node.heuristic_value = float('inf')


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


def depth_nearest_neighbour(node: Node):
    if len(node.remaining) == 0:
        node.path.append(node.path[0])
        node.cost += distances[node.path[-2], node.path[-1]]
        return node

    new_path = node.path.copy()
    new_path.append(node.remaining[0])
    nearest = Node(new_path, node.cost)
    nearest.cost += distances[nearest.path[-2], nearest.path[-1]]
    double_cost = float("inf")
    for i in node.remaining[1:]:
        new_path = node.path.copy()
        new_path.append(i)
        temp = Node(new_path, node.cost)
        temp.cost += distances[temp.path[-2], temp.path[-1]]
        for j in temp.remaining:
            new_path = temp.path.copy()
            new_path.append(j)
            temp2 = Node(new_path, temp.cost)
            temp2.cost += distances[temp2.path[-2], temp2.path[-1]]
            if temp2.cost < double_cost:
                double_cost = temp2.cost
                nearest = temp

    return depth_nearest_neighbour(nearest)


def a_star(node: Node, heuristic: Callable) -> Node:
    calculate_heuristic(node, heuristic)
    nodes = deque([node])
    while True:
        current = min(nodes, key=lambda x: x.heuristic_value)
        if len(current.remaining) == 0:
            current.path.append(current.path[0])
            current.cost += distances[current.path[-2], current.path[-1]]
            return current

        for i in current.remaining:
            new_temp_path = current.path.copy()
            new_temp_path.append(i)
            child = Node(new_temp_path, current.cost)
            child.cost += distances[child.path[-2], child.path[-1]]
            if child.cost != float('inf'):
                calculate_heuristic(child, heuristic)
                nodes.append(child)
        nodes.remove(current)


def aco(n_iter=100, n_ants=100, alpha=0.75, beta=1.25, Q=5, ro=0.05):
    pheromones = np.ones((SIZE, SIZE))
    dists = distances.copy()
    best_ant = None
    best_cost = float('inf')
    for t in range(n_iter):
        for a in range(n_ants):
            start = random.choice(range(SIZE))
            ant = Node([start], 0)
            while ant.remaining:
                p = {}
                for j in ant.remaining:
                    if dists[ant.path[-1], j] == float('inf'):
                        continue
                    p[j] = pheromones[ant.path[-1], j] ** alpha * (1 / dists[ant.path[-1], j]) ** beta
                if not list(p.keys()):
                    ant.cost = float('inf')
                    ant.path = []
                    break
                probs = list(p.values())
                probs = softmax(probs)
                choice = np.random.choice(list(p.keys()), p=probs)
                ant.path.append(choice)
                ant.cost += dists[ant.path[-2], ant.path[-1]]
                ant.remaining.remove(choice)

            for i in range(1, len(ant.path)):
                pheromones[ant.path[i - 1], ant.path[i]] += Q * (dists[ant.path[i - 1], ant.path[i]] / ant.cost)
            pheromones *= 1 - ro
            if ant.cost < best_cost:
                best_ant = ant
    best_ant.path.append(best_ant.path[0])
    best_ant.cost += dists[best_ant.path[-2], best_ant.path[-1]]
    return best_ant


@timeit
def run_dfs():
    # print('Algorytm DFS')
    depth_leaves_full_asym: List[Node] = []
    root = Node([0], 0)
    depth_generate_tree_leaves(root, depth_leaves_full_asym)

    shortest = depth_leaves_full_asym[0]
    for leaf in depth_leaves_full_asym:
        if leaf.cost < shortest.cost:
            shortest = leaf
    # print(shortest)
    # plot_results([cities[i] for i in shortest.path],
    #              title=f'Trasa wyznaczona przez DFS, koszt={round(shortest.cost, 2)}')
    return shortest.cost


@timeit
def run_bfs():
    # print('Algorytm BFS')
    root = Node([0], 0)
    breadth_leaves_full_asym: List[Node] = []
    breadth_generate_tree_leaves([root], breadth_leaves_full_asym)
    shortest = breadth_leaves_full_asym[0]
    for leaf in breadth_leaves_full_asym:
        if leaf.cost < shortest.cost:
            shortest = leaf
    # print(shortest)
    # plot_results([cities[i] for i in shortest.path],
    #              title=f'Trasa wyznaczona przez algorytm BFS, koszt={round(shortest.cost, 2)}')
    return shortest.cost


@timeit
def run_nearest_neighbour():
    # print('Algorytm najblizszego sasiada')
    shortest = float('inf')
    i = 0
    while shortest == float('inf'):
        root = Node([i], 0)
        shortest = nearest_neighbour(root)
        i += 1
    # print(shortest)
    # plot_results([cities[i] for i in shortest.path],
    #              title=f'Trasa wyznaczona przez nearest neighboiur, koszt={round(shortest.cost, 2)}')
    return shortest.cost


@timeit
def run_depth_nearest_neighbour():
    # print('Algorytm najblizszego sasiadaz glebia 2')
    shortest = float('inf')
    i = 0
    while shortest == float('inf'):
        root = Node([i], 0)
        shortest = depth_nearest_neighbour(root)
        i += 1
    # print(shortest)
    # plot_results([cities[i] for i in shortest.path],
    #              title=f'Trasa wyznaczona przez nearest neighboiur z glebia 2, koszt={round(shortest.cost, 2)}')
    return shortest.cost


@timeit
def run_a_star(heuristic: Callable):
    # print(f'Algorytm A* korzystajacy z heurystyki {heuristic.__name__}')
    root = Node([0], 0)
    shortest = a_star(root, heuristic)
    # print(shortest)
    # plot_results([cities[i] for i in shortest.path],
    #              title=f'Trasa wyznaczona przez a* z heurystyką {heuristic.__name__}, koszt={round(shortest.cost, 2)}')
    return shortest.cost


@timeit
def run_aco(n_iter=100, n_ants=100, alpha=1, beta=1, Q=5, ro=0.05):
    # print('ALgorytm aco')
    shortest = aco(n_iter, n_ants, alpha, beta, Q, ro)
    # print(shortest)
    # plot_results([cities[i] for i in shortest.path],
    #              title=f'Trasa wyznaczona przez algorytm aco, koszt={round(shortest.cost, 2)}')
    return shortest.cost


def main():
    global cities, SIZE, distances, total
    n_cities = [5, 10, 15, 20, 25]
    metody = ['dfs', 'bfs', 'nearest_neighbour', 'depth_nearest_neighbour', 'a_star_avg', 'a_star_min', 'aco']
    wyniki = {metoda: [(0, 0) for _ in range(max(n_cities)+1)] for metoda in metody}
    for i in n_cities:
        print(f'starting for {i} cities')
        SIZE = i
        cities = []
        total = np.math.factorial(SIZE - 1) / 2
        distances = np.zeros((SIZE, SIZE))

        cities = [rand_coords() for _ in range(SIZE)]
        calculate_costs(distances)
        if broken:
            prune_branches(distances)

        if SIZE <= 11:
            wyniki['dfs'][i] = run_dfs()
            wyniki['bfs'][i] = run_bfs()
        else:
            wyniki['dfs'][i] = (float('inf'), float('inf'))
            wyniki['bfs'][i] = (float('inf'), float('inf'))

        print(f'current: nearest neighbour')
        wyniki['nearest_neighbour'][i] = run_nearest_neighbour()
        print(f'current: depth nearest neighbour')
        wyniki['depth_nearest_neighbour'][i] = run_depth_nearest_neighbour()
        print(f'current: a* avg')
        wyniki['a_star_avg'][i] = run_a_star(heuristic=np.average)
        print(f'current: a* min')
        wyniki['a_star_min'][i] = run_a_star(heuristic=np.min)
        print(f'current: aco')
        wyniki['aco'][i] = run_aco()

    plt.figure(figsize=(10, 10))
    plt.title('Czas wykonania algorytmów dla różnych liczby miast')
    plt.xlabel('Liczba miast')
    plt.ylabel('Czas wykonania [s]')
    plt.xlim(min(n_cities), max(n_cities))
    plt.grid(True)
    for metoda in metody:
        temp = [wyniki[metoda][i][1] for i in n_cities]
        plt.plot(n_cities, temp, label=metoda)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.title('Czas wykonania algorytmów dla różnych liczby miast 2')
    plt.xlabel('Liczba miast')
    plt.ylabel('Czas wykonania [s]')
    plt.xlim(min(n_cities), max(n_cities))
    plt.grid(True)
    for metoda in metody:
        if metoda not in ['dfs', 'bfs', 'aco']:
            temp = [wyniki[metoda][i][1] for i in n_cities]
            plt.plot(n_cities, temp, label=metoda)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.title('Koszt uzyskany przez algorytmy dla różnych liczby miast')
    plt.xlabel('Liczba miast')
    plt.ylabel('Koszt')
    plt.xlim(min(n_cities), max(n_cities))
    plt.grid(True)
    for metoda in metody:
        temp = [wyniki[metoda][i][0] for i in n_cities]
        plt.plot(n_cities, temp, label=metoda, alpha=0.75)
    plt.legend()
    plt.show()

    # print(wyniki)


if __name__ == '__main__':
    main()
