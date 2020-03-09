# TODO: Measure time of functions executions

import random
import math
import os
import matplotlib.pyplot as plt
from itertools import permutations
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

sample = [
    (288, 149), (288, 129), (270, 133), (256, 141), (256, 157), (246, 157),
    (236, 169), (228, 169), (228, 161), (220, 169), (212, 169), (204, 169),
    (196, 169), (188, 169), (196, 161), (188, 145), (172, 145), (164, 145),
    (156, 145), (148, 145), (140, 145), (148, 169), (164, 169), (172, 169),
    (156, 169), (140, 169), (132, 169), (124, 169), (116, 161), (104, 153),
    (104, 161), (104, 169), (90, 165), (80, 157), (64, 157), (64, 165),
    (56, 169), (56, 161), (56, 153), (56, 145), (56, 137), (56, 129),
    (56, 121), (40, 121), (40, 129), (40, 137), (40, 145), (40, 153),
    (40, 161), (40, 169), (32, 169), (32, 161), (32, 153), (32, 145),
    (32, 137), (32, 129), (32, 121), (32, 113), (40, 113), (56, 113),
    (56, 105), (48, 99), (40, 99), (32, 97), (32, 89), (24, 89),
    (16, 97), (16, 109), (8, 109), (8, 97), (8, 89), (8, 81),
    (8, 73), (8, 65), (8, 57), (16, 57), (8, 49), (8, 41),
    (24, 45), (32, 41), (32, 49), (32, 57), (32, 65), (32, 73),
    (32, 81), (40, 83), (40, 73), (40, 63), (40, 51), (44, 43),
    (44, 35), (44, 27), (32, 25), (24, 25), (16, 25), (16, 17),
    (24, 17), (32, 17), (44, 11), (56, 9), (56, 17), (56, 25),
    (56, 33), (56, 41), (64, 41), (72, 41), (72, 49), (56, 49),
    (48, 51), (56, 57), (56, 65), (48, 63), (48, 73), (56, 73),
    (56, 81), (48, 83), (56, 89), (56, 97), (104, 97), (104, 105),
    (104, 113), (104, 121), (104, 129), (104, 137), (104, 145), (116, 145),
    (124, 145), (132, 145), (132, 137), (140, 137), (148, 137), (156, 137),
    (164, 137), (172, 125), (172, 117), (172, 109), (172, 101), (172, 93),
    (172, 85), (180, 85), (180, 77), (180, 69), (180, 61), (180, 53),
    (172, 53), (172, 61), (172, 69), (172, 77), (164, 81), (148, 85)]


class suppress_stderr(object):

    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[1], 2)
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def rand_coords():
    random.seed()
    return random.randrange(10000), random.randrange(10000)


def brute_force():
    to_go = cities.copy()
    to_go.remove(to_go[0])

    perm = names.copy()
    perm.remove(0)
    routes = {}
    for perm_index, perm in enumerate(permutations(list(map(str, perm)))):
        routes[perm_index] = []
        routes[perm_index].append(0)
        for j, p in enumerate(perm):
            routes[perm_index].append(int(p))
        routes[perm_index].append(0)

    distances = {}

    for from_index, from_coords in enumerate(cities):
        distances[from_index] = {}
        for to_index, to_coords in enumerate(cities):
            if from_index == to_index:
                distances[from_index][to_index] = 0
            else:
                distances[from_index][to_index] = \
                    (math.hypot((from_coords[0] - to_coords[0]), (from_coords[1] - to_coords[1])))

    routes_dist = {}

    for route in routes:
        dist = 0
        for route_i in range(len(routes[route])-1):
            dist += distances[routes[route][route_i]][routes[route][route_i+1]]
        routes_dist[route] = dist

    shortest_dist = routes_dist[0]
    shortest_route = 0
    for route in routes_dist:
        if routes_dist[route] < shortest_dist:
            shortest_dist = routes_dist[route]
            shortest_route = route

    print("Shortest Route: ")
    print(routes[shortest_route])
    print("Distance: " + str(round(shortest_dist, 3)))

    x = []
    y = []
    for city in routes[shortest_route]:
        x.append(cities[city][0])
        y.append(cities[city][1])

    plt.plot(x, y, linestyle='--', marker='.', color='r')
    plt.plot(x[0], y[0], marker='o', color='r')
    plt.title("Route by Brute-Force. Distance: " + str(round(shortest_dist, 3)))
    plt.autoscale()
    plt.show()


def nearest_neighbour():
    global cities
    to_go = names.copy()
    to_go.remove(0)
    route = [0]
    distance = 0
    x = [cities[0][0]]
    y = [cities[0][1]]
    while len(to_go) > 0:
        closest = to_go[0]
        way = math.hypot((cities[route[-1]][0] - cities[to_go[0]][0]),
                         (cities[route[-1]][1] - cities[to_go[0]][1]))

        for city in to_go[1:]:
            temp = math.hypot((cities[route[-1]][0] - cities[city][0]),
                              (cities[route[-1]][1] - cities[city][1]))
            if temp < way:
                closest = city
                way = temp

        x.append(cities[closest][0])
        y.append(cities[closest][1])

        route.append(closest)
        distance += way
        to_go.remove(closest)

    distance += math.hypot((cities[route[-1]][0] - cities[0][0]),
                           (cities[route[-1]][1] - cities[0][1]))
    route.append(0)
    x.append(cities[0][0])
    y.append(cities[0][1])

    print("Designated route: " + str(route))
    print("Distance: " + str(round(distance, 3)))

    plt.plot(x, y, linestyle='--', marker='.', color='g')
    plt.plot(x[0], y[0], marker='o', color='brown')
    plt.title("Route by nearest neighbour. Distance: " + str(round(distance, 3)))
    plt.show()


def compute_distances(locations):
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                distances[from_counter][to_counter] = (int(
                    math.hypot((from_node[0] - to_node[0]),
                               (from_node[1] - to_node[1]))))

    print(distances)
    return distances


def display_solution(manager, routing, solution):
    index = routing.Start(0)
    plan_output = 'Route:\n['
    x = [data['cities'][index][0] / 1000]
    y = [data['cities'][index][1] / 1000]
    while not routing.IsEnd(index):
        plan_output += '{}, '.format(manager.IndexToNode(index))
        x.append(data['cities'][index][0] / 1000)
        y.append(data['cities'][index][1] / 1000)
        index = solution.Value(routing.NextVar(index))
    plan_output += '{}]\n'.format(manager.IndexToNode(index))
    x.append(data['cities'][0][0] / 1000)
    y.append(data['cities'][0][1] / 1000)
    plan_output += 'Distance: {}\n'.format(solution.ObjectiveValue() / 1000)
    print(plan_output)

    plt.plot(x, y, linestyle='--', marker='.', color='b')
    plt.plot(x[0], y[0], marker='o', color='b')
    plt.title("Route by Google OR-Tools. Distance: " + str(solution.ObjectiveValue() / 1000))
    plt.show()


def ortools_solver():
    data['cities'] = []
    data['salesmen'] = 1
    data['start'] = 0

    for pair in cities:
        x = pair[0] * 1000
        y = pair[1] * 1000
        data['cities'].append((x, y))

    manager = pywrapcp.RoutingIndexManager(len(data['cities']), data['salesmen'], data['start'])
    routing = pywrapcp.RoutingModel(manager)

    distances = compute_distances(data['cities'])

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distances[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_params = pywrapcp.DefaultRoutingSearchParameters()

    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    # search_params.time_limit.seconds = 1
    # search_params.log_search = True

    solution = routing.SolveWithParameters(search_params)

    display_solution(manager, routing, solution)


if __name__ == '__main__':
    n_cities = 10
    names = []
    data = {}
    for i in range(n_cities):
        names.append(i)

    cities = []

    i = 0
    for name in names:
        cities.append(rand_coords())
        # cities.append(sample[i])
        # i += 1

    print(cities)

    print("Brute-force: ")
    try:
        brute_force()
    except MemoryError:
        print("Memory Error detected. Model too big for brute force on this device")
    print("======================================")

    print("Nearest Neighbor: ")
    nearest_neighbour()
    print("======================================")

    print("OR-Tools solution")
    with suppress_stderr():
        ortools_solver()
