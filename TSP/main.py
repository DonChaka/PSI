import random
import math
import os
import matplotlib.pyplot as plt
from itertools import permutations
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from time import time

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

sample2 = [(17, 84), (95, 67), (85, 77), (82, 95), (36, 35), (7, 81), (63, 7), (93, 10), (59, 30), (70, 12), (52, 55), (59, 10), (87, 64), (15, 79), (35, 96), (6, 29), (25, 47), (88, 41), (40, 12), (78, 45), (9, 74), (62, 85), (98, 26), (42, 58), (92, 10), (27, 67), (83, 54), (9, 81), (59, 7), (57, 23), (8, 31), (31, 17), (27, 33), (10, 44), (49, 2), (14, 70), (26, 17), (61, 27), (11, 71), (33, 46), (72, 45), (16, 94), (39, 69), (58, 64), (86, 81), (36, 97), (3, 31), (64, 63), (43, 75), (32, 87), (37, 83), (14, 91), (15, 58), (51, 95), (2, 97), (43, 74), (26, 69), (19, 82), (92, 8), (57, 44), (45, 56), (7, 76), (85, 36), (31, 19), (39, 43), (74, 79), (74, 2), (76, 97), (28, 91), (3, 12), (38, 68), (84, 68), (19, 72), (59, 49), (89, 67), (44, 52), (29, 3), (74, 66), (66, 91), (77, 75), (96, 44), (50, 43), (80, 50), (58, 43), (89, 54), (27, 10), (32, 45), (65, 16), (14, 82), (97, 72), (96, 76), (46, 79), (6, 62), (30, 11), (98, 73), (74, 56), (46, 92), (94, 87), (61, 40), (65, 41), (63, 60), (91, 8), (4, 96), (64, 89), (25, 90), (22, 71), (82, 21), (19, 70), (18, 4), (20, 99), (78, 3), (0, 76), (71, 32), (35, 77), (74, 0), (93, 48), (20, 79), (32, 14), (54, 52), (74, 86), (55, 51), (92, 32), (73, 77), (33, 87), (76, 59), (54, 65), (72, 98), (19, 98), (40, 33), (89, 26), (47, 0), (35, 52), (32, 83), (55, 83), (91, 99), (56, 65), (85, 91), (80, 14), (95, 1), (0, 33), (63, 35), (53, 85), (26, 1), (35, 23), (78, 72), (64, 42), (69, 7), (47, 96), (79, 94), (15, 9), (69, 32), (19, 75), (5, 18), (80, 45), (94, 30), (56, 24), (72, 62), (85, 57), (92, 2), (11, 7), (21, 22), (38, 4), (1, 70), (80, 15), (46, 22), (5, 4), (8, 57), (61, 83), (9, 22), (84, 90), (70, 55), (3, 74), (76, 36), (0, 97), (13, 70), (16, 94), (50, 51), (41, 55), (86, 4), (88, 95), (64, 57), (24, 65), (79, 87), (28, 93), (9, 53), (10, 26), (7, 79), (95, 10), (57, 81), (50, 96), (88, 95), (45, 85), (5, 89), (37, 54), (3, 11), (88, 56), (59, 34), (47, 28), (34, 17), (28, 70), (22, 52), (26, 73), (11, 29), (85, 86), (53, 81), (0, 67), (69, 1), (99, 58), (5, 95), (12, 95), (7, 17), (56, 98), (33, 70), (12, 12), (5, 81), (46, 83), (18, 58), (22, 35), (92, 77), (20, 6), (41, 69), (82, 66), (35, 90), (67, 43), (58, 64), (81, 82), (85, 13), (30, 23), (88, 14), (64, 53), (41, 37), (45, 72), (82, 9), (56, 3), (67, 28), (89, 35), (57, 41), (76, 9), (52, 77), (47, 1), (5, 34), (22, 59), (7, 38), (64, 73), (73, 92), (68, 46), (47, 15), (24, 63), (11, 53), (7, 8), (21, 1), (21, 58), (52, 22), (53, 20), (52, 72), (56, 17), (37, 95), (46, 1), (76, 58), (11, 34), (47, 59), (37, 38), (85, 1), (78, 2), (52, 66), (50, 68), (81, 7), (51, 55), (73, 64), (87, 36), (11, 31), (4, 53), (94, 95), (93, 96), (4, 78), (91, 13), (80, 44), (24, 35), (53, 79), (11, 93), (19, 10), (9, 59), (49, 53), (40, 59), (68, 57), (47, 25), (17, 79), (18, 96), (76, 40), (11, 50), (27, 98), (61, 82), (32, 16), (27, 54), (19, 9), (51, 84), (31, 35), (99, 23), (22, 51), (55, 12), (30, 1), (94, 47), (69, 44), (95, 8), (32, 59), (66, 22), (62, 16), (11, 51), (57, 26), (10, 90), (54, 74), (67, 74), (70, 53), (89, 57), (41, 6), (53, 72), (55, 75), (53, 8), (77, 82), (35, 0), (91, 11), (2, 21), (90, 75), (62, 8), (13, 96), (96, 3), (30, 0), (10, 68), (39, 74), (51, 51), (54, 59), (37, 47), (33, 69), (2, 78), (24, 35), (32, 23), (6, 94), (7, 66), (90, 74), (64, 59), (48, 29), (19, 35), (74, 89), (27, 57), (0, 63), (39, 71), (63, 21), (20, 24), (88, 60), (50, 59), (21, 67), (33, 67), (40, 12), (19, 77), (6, 14), (65, 17), (64, 21), (4, 49), (8, 72), (24, 12), (34, 97), (65, 52), (25, 39), (53, 15), (85, 67), (12, 94), (27, 33), (41, 22), (58, 28), (2, 43), (50, 55), (91, 80), (54, 20), (22, 61), (57, 88), (2, 61), (84, 5), (47, 13), (95, 24), (77, 8), (70, 37), (85, 9), (42, 69), (65, 98), (61, 18), (45, 5), (42, 6), (70, 69), (65, 77), (4, 88), (26, 13), (83, 85), (85, 55), (23, 42), (16, 54), (68, 95), (9, 39), (73, 90), (9, 51), (95, 13)]


def rand_coords():
    random.seed()
    return random.randrange(100), random.randrange(100)


def compute_distances_float(locations):
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                distances[from_counter][to_counter] = (
                    math.hypot((from_node[0] - to_node[0]),
                               (from_node[1] - to_node[1])))

    return distances


def brute_force():
    time_start = time()
    to_go = cities.copy()
    to_go.remove(to_go[0])

    perm = names.copy()
    perm.remove(0)

    min_distance = float("inf")
    min_route = []

    distances = compute_distances_float(cities)

    for index, permutation in enumerate(permutations(list(map(str, perm)))):

        route = [0]
        for p in list(map(int, permutation)):
            route.append(p)
        route.append(0)

        distance = 0
        for r in range(len(route)-1):
            distance += distances[route[r]][route[r+1]]

        if distance < min_distance:
            min_distance = distance
            min_route = route.copy()

    time_stop = time()
    print("Shortest Route: ")
    print(min_route)
    print("Distance: " + str(round(min_distance, 3)) + " Found in: " + str(round(time_stop - time_start, 5)) + "s")

    x = []
    y = []
    for city in min_route:
        x.append(cities[city][0])
        y.append(cities[city][1])

    plt.plot(x, y, linestyle='--', marker='.', color='r')
    plt.plot(x[0], y[0], marker='o', color='r')
    plt.title("Route by Brute-Force. Distance: " + str(round(min_distance, 3)) + " Time: " + str(round(time_stop - time_start, 5)) + "s")
    plt.autoscale()
    plt.show()


def nearest_neighbour():
    time_start = time()
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
    time_stop = time()

    print("Designated route: " + str(route))
    print("Distance: " + str(round(distance, 3)) + " Found in: " + str(round(time_stop - time_start, 5)) + "s")

    plt.plot(x, y, linestyle='--', marker='.', color='g')
    plt.plot(x[0], y[0], marker='o', color='brown')
    plt.title("Route by nearest neighbour. Distance: " + str(round(distance, 3)) + " Time: " + str(round(time_stop - time_start, 5)) + "s")
    plt.show()


def display_solution(manager, routing, solution, duration):
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
    plan_output += 'Distance: {}\n'.format(solution.ObjectiveValue() / 1000) + " Found in: " + str(duration) + "s"
    print(plan_output)

    plt.plot(x, y, linestyle='--', marker='.', color='b')
    plt.plot(x[0], y[0], marker='o', color='b')
    plt.title("Route by Google OR-Tools. Distance: " + str(solution.ObjectiveValue() / 1000) + " Time: " + str(duration) + "s")
    plt.show()


def compute_distances_int(locations):
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

    return distances


def ortools_solver():
    start_time = time()
    data['cities'] = []
    data['salesmen'] = 1
    data['start'] = 0

    for pair in cities:
        x = pair[0] * 1000
        y = pair[1] * 1000
        data['cities'].append((x, y))

    manager = pywrapcp.RoutingIndexManager(len(data['cities']), data['salesmen'], data['start'])
    routing = pywrapcp.RoutingModel(manager)

    distances = compute_distances_int(data['cities'])

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
    # search_params.time_limit.seconds = 15

    solution = routing.SolveWithParameters(search_params)
    stop_time = time()
    display_solution(manager, routing, solution, round(stop_time - start_time, 5))


if __name__ == '__main__':
    n_cities = 12
    names = []
    data = {}
    for i in range(n_cities):
        names.append(i)

    cities = []

    i = 0

    for name in names:
        # cities.append(rand_coords())
        cities.append(sample2[i])
        i += 1

    print(cities)

    if n_cities < 12:
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
    ortools_solver()
