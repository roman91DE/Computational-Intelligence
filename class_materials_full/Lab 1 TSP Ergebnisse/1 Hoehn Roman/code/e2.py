import facilities
import random
import time

db = facilities.DB(
    "/Users/roman/computational_intelligence/class_materials_full/Lab 1 TSP Ergebnisse/1 Hoehn Roman/data/citiesAndDistances.txt"
)

# create initial solution as starting point for two-opt using nearest neighbour algorithm
def nearest_neighbour(n_cities: int) -> list:
    global db
    unvisited_cities, visited_cities = [n for n in range(n_cities)], []
    start = unvisited_cities.pop(
        random.randint(0, n_cities - 1)
    )  # pick random starting point
    visited_cities.append(start)
    while len(unvisited_cities) > 0:
        assert len(visited_cities) > 0
        current_city = visited_cities[-1]
        min_distance, index = None, None
        for city in unvisited_cities:
            if min_distance is None and index is None:
                min_distance, index = db.distance_matrix[current_city][city], city
            elif db.distance_matrix[current_city][city] < min_distance:
                min_distance, index = db.distance_matrix[current_city][city], city
        else:
            visited_cities.append(index)
            unvisited_cities.remove(index)
    else:
        return visited_cities


# reverses the order of cities i . . . k in a tour (returns new route)
def two_opt_xchg(route: list, i: int, k: int) -> list:
    new_route = list(route)
    while not i >= k:
        new_route[i], new_route[k] = new_route[k], new_route[i]
        i += 1
        k -= 1
    return new_route


# implements the best improvement strategy by creating deterministically all possible and useful 2-opt swaps,
# measuring the length of the resulting tour and the length of the reversed tour (asymmetric TSP)
# also counts the number of visited/measured tours
def two_opt_step(route: list) -> tuple:
    global db
    cur_route, cur_distance = list(route), facilities.get_distance(route)
    counter = 0
    for i in range(len(cur_route) - 1):
        for k in range(i + 1, len(cur_route)):
            counter += 2
            swap_route = two_opt_xchg(cur_route, i, k)
            swap_distance = facilities.get_distance(swap_route)
            rev_route = facilities.reverse_route(swap_route)
            rev_distance = facilities.get_distance(rev_route)
            if rev_distance < swap_distance:
                swap_route, swap_distance = list(rev_route), rev_distance
            if swap_distance <= cur_distance:  # <= because of hill climber approach?
                cur_route, cur_distance = list(swap_route), swap_distance
    if cur_route == route:  # no changes during step
        return (
            False,
            cur_route,
            cur_distance,
            counter,
        )  # returns [improvement, solution, distance, swap counter]
    else:
        return True, cur_route, cur_distance, counter


# implements a hill climber applying two_opt_step(tour) as long as an improvement is possible. two_opt() should return
# & print the shortest path found, its length, and the number of visited/measured tours.
def two_opt(route: list) -> tuple:
    global db
    total_counter = 0
    cur_route = list(route)
    cur_distance = facilities.get_distance(cur_route)
    improved = True
    while improved:
        improved, cur_route, cur_distance, round_counter = two_opt_step(cur_route)
        total_counter += round_counter
    else:
        facilities.print_route(cur_route, cur_distance)
        print("Total number of swaps: {}".format(total_counter))
        return cur_route, cur_distance, total_counter


if __name__ == "__main__":

    n = -1
    while not (2 < n <= 16):
        try:
            n = int(
                input(
                    "Enter the number of cities for which you want to calculate TSP solution\nn = "
                )
            )
            assert 2 < n <= 16
        except AssertionError:
            print("Only values in the range 2 < n <= 16 are possible")

    facilities.print_line()

    # random route
    print("Initial random route")
    rand_route = facilities.rand_route(n)
    facilities.print_route(rand_route, facilities.get_distance(rand_route))

    # try to optimize random solution using 2-opt
    print("\n2-opt solution for random starting route")
    t0 = time.time()
    opt_route1, dist, counter = two_opt(rand_route)
    t1 = time.time()
    print("2-opt time: {:7f} seconds".format(t1 - t0))

    facilities.print_line()

    # next neighbour algorithm route
    print("Initial greedy route (nearest neighbour)")
    t0 = time.time()
    nn_route = nearest_neighbour(n)
    t1 = time.time()
    t_nearest_neighbour = t1 - t0
    facilities.print_route(nn_route, facilities.get_distance(nn_route))

    # try to optimize route from next neighbour using 2-opt
    print("\n2-opt solution for next neighbour Initialisation")
    t0 = time.time()
    opt_route2, dist, counter = two_opt(nn_route)
    t1 = time.time()
    t_two_opt = t1 - t0

    # print time for each algorithm
    print(
        "Time:\nNearest Neighbour: {:7f} seconds\n2-opt: {:7f} seconds\nCombined: {:7f} seconds".format(
            t_nearest_neighbour, t_two_opt, (t_nearest_neighbour + t_two_opt)
        )
    )

    facilities.print_line()
