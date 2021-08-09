import facilities
import itertools
import time
import math
import matplotlib.pyplot as plt
from typing import Callable


# search the shortest path for first n cities, returns optimal route (array) and distance (int)
def bruteforce_tsp(n_cities: int, objective_function: Callable) -> tuple:
    global db
    decision_space = list(itertools.permutations([_ for _ in range(n_cities)]))
    # print("n = {}\nTotal count of tours to check: {}".format(n_cities, len(decision_space)))
    cur_best_route, cur_best_score = list(decision_space[0]), objective_function(
        decision_space[0]
    )
    for route in decision_space:
        temp_score = objective_function(route)
        if temp_score < cur_best_score:
            cur_best_route, cur_best_score = list(route), temp_score
    return cur_best_route, cur_best_score


if __name__ == "__main__":

    # compute for the first n cities
    n = 10

    # find optimal solution and measure time for different number of cities, print results
    num_cities, bf_time = [], []
    for k in range(2, n):
        t_start = time.time()
        arr, dist = bruteforce_tsp(k, facilities.get_distance)
        t_total = time.time() - t_start
        num_cities.append(k)
        bf_time.append(t_total)
        print("For first {} cities".format(k))
        facilities.print_route(arr, dist)
        print("Time = {:7f} seconds\n".format(t_total))

    # write runtimes to file for further analysis
    output = open("code/predictions/runtime.csv", "w")
    output.write("n\truntime\tlogruntime\n")
    for t, n in zip(bf_time, num_cities):
        output.write("{}\t{}\t{}\n".format(n, t, math.log(t)))
    output.close()
