import facilities
import math
import random
from e2 import two_opt_xchg
from matplotlib import pyplot as plt
from typing import Callable
import time


# adjust parameters for simulated annealing
t_start = 1000000  # initial temperature
t_break = 100  # break if temperature <= this level
cooling_frequency = 2  # cool down every k.th round
limit_no_improvement = 3  # quit after n rounds without improvement

# adjust parameters for cooling functions
geometric_factor = 0.9
linear_subtrahend = 5000
quadratic_factor = 1.05
exponential_factor = 0.95

# cooling functions
def geometric_cooling(cur_temp: int, cur_round: int) -> int:
    global geometric_factor
    assert 0 < geometric_factor < 1
    return cur_temp * geometric_factor


def linear_cooling(cur_temp: int, cur_round: int) -> int:
    global linear_subtrahend
    assert linear_subtrahend > 0
    return max(cur_temp - linear_subtrahend, 0)


def quadratic_cooling(cur_temp: int, cur_round: int) -> int:
    global quadratic_factor, t_start
    assert quadratic_factor > 1
    return t_start / (1 + (quadratic_factor * cur_round) ** 2)


def exponential_cooling(cur_temp: int, cur_round: int) -> int:
    global exponential_factor, t_start
    assert 0 < exponential_factor < 1
    return t_start * (exponential_factor ** cur_round)


# plot cooling functions
def plot_cooling_function(funct: callable) -> None:
    global t_start, t_break
    rounds, temps = [], []
    ind = 0
    cur_temp = t_start
    while cur_temp > t_break:
        ind += 1
        rounds.append(ind)
        temps.append(cur_temp)
        cur_temp = funct(cur_temp, ind)
    else:
        plt.plot(rounds, temps)
        plt.xlabel("Round")
        plt.ylabel("Temperature")
        plt.title("Cooling Function: {}".format(funct.__name__))
        plt.show()


# pertubation functions

# switch two random nodes
def node_xchange(input_route: list) -> list:
    zv1, zv2 = random.randint(0, len(input_route) - 1), random.randint(
        0, len(input_route) - 1
    )
    input_route.insert(zv1, input_route.pop(zv2))
    return input_route


# pop and prepend random node
def node_insert(input_route: list) -> list:
    zv = random.randint(0, len(input_route) - 1)
    input_route.insert(0, input_route.pop(zv))
    return input_route


# slightly modified version of two_opt_step from excercise 2
def two_opt_step(route: list) -> list:
    global db
    cur_route, cur_distance = list(route), facilities.get_distance(route)
    for i in range(len(cur_route) - 1):
        for k in range(i + 1, len(cur_route)):
            swap_route = two_opt_xchg(cur_route, i, k)
            swap_distance = facilities.get_distance(swap_route)
            rev_route = facilities.reverse_route(swap_route)
            rev_distance = facilities.get_distance(rev_route)
            if rev_distance < swap_distance:
                swap_route, swap_distance = list(rev_route), rev_distance
            if swap_distance <= cur_distance:
                cur_route, cur_distance = list(swap_route), swap_distance
    return cur_route


# randomly picks one of the 3 pertubation functions above
def pertubation(input_route: list) -> list:
    global db
    assert len(input_route) >= 2
    zv = random.randint(1, 3)
    if zv == 1:
        return node_xchange(input_route)
    elif zv == 2:
        return node_insert(input_route)
    else:
        return two_opt_step(input_route)


# simulated annealing algorithm
def simulated_annealing(
    n_cities: int, init_route: list, cooling_function: Callable
) -> tuple:
    global db, t_start, t_break, cooling_frequency, limit_no_improvement
    cur_temp = t_start
    cur_round = 0
    cur_best_route, cur_route = init_route, init_route
    rounds_without_improvement = 0
    # loop while temperature above treshold and not more than 3 previous rounds without improvement
    while (cur_temp > t_break) and (rounds_without_improvement < limit_no_improvement):
        rounds_without_improvement += 1
        cur_round += 1
        temp_route = pertubation(cur_route)
        old_dist, new_dist = facilities.get_distance(
            cur_route
        ), facilities.get_distance(temp_route)
        # check if distance of reverse route is shorter
        if True:
            pass
        # check if pertubation is equal or better than current solution, if so accept it
        if new_dist <= old_dist:
            cur_route = list(temp_route)
            rounds_without_improvement = 0
            # check if better than our best solution so far, if so update best solution
            if new_dist < facilities.get_distance(cur_best_route):
                cur_best_route = list(cur_route)
        else:
            if random.random() <= math.exp(-((new_dist - old_dist) / cur_temp)):
                cur_route = list(temp_route)
        # cool down every n.th round
        if cur_round % cooling_frequency == 0:
            cur_temp = cooling_function(cur_temp, cur_round)
    else:
        return cur_best_route, cur_round


if __name__ == "__main__":

    # plotting cooling functions
    fl = input("\nShow plots for cooling functions? [y/n]")
    if fl == "y":
        plot_cooling_function(linear_cooling)
        plot_cooling_function(geometric_cooling)
        plot_cooling_function(quadratic_cooling)
        plot_cooling_function(exponential_cooling)

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

    # initialize random starting round
    starting_route = facilities.rand_route(n)
    print("Random starting route:")
    facilities.print_route(starting_route, facilities.get_distance(starting_route))
    facilities.print_line()

    t_0 = time.time()
    sol1, rounds1 = simulated_annealing(16, starting_route, linear_cooling)
    t_lin = time.time() - t_0

    t_0 = time.time()
    sol2, rounds2 = simulated_annealing(16, starting_route, geometric_cooling)
    t_geo = time.time() - t_0

    t_0 = time.time()
    sol3, rounds3 = simulated_annealing(16, starting_route, quadratic_cooling)
    t_quad = time.time() - t_0

    t_0 = time.time()
    sol4, rounds4 = simulated_annealing(16, starting_route, exponential_cooling)
    t_exp = time.time() - t_0

    print(
        "Simulated Annealing solutions with global parameters:\nT_init={}\nT_end={}\nCooling frequency every {}. round\nBreak after {} rounds without improvement".format(
            t_start, t_break, cooling_frequency, limit_no_improvement
        )
    )
    facilities.print_line()

    print("\nUsing linear cooling function with parameter={}".format(linear_subtrahend))
    facilities.print_route(sol1, facilities.get_distance(sol1))
    print(
        "Total number of rounds: {}\nTotal Time: {:7f} seconds\n".format(rounds1, t_lin)
    )
    facilities.print_line()

    print(
        "\nUsing geometric cooling function with parameter={}".format(geometric_factor)
    )
    facilities.print_route(sol2, facilities.get_distance(sol2))
    print(
        "Total number of rounds: {}\nTotal Time: {:7f} seconds\n".format(rounds2, t_geo)
    )
    facilities.print_line()

    print(
        "\nUsing quadratic cooling function with parameter={}".format(quadratic_factor)
    )
    facilities.print_route(sol3, facilities.get_distance(sol3))
    print(
        "Total number of rounds: {}\nTotal Time: {:7f} seconds\n".format(
            rounds3, t_quad
        )
    )
    facilities.print_line()

    print(
        "\nUsing exponential cooling function with parameter={}".format(
            exponential_factor
        )
    )
    facilities.print_route(sol4, facilities.get_distance(sol4))
    print(
        "Total number of rounds: {}\nTotal Time: {:7f} seconds\n".format(rounds4, t_exp)
    )
    facilities.print_line()
