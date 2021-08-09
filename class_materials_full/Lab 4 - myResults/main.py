from typing import Callable, Dict, List
from pypower.api import *
from case300extended2 import *
import numpy as np
from math import exp
import random
import copy
import time
import os


def get_violated_busses(case):
    # create an empty array of sufficient size
    violated = np.empty_like(case["bus"])
    j = 0
    # go through all busses
    for i in range(0, len(case["bus"])):
        # if voltage is lower or larger than allowed
        if (
            case["bus"][i][7] < case["bus"][i][12]
            or case["bus"][i][7] > case["bus"][i][11]
        ):
            # print(str(case["bus"][i][0]))
            # copy information of the bus with a violation to the data structured that will be returned
            np.copyto(violated[j], case["bus"][i])
            j += 1
    return violated[0:j]


def active_extension_line_resistances(case):
    cumulated_resistance = 0
    for i in range(0, len(case["extensions"])):
        if case["extensions"][i][10] > 0:
            cumulated_resistance += case["extensions"][i][2]
    return cumulated_resistance


def evaluate_objective_function(network):
    # tell runpf to be quite
    opt = ppoption(VERBOSE=0, OUT_ALL=0)

    # copy original network
    case = copy.deepcopy(network)

    # extend busses of the child
    case["branch"] = np.concatenate((case["branch"], case["extensions"]))

    # compute the power flow
    r = runpf(case, opt)

    ret = np.zeros(3)

    # check if run was successful
    if r[0]["success"] == 1:
        # figure out busses with under and overvoltages
        v = get_violated_busses(r[0])

        # number of busses with violations
        ret[1] = len(v)

        # sum of electric resistances, which can be seen as geometrical line length, which can be associcated with line costs
        ret[2] = active_extension_line_resistances(r[0])

        if ret[1] > 0:
            ret[0] = 1 / (1 + ret[1])
        else:
            ret[0] = 0.5 + 0.5 * 1 / (1 + ret[2])

    return ret


def extension_measures_to_str(case):
    s = ""
    for i in range(0, len(case["extensions"])):
        # is the extension branch acitve?
        if case["extensions"][i][10] == 1:
            s += "1"
        else:
            s += "0"
    return s


def hill_climber(
    case: dict,
    fitness_function: Callable,
    time_limit_secs: int,
    print_stats: bool = True,
):

    # evaluate parent
    fitness = fitness_function(case)

    gen = 0
    t_start = time.time()

    # i replaced the termination condition for a timer for easier comparison
    while (time.time() - t_start) < time_limit_secs:

        gen += 1

        # clone parent
        child = copy.deepcopy(case)

        # mutate one extension
        rindex = random.randint(0, len(child["extensions"]) - 1)
        child["extensions"][rindex][10] = 1 - child["extensions"][rindex][10]

        # evaluate fitness
        child_fitness = fitness_function(child)

        # found a better solution or a solution with same fitness?
        if child_fitness[0] >= fitness[0]:
            case = copy.deepcopy(child)
            fitness = copy.deepcopy(child_fitness)
            # fitness = np.copy(child_fitness)

        if print_stats:
            print(
                "gen %5d fit %1.5f #viol %3d costs %1.5f %s"
                % (
                    gen,
                    fitness[0],
                    fitness[1],
                    fitness[2],
                    extension_measures_to_str(case),
                )
            )

    else:

        return (fitness, case)


# return a copy of a network where the activation of each extensions is flipped with probability p
def random_init(network: Dict, p: float = 0.5) -> Dict:

    case = copy.deepcopy(network)

    for i in range(len(case["extensions"])):
        if random.random() > p:
            case["extensions"][i][10] = 1 - case["extensions"][i][10]

    return case


# mutation operator - flips the state of extensions
def flip_mutation(network: Dict) -> Dict:

    n = len(network["extensions"])

    # flip one gene per mutation on average
    m_rate = 1 / n

    for i in range(n):
        if random.random() < m_rate:
            network["extensions"][i][10] = 1 - network["extensions"][i][10]


# mutation operator - swaps the state of two extensions (not used)
def swap_mutation(network: Dict):

    n = len(network["extensions"])

    # swap one gene per mutation on average
    m_rate = 1 / n

    for i in range(n):
        if random.random() < m_rate:
            zv = random.randint(0, n - 1)
            while zv == i:
                zv = random.randint(0, n - 1)
            network["extensions"][i][10], network["extensions"][zv][10] = (
                network["extensions"][zv][10],
                network["extensions"][i][10],
            )


# randomly pick one mutation operator (not used)
def random_mutation(network: dict):

    if random.random() > 0.5:
        swap_mutation(network)

    else:
        flip_mutation(network)


# crossover operator (one point)
def crossover_onepoint(mother: dict, father: dict) -> dict:

    child = copy.deepcopy(mother)

    n = len(child["extensions"])
    zv = random.randint(0, n - 1)

    for i in range(zv, n):
        child["extensions"][i][10] = father["extensions"][i][10]

    return child


# alternative fitness evaluation, punishes each violation by factor m
def evaluate_fitness(network: Dict) -> list:

    m = 100  # punish each violation by factor m

    # (copied from previous fitness function)
    opt = ppoption(VERBOSE=0, OUT_ALL=0)
    case = copy.deepcopy(network)
    case["branch"] = np.concatenate((case["branch"], case["extensions"]))
    r = runpf(case, opt)

    assert r[0]["success"] == 1

    # compute fitness f
    v = len(get_violated_busses(r[0]))
    s = active_extension_line_resistances(r[0])
    f = -(s + m * v)

    ret = np.zeros(3)
    ret[0], ret[1], ret[2] = f, v, s

    return ret  # ret -> [fitness, number violations, cost/length]


# search solution with evolutionary algorithm
def evolutionary_algorithm(
    case: Dict,
    time_limit_secs: int,
    population_size: int,
    mutation_func: Callable,
    crossover_func: Callable,
    fitness_func: Callable,
    elite_size: int = 2,
    tournament_size: int = 4,
    print_stats: bool = True,
) -> tuple:

    main_pop: list = []

    for _ in range(population_size):

        # fill population with randomly generated solutions
        main_pop.append(random_init(case))

    cur_generation: int = 0
    t_start = time.time()

    while (time.time() - t_start) < time_limit_secs:

        cur_generation += 1

        main_pop.sort(key=lambda x: fitness_func(x)[0], reverse=True)

        if print_stats:
            temp = fitness_func(main_pop[0])
            print(
                "Generation: {:<3.0f}\tBest Solution: fitness={:<6.2f}\tviolations={:<3.0f}\tlength={:<6.2f}\n".format(
                    cur_generation, temp[0], temp[1], temp[2]
                )
            )

        breeding_pop = []

        while len(breeding_pop) < 6:

            tournament = random.choices(main_pop, k=tournament_size)
            tournament.sort(key=lambda x: fitness_func(x)[0], reverse=True)

            breeding_pop.append(tournament[0])

        main_pop = main_pop[:elite_size]

        while len(main_pop) < population_size:

            mother, father = random.choices(breeding_pop, k=2)

            child = crossover_func(mother, father)
            mutation_func(child)

            main_pop.append(child)

    else:

        main_pop.sort(key=lambda x: fitness_func(x)[0], reverse=True)

        if print_stats:
            temp = fitness_func(main_pop[0])
            print(
                "Terminated after {} generations\nBest Solution: fitnes={}; violations={}; length={}\n".format(
                    cur_generation, temp[0], temp[1], temp[2]
                )
            )

        return (fitness_func(main_pop[0]), main_pop[0])


# search solution with simulated annealing algorithm
def simulated_annealing(
    case: Dict,
    time_limit_secs: int,
    fitness_function: Callable,
    mutation_function: Callable,
    t_start: float,
    t_end: float,
    print_stats: bool,
) -> tuple:

    current_solution = copy.deepcopy(case)
    current_fitness = fitness_function(current_solution)

    best_solution = copy.deepcopy(current_solution)
    best_fitness = copy.deepcopy(current_fitness)

    t = t_start
    rounds_no_improvement = 0

    t_start = time.time()

    while ((time.time() - t_start) < time_limit_secs) and (t > t_end):

        # use geometric cooling scheme
        t *= 0.99

        if print_stats:
            print(
                "Temp: {:<10.2f}\tCurrent Fitness: {:<10.2f}\tBest Fitness: {:<10.2f}\tViolations (current): {:<10}\tViolations (best): {:<10}\n".format(
                    t,
                    current_fitness[0],
                    best_fitness[0],
                    current_fitness[1],
                    best_fitness[1],
                )
            )

        # generate new solution through mutation and evaluate fitness
        new_solution = copy.deepcopy(current_solution)
        mutation_function(new_solution)
        new_fitness = fitness_function(new_solution)

        # new solution is better than current solution
        if new_fitness[0] >= current_fitness[0]:

            current_solution = copy.deepcopy(new_solution)
            current_fitness = copy.deepcopy(new_fitness)
            rounds_no_improvement = 0

            # new solution is best solution found so far
            if current_fitness[0] > best_fitness[0]:

                best_solution = copy.deepcopy(current_solution)
                best_fitness = copy.deepcopy(current_fitness)

        # fitness if new solution is worse than current solution
        else:

            # if no improvement for 10 rounds, switch current solution back to previous best solution
            if rounds_no_improvement > 10:

                rounds_no_improvement = 0
                current_solution = copy.deepcopy(best_solution)
                current_fitness = copy.deepcopy(best_fitness)
                continue

            else:
                rounds_no_improvement += 1
                p = exp(-((new_fitness[0] - current_fitness[0]) / t))

                if random.random() >= p:

                    current_solution = copy.deepcopy(new_solution)
                    current_fitness = copy.deepcopy(new_fitness)

    # termination condition reached
    else:

        if print_stats:
            print(
                "Fitness of best solution found: {}\nNumber of remaining violations: {}\nLength of active Extensions: {}\n".format(
                    best_fitness[0], best_fitness[1], best_fitness[2]
                )
            )
            print("Extension Solution: ", extension_measures_to_str(best_solution))

        # return tuple (Fitness, Solution)
        return (best_fitness, best_solution)


# search solution with metropolis algorithm
def metropolis_algorithm(
    case: Dict,
    time_limit_secs: int,
    fitness_function: Callable,
    mutation_function: Callable,
    temperature: float,
    print_stats: bool,
) -> tuple:

    current_solution = copy.deepcopy(case)
    current_fitness = fitness_function(current_solution)

    best_solution = copy.deepcopy(current_solution)
    best_fitness = copy.deepcopy(current_fitness)

    rounds_no_improvement = 0

    t_start = time.time()

    while (time.time() - t_start) < time_limit_secs:

        if print_stats:
            print(
                "Temp: {:<10.2f}\tCurrent Fitness: {:<10.2f}\tBest Fitness: {:<10.2f}\tViolations (current): {:<10}\tViolations (best): {:<10}\n".format(
                    temperature,
                    current_fitness[0],
                    best_fitness[0],
                    current_fitness[1],
                    best_fitness[1],
                )
            )

        # generate new solution through mutation and evaluate fitness
        new_solution = copy.deepcopy(current_solution)
        mutation_function(new_solution)
        new_fitness = fitness_function(new_solution)

        # new solution is better than current solution
        if new_fitness[0] >= current_fitness[0]:

            current_solution = copy.deepcopy(new_solution)
            current_fitness = copy.deepcopy(new_fitness)
            rounds_no_improvement = 0

            # new solution is best solution found so far
            if current_fitness[0] > best_fitness[0]:

                best_solution = copy.deepcopy(current_solution)
                best_fitness = copy.deepcopy(current_fitness)

        # fitness if new solution is worse than current solution
        else:

            rounds_no_improvement += 1

            # if no improvement for n rounds, switch current solution back to previous best solution
            if rounds_no_improvement > 10:

                rounds_no_improvement = 0
                current_solution = copy.deepcopy(best_solution)
                current_fitness = copy.deepcopy(best_fitness)
                continue

            else:
                p = exp(-((new_fitness[0] - current_fitness[0]) / temperature))
                if random.random() > p:

                    current_solution = copy.deepcopy(new_solution)
                    current_fitness = copy.deepcopy(new_fitness)

    # termination condition reached
    else:

        if print_stats:
            print(
                "Fitness of best solution found: {}\nNumber of remaining violations: {}\nLength of active Extensions: {}\n".format(
                    best_fitness[0], best_fitness[1], best_fitness[2]
                )
            )
            print("Extension Solution: ", extension_measures_to_str(best_solution))

        # return tuple (Fitness, Solution)
        return (best_fitness, best_solution)


# modified simulated annealing, creates a population in each iteration and chooses the one with highest fitness
def simulated_annealing_population(
    case: Dict,
    time_limit_secs: int,
    population_size: int,
    fitness_function: Callable,
    mutation_function: Callable,
    t_start: float,
    t_end: float,
    print_stats: bool,
) -> tuple:

    current_solution = case
    current_fitness = fitness_function(current_solution)

    best_solution = copy.deepcopy(current_solution)
    best_fitness = copy.deepcopy(current_fitness)

    t = t_start
    rounds_no_improvement = 0

    t_start = time.time()

    while ((time.time() - t_start) < time_limit_secs) and (t > t_end):

        # use geometric cooling scheme
        t *= 0.9

        if print_stats:
            print(
                "Temp: {:<10.2f}\tCurrent Fitness: {:<10.2f}\tBest Fitness: {:<10.2f}\tViolations (current): {:<10}\tViolations (best): {:<10}\n".format(
                    t,
                    current_fitness[0],
                    best_fitness[0],
                    current_fitness[1],
                    best_fitness[1],
                )
            )

        # generate new solution through mutation and evaluate fitness

        population = []

        for _ in range(population_size):
            new_solution = copy.deepcopy(current_solution)
            mutation_function(new_solution)
            population.append(new_solution)

        population.sort(key=lambda x: fitness_function(x)[0], reverse=True)

        new_solution = copy.deepcopy(population[0])
        new_fitness = fitness_function(new_solution)

        # new solution is better than current solution
        if new_fitness[0] >= current_fitness[0]:

            current_solution = copy.deepcopy(new_solution)
            current_fitness = copy.deepcopy(new_fitness)
            rounds_no_improvement = 0

            # new solution is best solution found so far
            if current_fitness[0] > best_fitness[0]:

                best_solution = copy.deepcopy(current_solution)
                best_fitness = copy.deepcopy(current_fitness)

        # fitness if new solution is worse than current solution
        else:

            rounds_no_improvement += 1

            # if no improvement for 10 rounds, switch current solution back to previous best solution
            if rounds_no_improvement > 10:

                rounds_no_improvement = 0
                current_solution = copy.deepcopy(best_solution)
                current_fitness = copy.deepcopy(best_fitness)
                continue

            else:

                p = exp(-((new_fitness[0] - current_fitness[0]) / t))

                if random.random() > p:

                    current_solution = copy.deepcopy(new_solution)
                    current_fitness = copy.deepcopy(new_fitness)

    # termination condition reached
    else:

        if print_stats:
            print(
                "Fitness of best solution found: {}\nNumber of remaining violations: {}\nLength of active Extensions: {}\n".format(
                    best_fitness[0], best_fitness[1], best_fitness[2]
                )
            )
            print("Extension Solution: ", extension_measures_to_str(best_solution))

        # return tuple (Fitness, Solution)
        return (best_fitness, best_solution)


# modified metropolis algorithm, creates a population in each iteration and chooses the one with highest fitness
def metropolis_algorithm_population(
    case: Dict,
    time_limit_secs: int,
    population_size: int,
    fitness_function: Callable,
    mutation_function: Callable,
    temperature: float,
    print_stats: bool,
) -> tuple:

    current_solution = copy.deepcopy(case)
    current_fitness = fitness_function(current_solution)

    best_solution = copy.deepcopy(current_solution)
    best_fitness = copy.deepcopy(current_fitness)

    rounds_no_improvement = 0

    t_start = time.time()

    while (time.time() - t_start) < time_limit_secs:

        if print_stats:
            print(
                "Temp: {:<10.2f}\tCurrent Fitness: {:<10.2f}\tBest Fitness: {:<10.2f}\tViolations (current): {:<10}\tViolations (best): {:<10}\n".format(
                    temperature,
                    current_fitness[0],
                    best_fitness[0],
                    current_fitness[1],
                    best_fitness[1],
                )
            )

        population = []

        for _ in range(population_size):
            new_solution = copy.deepcopy(current_solution)
            mutation_function(new_solution)
            population.append(new_solution)

        population.sort(key=lambda x: fitness_function(x)[0], reverse=True)

        new_solution = copy.deepcopy(population[0])
        new_fitness = fitness_function(new_solution)

        # new solution is better than current solution
        if new_fitness[0] >= current_fitness[0]:

            current_solution = copy.deepcopy(new_solution)
            current_fitness = copy.deepcopy(new_fitness)
            rounds_no_improvement = 0

            # new solution is best solution found so far
            if current_fitness[0] > best_fitness[0]:

                best_solution = copy.deepcopy(current_solution)
                best_fitness = copy.deepcopy(current_fitness)

        # fitness if new solution is worse than current solution
        else:

            rounds_no_improvement += 1

            p = exp(-((new_fitness[0] - current_fitness[0]) / temperature))

            if random.random() <= p:

                current_solution = copy.deepcopy(new_solution)
                current_fitness = copy.deepcopy(new_fitness)
                continue

            # if no improvement for n rounds, switch current solution back to previous best solution
            elif rounds_no_improvement > 6:

                rounds_no_improvement = 0
                current_solution = copy.deepcopy(best_solution)
                current_solution = copy.deepcopy(best_fitness)

    # termination condition reached
    else:

        if print_stats:
            print(
                "Fitness of best solution found: {}\nNumber of remaining violations: {}\nLength of active Extensions: {}\n".format(
                    best_fitness[0], best_fitness[1], best_fitness[2]
                )
            )
            print("Extension Solution: ", extension_measures_to_str(best_solution))

        # return tuple (Fitness, Solution)
        return (best_fitness, best_solution)


# allowed time for each algorithm [seconds]
time_limit = 15

# Choose if algorithms print stats to console during execution
printing = True


# wrapper functions for each algorithm


def run_hillclimber():

    case = case300extended2()
    return hill_climber(
        random_init(case), evaluate_objective_function, time_limit, printing
    )


def run_hillclimber_custom():

    case = case300extended2()
    return hill_climber(random_init(case), evaluate_fitness, time_limit, printing)


def run_sa():

    case = case300extended2()
    return simulated_annealing(
        random_init(case),
        time_limit,
        evaluate_objective_function,
        flip_mutation,
        1000000,
        10,
        printing,
    )


def run_sa_custom():

    case = case300extended2()
    return simulated_annealing(
        random_init(case),
        time_limit,
        evaluate_fitness,
        flip_mutation,
        1000000,
        10,
        printing,
    )


def run_ea():

    case = case300extended2()
    return evolutionary_algorithm(
        case,
        time_limit,
        25,
        flip_mutation,
        crossover_onepoint,
        evaluate_objective_function,
        print_stats=printing,
    )


def run_ea_custom():

    case = case300extended2()
    return evolutionary_algorithm(
        case,
        time_limit,
        25,
        flip_mutation,
        crossover_onepoint,
        evaluate_fitness,
        print_stats=printing,
    )


def run_metropolis():

    case = case300extended2()
    return metropolis_algorithm(
        random_init(case),
        time_limit,
        evaluate_objective_function,
        flip_mutation,
        2000,
        printing,
    )


def run_metropolis_custom():

    case = case300extended2()
    return metropolis_algorithm(
        random_init(case), time_limit, evaluate_fitness, flip_mutation, 2000, printing
    )


# run all algorithms and create a log file of the results
def run_all_algorithms():

    results = open("results.tsv", "a")

    if os.stat("results.tsv").st_size == 0:
        results.write(
            "algorithm\tfitness(basic)\tfitness(custom)\tnum_violations\tlength_extensions\textension_pattern\n"
        )

    algorithms = (
        run_hillclimber,
        run_hillclimber_custom,
        run_ea,
        run_ea_custom,
        run_sa,
        run_sa_custom,
        run_metropolis,
        run_metropolis_custom,
    )

    n_runs = 10  # number of runs for each algorithm

    for algorithm in algorithms:

        for i in range(n_runs):

            print("\nRound {} - {}()".format(i + 1, algorithm.__name__))
            return_tuple = algorithm()
            results.write(
                "{}\t{}\t{}\t{}\t{}\t{}\n".format(
                    algorithm.__name__,
                    evaluate_objective_function(return_tuple[1])[0],
                    evaluate_fitness(return_tuple[1])[0],
                    return_tuple[0][1],
                    return_tuple[0][2],
                    extension_measures_to_str(return_tuple[1]),
                )
            )

    print("\nFinished all Runs!")
    results.close()


# final contestors

# hc algorithm with max_iterations
def hc_maxIterations(case: dict, max_iterations: int = 10000) -> tuple:

    # evaluate parent
    fitness = evaluate_objective_function(case)

    gen = 0

    # i replaced the termination condition for a timer for easier comparison
    while gen < max_iterations:

        gen += 1

        # clone parent
        child = copy.deepcopy(case)

        # mutate one extension
        rindex = random.randint(0, len(child["extensions"]) - 1)
        child["extensions"][rindex][10] = 1 - child["extensions"][rindex][10]

        # evaluate fitness
        child_fitness = evaluate_objective_function(child)

        # found a better solution or a solution with same fitness?
        if child_fitness[0] >= fitness[0]:
            case = copy.deepcopy(child)
            fitness = copy.deepcopy(child_fitness)
            # fitness = np.copy(child_fitness)

    else:
        return (fitness, case)


# metropolis algorithm
def metropolis_maxIterations(
    case: Dict, temperature: float = 2000.0, max_iterations: int = 10000
) -> tuple:

    current_solution = copy.deepcopy(case)
    current_fitness = evaluate_objective_function(current_solution)

    best_solution = copy.deepcopy(current_solution)
    best_fitness = copy.deepcopy(current_fitness)

    rounds_no_improvement = 0
    cur_round = 0

    while cur_round < max_iterations:

        cur_round += 1

        # generate new solution through mutation and evaluate fitness
        new_solution = copy.deepcopy(current_solution)
        flip_mutation(new_solution)
        new_fitness = evaluate_objective_function(new_solution)

        # new solution is better than current solution
        if new_fitness[0] >= current_fitness[0]:

            current_solution = copy.deepcopy(new_solution)
            current_fitness = copy.deepcopy(new_fitness)
            rounds_no_improvement = 0

            # new solution is best solution found so far
            if current_fitness[0] > best_fitness[0]:

                best_solution = copy.deepcopy(current_solution)
                best_fitness = copy.deepcopy(current_fitness)

        # fitness if new solution is worse than current solution
        else:

            rounds_no_improvement += 1

            # if no improvement for n rounds, switch current solution back to previous best solution
            if rounds_no_improvement > 10:

                rounds_no_improvement = 0
                current_solution = copy.deepcopy(best_solution)
                current_fitness = copy.deepcopy(best_fitness)
                continue

            else:
                p = exp(-((new_fitness[0] - current_fitness[0]) / temperature))
                if random.random() > p:

                    current_solution = copy.deepcopy(new_solution)
                    current_fitness = copy.deepcopy(new_fitness)

    # termination condition reached
    else:

        # return tuple (Fitness, Solution)
        return (best_fitness, best_solution)


# simulated annealing
def simulated_annealing_maxIterations(
    case: Dict, t_start: float = 1000000, t_end: float = 1, max_iterations: int = 10000
) -> tuple:

    current_solution = copy.deepcopy(case)
    current_fitness = evaluate_objective_function(current_solution)

    best_solution = copy.deepcopy(current_solution)
    best_fitness = copy.deepcopy(current_fitness)

    t = t_start
    rounds_no_improvement = 0
    cur_round = 0

    while (cur_round < max_iterations) and (t > t_end):

        cur_round += 1

        # use geometric cooling scheme
        t *= 0.99

        # generate new solution through mutation and evaluate fitness
        new_solution = copy.deepcopy(current_solution)
        flip_mutation(new_solution)
        new_fitness = evaluate_objective_function(new_solution)

        # new solution is better than current solution
        if new_fitness[0] >= current_fitness[0]:

            current_solution = copy.deepcopy(new_solution)
            current_fitness = copy.deepcopy(new_fitness)
            rounds_no_improvement = 0

            # new solution is best solution found so far
            if current_fitness[0] > best_fitness[0]:

                best_solution = copy.deepcopy(current_solution)
                best_fitness = copy.deepcopy(current_fitness)

        # fitness if new solution is worse than current solution
        else:

            # if no improvement for 10 rounds, switch current solution back to previous best solution
            if rounds_no_improvement > 10:

                rounds_no_improvement = 0
                current_solution = copy.deepcopy(best_solution)
                current_fitness = copy.deepcopy(best_fitness)
                continue

            else:
                rounds_no_improvement += 1
                p = exp(-((new_fitness[0] - current_fitness[0]) / t))

                if random.random() >= p:

                    current_solution = copy.deepcopy(new_solution)
                    current_fitness = copy.deepcopy(new_fitness)

    # termination condition reached
    else:

        # return tuple (Fitness, Solution)
        return (best_fitness, best_solution)


# test the 3 best algorithms for 10k iterations, log results for statistical tests
def contest():

    case = case300extended2()

    rounds = 10

    log_file = open("final_contest.tsv", "w")

    algorithms = (
        hc_maxIterations,
        metropolis_maxIterations,
        simulated_annealing_maxIterations,
    )

    for algorithm in algorithms:

        for _ in range(rounds):
            print("round ", _, " algorithm ", algorithm.__name__)

            ret = algorithm(copy.deepcopy(case))
            log_file.write(
                "{}\t{}\t{}\t{}\n".format(
                    algorithm.__name__, ret[0][0], ret[0][1], ret[0][2]
                )
            )

    log_file.close()


if __name__ == "__main__":

    # first run
    # run_all()

    # second run
    contest()
