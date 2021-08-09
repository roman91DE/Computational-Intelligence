from city import distance
import time
import random
import algorithms_insertion_heuristics
from typing import Callable, List


def calc_tour_length(tsp, tour):
    cities = tsp["CITIES"]
    length = 0
    for i in range(tsp["DIMENSION"]):
        length += distance(cities[tour[i - 1]], cities[tour[i]])
    return length


def node_xchg_step(tour):
    i = random.randint(0, len(tour) - 1)
    k = random.randint(0, len(tour) - 1)
    tour[i], tour[k] = tour[k], tour[i]
    return tour


# implementation of a hill climber
def HC_tour(tsp, max_iterations):
    start_time = time.time()
    tour = [i for i in range(tsp["DIMENSION"])]

    random.shuffle(tour)
    tour_len = calc_tour_length(tsp, tour)
    visited_tours = 1

    # best solution found so far
    best_tour = tour
    best_tour_len = tour_len

    # iterate max_iterations times
    while visited_tours < max_iterations:
        # derive a new tour
        new_tour = node_xchg_step(list(best_tour))
        new_tour_len = calc_tour_length(tsp, new_tour)
        visited_tours += 1

        # found a better one?
        if new_tour_len < best_tour_len:
            print(
                "improved from",
                best_tour_len,
                "to",
                new_tour_len,
                "by",
                best_tour_len - new_tour_len,
                "visited tours",
                visited_tours,
            )
            best_tour = new_tour
            best_tour_len = new_tour_len

    time_consumed = time.time() - start_time
    print(
        "time consumed",
        time_consumed,
        "tours visited",
        visited_tours,
        "number of tours per second",
        visited_tours / time_consumed,
    )
    return (best_tour_len), best_tour


def calc_HC_tour(tsp):
    return HC_tour(tsp, 100000)


# -------------------
# my code starts here
# -------------------


# class to represent and modify a solution
class Solution:

    # class constructor:
    # assigns random permutation as chromosome if no sequence is passed
    def __init__(
        self, tsp_obj: dict, m_rate: float, c_rate: float, chromosome: list = None
    ) -> None:
        assert (0 <= m_rate <= 1) and (0 <= c_rate <= 1)
        self.tsp_obj = tsp_obj
        self.n = tsp_obj["DIMENSION"]
        self.m_rate = m_rate
        self.c_rate = c_rate
        if chromosome is None:
            self.chromosome = [i for i in range(self.n)]
            random.shuffle(self.chromosome)
        else:
            self.chromosome = list(chromosome)
        self.fit = self.fitness()

    # check if a solution is valid
    def is_valid(self):
        return (len(self.chromosome) == len(set(self.chromosome))) and (
            (len(self.chromosome) == self.n)
        )

    # return negative of distance measured by dist_function for a given chromosome and a given tsp dictionary
    def fitness(self) -> int:
        return -(calc_tour_length(self.tsp_obj, self.chromosome))

    # Mutation Operators
    # ------------------

    # swap current node with another random node
    def mutate1(self) -> None:
        for i in range(self.n):
            if random.random() < self.m_rate:
                zv = random.randint(0, self.n - 1)
                while zv == i:
                    zv = random.randint(0, self.n - 1)
                self.chromosome[i], self.chromosome[zv] = (
                    self.chromosome[zv],
                    self.chromosome[i],
                )
        self.fit = self.fitness()

    # swap current node with an adjacent node
    def mutate2(self) -> None:
        for i in range(1, self.n - 2):  # skip first and last node
            if random.random() < self.m_rate:
                if random.random() > 0.5:
                    offset = 1
                else:
                    offset = -1
                self.chromosome[i], self.chromosome[i + offset] = (
                    self.chromosome[i + offset],
                    self.chromosome[i],
                )
        self.fit = self.fitness()

    # randomly picks one of three crossover operators
    def random_mutation(self) -> None:
        zv = random.random()
        if zv < 0.55:
            self.mutate1()
        else:
            self.mutate2()

    # Crossover Operators
    # -------------------

    # randomized one point crossover between parents chromosomes
    def crossover1(mother: "Solution", father: "Solution") -> "Solution":
        assert (
            (mother.n == father.n)
            and (mother.c_rate == father.c_rate)
            and (mother.m_rate == father.m_rate)
        )
        if random.random() < mother.c_rate:
            pivot_a, pivot_b = random.randint(0, mother.n - 2), random.randint(
                0, mother.n - 1
            )
            while pivot_a > pivot_b:
                pivot_b = random.randint(0, mother.n - 1)
            crossover_part = list(father.chromosome[pivot_a:pivot_b])
            base_part = list(mother.chromosome)
            for gene in crossover_part:
                if gene in base_part:
                    base_part.remove(gene)
            return Solution(
                mother.tsp_obj,
                mother.m_rate,
                mother.c_rate,
                list(base_part + crossover_part),
            )
        else:
            if random.random() > 0.5:
                return Solution(
                    mother.tsp_obj, mother.m_rate, mother.c_rate, mother.chromosome
                )
            else:
                return Solution(
                    father.tsp_obj, father.m_rate, father.c_rate, father.chromosome
                )

    # randomized multiple point crossover of gene pairs
    def crossover2(mother: "Solution", father: "Solution") -> "Solution":
        assert (
            (mother.n == father.n)
            and (mother.c_rate == father.c_rate)
            and (mother.m_rate == father.m_rate)
        )
        if random.random() < mother.c_rate:
            pivot_a, pivot_b = random.randint(0, (mother.n // 2) - 1), random.randint(
                (mother.n // 2) + 1, mother.n - 1
            )
            crossover_part = list(father.chromosome[pivot_a:pivot_b])
            base_part = list(mother.chromosome)
            for gene in crossover_part:
                if gene in base_part:
                    base_part.remove(gene)
            while len(crossover_part) > 0:
                if len(crossover_part) % 2 == 0:
                    pair = [crossover_part.pop(), crossover_part.pop()]
                    if random.random() > 0.5:
                        base_part.append(pair[0])
                        base_part.append(pair[1])
                    else:
                        base_part.insert(0, pair[1])
                        base_part.insert(0, pair[0])
                else:
                    if random.random() > 0.5:
                        base_part.append(crossover_part.pop())
                    else:
                        base_part.insert(0, crossover_part.pop())
            assert len(base_part) == len(mother.chromosome)
            return Solution(mother.tsp_obj, mother.m_rate, mother.c_rate, base_part)
        else:
            if random.random() > 0.5:
                return Solution(
                    mother.tsp_obj, mother.m_rate, mother.c_rate, mother.chromosome
                )
            else:
                return Solution(
                    father.tsp_obj, father.m_rate, father.c_rate, father.chromosome
                )

    # one point crossover at random position
    def crossover3(mother: "Solution", father: "Solution") -> "Solution":
        assert (
            (mother.n == father.n)
            and (mother.c_rate == father.c_rate)
            and (mother.m_rate == father.m_rate)
        )
        if random.random() < mother.c_rate:
            pivot_a, pivot_b = random.randint(0, (mother.n // 2) - 1), random.randint(
                (mother.n // 2) + 1, mother.n - 1
            )
            crossover_part = list(father.chromosome[pivot_a:pivot_b])
            base_part = list(mother.chromosome)
            for gene in crossover_part:
                if gene in base_part:
                    base_part.remove(gene)
            position = random.randint(0, (mother.n) - 1)
            new_chromosome = (
                base_part[:position] + crossover_part + base_part[position:]
            )
            assert len(new_chromosome) == len(mother.chromosome)
            return Solution(
                mother.tsp_obj, mother.m_rate, mother.c_rate, new_chromosome
            )
        else:
            if random.random() > 0.5:
                return Solution(
                    mother.tsp_obj, mother.m_rate, mother.c_rate, mother.chromosome
                )
            else:
                return Solution(
                    father.tsp_obj, father.m_rate, father.c_rate, father.chromosome
                )

    # randomly picks one of three crossover operators
    def random_crossover(mother: "Solution", father: "Solution") -> "Solution":
        zv = random.random()
        if zv < 0.33:
            return Solution.crossover1(mother, father)
        elif zv < 0.66:
            return Solution.crossover2(mother, father)
        else:
            return Solution.crossover3(mother, father)

    # print route and fitness
    def __str__(self):
        s = ""
        for val in self.chromosome:
            s += str(val + 1) + " "
        s += "\nFitness={}".format(self.fit)
        return s


# wrapper class to save and sort solutions
class Population:
    def __init__(self) -> None:
        self.population = []

    # sort population for fitness
    def sort_population(self):
        self.population.sort(key=lambda x: x.fit, reverse=True)

    # calculate and return the average fitness score of a vector of solutions
    def average_fitness(self) -> float:
        total = 0
        for solution in self.population:
            total += solution.fit
        return total / len(self.population)


# create an array of weights for random.choices()
def probability_dist(n_elements: int) -> List[int]:
    dist = []
    for i in range(n_elements, 0, -1):
        dist.append(i)
    return dist


# first implementation of an (basic) Evolutionary Algorithm
def __ea1__(
    tsp: dict, population_size: int, max_generations: int, log_results: bool = False
) -> int:

    t0 = time.time()

    # setting parameters
    m_rate = 0.05
    c_rate = 0.95
    size_breading_pop = 6
    tournament_size = 2
    elite_size = 1

    breading_dist = probability_dist(size_breading_pop)
    current_generation = 0

    # initialization of algorithm
    main_pop = Population()
    for _ in range(population_size):
        main_pop.population.append(Solution(tsp, m_rate, c_rate))

    # optionally log results
    if log_results:
        log_file = open("log.tsv", "a")
        log_file.write(
            "EA1\nPopulation Size\tGenerations\tMutation Rate\tCrossover Rate\tBreading Population Size\tTournament Size\tElite Size\n"
        )
        log_file.write(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                population_size,
                max_generations,
                m_rate,
                c_rate,
                size_breading_pop,
                tournament_size,
                elite_size,
            )
        )
        log_file.write("Current Generation\tAverage Fitness\tBest Fitness\n")

    # main loop
    while current_generation < max_generations:

        main_pop.sort_population()
        average_fitness = main_pop.average_fitness()

        if log_results and current_generation % 100 == 0:
            log_file.write(
                "{}\t{}\t{}\n".format(
                    current_generation, average_fitness, main_pop.population[0].fit
                )
            )

        current_generation += 1

        # select breading population by ...
        breading_pop = Population()
        while len(breading_pop.population) < size_breading_pop:

            # ...random tournament selection
            tournament_pool = Population()
            tournament_pool.population = random.choices(
                main_pop.population, k=tournament_size
            )
            tournament_pool.sort_population()
            breading_pop.population.append(
                tournament_pool.population[0]
            )  # selects Solution with highest fitness

        # initialize next generation
        else:
            new_population = Population()

            # select elite from previous population
            for i in range(elite_size):
                new_population.population.append(main_pop.population[i])

            # generate offspring
            breading_pop.sort_population()
            while len(new_population.population) < population_size:

                # selects parents with higher chance for fitter solutions..
                mother, father = random.choices(
                    breading_pop.population, weights=breading_dist, k=2
                )
                # .. and picks random crossover and mutation operators
                child = Solution.random_crossover(mother, father)
                child.random_mutation()
                new_population.population.append(child)

            else:
                main_pop.population = list(new_population.population)

    else:
        # reached limit of generations, return best solution
        main_pop.sort_population()
        assert main_pop.population[0].is_valid()

        if log_results:
            log_file.write(
                "Total Time: {}\nBest Route: {}\n\n".format(
                    time.time() - t0, str(main_pop.population[0])
                )
            )
            log_file.close()

        return -(main_pop.population[0].fit)


# second implementation of an Evolutionary Algorithm, dynamic adjustments of mutation rate
# Source: Damiani et. al, Evolutionary Design of Hashing Function Circuits using an FPGA, 2002


def __ea2__(
    tsp: dict, population_size: int, max_generations: int, log_results: bool = False
) -> int:

    t0 = time.time()

    # setting parameters
    m_rate = 0.0125
    c_rate = 0.95
    size_breading_pop = 6
    tournament_size = 2
    elite_size = 1

    breading_dist = probability_dist(size_breading_pop)
    current_generation = 0
    optimal_ratio = 3 / 2  # best fitness/ average fitness

    # initialization of algorithm
    main_pop = Population()
    for _ in range(population_size):
        main_pop.population.append(Solution(tsp, m_rate, c_rate))

    # optionally log results
    if log_results:
        log_file = open("log.tsv", "a")
        log_file.write(
            "EA2\nPopulation Size\tGenerations\tMutation Rate\tCrossover Rate\tBreading Population Size\tTournament Size\tElite Size\n"
        )
        log_file.write(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                population_size,
                max_generations,
                m_rate,
                c_rate,
                size_breading_pop,
                tournament_size,
                elite_size,
            )
        )
        log_file.write(
            "Current Generation\tAverage Fitness\tBest Fitness\tMutation Rate\n"
        )

    # main loop
    while current_generation < max_generations:

        main_pop.sort_population()

        # calculate ratio -> best fitness : average fitness
        cur_ratio = main_pop.population[0].fit / main_pop.average_fitness()
        # adjust mutation rate
        m_rate = (optimal_ratio / cur_ratio) * m_rate

        if log_results and current_generation % 100 == 0:
            log_file.write(
                "{}\t{}\t{}\t{}\n".format(
                    current_generation,
                    main_pop.average_fitness(),
                    main_pop.population[0].fit,
                    m_rate,
                )
            )

        current_generation += 1

        # select breading population by ...
        breading_pop = Population()
        while len(breading_pop.population) < size_breading_pop:

            # ...random tournament selection
            tournament_pool = Population()
            tournament_pool.population = random.choices(
                main_pop.population, k=tournament_size
            )
            tournament_pool.sort_population()
            breading_pop.population.append(
                tournament_pool.population[0]
            )  # selects Solution with highest fitness

        # initialize next generation
        else:
            new_population = Population()

            # select elite from previous population
            for i in range(elite_size):
                new_population.population.append(main_pop.population[i])

            # generate offspring
            breading_pop.sort_population()
            while len(new_population.population) < population_size:

                # selects parents with higher chance for fitter solutions..
                mother, father = random.choices(
                    breading_pop.population, weights=breading_dist, k=2
                )
                # .. uses basic crossover and mutation operator
                child = Solution.random_crossover(mother, father)
                child.random_mutation()
                new_population.population.append(child)

            else:
                main_pop.population = list(new_population.population)

    else:
        # reached limit of generations, return best solution
        main_pop.sort_population()
        assert main_pop.population[0].is_valid()

        if log_results:
            log_file.write(
                "Total Time: {}\nBest Route: {}\n\n".format(
                    time.time() - t0, str(main_pop.population[0])
                )
            )
            log_file.close()

        return -(main_pop.population[0].fit)


# third implementation of an (basic) Evolutionary Algorithm, initialized with 1 solution from next neighbour heuristic
def __ea3__(
    tsp: dict, population_size: int, max_generations: int, log_results: bool = False
) -> int:

    t0 = time.time()

    # setting parameters
    m_rate = 0.1
    c_rate = 0.95
    size_breading_pop = 5
    tournament_size = 2
    elite_size = 1

    breading_dist = probability_dist(size_breading_pop)
    current_generation = 0

    main_pop = Population()

    # 1 initial solutions is generated by next neighbour heuristics
    nn = algorithms_insertion_heuristics.nearest_neighbor_tour(tsp)
    # adjust from 1 indexed to 0 indexed
    for i in range(len(nn)):
        nn[i] -= 1
    nn.pop()  # pop last city because it is equal to the first (different notation)
    main_pop.population.append(Solution(tsp, m_rate, c_rate, nn))

    while len(main_pop.population) < population_size:
        main_pop.population.append(Solution(tsp, m_rate, c_rate))

    # optionally log results
    if log_results:
        log_file = open("log.tsv", "a")
        log_file.write(
            "EA3\nPopulation Size\tGenerations\tMutation Rate\tCrossover Rate\tBreading Population Size\tTournament Size\tElite Size\n"
        )
        log_file.write(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                population_size,
                max_generations,
                m_rate,
                c_rate,
                size_breading_pop,
                tournament_size,
                elite_size,
            )
        )
        log_file.write(
            "Current Generation\tAverage Fitness\tBest Fitness\tMutation Rate\n"
        )

    # main loop
    while current_generation < max_generations:

        main_pop.sort_population()

        if log_results and current_generation % 100 == 0:
            log_file.write(
                "{}\t{}\t{}\t{}\n".format(
                    current_generation,
                    main_pop.average_fitness(),
                    main_pop.population[0].fit,
                    m_rate,
                )
            )

        current_generation += 1

        # select breading population by ...
        breading_pop = Population()
        while len(breading_pop.population) < size_breading_pop:

            # ...random tournament selection
            tournament_pool = Population()
            tournament_pool.population = random.choices(
                main_pop.population, k=tournament_size
            )
            tournament_pool.sort_population()
            breading_pop.population.append(
                tournament_pool.population[0]
            )  # selects Solution with highest fitness

        # initialize next generation
        else:
            new_population = Population()

            # select elite from previous population
            for i in range(elite_size):
                new_population.population.append(main_pop.population[i])

            # generate offspring
            breading_pop.sort_population()
            while len(new_population.population) < population_size:

                # selects parents with higher chance for fitter solutions..
                mother, father = random.choices(
                    breading_pop.population, weights=breading_dist, k=2
                )
                # .. uses basic crossover and mutation operator
                child = Solution.random_crossover(mother, father)
                child.random_mutation()
                new_population.population.append(child)

            else:
                main_pop.population = list(new_population.population)

    else:
        # reached limit of generations, return best solution
        main_pop.sort_population()
        assert main_pop.population[0].is_valid()

        if log_results:
            log_file.write(
                "Total Time: {}\nBest Route: {}\n\n".format(
                    time.time() - t0, str(main_pop.population[0])
                )
            )
            log_file.close()

        return -(main_pop.population[0].fit)


# run and print ea
def EA_tour(tsp, population_size, max_generations):

    start_time = time.time()

    # uncomment to choose which EA to run
    # ---

    # tour_len = __ea1__(tsp, population_size, max_generations,log_results=True)
    # tour_len = __ea2__(tsp, population_size, max_generations,log_results=True)
    tour_len = __ea3__(tsp, population_size, max_generations, log_results=True)

    time_consumed = time.time() - start_time

    print("time consumed", time_consumed)
    return tour_len


# -------------------
# my code ends here
# -------------------


def calc_EA_tour(tsp):
    return EA_tour(tsp, 20, 5000)


def calc_EA_tour_txt(tsp):
    file = open("bestresults.txt", "w")
    runs = 30
    tour_len_sum = 0
    for i in range(runs):
        best_tour_len = calc_EA_tour(tsp)
        print("EA LENGTH RUN", i + 1, ":        {}".format(best_tour_len))
        file.write(str(best_tour_len))
        file.write("\n")
        tour_len_sum += best_tour_len
    avg_tour_len = round(tour_len_sum / runs, 2)
    return avg_tour_len


# run geneteic algorithm:
# python3.9 main.py -g ./tspfiles


# run on ubuntu:
# /usr/bin/python3 /home/roman/github/Lab2-main/code_for_Students/main.py -g ./tspfiles


# run on windows
# C:/Users/Dozent/AppData/Local/Programs/Python/Python38-32/python.exe c:/Users/Dozent/Desktop/Roman/Lab2-main/code_for_Students/main.py -g ./code_for_Students\tspfiles
