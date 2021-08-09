import random
import math


class DB:
    def __init__(self, filepath: str) -> None:

        file = open(filepath)
        self.cities, self.distance_matrix, self.pheromone_matrix = [], [], []
        first_line = True
        self.n = None

        for line in file:
            row = line.split()
            if first_line:
                for element in row:
                    self.cities.append(element)
                first_line = False
                self.n = len(self.cities)
            else:
                self.distance_matrix.append(list(map(int, row[1:])))

                self.pheromone_matrix.append(
                    [(1 / (self.n ** 2)) for _ in range(self.n)]
                )

        file.close()

    def sumup_pheromones(self):
        p = 0
        for row in self.pheromone_matrix:
            for col in row:
                p += col
        return p

    def str_both_matrices(self) -> str:

        dm, pm = "\t", "\t"
        for city in self.cities:
            dm += city + "\t"
            pm += city + "\t"
        dm += "\n"
        pm += "\n"
        for row_d, row_p, city in zip(
            self.distance_matrix, self.pheromone_matrix, self.cities
        ):
            dm += city + "\t"
            pm += city + "\t"
            for val_d, val_p in zip(row_d, row_p):
                dm += str(val_d) + "\t"
                pm += str(val_p) + "\t"
            dm += "\n"
            pm += "\n"

        return f"Distance Matrix\n---------------\n{dm}\nPheromone Matrix\n---------------\n{pm}\n"

    def both_matrices_to_tsv(self, id: str) -> None:

        tsv_file = open(f"matrices_{id}.tsv", "a")
        tsv_file.write(self.str_both_matrices())
        tsv_file.close()

    def str_pheromone_matrix(self) -> str:

        pm = "\t"
        for city in self.cities:
            pm += city + "\t"
        pm += "\n"
        for row_p, city in zip(self.pheromone_matrix, self.cities):
            pm += city + "\t"
            for val_p in row_p:
                pm += f"{val_p:.16f}\t"
            pm += "\n"

        return f"Pheromone Matrix\n---------------\n{pm}\nsum of p={self.sumup_pheromones()}\n"

    def pheromone_to_tsv(self, id: str) -> None:

        tsv_file = open(f"pheromones_{id}.tsv", "a")
        tsv_file.write(self.str_pheromone_matrix())
        tsv_file.close()

    def get_distance(self, route: list) -> int:

        assert len(route) == self.n

        total_distance = 0
        for i in range(len(route) - 1):
            current, next = route[i], route[i + 1]
            total_distance += self.distance_matrix[current][next]
        total_distance += self.distance_matrix[route[-1]][route[0]]

        return total_distance

    def get_pheromone_concentration(self, i, j):
        return self.pheromone_matrix[i][j]

    def evaporate(self):

        sum_p = self.sumup_pheromones()

        for r in range(len(self.pheromone_matrix)):
            for c in range(len(self.pheromone_matrix[r])):
                self.pheromone_matrix[r][c] /= sum_p

    def place_pheromones(self, route: list) -> None:

        distance = self.get_distance(route)
        amount = math.exp(-(distance / 500))

        for i in range(len(route) - 1):
            current, next = route[i], route[i + 1]
            self.pheromone_matrix[current][next] += amount

    def get_final_solution(self):

        visited_cities, unvisited_cities = [], [i for i in range(self.n)]
        visited_cities.append(unvisited_cities.pop(random.randint(0, self.n - 1)))

        while unvisited_cities:

            cur_city = visited_cities[-1]
            max_p_score, best_city = -1, unvisited_cities[0]

            for city in unvisited_cities:

                temp_p_score = self.get_pheromone_concentration(cur_city, city)

                if temp_p_score > max_p_score:
                    max_p_score = temp_p_score
                    best_city = city

            unvisited_cities.remove(best_city)
            visited_cities.append(best_city)

        print(f"Fitness: {self.get_distance(visited_cities)}\nTour: {visited_cities}\n")


class Ant:
    def __init__(self, db: "DB") -> None:

        self.adb = db
        self.unvisited_cities = [n for n in range(db.n)]
        self.visited_cities = []

    def run(self):

        if len(self.unvisited_cities) == self.adb.n:
            self.visited_cities.append(
                self.unvisited_cities.pop(
                    random.randint(0, len(self.unvisited_cities) - 1)
                )
            )

        while len(self.unvisited_cities) > 1:

            cur_pos = self.visited_cities[-1]
            next_pos = random.randint(0, len(self.unvisited_cities) - 1)
            if (random.random() / self.adb.n) < self.adb.get_pheromone_concentration(
                cur_pos, next_pos
            ):
                self.visited_cities.append(self.unvisited_cities.pop(next_pos))

        self.visited_cities.append(self.unvisited_cities.pop())

    def get_path(self):

        assert (
            len(self.unvisited_cities) == 0 and len(self.visited_cities) == self.adb.n
        )
        return self.visited_cities


def ant_colony_optimization(
    filepath: str,
    num_ants: int,
    iterations: int,
    log_pheromones: bool = False,
    id: int = "",
):

    db = DB(filepath)
    print(db.str_both_matrices())

    for i in range(iterations):

        print(f"{i:<2}.th iteration of {num_ants} Ants")

        swarm = [Ant(db) for _ in range(num_ants)]

        for ant in swarm:
            ant.run()

        for ant in swarm:
            db.place_pheromones(ant.get_path())

        db.evaporate()

        if log_pheromones and (i % 100 == 0):
            db.pheromone_to_tsv(id)

    db.get_final_solution()


if __name__ == "__main__":

    ant_colony_optimization(
        "ant_colony_algorithm/citiesAndDistances.txt", 50, 1500, True
    )
