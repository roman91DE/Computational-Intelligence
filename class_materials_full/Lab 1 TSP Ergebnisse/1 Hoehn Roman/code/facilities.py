import random


# database class reads from txt file, saves distance matrix and names of cities
class DB:
    def __init__(self, filepath: str) -> None:
        file = open(filepath)
        self.cities, self.distance_matrix = [], []
        first_line = True
        for line in file:
            row = line.split()
            if first_line:
                for element in row:
                    self.cities.append(element)
                first_line = False
            else:
                self.distance_matrix.append(list(map(int, row[1:])))
        file.close()


# set global database variable for all excercises
db = DB(
    "/Users/roman/computational_intelligence/class_materials_full/Lab 1 TSP Ergebnisse/1 Hoehn Roman/data/citiesAndDistances.txt"
)


# compute the distance for a single route
def get_distance(route: list) -> int:
    global db
    total_distance = 0
    for i in range(len(route) - 1):
        current, next = route[i], route[i + 1]
        total_distance += db.distance_matrix[current][next]
    total_distance += db.distance_matrix[route[-1]][
        route[0]
    ]  # travel back to starting point
    return total_distance


# print city names & total distance for input route
def print_route(route: list, dist: int) -> None:
    global db
    for element in route:
        print(db.cities[element], end=" - ")
    print(db.cities[route[0]])
    print("Total Distance = {}km".format(dist))


# helper function returns reverse of input route
def reverse_route(route: list) -> list:
    arr = []
    for element in route:
        arr.insert(0, element)
    return arr


# returns a random route
def rand_route(n: int) -> list:
    arr = [_ for _ in range(n)]
    random.shuffle(arr)
    return arr


# prints line to console
def print_line() -> None:
    print("\n-----------------------------------------------\n")
