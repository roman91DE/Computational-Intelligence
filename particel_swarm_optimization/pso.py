from matplotlib import pyplot as plt
import random
from typing import Callable


def f(x: float) -> float:
    return (
        5 * (x ** 3)
        - 42 * (x ** 2)
        + 36 * x
        - 18798
        + (130 * (x ** 0.5))
        - (x * x)
        + 100000000
    )


def plot_f(f, min, max):

    x_vals = [x for x in range(min, max, (max - min) // 100)]
    y_vals = [f(x) for x in x_vals]

    plt.plot(x_vals, y_vals)
    plt.show()


# objectiv: find x for which f(x) -> 0


class Particle:
    def __init__(
        self, x_init, fun: Callable[[float], float], swarm: "Swarm", inertia_init=10
    ):

        self.f = fun

        self.swarm = swarm

        self.cur_x = x_init
        self.cur_y = self.f(x_init)

        self.best_x = self.cur_x
        self.best_y = self.cur_y

        self.alpha = inertia_init
        self.beta1, self.beta2 = random.random(), random.random()

        self.eval()

        self.velocity = 1
        # v_i(t+1) = alpha * v_i(t) + beta1 * [x_i(local)(t) - x_i(t)] + beta2 * [x(global)(t) - x_i(t)]

    def update_coef(self):

        self.alpha *= 0.95
        self.beta1, self.beta2 = random.random(), random.random()

    def update_vel(self):

        inertia_part = self.alpha * self.velocity
        cognitive_part = self.beta1 * (self.best_x - self.cur_x)
        social_part = self.beta2 * (self.swarm.swarm_best_x - self.cur_x)

        self.velocity = inertia_part + cognitive_part + social_part

    def update_pos(self):
        self.cur_x = self.cur_x + self.velocity

    def run(self):

        self.update_coef()
        self.update_vel()
        self.update_pos()
        self.eval()

    def eval(self):

        self.cur_y = self.f(self.cur_x)

        if abs(self.cur_y) <= abs(self.best_y):
            self.best_y, self.best_x = self.cur_y, self.cur_x

            if abs(self.best_y) < abs(self.swarm.swarm_best_y):
                self.swarm.swarm_best_x = self.best_x
                self.swarm.swarm_best_y = self.best_y

    def __str__(self):

        return f"x_cur = {self.cur_x:<12.7f}; y_cur = {self.cur_y:<12.7f}"


class Swarm:
    def __init__(self, size, fun: Callable[[float], float], min_x, max_x):

        self.f = fun
        self.swarm_best_x, self.swarm_best_y = min_x, fun(min_x)
        self.swarm = [
            Particle(random.randrange(min_x, max_x - 1) + random.random(), self.f, self)
            for _ in range(size)
        ]

    def print_best_particle(self):

        best = self.swarm[0]

        for particle in self.swarm:

            if abs(particle.best_y) < abs(best.best_y):

                best = particle

        print("Best Solution:")
        print(best)

    def print_total(self):

        s = ""

        for particle in self.swarm:
            s += str(particle) + "\n"

        s += f"---\nSwarm Memory: x_best = {str(self.swarm_best_x)}; y_best = {str(self.swarm_best_y)}\n"

        return s

    def print_best(self):

        return f"---\nSwarm Memory: x_best = {str(self.swarm_best_x)}; y_best = {str(self.swarm_best_y)}\n"

    def run(self, max_iterations):

        for _ in range(max_iterations):

            self.print_total()

            for particle in self.swarm:

                particle.run()

        else:

            self.print_best_particle()


if __name__ == "__main__":

    mi, ma = -1000000, 1000000
    plot_f(f, mi, ma)

    s = Swarm(100, f, mi, ma)
    s.run(1000)
