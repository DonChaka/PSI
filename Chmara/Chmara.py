from typing import Callable, List
import matplotlib
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from drawnow import drawnow
from numpy.random import rand, uniform

# f(x,y) = (1.5 – x -xy)^2 + (2.25 – x + xy^2)^2 + (2.625 – x + xy^3)^2
# x,y in range <-4.5, 4.5>
N_UNITS = 100
X_MIN = -4.5
X_MAX = 4.5
Y_MIN = -4.5
Y_MAX = 4.5


def f(x, y):
    return -((1.5 - x - y * x) ** 2 + (2.25 - x + (x * y) ** 2) ** 2 + (2.625 - x + (x * y) ** 3) ** 2)


class Unit:
    best_value: float = -float("inf")
    best_x = None
    best_y = None
    exploration = True

    def __init__(self, start_x: float, start_y: float, function: Callable[[float, float], float], x_range: tuple,
                 y_range: tuple, weight: float = 1.0, c1: float = 1.0, c2: float = 1.0):
        self.x = start_x
        self.y = start_y
        self.function = function
        self.min_x = x_range[0]
        self.max_x = x_range[1]
        self.min_y = y_range[0]
        self.max_y = y_range[1]
        self.weight = weight
        self.c1 = c1
        self.c2 = c2
        self.velocity_x = uniform(-1, 1)
        self.velocity_y = uniform(-1, 1)
        self.best_value = self.function(self.x, self.y)
        self.best_x = self.x
        self.best_y = self.y

        if self.best_value > Unit.best_value:
            Unit.best_value = self.best_value
            Unit.best_x = self.best_x
            Unit.best_y = self.best_y

    def update_velocity(self):
        if not self.exploration:
            r1 = np.random.rand()
            r2 = np.random.rand()
            self.velocity_x = self.weight * self.velocity_x + self.c1 * r1 * (self.best_x - self.x) + self.c2 * r2 * (
                    Unit.best_x - self.x)
            self.velocity_y = self.weight * self.velocity_y + self.c1 * r1 * (self.best_y - self.y) + self.c2 * r2 * (
                    Unit.best_y - self.y)

    def update_position(self):
        self.x += self.velocity_x * 0.1
        self.y += self.velocity_y * 0.1

        if self.x < self.min_x:
            self.x = self.min_x
            self.velocity_x *= -0.5
        if self.x > self.max_x:
            self.x = self.max_x
            self.velocity_x *= -0.5
        if self.y < self.min_y:
            self.y = self.min_y
            self.velocity_y *= -0.5
        if self.y > self.max_y:
            self.y = self.max_y
            self.velocity_y *= -0.5

    def update_best(self):
        if self.function(self.x, self.y) > self.best_value:
            self.best_value = self.function(self.x, self.y)
            self.best_x = self.x
            self.best_y = self.y
            if self.best_value > Unit.best_value:
                Unit.best_value = self.best_value
                Unit.best_x = self.best_x
                Unit.best_y = self.best_y

    def update(self):
        self.update_velocity()
        self.update_position()
        self.update_best()

    def __str__(self):
        return f"x: {self.x}, y: {self.y}, value: {self.function(self.x, self.y)}"

    def __repr__(self):
        return f"x: {self.x}, y: {self.y}, value: {self.function(self.x, self.y)}"

    def plot(self, ax):
        ax.plot(self.x, self.y, "bo")
        ax.arrow(self.x, self.y, self.velocity_x, self.velocity_y, head_width=0.1, head_length=0.1,
                 fc='k', ec='k')


def update(units: List[Unit], ax: Axes):
    for unit in units:
        unit.plot(ax)
        unit.update()


def main():
    m = rand(1)[0]
    c1 = rand(1)[0]
    c2 = rand(1)[0]
    units = [Unit(uniform(X_MIN, X_MAX), uniform(Y_MIN, Y_MAX), f, (X_MIN, X_MAX), (Y_MIN, Y_MAX), m, c1, c2) for _ in range(N_UNITS)]
    fig, ax = plt.subplots()

    # xx, yy = np.meshgrid(np.arange(X_MIN, X_MAX, 0.01), np.arange(Y_MIN, Y_MAX, 0.01))
    # mesh_data = np.c_[xx.ravel(), yy.ravel()]
    # Z = np.array([f(x, y) for x, y in mesh_data]).reshape(xx.shape)

    # for _ in range(100):
    i = 0
    while True:
        i += 1
        if i > 50:
            Unit.exploration = False
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_title(f"Iteration: {i}, Best value: {Unit.best_value:.2f}, at x: {Unit.best_x:.2f}, y: {Unit.best_y:.2f}")
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.grid()
        # ax.contour(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 10))
        update(units, ax)
        plt.draw()
        plt.pause(0.001)
        plt.cla()


if __name__ == "__main__":
    main()
