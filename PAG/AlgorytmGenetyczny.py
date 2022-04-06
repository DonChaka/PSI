from random import random, uniform
from typing import Callable
import numpy as np

population_size = 100
crossover_probability = 0.8
mutation_probability = 0.2
epochs = 100


def paraboloid(x, y):
    return (2 * x ** 2) / 3 + (7 * y ** 2) / 2 + 1


def d2b(f: int, b: int) -> str:
    n = int(f)
    base = int(b)
    ret = ""
    for y in range(base - 1, -1, -1):
        ret += str((n >> y) & 1)
    return ret


def inv_chr(string: str, position: int) -> str:
    if int(string[position]) == 1:
        string = string[:position] + '0' + string[position + 1:]
    else:
        string = string[:position] + '1' + string[position + 1:]
    return string


class Unit:
    def __init__(self, genotype: int = None, gen_bit_size: int = 16,
                 optimized_function: Callable[[int, int], float] = paraboloid, feature_size: int = 8):
        self.genotype = genotype
        self.gen_bitsize = gen_bit_size
        self.feature_size = feature_size
        self.optimized_function = optimized_function
        self.max_genotype = (1 << gen_bit_size) - 1
        if genotype is None:
            self.genotype = int(uniform(0, self.max_genotype))

    @property
    def fitness(self):
        return paraboloid((self.genotype & (((1 << self.feature_size) - 1) << 8)) >> self.feature_size,
                          self.genotype & ((1 << self.feature_size) - 1))

    def __str__(self):
        return f'(x: {(self.genotype & (((1 << self.feature_size) - 1) << 8)) >> self.feature_size}, y: {self.genotype & ((1 << self.feature_size) - 1)})'

    def __repr__(self):
        return f'(x: {(self.genotype & (((1 << self.feature_size) - 1) << 8)) >> self.feature_size}, y: {self.genotype & ((1 << self.feature_size) - 1)})'

    def cross(self, other, pivot: int = None):
        if self.gen_bitsize != self.gen_bitsize:
            raise ValueError("Different genotype sizes")
        if pivot is None:
            pivot = int(uniform(1, self.gen_bitsize - 1))
        self_str_gen = d2b(self.genotype, self.gen_bitsize)
        other_str_gen = d2b(other.genotype, self.gen_bitsize)
        return Unit(int(self_str_gen[:pivot] + other_str_gen[pivot:], 2), self.gen_bitsize), Unit(
            int(other_str_gen[:pivot] + self_str_gen[pivot:], 2), other.gen_bitsize)

    def mutate(self, pivot: int = None):
        if self.gen_bitsize != self.gen_bitsize:
            raise ValueError("Different genotype sizes")
        if pivot is None:
            pivot = int(uniform(0, self.gen_bitsize))
        self_str_gen = d2b(self.genotype, self.gen_bitsize)
        self.genotype = int(inv_chr(self_str_gen, pivot), 2)
        return self


def main():
    population = [Unit() for _ in range(population_size)]

    for epoch in range(epochs):
        fitness = [unit.fitness for unit in population]
        fitness_sum = sum(fitness)
        temp = [fitness_sum / f for f in fitness]
        temp_sum = sum(temp)
        fitness_prob = [f / temp_sum for f in temp]
        new_population = []
        if epoch % (epochs // 10) == 0:
            print(f'Epoch: {epoch}')
            print(f'Avg Fitness: {fitness_sum / population_size}')
            print(f'Best Fitness: {min(fitness)}')
            print(population)
        for _ in range(population_size // 2):
            choices = np.random.choice(population, 2, p=fitness_prob, replace=False)
            parent1, parent2 = choices[0], choices[1]

            if random() < crossover_probability:
                child1, child2 = parent1.cross(parent2)
            else:
                child1, child2 = parent1, parent2

            if random() < mutation_probability:
                child1.mutate()
            if random() < mutation_probability:
                child2.mutate()
            new_population.append(child1)
            new_population.append(child2)

        population = new_population
        population.sort(key=lambda x: x.fitness, reverse=False)
        population = population[:population_size]
    print(population)
    print(population[0].fitness)
    print(population[0])


if __name__ == '__main__':
    main()
