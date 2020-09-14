from random import *
from numpy.ma import arange
import matplotlib.pyplot as plt


def x_function(x):
    return x*x + 2*x - 3


def den2bin(f):
    bStr = ''
    n = int(f)
    if n < 0: raise
    if n == 0: return '0'
    while n > 0:
        bStr = str(n % 2) + bStr
        n = n >> 1
    return bStr


def d2b(f, b):
    n = int(f)
    base = int(b)
    ret = ""
    for y in range(base - 1, -1, -1):
        ret += str((n >> y) & 1)
    return ret


def invchr(string, position):
    if int(string[position]) == 1:
        string = string[:position] + '0' + string[position + 1:]
    else:
        string = string[:position] + '1' + string[position + 1:]
    return string


def roulette(values, fitness):
    n_rand = random() * fitness
    sum_fit = 0
    for i in range(len(values)):
        sum_fit += values[i]
        if sum_fit >= n_rand:
            break
    return i


if __name__ == "__main__":
    x_max = 127
    x_min = 0

    batch_size = 10
    mutation_probability = 0.3
    hybridization_probability = 0.6
    evolutions = 10

    gen_1_xvalues = []
    gen_1_fvalues = []
    generations_x = []
    generations_f = []
    fitness = 0

    x_size = int(len(den2bin(x_max)))

    # print("Maximum chromosome size of x is", x_size, "bits, i.e.,", pow(2, x_size), "variables.")

    for i in range(batch_size):
        x_tmp = int(uniform(x_min, x_max))
        gen_1_xvalues.append(x_tmp)

        f_tmp = x_function(x_tmp)
        gen_1_fvalues.append(f_tmp)

        fitness += f_tmp

    max_f_gen1 = gen_1_fvalues[0]
    max_x_gen1 = gen_1_xvalues[0]
    for i in range(batch_size):
        if gen_1_fvalues[i] >= max_f_gen1:
            max_f_gen1 = gen_1_fvalues[i]
            max_x_gen1 = gen_1_xvalues[i]

    for i in range(evolutions):
        nextgen_xvalues = []
        nextgen_fvalues = []
        selected = []

        for j in range(batch_size):
            ind_sel = roulette(gen_1_fvalues, fitness)
            selected.append(gen_1_xvalues[ind_sel])

        for j in range(0, batch_size, 2):
            sel_ind_A = d2b(selected[j], x_size)
            sel_ind_B = d2b(selected[j+1], x_size)

            ran_hyb = random()
            if ran_hyb > hybridization_probability:
                ind_AB = sel_ind_A
                ind_BA = sel_ind_B
            else:
                cut_point = int(uniform(1, x_size))

                ind_AB = sel_ind_A[:cut_point] + sel_ind_B[cut_point:]

                ran_mut = random()
                if ran_mut < mutation_probability:
                    gene_position = int(uniform(0, x_size))
                    int_mut = invchr(ind_AB, gene_position)
                    ind_AB = int_mut

                ind_BA = sel_ind_B[:cut_point] + sel_ind_A[cut_point:]

                ran_mut = random()
                if ran_mut < mutation_probability:
                    gene_position = int(uniform(0, x_size))
                    int_mut = invchr(ind_BA, gene_position)
                    ind_BA = int_mut

            new_AB = int(ind_AB, 2)
            nextgen_xvalues.append(new_AB)
            new_f_AB = x_function(new_AB)
            nextgen_fvalues.append(new_f_AB)

            new_BA = int(ind_BA, 2)
            nextgen_xvalues.append(new_BA)
            new_f_BA = x_function(new_BA)
            nextgen_fvalues.append(new_f_BA)

        # print('gen', i + 2, nextgen_xvalues)

        max_f_nextgen = nextgen_fvalues[0]
        max_x_nextgen = nextgen_xvalues[0]
        for j in range(batch_size):
            if nextgen_fvalues[j] >= max_f_nextgen:
                max_f_nextgen = nextgen_fvalues[j]
                max_x_nextgen = nextgen_xvalues[j]

        if max_f_gen1 > max_f_nextgen:
            max_f_nextgen = max_f_gen1
            max_x_nextgen = max_x_gen1
            nextgen_fvalues[0] = max_f_gen1
            nextgen_xvalues[0] = max_x_gen1

        gen_1_xvalues = nextgen_xvalues
        gen_1_fvalues = nextgen_fvalues
        max_x_gen1 = max_x_nextgen
        max_f_gen1 = max_f_nextgen
        generations_x.append(max_x_nextgen)
        generations_f.append(max_f_nextgen)

        fitness = 0
        for j in range(batch_size):
            f_tmp = x_function(gen_1_xvalues[j])
            fitness += f_tmp

    print("Max value found by GA is F(x) = " + str(max_f_nextgen) + " with x = " + str(max_x_nextgen))

    x = arange(x_min, x_max, 0.1)
    y = x_function(x)

    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('F(x)')

    plt.figure(2)
    plt.plot(nextgen_xvalues, nextgen_fvalues, 'bo')
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.title(r'Data for last generation')

    plt.figure(3)
    plt.plot(range(evolutions), generations_f, 'ro')
    plt.xlabel('Generations')
    plt.ylabel('F(x) Max')

    plt.show()
