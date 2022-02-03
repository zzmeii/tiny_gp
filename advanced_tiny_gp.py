# tiny genetic programming plus, by © moshe sipper, www.moshesipper.com
# graphic output, dynamic progress display, bloat-control option 
# need to install https://pypi.org/project/graphviz/

from random import random, randint, seed
from statistics import mean
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython.display import Image, display
from graphviz import Digraph, Source


def add(x, y): return x + y


def sub(x, y): return x - y


def mul(x, y): return x * y


FUNCTIONS = [add, sub, mul]
TERMINALS = [-2, -1, 1, 2, 0.1]


def target_func(x):  # evolution's target
    return x * x * x * x + x * x * x + x * x + x + 1


def target_2d(x, y):
    return x + y * y + x * 2 * x * x


def generate_dataset():  # generate 101 data points from target_func
    dataset = []
    for x in range(-100, 101, 2):
        for y in range(-100, 101, 2):
            x /= 100
            y /= 100
            dataset.append([x, y, target_2d(x, y)])
    return dataset


class GPTree:
    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def __str__(self):
        names = {'sub': '-', 'add': '+', 'mul': '*'}
        name_keys = list(names.keys())
        if str(type(self.data)) != "<class 'function'>":
            return str(self.data)
        left = str(self.left)
        right = str(self.right)
        if (left == 0 or right == 0) and self.data.__name__ == 'mult':
            return 0
        return f'({left} {names[self.data.__name__]} {right})'

    def node_label(self):  # return string label
        if self.data in FUNCTIONS:
            return self.data.__name__
        else:
            return str(self.data)

    def draw(self, dot, count):  # dot & count are lists in order to pass "by reference"
        node_name = str(count[0])
        dot[0].node(node_name, self.node_label())
        if self.left:
            count[0] += 1
            dot[0].edge(node_name, str(count[0]))
            self.left.draw(dot, count)
        if self.right:
            count[0] += 1
            dot[0].edge(node_name, str(count[0]))
            self.right.draw(dot, count)

    def draw_tree(self, fname, footer):
        dot = [Digraph()]
        dot[0].attr(kw='graph', label=footer)
        count = [0]
        self.draw(dot, count)
        Source(dot[0], filename=fname + ".gv", format="png").render()
        display(Image(filename=fname + ".gv.png"))

    def compute_tree(self, x):
        if self.data in FUNCTIONS:
            return self.data(self.left.compute_tree(x), self.right.compute_tree(x))
        elif type(self.data) is str:
            return x[int(self.data[1:])]
        else:
            return self.data

    def random_tree(self, grow, min_depth, max_depth, depth=0):  # create random tree using either grow or full method
        if depth < min_depth or (depth < max_depth and not grow):
            self.data = FUNCTIONS[randint(0, len(FUNCTIONS) - 1)]
        elif depth >= max_depth:
            self.data = TERMINALS[randint(0, len(TERMINALS) - 1)]
        else:  # intermediate depth, grow
            if random() > 0.5:
                self.data = TERMINALS[randint(0, len(TERMINALS) - 1)]
            else:
                self.data = FUNCTIONS[randint(0, len(FUNCTIONS) - 1)]
        if self.data in FUNCTIONS:
            self.left = GPTree()
            self.left.random_tree(grow, min_depth, max_depth, depth=depth + 1)
            self.right = GPTree()
            self.right.random_tree(grow, min_depth, max_depth, depth=depth + 1)

    def mutation(self, prob_mutation, min_depth):
        if random() < prob_mutation:  # mutate at this node
            self.random_tree(grow=True, min_depth=min_depth, max_depth=2)
        elif self.left:
            self.left.mutation(prob_mutation, min_depth)
        elif self.right:
            self.right.mutation(prob_mutation, min_depth)

    #    def depth(self):
    #        if self.data in TERMINALS: return 0
    #        l = self.left.depth()  if self.left  else 0
    #        r = self.right.depth() if self.right else 0
    #        return 1 + max(l, r)

    def size(self):  # tree size in nodes
        if self.data in TERMINALS: return 1
        l = self.left.size() if self.left else 0
        r = self.right.size() if self.right else 0
        return 1 + l + r

    def build_subtree(self):  # count is list in order to pass "by reference"
        t = GPTree()
        t.data = self.data
        if self.left:  t.left = self.left.build_subtree()
        if self.right: t.right = self.right.build_subtree()
        return t

    def scan_tree(self, count, second):  # note: count is list, so it's passed "by reference"
        count[0] -= 1
        if count[0] <= 1:
            if not second:  # return subtree rooted here
                return self.build_subtree()
            else:  # glue subtree here
                self.data = second.data
                self.left = second.left
                self.right = second.right
        else:
            ret = None
            if self.left and count[0] > 1: ret = self.left.scan_tree(count, second)
            if self.right and count[0] > 1: ret = self.right.scan_tree(count, second)
            return ret

    def crossover(self, other, xo_rate):  # xo 2 trees at random nodes
        if random() < xo_rate:
            second = other.scan_tree([randint(1, other.size())], None)  # 2nd random subtree
            self.scan_tree([randint(1, self.size())], second)  # 2nd subtree "glued" inside 1st tree


# end class GPTree

def init_population(min_depth, max_depth, pop_size):  # ramped half-and-half
    pop = []
    for md in range(3, max_depth + 1):
        for i in range(int(pop_size / 6)):
            t = GPTree()
            t.random_tree(grow=True, min_depth=min_depth, max_depth=md)  # grow
            pop.append(t)
        for i in range(int(pop_size / 6)):
            t = GPTree()
            t.random_tree(grow=False, min_depth=min_depth, max_depth=md)  # full
            pop.append(t)
    return pop


def error(individual, dataset):
    return mean([abs(individual.compute_tree(ds[:-1]) - ds[-1]) for ds in dataset])


def fitness(individual, dataset, bloat_control):
    if bloat_control:
        return 1 / (1 + error(individual, dataset) + 0.01 * individual.size())
    else:
        return 1 / (1 + error(individual, dataset))


def selection(population, fitnesses, tournament_size):  # select one individual using tournament selection
    tournament = [randint(0, len(population) - 1) for i in range(tournament_size)]  # select tournament contenders
    tournament_fitnesses = [fitnesses[tournament[i]] for i in range(tournament_size)]
    return deepcopy(population[tournament[tournament_fitnesses.index(max(tournament_fitnesses))]])


def prepare_plots(generations):
    # noinspection PyTypeChecker
    fig, axarr = plt.subplots(2, sharex=True)
    fig.canvas.set_window_title('EVOLUTIONARY PROGRESS')
    fig.subplots_adjust(hspace=0.5)
    axarr[0].set_title('error', fontsize=14)
    axarr[1].set_title('mean size', fontsize=14)
    plt.xlabel('generation', fontsize=18)
    plt.ion()  # interactive mode for plot
    axarr[0].set_xlim(0, generations)
    axarr[0].set_ylim(0, 1)  # fitness range
    xdata = []
    ydata = [[], []]
    line = [None, None]
    line[0], = axarr[0].plot(xdata, ydata[0], 'b')  # 'b-' = blue line
    line[1], = axarr[1].plot(xdata, ydata[1], 'r')  # 'r-' = red line
    return axarr, line, xdata, ydata


def plot(axarr, line, xdata, ydata, gen, pop, errors, max_mean_size):
    xdata.append(gen)
    ydata[0].append(min(errors))
    line[0].set_xdata(xdata)
    line[0].set_ydata(ydata[0])
    sizes = [ind.size() for ind in pop]
    if mean(sizes) > max_mean_size[0]:
        max_mean_size[0] = mean(sizes)
        axarr[1].set_ylim(0, max_mean_size[0])
    ydata[1].append(mean(sizes))
    line[1].set_xdata(xdata)
    line[1].set_ydata(ydata[1])
    plt.draw()
    plt.pause(0.01)


def symbol_regression(dataset, pop_size=60, min_depth=2, max_depth=5,
                      generations=250, tournament_size=5, xo_rate=0.8,
                      prob_mutation=0.2, bloat_control=False, add_x=True, fitness=fitness):
    # init stuff
    if add_x:
        for i in range(len(dataset[0]) - 1):
            # noinspection PyTypeChecker
            TERMINALS.append('x' + str(i))
    population = init_population(min_depth, max_depth, pop_size)
    best_of_run = None
    best_of_run_error = 1e20
    best_of_run_gen = 0
    fitnesses = [fitness(ind, dataset, bloat_control) for ind in population]
    max_mean_size = [0]  # track maximal mean size for plotting
    axarr, line, xdata, ydata = prepare_plots(generations)

    # go evolution!
    for gen in range(generations):
        nextgen_population = []
        for i in range(pop_size):
            parent1 = selection(population, fitnesses, tournament_size)
            parent2 = selection(population, fitnesses, tournament_size)
            parent1.crossover(parent2, xo_rate)
            parent1.mutation(prob_mutation, min_depth)
            nextgen_population.append(parent1)
        population = nextgen_population
        fitnesses = [fitness(ind, dataset, bloat_control) for ind in population]
        errors = [error(ind, dataset) for ind in population]
        if min(errors) < best_of_run_error:
            best_of_run_error = min(errors)
            best_of_run_gen = gen
            best_of_run = deepcopy(population[errors.index(min(errors))])
            print("________________________")
            best_of_run.draw_tree("best_of_run", "gen: " + str(gen) + ", error: " + str(round(best_of_run_error, 3)))
        plot(axarr, line, xdata, ydata, gen, population, errors, max_mean_size)
        if best_of_run_error <= 1e-5: break

    endrun = "_________________________________________________\n" \
             "END OF RUN (bloat control was ON)" if bloat_control else "OFF)"
    print(endrun)
    s = "\n\nbest_of_run attained at gen " + str(best_of_run_gen) + " and has error=" + str(round(best_of_run_error, 3))
    best_of_run.draw_tree("best_of_run", s)
    return best_of_run


if __name__ == "__main__":
    seed()  # init internal state of random number generator
    dataset = generate_dataset()
    res = symbol_regression(dataset, bloat_control=True)
    print(str(res))
