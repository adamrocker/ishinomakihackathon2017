# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import math
import sys
sys.path.append('/lib/python2.7/site-packages')

import random
import numpy as np
import tensorflow as tf


def print_debug(msg):
    if False:
        print(msg)

WIDTH = 200
HEIGHT = 200

SELECTION_SIZE = 2    # 4
POPULATION_SIZE = 10  # 10
STEP = 200            # 200
GENERATION = 1000     # 1000
NUM_FOOD = 50         # 50
NUM_POISON = 0        # 0
MUTATION_BIAS = 0.1
SENSOR_LENGTH_WALL = 40
SENSOR_LENGTH_FOOD = 40
SENSOR_LENGTH_POISON = 40
AGENT_RADIUS = 5
FOOD_RADIUS = 2
POISON_RADIUS = 2


class Genome(object):

    NUM_FEATURE = 6  # [壁距離, 壁角度, 餌距離, 餌角度, 敵距離, 敵距離]  # それぞれ1つしか認識できない
    NUM_HIDDEN_NODE = [4]  # 各レイヤーのノード数
    NUM_HIDDEN = len(NUM_HIDDEN_NODE)  # レイヤー数
    NUM_OUTPUT = 2
    ARRAY = [NUM_FEATURE] + NUM_HIDDEN_NODE + [NUM_OUTPUT]
    LAYER = len(ARRAY) - 1
    GENE_LENGTH = -1
    MUTATION_RATE = 0.01

    @classmethod
    def gene_length(cls):
        if 0 <= cls.GENE_LENGTH:
            return cls.GENE_LENGTH

        length = 0
        for index, i in enumerate(cls.ARRAY[0: -1]):
            length += i * cls.ARRAY[index + 1]
        cls.GENE_LENGTH = length
        return cls.GENE_LENGTH

    def __init__(self, mutation_rate=None):
        length = self.gene_length()
        self._gene = np.random.rand(length).astype(np.float32) - np.random.rand(length).astype(np.float32)
        if mutation_rate:
            self._mutation_rate = mutation_rate
        else:
            self._mutation_rate = 1.0 / length
        self._fitness = 0.0

    def _gene_layer_offset(self, layer):
        # layerのスタート地点までのオフセット
        length = 0
        for index, i in enumerate(Genome.ARRAY[0: layer]):
            length += i * Genome.ARRAY[index + 1]
        return length

    def gene_layer(self, layer):
        start = self._gene_layer_offset(layer)
        end = self._gene_layer_offset(layer + 1)
        return self._gene[start:end]

    def mutate(self):
        length = len(self._gene)
        mutate = np.zeros(length).astype(np.float32)
        rand = np.random.rand(length)
        for i in range(length):
            if rand[i] <= self._mutation_rate:
                val = np.random.rand() - np.random.rand()
                print_debug("mutate[%d] = %f" % (i, val))
                mutate[i] = val * MUTATION_BIAS
            else:
                mutate[i] = 0.0

        self._gene += mutate

    def copy_gene(self):
        # deep copy
        return self._gene.copy()

    def set_gene(self, gene):
        self._gene = gene

    def set_fitness(self, fitness):
        self._fitness = fitness

    def get_fitness(self):
        return self._fitness


class NN(object):

    def __init__(self, population):
        self._x, self._model = self._build_nn(population)

    @classmethod
    def _flatten(cls, population):
        # make flatten all genome info for GPU calculation
        size = len(population)
        gene_length = Genome.gene_length()
        index = 0
        flat = np.zeros(size * gene_length)
        for layer in range(Genome.LAYER):
            for genome in population:
                arr = genome.gene_layer(layer)
                end = index + len(arr)
                flat[index:end] = arr
                index = end

        return flat

    @classmethod
    def _build_nn(cls, population):
        size = len(population)
        x = tf.placeholder(tf.float32, [size, None, Genome.NUM_FEATURE], name="input")
        genes = cls._flatten(population)
        print_debug("----genes-----")
        print_debug(genes)

        start = 0
        length = size * Genome.NUM_FEATURE * Genome.NUM_HIDDEN_NODE[0]
        c1 = tf.constant(genes[start:length],
                         dtype=tf.float32,
                         shape=[size, Genome.NUM_FEATURE, Genome.NUM_HIDDEN_NODE[0]],
                         name="layer1")
        w1 = tf.Variable(c1)
        x1 = 2 * tf.nn.sigmoid(tf.matmul(x, w1)) - 1
        print_debug("----layer1[{}:{}]----".format(start, length))
        print_debug(genes[start:length])

        start = length
        length = start + size * Genome.NUM_HIDDEN_NODE[0] * Genome.NUM_OUTPUT
        c2 = tf.constant(genes[start:length],
                         dtype=tf.float32,
                         shape=[size, Genome.NUM_HIDDEN_NODE[0], Genome.NUM_OUTPUT],
                         name="layer2")
        w2 = tf.Variable(c2)
        x2 = tf.nn.sigmoid(tf.matmul(x1, w2))
        print_debug("----layer2[{}:{}]----".format(start, length))
        print_debug(genes[start:length])
        return x, x2

    def eval(self, input):
        fetch = self._model.eval(feed_dict={self._x: input})
        return fetch[:, 0]


class GenePool(object):

    def __init__(self, size):
        self._generation = 0  # 世代数
        self._selection_size = SELECTION_SIZE
        self._size = size
        self._population = []  # List[Genome]
        for i in range(size):
            self._population.append(Genome())

        self._nn = None
        self._shuffle_arr = [i for i in range(self._size)]

    def save_population(self):
        import datetime
        ts = datetime.datetime.now().strftime("%H%M%S")
        filename = './data/population_{}_{}.npy'.format(ts, self._generation)
        population = [[genome._gene, genome.get_fitness()] for genome in self._population]
        np.save(filename, population)

    def load_population(self, filename):
        arr = np.load(filename)
        for index, item in enumerate(arr):
            gene, fitness = item
            genome = self._population[index]  # type Genome
            genome.set_gene(gene)
            genome.set_fitness(fitness)

    def get_genome(self, index):
        # type: (int) -> Genome
        return self._population[index]

    def print_all_genome(self):
        for genome in self._population:
            print(genome._gene)

    def init_world(self):
        self._nn = NN(self._population)

    def play(self, input):
        return self._nn.eval(input=input)

    def set_fitness(self, index, fitness):
        self._population[index].set_fitness(fitness)

    def get_fitness(self, index):
        return self._population[index].get_fitness()

    def get_elite_index(self):
        elite_index = 0
        elite_fitness = 0
        for index, genome in enumerate(self._population):
            fit = genome.get_fitness()
            if elite_fitness < fit:
                elite_fitness = fit
                elite_index = index

        return elite_index

    def mutation(self, elite_index=-1):
        for index, genome in enumerate(self._population):
            if index != elite_index:
                # print("Mutate[{}]: fitness={}".format(index, genome.get_fitness()))
                genome.mutate()

    def selection(self):
        self._generation += 1
        self._tournament_selection()

    def _tournament_selection(self):
        random.shuffle(self._shuffle_arr)
        index_arr = self._shuffle_arr[0: self._selection_size]
        # print("tournament_index: {}".format(index_arr))
        winner = self._population[index_arr[0]]  # type: Genome
        losers = []  # List[Genome]
        for i in range(1, self._selection_size):
            winner_fitness = winner.get_fitness()
            challenger_index = index_arr[i]
            # print("challenger_index: {}".format(challenger_index))
            challenger = self._population[challenger_index]  # type: Genome
            challenger_fitness = challenger.get_fitness()
            if winner_fitness < challenger_fitness:
                loser = winner
                winner = challenger
                losers.append(loser)
            else:
                losers.append(challenger)

        # print("winner:loser = {}:{}".format(winner.get_fitness(), [loser.get_fitness() for loser in losers]))

        for loser in losers:  # type: Genome
            winners_gene = winner.copy_gene()
            loser.set_gene(winners_gene)
            loser.set_fitness(winner.get_fitness())

        return


class World(object):
    POSITION_X = 0
    POSITION_Y = 1

    def __init__(self, id=0):
        self._id = id
        self._width = WIDTH
        self._height = HEIGHT
        self._agent_radius = AGENT_RADIUS  # エージェントの半径
        self._agent_speed = 5
        self._agent_step_theta = math.pi / 18  # (rad) 1stepでの最大回転角度(10度)
        self._agent_sensor_strength_wall = SENSOR_LENGTH_WALL
        self._agent_sensor_strength_food = SENSOR_LENGTH_FOOD
        self._agent_sensor_strength_poison = SENSOR_LENGTH_POISON
        self._food_point = 10
        self._poison_point = -10

        self._food_radius = FOOD_RADIUS
        self._poison_radius = POISON_RADIUS

    def init(self, foods, poisons):
        self._agent_position = [200, 200]  # スタート位置
        self._agent_direction = 0  # 向いている方向(rad)
        self._agent_fitness = 0
        self._foods = []  # [[1, 1], [1, 2]]  # foodの位置
        for food in foods:
            self._foods.append(list(food))
        self._poisons = []  # [[10, 11], [12, 13]]  # 毒位置
        for poison in poisons:
            self._poisons.append(list(poison))

    @staticmethod
    def meals(num):
        arr = np.random.rand(num * 2) * WIDTH
        meals = np.reshape(arr, (num, 2))
        return meals.astype(np.int32)

    def get_fitness(self):
        return self._agent_fitness

    def _move_length(self, left, right):
        diff = right - left  # 右方向が正
        drive_strength = 1.0 - math.fabs(diff)
        move_length = drive_strength * self._agent_speed
        return move_length

    def _rotate(self, left, right):
        diff = left - right  # 右方向が正
        rotate_theta = diff * self._agent_step_theta
        return rotate_theta

    def move(self, output):
        # 引数はNNの出力(output)
        left, right = output
        rotate = self._rotate(left, right)
        move_length = self._move_length(left, right)
        self._agent_direction += rotate
        diff_x = round(move_length * math.cos(self._agent_direction), 2)  # 小数点2桁まで
        diff_y = round(move_length * math.sin(self._agent_direction), 2)  # 小数点2桁まで
        current_x, current_y = self._agent_position
        next_x = current_x + diff_x
        next_y = current_y + diff_y
        next_x = min(self._width, max(0, next_x))
        next_y = min(self._height, max(0, next_y))
        self._agent_position[self.POSITION_X] = next_x
        self._agent_position[self.POSITION_Y] = next_y
        return next_x - current_x, next_y - current_y

    def _sensor_diff(self, p1, p2):
        p1x = p1[self.POSITION_X]
        p1y = p1[self.POSITION_Y]
        p2x = p2[self.POSITION_X]
        p2y = p2[self.POSITION_Y]
        distance = math.sqrt((p2x - p1x)**2 + (p2y - p1y)**2)
        radian = math.atan2(p2y - p1y, p2x - p1x)
        return distance, radian

    def _get_min_sensor_diff(self, target_arr, sensor_length):
        distance, radian = min(target_arr, key=lambda x: x[0])
        sensor_strength = 0.0
        sensor_theta = 0.0
        index = -1
        if distance < sensor_length:
            sensor_strength = (sensor_length - distance) / sensor_length
            sensor_theta = radian

            for i, item in enumerate(target_arr):
                d, r = item
                if d == distance and r == radian:
                    index = i
                    break

        return sensor_strength, sensor_theta, index

    def eat(self):
        # エージェントにぶつかったら食べる
        pos = self._agent_position

        # 餌との接触
        if 0 < len(self._foods):
            eat_area_food = self._agent_radius + self._food_radius
            food_diff_arr = [self._sensor_diff(pos, food) for food in self._foods]
            fd, fr, findex = self._get_min_sensor_diff(food_diff_arr, eat_area_food)
            if 0 <= findex:
                # print("agent[{}].eat: food[{}]".format(self._id, findex))
                self._foods.pop(findex)
                self._agent_fitness += self._food_point

        # 毒との接触
        if 0 < len(self._poisons):
            eat_area_poison = self._agent_radius + self._food_radius
            poison_diff_arr = [self._sensor_diff(pos, poison) for poison in self._poisons]
            pd, pr, pindex = self._get_min_sensor_diff(poison_diff_arr, eat_area_poison)
            if 0 <= pindex:
                # print("agent[{}].eat: poison[{}]".format(self._id, pindex))
                self._poisons.pop(pindex)
                self._agent_fitness += self._poison_point

    def sensing(self):
        pos = self._agent_position
        x = pos[self.POSITION_X]
        y = pos[self.POSITION_Y]

        # 壁との距離と角度
        wall_diff_arr = [self._sensor_diff(pos, wall) for wall in [[x, 0], [0, y], [x, self._height], [self._width, y]]]
        wall_sensor_strength, wall_sensor_theta, _ = self._get_min_sensor_diff(wall_diff_arr,
                                                                               self._agent_sensor_strength_wall)

        # 餌との距離と角度
        food_sensor_strength = 0.0
        food_sensor_theta = 0.0
        if 0 < len(self._foods):
            food_diff_arr = [self._sensor_diff(pos, food) for food in self._foods]
            food_sensor_strength, food_sensor_theta, _ = self._get_min_sensor_diff(food_diff_arr,
                                                                                   self._agent_sensor_strength_food)

        # 毒との距離と角度
        poison_sensor_strength = 0.0
        poison_sensor_theta = 1.0  # bias
        if 0 < len(self._poisons):
            poison_diff_arr = [self._sensor_diff(pos, poison) for poison in self._poisons]
            poison_sensor_strength, poison_sensor_theta, _ = self._get_min_sensor_diff(poison_diff_arr,
                                                                                       self._agent_sensor_strength_poison)

        return np.array([wall_sensor_strength, wall_sensor_theta,
                         food_sensor_strength, food_sensor_theta,
                         poison_sensor_strength, poison_sensor_theta]).astype(np.float32)


def run(gp, generation, size, step):
    num_food = NUM_FOOD
    num_poison = NUM_POISON
    foods = World.meals(num_food)
    poisons = World.meals(num_poison)
    worlds = [World(i) for i in range(size)]

    for i in range(generation):
        print("# Generation: %d" % i)
        if i % 10 == 0:
            gp.save_population()
        # gp.print_all_genome()

        # init world
        gp.init_world()
        for world in worlds:
            world.init(foods.copy(), poisons.copy())

        sess = tf.Session()
        with sess.as_default():
            tf.global_variables_initializer().run()
            input = np.zeros(Genome.NUM_FEATURE * size)

            for _ in range(step):
                move_arr = []
                # set input array
                start = 0
                for world in worlds:
                    inp = world.sensing()
                    end = start + len(inp)
                    input[start:end] = inp
                    start = end

                input_placeholder = np.reshape(input, (size, 1, Genome.NUM_FEATURE))
                command = gp.play(input_placeholder)
                for index, world in enumerate(worlds):
                    cmd = command[index]
                    diff_x, diff_y = world.move(cmd)
                    move_arr.append([diff_x, diff_y])
                    world.eat()

            # set fitness
            for index, world in enumerate(worlds):
                fit = world.get_fitness()
                print("Genome[{}]: fitness={}".format(index, fit))
                gp.set_fitness(index, fit)

            gp.selection()

            elite_index = gp.get_elite_index()  # elete strategy
            gp.mutation(elite_index=elite_index)


def play(gp, size, step, file, index):
    foods = World.meals(NUM_FOOD)
    poisons = World.meals(NUM_POISON)
    worlds = [World(i) for i in range(size)]

    gp.init_world()
    for world in worlds:
        world.init(foods.copy(), poisons.copy())

    gp.load_population(file)

    import Tkinter as tk
    c0 = tk.Canvas(width=WIDTH, height=HEIGHT)
    c0.pack()
    # create agent
    agent_tag = 'agent'
    x1 = WIDTH / 2 - AGENT_RADIUS / 2
    y1 = HEIGHT / 2 - AGENT_RADIUS / 2
    x2 = WIDTH / 2 + AGENT_RADIUS / 2
    y2 = HEIGHT / 2 + AGENT_RADIUS / 2
    c0.create_oval(x1, y1, x2, y2, fill='#ff0000', tags=agent_tag)
    # create food
    for index, food in enumerate(foods):
        x, y = food
        x1 = x - FOOD_RADIUS / 2
        y1 = y - FOOD_RADIUS / 2
        x2 = x + FOOD_RADIUS / 2
        y2 = y + FOOD_RADIUS / 2
        tag = "food%d" % index
        c0.create_oval(x1, y1, x2, y2, fill='#000000', tags=tag)

    sess = tf.Session()
    with sess.as_default():
        tf.global_variables_initializer().run()
        for _ in range(step):
            # set input array
            start = 0
            for world in worlds:
                inp = world.sensing()
                end = start + len(inp)
                input[start:end] = inp
                start = end

            input_placeholder = np.reshape(input, (size, 1, Genome.NUM_FEATURE))
            command = gp.play(input_placeholder)
            cmd = command[index]
            x, y = world[index].move(cmd)
            world[index].eat()

            time.sleep(0.1)
            c0.move(agent_tag, x, y)
            c0.update()

    tk.mainloop()


def show(filename):
    arr = np.load(filename)
    for item in arr:
        print(list(item))


tf.app.flags.DEFINE_string("command", "train", "train, play, show")
tf.app.flags.DEFINE_string("file", "./data/population.npy", "Population file")
tf.app.flags.DEFINE_integer("index", 0, "Agent index")


def main(args):
    flags = tf.app.flags.FLAGS

    time_start = time.time()
    np.random.seed(0)
    generation = GENERATION
    step = STEP  # 各個体が何ステップ動くか
    size = POPULATION_SIZE  # Population size
    gp = GenePool(size)

    if flags.command == 'train':
        run(gp, generation, size, step)
    elif flags.command == 'play':
        play(gp, size, step, flags.file, flags.index)
    elif flags.command == 'show':
        show(flags.file)

    time_end = time.time()
    print("time: {}s".format(time_end - time_start))


if __name__ == '__main__':
    tf.app.run()



