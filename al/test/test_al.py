# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('/lib/python2.7/site-packages')
sys.path.append('../')

import unittest
import numpy as np
import math
from app.al import World


class TestWorld(unittest.TestCase):

    def setUp(self):
        super(TestWorld, self).setUp()
        self.world = World()
        self.world.init([np.array([1, 10]), np.array([100, 200])], [np.array([10, 11]), np.array([12, 13])])
        self.world._width = 400
        self.world._height = 400
        self.world._agent_position = [10, 100]
        self.world._agent_speed = 10
        self.world._agent_direction = math.pi / 2
        self.world._agent_step_theta = math.pi / 18
        self.world._foods = [[1, 10], [100, 200]]  # foodの位置
        self.world._agent_sensor_strength_wall = 20
        self.world._agent_sensor_strength_food = 20
        self.world._agent_sensor_strength_poison = 20

    def test_sensor_diff(self):
        data = [([0, 0], [0, 10], (10, math.pi / 2)),
                ([100, 100], [100, 50], (50, 3 * math.pi / 2))]
        for item in data:
            p1 = item[0]
            p2 = item[1]
            expected = item[2]
            d, r = self.world._sensor_diff(p1, p2)
            self.assertEqual(d, expected[0])
            self.assertEqual(r, expected[1])

    def test_move_length(self):
        length = self.world._move_length(1, 1)
        self.assertEqual(length, 10)

        length = self.world._move_length(0, 1)
        self.assertEqual(length, 0)

        length = self.world._move_length(1, 0)
        self.assertEqual(length, 0)

        length = self.world._move_length(1, 0.5)
        self.assertEqual(length, 0.5 * self.world._agent_speed)

    def test_rotate(self):
        rotate = self.world._rotate(1, 1)
        self.assertEqual(rotate, 0)

        rotate = self.world._rotate(0, 1)
        self.assertEqual(rotate, -math.pi / 18)

        rotate = self.world._rotate(1, 0)
        self.assertEqual(rotate, math.pi / 18)

    def test_move(self):
        self.world.move([1, 1])
        self.assertEqual(self.world._agent_direction, math.pi / 2)
        self.assertEqual(self.world._agent_position, [10, 110])

        direction = self.world._agent_direction
        for _ in range(18):  # rotate PI
            self.world.move([0, 1])

        self.assertEqual(round(self.world._agent_direction, 4), round(math.pi * 3 / 2, 4))
        self.assertEqual(self.world._agent_position, [10, 110])

        for _ in range(27):  # rotate PI
            self.world.move([1, 0])
        self.assertEqual(round(self.world._agent_direction, 4), round(math.pi, 4))
        self.assertEqual(self.world._agent_position, [10, 110])

        self.world.move([1, 1])
        self.assertEqual(self.world._agent_position, [0, 110])

    def test_wall_distance(self):
        width = self.world._width
        height = self.world._height
        x = 10
        y = 100
        pos = [x, y]  # agent position
        wall_diff_arr = [self.world._sensor_diff(pos, wall) for wall in [[x, 0], [0, y], [x, height], [width, y]]]
        self.world._agent_direction = math.pi
        strength, theta, index = self.world._get_min_sensor_diff(wall_diff_arr, 20)
        self.assertEqual(strength, 0.5)
        self.assertEqual(theta, 0)
        self.assertEqual(index, 1)

    def test_food_distance(self):
        x = 15
        y = 24
        pos = [x, y]  # agent position
        food_diff_arr = [self.world._sensor_diff(pos, food) for food in self.world._foods]
        self.world._agent_direction = math.pi
        strength, theta, index = self.world._get_min_sensor_diff(food_diff_arr, 20)
        self.assertEqual(round(strength, 3), 0.01)
        self.assertEqual(theta, -0.5)
        self.assertEqual(index, 0)

    def test_meals(self):
        np.random.seed(0)
        meals = World.meals(2, 400)
        self.assertEqual(meals[0][0], 219)
        self.assertEqual(meals[0][1], 286)
        self.assertEqual(meals[1][0], 241)
        self.assertEqual(meals[1][1], 217)


if __name__ == "__main__":
    unittest.main()
