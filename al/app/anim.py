# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import Tkinter as tk
import time
import sys
sys.path.append('/lib/python2.7/site-packages')
sys.path.append('.')

from al import World

WIDTH = 200
HEIGHT = 200
NUM_FOOD = 50
FOOD_RADIAL = 2
AGENT_RADIAL = 5

foods = World.meals(NUM_FOOD)

c0 = tk.Canvas(width=WIDTH, height=HEIGHT)
c0.pack()

# create agent
x1 = WIDTH / 2 - AGENT_RADIAL / 2
y1 = HEIGHT / 2 - AGENT_RADIAL / 2
x2 = WIDTH / 2 + AGENT_RADIAL / 2
y2 = HEIGHT / 2 + AGENT_RADIAL / 2
c0.create_oval(x1, y1, x2, y2, fill='#ff0000', tags='o')

# create food
for index, food in enumerate(foods):
    x, y = food
    x1 = x - FOOD_RADIAL / 2
    y1 = y - FOOD_RADIAL / 2
    x2 = x + FOOD_RADIAL / 2
    y2 = y + FOOD_RADIAL / 2
    tag = "food%d" % index
    c0.create_oval(x1, y1, x2, y2, fill='#000000', tags=tag)

for i in range(0, 202, 2):
    time.sleep(0.1)
    c0.move('o', 2, 0)
    c0.update()

tk.mainloop()
