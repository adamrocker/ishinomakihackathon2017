# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import Tkinter as tk
import time

c0 = tk.Canvas(width=400, height=400)
c0.pack()
c0.create_oval(195, 195, 205, 205, fill='#ff0000', tags='o')
for i in range(0, 202, 2):
    time.sleep(0.1)
    c0.move('o', 2, 0)
    c0.update()

tk.mainloop()
