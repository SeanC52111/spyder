# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:43:54 2017

@author: Sean Chang
"""

import random

"""
calculate pi

"""

def pi(count):
    total = 0
    in_circle = 0
    for i in range(count):
        total += 1
        x = random.uniform(-4,4)
        y = random.uniform(-4,4)
        distance = x**2 + y**2
        if(distance <= 16):
            in_circle += 1
    return in_circle/total*4

print(pi(1000000))

