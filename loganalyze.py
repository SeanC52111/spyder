# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 13:13:55 2017

@author: Sean Chang
"""

#!/usr/bin/python

# Format of each line is:
# date\ttime\tstore name\titem description\tcost\tmethod of payment
#
# We want elements 2 (store name) and 4 (cost)
# We need to write them out to standard output, separated by a tab

import sys

line = '10.223.157.186 - - [15/Jul/2009:15:50:35 -0700] "GET /assets/js/lowpro.js HTTP/1.1" 200 10469'
data = line.strip().split()
print(len(data))