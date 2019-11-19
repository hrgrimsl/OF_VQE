import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import os

with open('Results/H4/adapt.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    y = []
    for row in plots:
        y.append(math.fabs(float(row[3])))
        iters_y = list(range(len(y)))
plt.semilogy(iters_y, y, '.-', label='qubit ADAPT')
path = 'Results/H4/hh'
x = 0
for filename in os.listdir(path):
    x += 1
    with open(path+'/'+filename, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        y = []
        for row in plots:
            y.append(math.fabs(float(row[3])))
            iters_y = list(range(len(y)))
    if x == len(os.listdir(path)):
        plt.semilogy(iters_y, y, 'g-', alpha=0.3, label='qubit ADAPT 1/4 pool')
    else:
        plt.semilogy(iters_y, y, 'g-', alpha=0.3)
path = 'Results/H4/hhhh'
x = 0
for filename in os.listdir(path):
    x += 1
    with open(path+'/'+filename, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        y = []
        for row in plots:
            y.append(math.fabs(float(row[3])))
            iters_y = list(range(len(y)))
    if x == len(os.listdir(path)):
        plt.semilogy(iters_y, y, 'r-', alpha=0.3, label='qubit ADAPT 1/16 pool')
    else:
        plt.semilogy(iters_y, y, 'r-', alpha=0.3)
path = 'Results/H4/hhhhh'
x = 0
for filename in os.listdir(path):
    x += 1
    with open(path+'/'+filename, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        y = []
        for row in plots:
            y.append(math.fabs(float(row[3])))
            iters_y = list(range(len(y)))
    if x == len(os.listdir(path)):
        plt.semilogy(iters_y, y, 'k-', alpha=0.3, label='qubit ADAPT 1/32 pool')
    else:
        plt.semilogy(iters_y, y, 'k-', alpha=0.3)
plt.xlabel('iterations',  fontsize=14)
plt.ylabel('E(iterations)-E(FCI) Hartree',  fontsize=14)
plt.ylim((1e-14,1))
# plt.annotate('(a)',  fontsize=14, xy=(plt.xlim()[1]*0.9,plt.ylim()[1]*0.002))
plt.legend(loc='lower center',  fontsize=14)
plt.show()