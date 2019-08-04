import numpy as np
import matplotlib.pyplot as plt
import csv
import math

x = []
y = []

with open('Result/max_cut/4_nodes_QAOA_NM.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(int(row[1]))
        y.append(math.fabs(float(row[0])))

x1 = []
y1 = []

with open('Result/max_cut/4_nodes_ADAPT_NM.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x1.append(int(row[1]))
        y1.append(math.fabs(float(row[0])))

x2 = []
y2 = []

with open('Result/max_cut/4_nodes_ADAPT_min_NM.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x2.append(int(row[1]))
        y2.append(math.fabs(float(row[0])))

x3 = []
y3 = []

with open('Result/max_cut/4_nodes_ADAPT_QAOA_NM.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x3.append(int(row[1]))
        y3.append(math.fabs(float(row[0])))

plt.semilogy(x, y, label='QAOA')
plt.semilogy(x1, y1, label='qADAPT')
plt.semilogy(x2, y2, label='qADAPT (minimizer)')
plt.semilogy(x3, y3, label='ADAPT QAOA')
plt.legend()
plt.show()