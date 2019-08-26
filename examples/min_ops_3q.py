import openfermion
import random
import numpy as np

from openfermion import *

from itertools import combinations

H =  QubitOperator('X0 X1 X2', random.random())
H += QubitOperator('X0 X1 Z2', random.random())
H += QubitOperator('X0 Z1 X2', random.random())
H += QubitOperator('Z0 X1 X2', random.random())
H += QubitOperator('Z0 Z1 X2', random.random())
H += QubitOperator('Z0 X1 Z2', random.random())
H += QubitOperator('X0 Z1 Z2', random.random())
H += QubitOperator('Z0 Z1 Z2', random.random())
H += QubitOperator('X0 Y1 Y2', random.random())
H += QubitOperator('Y0 X1 Y2', random.random())
H += QubitOperator('Y0 Y1 X2', random.random())
H += QubitOperator('Z0 Y1 Y2', random.random())
H += QubitOperator('Y0 Z1 Y2', random.random())
H += QubitOperator('Y0 Y1 Z2', random.random())
H += QubitOperator('Y1 Y2', random.random())
H += QubitOperator('Y0 Y2', random.random())
H += QubitOperator('Y0 Y1', random.random())
H += QubitOperator('X2', random.random())
H += QubitOperator('X1', random.random())
H += QubitOperator('X0', random.random())
H += QubitOperator('Z1 Z2', random.random())
H += QubitOperator('Z0 Z2', random.random())
H += QubitOperator('Z0 Z1', random.random())
H += QubitOperator('Z0', random.random())
H += QubitOperator('Z1', random.random())
H += QubitOperator('Z2', random.random())
H += QubitOperator('X1 X2', random.random())
H += QubitOperator('X0 X2', random.random())
H += QubitOperator('X0 X1', random.random())
H += QubitOperator('X0 Z1', random.random())
H += QubitOperator('X1 Z2', random.random())
H += QubitOperator('X0 Z2', random.random())
H += QubitOperator('Z1 X2', random.random())
H += QubitOperator('Z0 X2', random.random())
H += QubitOperator('Z0 X1', random.random())

pool = []

pool.append(QubitOperator('Y0', 1j))
pool.append(QubitOperator('Y1', 1j))
pool.append(QubitOperator('Y2', 1j))

pool.append(QubitOperator('Y0 Y1 Y2', 1j))

pool.append(QubitOperator('X0 X1 Y2', 1j))
pool.append(QubitOperator('X0 Y1 X2', 1j))
pool.append(QubitOperator('Y0 X1 X2', 1j))

pool.append(QubitOperator('Z0 Z1 Y2', 1j))
pool.append(QubitOperator('Z0 Y1 Z2', 1j))
pool.append(QubitOperator('Y0 Z1 Z2', 1j))

pool.append(QubitOperator('X0 Z1 Y2', 1j))
pool.append(QubitOperator('X0 Y1 Z2', 1j))
pool.append(QubitOperator('Y0 X1 Z2', 1j))

pool.append(QubitOperator('Z0 X1 Y2', 1j))
pool.append(QubitOperator('Z0 Y1 X2', 1j))
pool.append(QubitOperator('Y0 Z1 X2', 1j))

pool.append(QubitOperator('X1 Y2', 1j))
pool.append(QubitOperator('X0 Y1', 1j))
pool.append(QubitOperator('X0 Y2', 1j))

pool.append(QubitOperator('Y1 X2', 1j))
pool.append(QubitOperator('Y0 X1', 1j))
pool.append(QubitOperator('Y0 X2', 1j))

pool.append(QubitOperator('Z1 Y2', 1j))
pool.append(QubitOperator('Z0 Y1', 1j))
pool.append(QubitOperator('Z0 Y2', 1j))

pool.append(QubitOperator('Y1 Z2', 1j))
pool.append(QubitOperator('Y0 Z1', 1j))
pool.append(QubitOperator('Y0 Z2', 1j))

comb = list(combinations(list(range(len(pool))), 7))
print('number of combinations:',len(comb))

trial_pool
