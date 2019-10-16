import numpy as np
import matplotlib.pyplot as plt
import csv
import math

x = []
y = []
z = []

x1 = []
y1 = []
z1 = []

x2 = []
y2 = []
z2 = []

x3 = []
y3 = []
z3 = []

x4 = []
y4 = []
z4 = []

x5 = []
y5 = []
z5 = []

x6 = []
y6 = []
z6 = []

x7 = []
y7 = []
z7 = []

x8 = []
y8 = []
z8 = []

x9 = []
y9 = []
z9 = []

x10 = []
y10 = []
z10 = []

x11 = []
y11 = []
z11 = []

x12 = []
y12 = []
z12 = []

x13 = []
y13 = []
z13 = []

x14 = []
y14 = []
z14 = []

x15 = []
y15 = []
z15 = []

x16 = []
y16 = []
z16 = []

x17 = []
y17 = []
z17 = []

x18 = []
y18 = []
z18 = []

x19 = []
y19 = []
z19 = []

x20 = []
y20 = []
z20 = []

x21 = []
y21 = []
z21 = []

x22 = []
y22 = []
z22 = []

x23 = []
y23 = []
z23 = []

x24 = []
y24 = []
z24 = []

x25 = []
y25 = []
z25 = []

x26 = []
y26 = []
z26 = []

x27 = []
y27 = []
z27 = []

x28 = []
y28 = []
z28 = []

x29 = []
y29 = []
z29 = []

x30 = []
y30 = []
z30 = []

x31 = []
y31 = []
z31 = []

x32 = []
y32 = []
z32 = []

x33 = []
y33 = []
z33 = []

x34 = []
y34 = []
z34 = []

x35 = []
y35 = []

x36 = []
y36 = []
z36 = []

x37 = []
y37 = []

x38 = []
y38 = []

x39 = []
y39 = []

x40 = []
y40 = []

x41 = []
y41 = []

x42 = []
y42 = []

x43 = []
y43 = []

pars0 = []
pars1 = []
pars2 = []
pars3 = []
pars4 = []
numpars = []

ops = []
param = []
n = 0
n_gate = []

with open('Results/LiH/qubits_1e-9_gate.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in reversed(list(plots)):
        # row[0] = row[0].replace('%d, %s', '%s, %d')
        row[0] = row[0].replace('),', '##')
        row[0] = row[0].replace(',', '')
        row[0] = row[0].replace('##', '),')
        row[0] = row[0].replace( "'" , '')
        row[0] = row[0].replace('((', "['")
        row[0] = row[0].replace('))', "']")
        row[0] = row[0].replace('(', "")
        row[0] = row[0].replace(')', "")
        row[0] = row[0].replace('[','(')
        row[0] = row[0].replace( ']', ')')
        row[0] = row[0].replace( ' ', '')
        ops.append((row[0]))
        param.append(float(row[3]))
        for i in row[0]:
            if  i == "X" or i == "Y" or i == "Z":
                n +=1
        n -=1
        n_gate.append(2*n)

with open('Results/H4/qubits_Zs_1e-3.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x1.append(float(row[2]))
        z1.append(math.fabs(float(row[1])))
        y1.append(math.fabs(float(row[0])))

with open('Results/H4/qubits_nZs_1e-3.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x2.append(float(row[2]))
        z2.append(math.fabs(float(row[1])))
        y2.append(math.fabs(float(row[0])))

with open('Results/H4/qubits_Zs_1e-4.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x3.append(float(row[2]))
        z3.append(math.fabs(float(row[1])))
        y3.append(math.fabs(float(row[0])))

with open('Results/H4/qubits_nZs_1e-4.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x4.append(float(row[2]))
        z4.append(math.fabs(float(row[1])))
        y4.append(math.fabs(float(row[0])))

with open('Results/H4/SD_1e-3.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x5.append(float(row[2]))
        z5.append(math.fabs(float(row[1])))
        y5.append(math.fabs(float(row[0])))

with open('Results/H4/SD_1e-4.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x6.append(float(row[2]))
        z6.append(math.fabs(float(row[1])))
        y6.append(math.fabs(float(row[0])))

with open('Results/H4/GSD_1e-3.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x7.append(float(row[2]))
        z7.append(math.fabs(float(row[1])))
        y7.append(math.fabs(float(row[0])))

with open('Results/H4/GSD_1e-4.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x8.append(float(row[2]))
        z8.append(math.fabs(float(row[1])))
        y8.append(math.fabs(float(row[0])))

with open('Results/H4/qGSD_nZs_1e-9.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x9.append(float(row[2]))
        z9.append(math.fabs(float(row[1])))
        y9.append(math.fabs(float(row[0])))

with open('Results/H4/qubits_gp_1e-4.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x10.append(float(row[2]))
        z10.append(math.fabs(float(row[1])))
        y10.append(math.fabs(float(row[0])))

with open('Results/H4/qubits_gp_1e-3.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x11.append(float(row[2]))
        z11.append(math.fabs(float(row[1])))
        y11.append(math.fabs(float(row[0])))

with open('Results/H4/GSD_nspin_1e-4.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x12.append(float(row[2]))
        z12.append(math.fabs(float(row[1])))
        y12.append(math.fabs(float(row[0])))

with open('Results/H4/qubits_GP2_1e-4.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x13.append(float(row[2]))
        z13.append(math.fabs(float(row[1])))
        y13.append(math.fabs(float(row[0])))

with open('Results/H4/qubits2_1e-4.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x14.append(float(row[2]))
        z14.append(math.fabs(float(row[1])))
        y14.append(math.fabs(float(row[0])))

with open('Results/H4/qubits_spin_1e-9.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x15.append(float(row[2]))
        z15.append(math.fabs(float(row[1])))
        y15.append(math.fabs(float(row[0])))

with open('Results/H4/qubits_nZs_few_1e-4.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x16.append(float(row[2]))
        z16.append(math.fabs(float(row[1])))
        y16.append(math.fabs(float(row[0])))

with open('Results/H4/hess_plus_1e-4.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x17.append(float(row[2]))
        z17.append(math.fabs(float(row[1])))
        y17.append(math.fabs(float(row[0])))

with open('Results/H4/hess_minus_1e-4.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x18.append(float(row[2]))
        z18.append(math.fabs(float(row[1])))
        y18.append(math.fabs(float(row[0])))

with open('Results/H4/GSD_trot_diff_1e-4.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x19.append(float(row[2]))
        z19.append(math.fabs(float(row[1])))
        y19.append(math.fabs(float(row[0])))

with open('Results/H4/GSD_trot_same_old_1e-4.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x20.append(float(row[2]))
        z20.append(math.fabs(float(row[1])))
        y20.append(math.fabs(float(row[0])))

with open('Results/H4/GSD_trot_same_new_1e-4.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x21.append(float(row[2]))
        z21.append(math.fabs(float(row[1])))
        y21.append(math.fabs(float(row[0])))

with open('Results/H4/hess_plus_all_1e-4.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x22.append(float(row[2]))
        z22.append(math.fabs(float(row[1])))
        y22.append(math.fabs(float(row[0])))

with open('Results/H4/hess_GSD_1e-4.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:

        x23.append(float(row[2]))
        z23.append(math.fabs(float(row[1])))
        y23.append(math.fabs(float(row[0])))

with open('Results/H4/GSD_trot_same_old_1e-9.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x24.append(float(row[2]))
        z24.append(math.fabs(float(row[1])))
        y24.append(math.fabs(float(row[0])))

with open('Results/H4/Qubits_A_1e-4.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x25.append(float(row[2]))
        z25.append(math.fabs(float(row[1])))
        y25.append(math.fabs(float(row[0])))

with open('Results/H4/Qubits_rand.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x26.append(float(row[2]))
        # z26.append(math.fabs(float(row[1])))
        y26.append(math.fabs(float(row[0])))

with open('Results/H4/q_rand_0.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x27.append(float(row[1]))
        y27.append(math.fabs(float(row[0])))

with open('Results/H4/q_rand_1.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x28.append(float(row[3]))
        y28.append(math.fabs(float(row[2])))

with open('Results/H4/q_rand_2.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x29.append(float(row[5]))
        y29.append(math.fabs(float(row[4])))

with open('Results/H4/q_rand_3.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x30.append(float(row[7]))
        y30.append(math.fabs(float(row[6])))

with open('Results/H4/q_rand_4.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x31.append(float(row[9]))
        y31.append(math.fabs(float(row[8])))

with open('Results/H4/q_rand_5.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x32.append(float(row[11]))
        y32.append(math.fabs(float(row[10])))

with open('Results/H4/q_rand_6.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x33.append(float(row[13]))
        y33.append(math.fabs(float(row[12])))

with open('Results/H4/q_rand_7.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x34.append(float(row[15]))
        y34.append(math.fabs(float(row[14])))

with open('Results/H4/qubits_energy_step.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x35.append(float(row[1]))
        y35.append(math.fabs(float(row[0])))

with open('Results/H4/qubits_new.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x36.append(float(row[1]))
        y36.append(math.fabs(float(row[0])))
        z36.append(1-float(row[3]))

with open('Results/H4/GSD_new.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x37.append(float(row[1]))
        y37.append(math.fabs(float(row[0])))

with open('Results/H4/GSD_E_step.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x38.append(float(row[1]))
        y38.append(math.fabs(float(row[0])))

with open('Results/H4/qubits_reduce_XYYY_1e-7.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x39.append(float(row[1]))
        y39.append(math.fabs(float(row[0])))

with open('Results/H4/qubits_672_1e-7.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x40.append(float(row[1]))
        y40.append(math.fabs(float(row[0])))

with open('Results/H4/qubits_reduce_YXYY_1e-7.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x41.append(float(row[1]))
        y41.append(math.fabs(float(row[0])))

with open('Results/H4/qubits_reduce_YYXY_1e-7.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x42.append(float(row[1]))
        y42.append(math.fabs(float(row[0])))

with open('Results/H4/qubits_reduce_YYYX_1e-7.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x43.append(float(row[1]))
        y43.append(math.fabs(float(row[0])))

h4_q_mini = []
iter_h4_q_mini = []

with open('Results/H4/qubits_mini_1e-12.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        iter_h4_q_mini.append(float(row[1]))
        h4_q_mini.append(math.fabs(float(row[0])))

with open('Results/H4/Qubits_pars_dyn.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        pars0.append(float(row[0]))
        pars1.append(float(row[1]))
        pars2.append(float(row[2]))
        pars3.append(float(row[3]))
        pars4.append(float(row[4]))

lih_rand_1  = []
lih_rand_2  = []
lih_rand_3  = []
lih_rand_4  = []
lih_rand_5  = []
lih_rand_6  = []
lih_rand_7  = []
lih_rand_8  = []
lih_rand_9  = []
lih_rand_10 = []
lih_rand_11 = []
lih_rand_12 = []
iter_lih_rand_1  = []
iter_lih_rand_2  = []
iter_lih_rand_3  = []
iter_lih_rand_4  = []
iter_lih_rand_5  = []
iter_lih_rand_6  = []
iter_lih_rand_7  = []
iter_lih_rand_8  = []
iter_lih_rand_9  = []
iter_lih_rand_10 = []
iter_lih_rand_11 = []
iter_lih_rand_12 = []

with open('Results/LiH/qubits_rand_1.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        lih_rand_1.append(math.fabs(float(row[0])))
        iter_lih_rand_1.append(int(row[1]))

with open('Results/LiH/qubits_rand_2.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        lih_rand_2.append(math.fabs(float(row[0])))
        iter_lih_rand_2.append(int(row[1]))

with open('Results/LiH/qubits_rand_3.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        lih_rand_3.append(math.fabs(float(row[0])))
        iter_lih_rand_3.append(int(row[1]))

with open('Results/LiH/qubits_rand_4.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        lih_rand_4.append(math.fabs(float(row[0])))
        iter_lih_rand_4.append(int(row[1]))

with open('Results/LiH/qubits_rand_5.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        lih_rand_5.append(math.fabs(float(row[0])))
        iter_lih_rand_5.append(int(row[1]))

with open('Results/LiH/qubits_rand_6.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        lih_rand_6.append(math.fabs(float(row[0])))
        iter_lih_rand_6.append(int(row[1]))

with open('Results/LiH/qubits_rand_7.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        lih_rand_7.append(math.fabs(float(row[0])))
        iter_lih_rand_7.append(int(row[1]))

with open('Results/LiH/qubits_rand_8.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        lih_rand_8.append(math.fabs(float(row[0])))
        iter_lih_rand_8.append(int(row[1]))

with open('Results/LiH/qubits_rand_9.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        lih_rand_9.append(math.fabs(float(row[0])))
        iter_lih_rand_9.append(int(row[1]))

with open('Results/LiH/qubits_rand_10.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        lih_rand_10.append(math.fabs(float(row[0])))
        iter_lih_rand_10.append(int(row[1]))

with open('Results/LiH/qubits_rand_11.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        lih_rand_11.append(math.fabs(float(row[0])))
        iter_lih_rand_11.append(int(row[1]))

with open('Results/LiH/qubits_rand_12.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        lih_rand_12.append(math.fabs(float(row[0])))
        iter_lih_rand_12.append(int(row[1]))

half_1  = []
half_2  = []
half_3  = []
half_4  = []
half_5  = []
half_6  = []
half_7  = []
half_8  = []
half_9  = []
half_10 = []
half_11 = []
half_12 = []
iters_half = []

with open('Results/H4/qubits_half.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        half_1.append(math.fabs(float(row[0])))
        half_2.append(math.fabs(float(row[1])))
        half_3.append(math.fabs(float(row[2])))
        half_4.append(math.fabs(float(row[3])))
        half_5.append(math.fabs(float(row[4])))
        half_6.append(math.fabs(float(row[5])))
        half_7.append(math.fabs(float(row[6])))
        half_8.append(math.fabs(float(row[7])))
        half_9.append(math.fabs(float(row[8])))
        half_10.append(math.fabs(float(row[9])))
        half_11.append(math.fabs(float(row[10])))
        half_12.append(math.fabs(float(row[11])))
        iters_half.append(float(row[12]))

hh_1  = []
hh_2  = []
hh_3  = []
hh_4  = []
hh_5  = []
hh_6  = []
hh_7  = []
hh_8  = []
hh_9  = []
hh_10 = []
hh_11 = []
hh_12 = []
iters_hh_1  = []
iters_hh_2  = []
iters_hh_3  = []
iters_hh_4  = []
iters_hh_5  = []
iters_hh_6  = []
iters_hh_7  = []
iters_hh_8  = []
iters_hh_9  = []
iters_hh_10 = []
iters_hh_11 = []
iters_hh_12 = []

with open('Results/H4/qubits_hh_1.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hh_1.append(math.fabs(float(row[0])))
        iters_hh_1=list(range(len(hh_1)))

with open('Results/H4/qubits_hh_2.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hh_2.append(math.fabs(float(row[0])))
        iters_hh_2=list(range(len(hh_2)))

with open('Results/H4/qubits_hh_3.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hh_3.append(math.fabs(float(row[0])))
        iters_hh_3=list(range(len(hh_3)))

with open('Results/H4/qubits_hh_4.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hh_4.append(math.fabs(float(row[0])))
        iters_hh_4=list(range(len(hh_4)))

with open('Results/H4/qubits_hh_5.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hh_5.append(math.fabs(float(row[0])))
        iters_hh_5=list(range(len(hh_5)))

with open('Results/H4/qubits_hh_6.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hh_6.append(math.fabs(float(row[0])))
        iters_hh_6=list(range(len(hh_6)))

with open('Results/H4/qubits_hh_7.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hh_7.append(math.fabs(float(row[0])))
        iters_hh_7=list(range(len(hh_7)))

with open('Results/H4/qubits_hh_8.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hh_8.append(math.fabs(float(row[0])))
        iters_hh_8=list(range(len(hh_8)))

with open('Results/H4/qubits_hh_9.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hh_9.append(math.fabs(float(row[0])))
        iters_hh_9=list(range(len(hh_9)))

with open('Results/H4/qubits_hh_10.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hh_10.append(math.fabs(float(row[0])))
        iters_hh_10=list(range(len(hh_10)))

with open('Results/H4/qubits_hh_11.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hh_11.append(math.fabs(float(row[0])))
        iters_hh_11=list(range(len(hh_11)))

with open('Results/H4/qubits_hh_12.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hh_12.append(math.fabs(float(row[0])))
        iters_hh_12=list(range(len(hh_12)))

trim40_rand_1  = []
trim40_rand_2  = []
trim40_rand_3  = []
trim40_rand_4  = []
trim40_rand_5  = []
trim40_rand_6  = []
trim40_rand_7  = []
trim40_rand_8  = []
trim40_rand_9  = []
trim40_rand_10 = []
trim40_rand_11 = []
trim40_rand_12 = []
iters_trim40_1  = []
iters_trim40_2  = []
iters_trim40_3  = []
iters_trim40_4  = []
iters_trim40_5  = []
iters_trim40_6  = []
iters_trim40_7  = []
iters_trim40_8  = []
iters_trim40_9  = []
iters_trim40_10 = []
iters_trim40_11 = []
iters_trim40_12 = []

with open('Results/H4/qubits_trim40_rand.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        trim40_rand_1.append(math.fabs(float(row[0])))
        iters_trim40_1=list(range(len(trim40_rand_1)))

with open('Results/H4/qubits_trim40_rand_1.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        trim40_rand_2.append(math.fabs(float(row[0])))
        iters_trim40_2=list(range(len(trim40_rand_2)))

with open('Results/H4/qubits_trim40_rand_2.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        trim40_rand_3.append(math.fabs(float(row[0])))
        iters_trim40_3=list(range(len(trim40_rand_3)))

with open('Results/H4/qubits_trim40_rand_3.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        trim40_rand_4.append(math.fabs(float(row[0])))
        iters_trim40_4=list(range(len(trim40_rand_4)))

with open('Results/H4/qubits_trim40_rand_4.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        trim40_rand_5.append(math.fabs(float(row[0])))
        iters_trim40_5=list(range(len(trim40_rand_5)))

with open('Results/H4/qubits_trim40_rand_5.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        trim40_rand_6.append(math.fabs(float(row[0])))
        iters_trim40_6=list(range(len(trim40_rand_6)))

with open('Results/H4/qubits_trim40_rand_6.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        trim40_rand_7.append(math.fabs(float(row[0])))
        iters_trim40_7=list(range(len(trim40_rand_7)))

with open('Results/H4/qubits_trim40_rand_7.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        trim40_rand_8.append(math.fabs(float(row[0])))
        iters_trim40_8=list(range(len(trim40_rand_8)))

with open('Results/H4/qubits_trim40_rand_8.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        trim40_rand_9.append(math.fabs(float(row[0])))
        iters_trim40_9=list(range(len(trim40_rand_9)))

with open('Results/H4/qubits_trim40_rand_9.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        trim40_rand_10.append(math.fabs(float(row[0])))
        iters_trim40_10=list(range(len(trim40_rand_10)))

with open('Results/H4/qubits_trim40_rand_10.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        trim40_rand_11.append(math.fabs(float(row[0])))
        iters_trim40_11=list(range(len(trim40_rand_11)))

with open('Results/H4/qubits_trim40_rand_11.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        trim40_rand_12.append(math.fabs(float(row[0])))
        iters_trim40_12=list(range(len(trim40_rand_12)))

lih_q_x = []
lih_q_z = []
lih_q_y = []

with open('Results/LiH/qubits_1e-9.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        lih_q_x.append(float(row[1]))
        lih_q_z.append(math.fabs(float(row[1])))
        lih_q_y.append(math.fabs(float(row[0])))

lih_f_x = []
lih_f_z = []
lih_f_y = []

with open('Results/LiH/GSD_1e-7.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        lih_f_x.append(float(row[2]))
        lih_f_z.append(math.fabs(float(row[1])))
        lih_f_y.append(math.fabs(float(row[0])))

hhh_1  = []
hhh_2  = []
hhh_3  = []
hhh_4  = []
hhh_5  = []
hhh_6  = []
hhh_7  = []
hhh_8  = []
hhh_9  = []
hhh_10 = []
hhh_11 = []
hhh_12 = []
iters_hhh_1  = []
iters_hhh_2  = []
iters_hhh_3  = []
iters_hhh_4  = []
iters_hhh_5  = []
iters_hhh_6  = []
iters_hhh_7  = []
iters_hhh_8  = []
iters_hhh_9  = []
iters_hhh_10 = []
iters_hhh_11 = []
iters_hhh_12 = []

with open('Results/H4/qubits_hhh_1.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hhh_1.append(math.fabs(float(row[0])))
        iters_hhh_1=list(range(len(hhh_1)))

with open('Results/H4/qubits_hhh_2.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hhh_2.append(math.fabs(float(row[0])))
        iters_hhh_2=list(range(len(hhh_2)))

with open('Results/H4/qubits_hhh_3.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hhh_3.append(math.fabs(float(row[0])))
        iters_hhh_3=list(range(len(hhh_3)))

with open('Results/H4/qubits_hhh_4.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hhh_4.append(math.fabs(float(row[0])))
        iters_hhh_4=list(range(len(hhh_4)))

with open('Results/H4/qubits_hhh_5.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hhh_5.append(math.fabs(float(row[0])))
        iters_hhh_5=list(range(len(hhh_5)))

with open('Results/H4/qubits_hhh_6.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hhh_6.append(math.fabs(float(row[0])))
        iters_hhh_6=list(range(len(hhh_6)))

with open('Results/H4/qubits_hhh_7.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hhh_7.append(math.fabs(float(row[0])))
        iters_hhh_7=list(range(len(hhh_7)))

with open('Results/H4/qubits_hhh_8.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hhh_8.append(math.fabs(float(row[0])))
        iters_hhh_8=list(range(len(hhh_8)))

with open('Results/H4/qubits_hhh_9.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hhh_9.append(math.fabs(float(row[0])))
        iters_hhh_9=list(range(len(hhh_9)))

with open('Results/H4/qubits_hhh_10.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hhh_10.append(math.fabs(float(row[0])))
        iters_hhh_10=list(range(len(hhh_10)))

with open('Results/H4/qubits_hhh_11.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hhh_11.append(math.fabs(float(row[0])))
        iters_hhh_11=list(range(len(hhh_11)))

with open('Results/H4/qubits_hhh_12.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hhh_12.append(math.fabs(float(row[0])))
        iters_hhh_12=list(range(len(hhh_12)))

hhhh_1  = []
hhhh_2  = []
hhhh_3  = []
hhhh_4  = []
hhhh_5  = []
hhhh_6  = []
hhhh_7  = []
hhhh_8  = []
hhhh_9  = []
hhhh_10 = []
hhhh_11 = []
hhhh_12 = []
iters_hhhh_1  = []
iters_hhhh_2  = []
iters_hhhh_3  = []
iters_hhhh_4  = []
iters_hhhh_5  = []
iters_hhhh_6  = []
iters_hhhh_7  = []
iters_hhhh_8  = []
iters_hhhh_9  = []
iters_hhhh_10 = []
iters_hhhh_11 = []
iters_hhhh_12 = []

with open('Results/H4/qubits_hhhh_1.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hhhh_1.append(math.fabs(float(row[0])))
        iters_hhhh_1=list(range(len(hhhh_1)))

with open('Results/H4/qubits_hhhh_2.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hhhh_2.append(math.fabs(float(row[0])))
        iters_hhhh_2=list(range(len(hhhh_2)))

with open('Results/H4/qubits_hhhh_4.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hhhh_4.append(math.fabs(float(row[0])))
        iters_hhhh_4=list(range(len(hhhh_4)))

with open('Results/H4/qubits_hhhh_5.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hhhh_5.append(math.fabs(float(row[0])))
        iters_hhhh_5=list(range(len(hhhh_5)))

with open('Results/H4/qubits_hhhh_6.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hhhh_6.append(math.fabs(float(row[0])))
        iters_hhhh_6=list(range(len(hhhh_6)))

with open('Results/H4/qubits_hhhh_7.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hhhh_7.append(math.fabs(float(row[0])))
        iters_hhhh_7=list(range(len(hhhh_7)))

with open('Results/H4/qubits_hhhh_8.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hhhh_8.append(math.fabs(float(row[0])))
        iters_hhhh_8=list(range(len(hhhh_8)))

with open('Results/H4/qubits_hhhh_9.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hhhh_9.append(math.fabs(float(row[0])))
        iters_hhhh_9=list(range(len(hhhh_9)))

with open('Results/H4/qubits_hhhh_10.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hhhh_10.append(math.fabs(float(row[0])))
        iters_hhhh_10=list(range(len(hhhh_10)))

with open('Results/H4/qubits_hhhh_11.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hhhh_11.append(math.fabs(float(row[0])))
        iters_hhhh_11=list(range(len(hhhh_11)))

with open('Results/H4/qubits_hhhh_12.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        hhhh_12.append(math.fabs(float(row[0])))
        iters_hhhh_12=list(range(len(hhhh_12)))

remove_1 = []
remove_2 = []
remove_3 = []
iters_remove = []

with open('Results/H4/qubits_remove.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        remove_1.append(math.fabs(float(row[0])))
        remove_2.append(math.fabs(float(row[1])))
        remove_3.append(math.fabs(float(row[2])))
        iters_remove.append(float(row[3]))

qubits_ham = []
iters_ham = []

with open('Results/H4/qubits_ham.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        iters_ham.append(float(row[1]))
        qubits_ham.append(math.fabs(float(row[0])))

qubits_init = []
iters_init =[]

with open('Results/H4/qubits_init_zero.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        iters_init.append(float(row[1]))
        qubits_init.append(math.fabs(float(row[0])))

qubits_trim30 = []
iters_trim30 = []
overlap_trim30 = []

with open('Results/H4/qubits_trim30.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        iters_trim30.append(float(row[1]))
        qubits_trim30.append(math.fabs(float(row[0])))
        overlap_trim30.append(1-float(row[3]))

qubits_trim40 = []
iters_trim40 = []
overlap_trim40 = []

with open('Results/H4/qubits_trim40.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        iters_trim40.append(float(row[1]))
        qubits_trim40.append(math.fabs(float(row[0])))
        overlap_trim40.append(1-float(row[3]))

qubits_trim50 = []
iters_trim50 = []
overlap_trim50 = []

with open('Results/H4/qubits_trim50.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        iters_trim50.append(float(row[1]))
        qubits_trim50.append(math.fabs(float(row[0])))
        overlap_trim50.append(1-float(row[3]))

qubits_trim60 = []
iters_trim60 =[]

with open('Results/H4/qubits_trim60.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        iters_trim60.append(float(row[1]))
        qubits_trim60.append(math.fabs(float(row[0])))

qubits_trim_nonzero = []
iters_trim_nonzero =[]

with open('Results/H4/qubits_trim_nonzero.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        iters_trim_nonzero.append(float(row[1]))
        qubits_trim_nonzero.append(math.fabs(float(row[0])))

numpars = range(0,31)

f = open("GSD_Gate_1e-4.out", "r")
searchlines = f.readlines()
f.close()
num_param1 = 0
line_gate = []
line_gate.append(0)
CXGSD = []
xx = 0
for i, line in enumerate(searchlines):
    if "Deallocate | Qureg[0]" in line:
        num_param1 +=1
        line_gate.append(i)
for k in reversed(range(len(line_gate))):
    for l, line in enumerate(searchlines[int(line_gate[k-1]):int(line_gate[k])]):
        if "CX" in line:
            xx +=1
    if k > 0:
        CXGSD.append(xx)

f = open("Qubits_Gate_1e-4.out", "r")
searchlines = f.readlines()
f.close()
num_param1 = 0
line_gate = []
line_gate.append(0)
CXq = []
xx = 0
for i, line in enumerate(searchlines):
    if "Deallocate | Qureg[0]" in line:
        num_param1 +=1
        line_gate.append(i)
for k in reversed(range(len(line_gate))):
    for l, line in enumerate(searchlines[int(line_gate[k-1]):int(line_gate[k])]):
        if "CX" in line:
            xx +=1
    if k > 0:
        CXq.append(xx)

f = open("Qubits_fac_Gate_1e-4.out", "r")
searchlines = f.readlines()
f.close()
num_param1 = 0
line_gate = []
line_gate.append(0)
CXqf = []
xx = 0
for i, line in enumerate(searchlines):
    if "Deallocate | Qureg[0]" in line:
        num_param1 +=1
        line_gate.append(i)
for k in reversed(range(len(line_gate))):
    for l, line in enumerate(searchlines[int(line_gate[k-1]):int(line_gate[k])]):
        if "CX" in line:
            xx +=1
    if k > 0:
        CXqf.append(xx)

f = open("h4_qubits_Z_1e-7_gate.out", "r")
searchlines = f.readlines()
f.close()
num_param1 = 0
line_gate = []
line_gate.append(0)
CXqZ = []
xx = 0
for i, line in enumerate(searchlines):
    if "Deallocate | Qureg[0]" in line:
        num_param1 +=1
        line_gate.append(i)
for k in reversed(range(len(line_gate))):
    for l, line in enumerate(searchlines[int(line_gate[k-1]):int(line_gate[k])]):
        if "CX" in line:
            xx +=1
    if k > 0:
        CXqZ.append(xx)

f = open("h4_GSD_no_spin_1e-7_gate.out", "r")
searchlines = f.readlines()
f.close()
num_param1 = 0
line_gate = []
line_gate.append(0)
CXnS = []
xx = 0
for i, line in enumerate(searchlines):
    if "Deallocate | Qureg[0]" in line:
        num_param1 +=1
        line_gate.append(i)
for k in reversed(range(len(line_gate))):
    for l, line in enumerate(searchlines[int(line_gate[k-1]):int(line_gate[k])]):
        if "CX" in line:
            xx +=1
    if k > 0:
        CXnS.append(xx)

CXrand_total = [0]*len(x26)
CXrand_store = [0]*len(x26)
f = open("h4_qubits_1e-7_rand_0.out", "r")
searchlines = f.readlines()
f.close()
num_param1 = 0
line_gate = []
line_gate.append(0)
CXrand = []
CXrand0 = []
xx = 0
for i, line in enumerate(searchlines):
    if "Deallocate | Qureg[0]" in line:
        num_param1 +=1
        line_gate.append(i)
for k in reversed(range(len(line_gate))):
    for l, line in enumerate(searchlines[int(line_gate[k-1]):int(line_gate[k])]):
        if "CX" in line:
            xx +=1
    if k > 0:
        CXrand.append(xx)
CXrand0 = [x for x in CXrand]
for i in range(len(x26)):
    if i > len(CXrand)-1:
        CXrand.append(CXrand[-1])
CXrand_total = [CXrand_store[j]+CXrand[j] for j in range(len(CXrand))]
CXrand_store = CXrand_total

f = open("h4_qubits_1e-7_rand_1.out", "r")
searchlines = f.readlines()
f.close()
num_param1 = 0
line_gate = []
line_gate.append(0)
CXrand = []
CXrand1 = []
xx = 0
for i, line in enumerate(searchlines):
    if "Deallocate | Qureg[0]" in line:
        num_param1 +=1
        line_gate.append(i)
for k in reversed(range(len(line_gate))):
    for l, line in enumerate(searchlines[int(line_gate[k-1]):int(line_gate[k])]):
        if "CX" in line:
            xx +=1
    if k > 0:
        CXrand.append(xx)
CXrand1 = [x for x in CXrand]
for i in range(len(x26)):
    if i > len(CXrand)-1:
        CXrand.append(CXrand[-1])
CXrand_total = [CXrand_store[j]+CXrand[j] for j in range(len(CXrand))]
CXrand_store = CXrand_total

f = open("h4_qubits_1e-7_rand_2.out", "r")
searchlines = f.readlines()
f.close()
num_param1 = 0
line_gate = []
line_gate.append(0)
CXrand = []
CXrand2 = []
xx = 0
for i, line in enumerate(searchlines):
    if "Deallocate | Qureg[0]" in line:
        num_param1 +=1
        line_gate.append(i)
for k in reversed(range(len(line_gate))):
    for l, line in enumerate(searchlines[int(line_gate[k-1]):int(line_gate[k])]):
        if "CX" in line:
            xx +=1
    if k > 0:
        CXrand.append(xx)
CXrand2 = [x for x in CXrand]
for i in range(len(x26)):
    if i > len(CXrand)-1:
        CXrand.append(CXrand[-1])
CXrand_total = [CXrand_store[j]+CXrand[j] for j in range(len(CXrand))]
CXrand_store = CXrand_total

f = open("h4_qubits_1e-7_rand_3.out", "r")
searchlines = f.readlines()
f.close()
num_param1 = 0
line_gate = []
line_gate.append(0)
CXrand = []
CXrand3 = []
xx = 0
for i, line in enumerate(searchlines):
    if "Deallocate | Qureg[0]" in line:
        num_param1 +=1
        line_gate.append(i)
for k in reversed(range(len(line_gate))):
    for l, line in enumerate(searchlines[int(line_gate[k-1]):int(line_gate[k])]):
        if "CX" in line:
            xx +=1
    if k > 0:
        CXrand.append(xx)
CXrand3 = [x for x in CXrand]
for i in range(len(x26)):
    if i > len(CXrand)-1:
        CXrand.append(CXrand[-1])
CXrand_total = [CXrand_store[j]+CXrand[j] for j in range(len(CXrand))]
CXrand_store = CXrand_total

f = open("h4_qubits_1e-7_rand_4.out", "r")
searchlines = f.readlines()
f.close()
num_param1 = 0
line_gate = []
line_gate.append(0)
CXrand = []
CXrand4 = []
xx = 0
for i, line in enumerate(searchlines):
    if "Deallocate | Qureg[0]" in line:
        num_param1 +=1
        line_gate.append(i)
for k in reversed(range(len(line_gate))):
    for l, line in enumerate(searchlines[int(line_gate[k-1]):int(line_gate[k])]):
        if "CX" in line:
            xx +=1
    if k > 0:
        CXrand.append(xx)
CXrand4 = [x for x in CXrand]
for i in range(len(x26)):
    if i > len(CXrand)-1:
        CXrand.append(CXrand[-1])
CXrand_total = [CXrand_store[j]+CXrand[j] for j in range(len(CXrand))]
CXrand_store = CXrand_total

f = open("h4_qubits_1e-7_rand_5.out", "r")
searchlines = f.readlines()
f.close()
num_param1 = 0
line_gate = []
line_gate.append(0)
CXrand = []
CXrand5 = []
xx = 0
for i, line in enumerate(searchlines):
    if "Deallocate | Qureg[0]" in line:
        num_param1 +=1
        line_gate.append(i)
for k in reversed(range(len(line_gate))):
    for l, line in enumerate(searchlines[int(line_gate[k-1]):int(line_gate[k])]):
        if "CX" in line:
            xx +=1
    if k > 0:
        CXrand.append(xx)
CXrand5 = [x for x in CXrand]
for i in range(len(x26)):
    if i > len(CXrand)-1:
        CXrand.append(CXrand[-1])
CXrand_total = [CXrand_store[j]+CXrand[j] for j in range(len(CXrand))]
CXrand_store = CXrand_total

f = open("h4_qubits_1e-7_rand_6.out", "r")
searchlines = f.readlines()
f.close()
num_param1 = 0
line_gate = []
line_gate.append(0)
CXrand = []
CXrand6 = []
xx = 0
for i, line in enumerate(searchlines):
    if "Deallocate | Qureg[0]" in line:
        num_param1 +=1
        line_gate.append(i)
for k in reversed(range(len(line_gate))):
    for l, line in enumerate(searchlines[int(line_gate[k-1]):int(line_gate[k])]):
        if "CX" in line:
            xx +=1
    if k > 0:
        CXrand.append(xx)
CXrand6 = [x for x in CXrand]
for i in range(len(x26)):
    if i > len(CXrand)-1:
        CXrand.append(CXrand[-1])
CXrand_total = [CXrand_store[j]+CXrand[j] for j in range(len(CXrand))]
CXrand_store = CXrand_total

f = open("h4_qubits_1e-7_rand_7.out", "r")
searchlines = f.readlines()
f.close()
num_param1 = 0
line_gate = []
line_gate.append(0)
CXrand = []
CXrand7 = []
xx = 0
for i, line in enumerate(searchlines):
    if "Deallocate | Qureg[0]" in line:
        num_param1 +=1
        line_gate.append(i)
for k in reversed(range(len(line_gate))):
    for l, line in enumerate(searchlines[int(line_gate[k-1]):int(line_gate[k])]):
        if "CX" in line:
            xx +=1
    if k > 0:
        CXrand.append(xx)
CXrand7 = [x for x in CXrand]
for i in range(len(x26)):
    if i > len(CXrand)-1:
        CXrand.append(CXrand[-1])
CXrand_total = [CXrand_store[j]+CXrand[j] for j in range(len(CXrand))]

f = open("h4_qubits_trim30.out", "r")
searchlines = f.readlines()
f.close()
num_param1 = 0
line_gate = []
line_gate.append(0)
CXtrim30 = []
xx = 0
for i, line in enumerate(searchlines):
    if "Deallocate | Qureg[0]" in line:
        num_param1 +=1
        line_gate.append(i)
for k in reversed(range(len(line_gate))):
    for l, line in enumerate(searchlines[int(line_gate[k-1]):int(line_gate[k])]):
        if "CX" in line:
            xx +=1
    if k > 0:
        CXtrim30.append(xx)


f = open("h4_qubits_trim40.out", "r")
searchlines = f.readlines()
f.close()
num_param1 = 0
line_gate = []
line_gate.append(0)
CXtrim40 = []
xx = 0
for i, line in enumerate(searchlines):
    if "Deallocate | Qureg[0]" in line:
        num_param1 +=1
        line_gate.append(i)
for k in reversed(range(len(line_gate))):
    for l, line in enumerate(searchlines[int(line_gate[k-1]):int(line_gate[k])]):
        if "CX" in line:
            xx +=1
    if k > 0:
        CXtrim40.append(xx)

f = open("h4_qubits_trim50.out", "r")
searchlines = f.readlines()
f.close()
num_param1 = 0
line_gate = []
line_gate.append(0)
CXtrim50 = []
xx = 0
for i, line in enumerate(searchlines):
    if "Deallocate | Qureg[0]" in line:
        num_param1 +=1
        line_gate.append(i)
for k in reversed(range(len(line_gate))):
    for l, line in enumerate(searchlines[int(line_gate[k-1]):int(line_gate[k])]):
        if "CX" in line:
            xx +=1
    if k > 0:
        CXtrim50.append(xx)

f = open("h4_qubits_trim50.out", "r")
searchlines = f.readlines()
f.close()
num_param1 = 0
line_gate = []
line_gate.append(0)
CXtrim50 = []
xx = 0
for i, line in enumerate(searchlines):
    if "Deallocate | Qureg[0]" in line:
        num_param1 +=1
        line_gate.append(i)
for k in reversed(range(len(line_gate))):
    for l, line in enumerate(searchlines[int(line_gate[k-1]):int(line_gate[k])]):
        if "CX" in line:
            xx +=1
    if k > 0:
        CXtrim50.append(xx)

f = open("h4_qubits_trim60.out", "r")
searchlines = f.readlines()
f.close()
num_param1 = 0
line_gate = []
line_gate.append(0)
CXtrim60 = []
xx = 0
for i, line in enumerate(searchlines):
    if "Deallocate | Qureg[0]" in line:
        num_param1 +=1
        line_gate.append(i)
for k in reversed(range(len(line_gate))):
    for l, line in enumerate(searchlines[int(line_gate[k-1]):int(line_gate[k])]):
        if "CX" in line:
            xx +=1
    if k > 0:
        CXtrim60.append(xx)

f = open("h4_qubits_trim_nonzero.out", "r")
searchlines = f.readlines()
f.close()
num_param1 = 0
line_gate = []
line_gate.append(0)
CXtrim_nonzero = []
xx = 0
for i, line in enumerate(searchlines):
    if "Deallocate | Qureg[0]" in line:
        num_param1 +=1
        line_gate.append(i)
for k in reversed(range(len(line_gate))):
    for l, line in enumerate(searchlines[int(line_gate[k-1]):int(line_gate[k])]):
        if "CX" in line:
            xx +=1
    if k > 0:
        CXtrim_nonzero.append(xx)

CXrand_average = [x/8 for x in CXrand_total]

f = open("LIH_GSD_gates_1e-7.out", "r")
searchlines = f.readlines()
f.close()
num_param1 = 0
line_gate = []
line_gate.append(0)
CX3 = []
xx = 0
for i, line in enumerate(searchlines):
    if "Deallocate | Qureg[0]" in line:
        num_param1 +=1
        line_gate.append(i)
for k in range(len(line_gate)-1):
    for l, line in enumerate(searchlines[int(line_gate[k]):int(line_gate[k+1])]):
        if "CX" in line:
            xx +=1
    CX3.append(xx)

ops = []
param = []
n = 0
n_gate_h6 = []

with open('Results/H6/qubits_gate_1.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in reversed(list(plots)):
        # row[0] = row[0].replace('%d, %s', '%s, %d')
        row[0] = row[0].replace('),', '##')
        row[0] = row[0].replace(',', '')
        row[0] = row[0].replace('##', '),')
        row[0] = row[0].replace( "'" , '')
        row[0] = row[0].replace('((', "['")
        row[0] = row[0].replace('))', "']")
        row[0] = row[0].replace('(', "")
        row[0] = row[0].replace(')', "")
        row[0] = row[0].replace('[','(')
        row[0] = row[0].replace( ']', ')')
        row[0] = row[0].replace( ' ', '')
        ops.append((row[0]))
        param.append(float(row[3]))
        for i in row[0]:
            if  i == "X" or i == "Y" or i == "Z":
                n +=1
        n -=1
        n_gate_h6.append(2*n)

h6_q_z = []
h6_q_y = []

with open('Results/H6/qubits_1e-9.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        h6_q_z.append(math.fabs(float(row[1])))
        h6_q_y.append(math.fabs(float(row[0])))

h6_f_z = []
h6_f_y = []

with open('Results/H6/GSD_1e-9.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        h6_f_z.append(math.fabs(float(row[1])))
        h6_f_y.append(math.fabs(float(row[0])))

f = open("h6_GSD_1e-9.out", "r")
searchlines = f.readlines()
f.close()
num_param1 = 0
line_gate = []
line_gate.append(0)
CX_h6 = []
xx = 0
for i, line in enumerate(searchlines):
    if "Deallocate | Qureg[0]" in line:
        num_param1 +=1
        line_gate.append(i)
for k in range(len(line_gate)-1):
    for l, line in enumerate(searchlines[int(line_gate[k]):int(line_gate[k+1])]):
        if "CX" in line:
            xx +=1
    CX_h6.append(xx)
# print(num_param1)
# print(CX_h6)

print(lih_f_x)
print(h6_f_z)

# plt.semilogy(x36,y36, '.-', label='qubit ADAPT (672)')
# plt.semilogy(x39,y39, '.-', label='qubit ADAPT (266 XYYY)')
# plt.semilogy(x41,y41, '.-', label='qubit ADAPT (266 YXYY)')
# plt.semilogy(x42,y42, '.-', label='qubit ADAPT (266 YYXY)')
# plt.semilogy(x43,y43, '.-', label='qubit ADAPT (266 YYYX)')
# plt.legend()

# plt.semilogy(x36,y36, '.-', label='qubit ADAPT')
# plt.semilogy(iter_h4_q_mini,h4_q_mini, '.-', label='qubits ADAPT minimizer')
# plt.xlabel('iterations',  fontsize=12)
# plt.ylabel('E(iterations)-E(FCI) Hartree',  fontsize=12)
# plt.legend()

# fig, axes = plt.subplots(2,3)
# fig.subplots_adjust(wspace=0.05)
# axes[0,0].semilogy(x36,y36, '.-', label='qubit ADAPT')
# axes[0,0].semilogy(x37,y37, '.-', label='ADAPT')
# axes[0,0].set_ylabel('E(iterations)-E(FCI) Hartree',  fontsize=12)
# axes[0,0].set_ylim((1e-14,1))
# axes[0,0].annotate('(a)',  fontsize=12, xy=(axes[0,0].get_xlim()[1]*0.85,axes[0,0].get_ylim()[1]*0.02))

# axes[1,0].semilogy(CXq,y4, '.-', label='qubit ADAPT')
# axes[1,0].semilogy(CXGSD,y8, '.-', label='ADAPT')
# axes[1,0].set_ylabel('E(iterations)-E(FCI) Hartree',  fontsize=12)
# plt.sca(axes[1, 0])
# plt.xticks((0,1000,2000),('0','1','2'))
# axes[1,0].set_ylim((1e-14,1))
# axes[1,0].annotate('(d)',  fontsize=12, xy=(axes[1,0].get_xlim()[1]*0.85,axes[1,0].get_ylim()[1]*0.02))

# axes[0,1].semilogy(lih_q_x,lih_q_y, '.-', label='qubit ADAPT')
# axes[0,1].semilogy(lih_f_x,lih_f_y, '.-', label='fermionic ADAPT')
# axes[0,1].set_xlabel('iterations',  fontsize=12)
# axes[0,1].set_yticklabels([])
# axes[0,1].set_ylim((1e-14,1))
# axes[0,1].annotate('(b)',  fontsize=12, xy=(axes[0,1].get_xlim()[1]*0.85,axes[0,1].get_ylim()[1]*0.02))
# axes[0,1].legend(loc='lower right',  fontsize=12)

# axes[1,1].semilogy(n_gate, lih_q_y, '.-', label='qubit ADAPT')
# axes[1,1].semilogy(CX3, lih_f_y, '.-', label='ADAPT')
# axes[1,1].set_xlabel('number of CNOTs ('r'$10^{3}$'')', fontsize=12)
# axes[1,1].set_yticklabels([])
# plt.sca(axes[1, 1])
# plt.xticks((0,3000,6000),('0','3','6'))
# axes[1,1].set_ylim((1e-14,1))
# axes[1,1].annotate('(e)',  fontsize=12, xy=(axes[1,1].get_xlim()[1]*0.85,axes[1,1].get_ylim()[1]*0.02))

# axes[0,2].semilogy(h6_q_z,h6_q_y, '.-', label='qubit ADAPT')
# axes[0,2].semilogy(h6_f_z,h6_f_y, '.-', label='ADAPT')
# axes[0,2].set_yticklabels([])
# axes[0,2].set_ylim((1e-14,1))
# axes[0,2].annotate('(c)',  fontsize=12, xy=(axes[0,2].get_xlim()[1]*0.85,axes[0,2].get_ylim()[1]*0.02))

# axes[1,2].semilogy(n_gate_h6, h6_q_y, '.-', label='qubit ADAPT')
# axes[1,2].semilogy(CX_h6, h6_f_y, '.-', label='ADAPT')
# axes[1,2].set_yticklabels([])
# plt.sca(axes[1, 2])
# plt.xticks((0,10000,20000,30000),('0','10','20','30'))
# axes[1,2].set_ylim((1e-14,1))
# axes[1,2].annotate('(f)',  fontsize=12, xy=(axes[1,2].get_xlim()[1]*0.85,axes[1,2].get_ylim()[1]*0.02))

# fig, axes = plt.subplots(1,2)
# fig.subplots_adjust(wspace=0.05)
# axes[0].semilogy(x36,y36, '.-', label='qubit ADAPT')
# axes[0].semilogy(x27,y27, 'c-', alpha=0.3)
# axes[0].semilogy(x28,y28, 'c-', alpha=0.3)
# axes[0].semilogy(x29,y29, 'c-', alpha=0.3)
# axes[0].semilogy(x30,y30, 'c-', alpha=0.3)
# axes[0].semilogy(x31,y31, 'c-', alpha=0.3)
# axes[0].semilogy(x32,y32, 'c-', alpha=0.3)
# axes[0].semilogy(x33,y33, 'c-', alpha=0.3)
# axes[0].semilogy(x34,y34, 'c-', alpha=0.3, label='random ordering')
# axes[0].set_title(r'$H_4$')
# axes[0].set_xlabel('iterations',  fontsize=12)
# axes[0].set_ylabel('E(iterations)-E(FCI) Hartree',  fontsize=12)
# axes[0].set_ylim((1e-14,1))
# axes[0].annotate('(a)',  fontsize=12, xy=(axes[0].get_xlim()[1]*0.85,axes[0].get_ylim()[1]*0.02))
# axes[0].legend(loc='lower center',  fontsize=12)

# axes[1].semilogy(lih_q_x,lih_q_y, '.-', label='qubit ADAPT')
# axes[1].semilogy(iter_lih_rand_1 ,lih_rand_1 , 'c-', alpha=0.3)
# axes[1].semilogy(iter_lih_rand_2 ,lih_rand_2 , 'c-', alpha=0.3)
# axes[1].semilogy(iter_lih_rand_3 ,lih_rand_3 , 'c-', alpha=0.3)
# axes[1].semilogy(iter_lih_rand_4 ,lih_rand_4 , 'c-', alpha=0.3)
# axes[1].semilogy(iter_lih_rand_5 ,lih_rand_5 , 'c-', alpha=0.3)
# axes[1].semilogy(iter_lih_rand_6 ,lih_rand_6 , 'c-', alpha=0.3)
# axes[1].semilogy(iter_lih_rand_7 ,lih_rand_7 , 'c-', alpha=0.3)
# axes[1].semilogy(iter_lih_rand_8 ,lih_rand_8 , 'c-', alpha=0.3)
# axes[1].semilogy(iter_lih_rand_9 ,lih_rand_9 , 'c-', alpha=0.3)
# axes[1].semilogy(iter_lih_rand_10,lih_rand_10, 'c-', alpha=0.3)
# axes[1].semilogy(iter_lih_rand_11,lih_rand_11, 'c-', alpha=0.3)
# axes[1].semilogy(iter_lih_rand_12,lih_rand_12, 'c-', alpha=0.3, label='random ordering')
# axes[1].set_xlabel('iterations',  fontsize=12)
# axes[1].set_title(r'$LiH$')
# axes[1].set_yticklabels([])
# axes[1].set_ylim((1e-14,1))
# axes[1].annotate('(b)',  fontsize=12, xy=(axes[1].get_xlim()[1]*0.85,axes[1].get_ylim()[1]*0.02))
# axes[1].legend(loc='lower right',  fontsize=12)

fig, axes = plt.subplots(1,2)
fig.subplots_adjust(wspace=0.2)
fig.suptitle(r'$H_4$',fontsize=14)
axes[0].semilogy(x36,y36, '.-', label='qubit ADAPT')
axes[0].semilogy(iters_hh_1,hh_1, 'g-', alpha=0.3)
axes[0].semilogy(iters_hh_2,hh_2, 'g-', alpha=0.3)
axes[0].semilogy(iters_hh_3,hh_3, 'g-', alpha=0.3)
axes[0].semilogy(iters_hh_4,hh_4, 'g-', alpha=0.3)
axes[0].semilogy(iters_hh_5,hh_5, 'g-', alpha=0.3)
axes[0].semilogy(iters_hh_6,hh_6, 'g-', alpha=0.3)
axes[0].semilogy(iters_hh_7,hh_7, 'g-', alpha=0.3)
axes[0].semilogy(iters_hh_8,hh_8, 'g-', alpha=0.3)
axes[0].semilogy(iters_hh_9,hh_9, 'g-', alpha=0.3)
axes[0].semilogy(iters_hh_12,hh_12, 'g-', alpha=0.3, label='qubit ADAPT 1/4 pool')
axes[0].semilogy(iters_hhhh_1,hhhh_1, 'r-', alpha=0.3)
axes[0].semilogy(iters_hhhh_2,hhhh_2, 'r-', alpha=0.3)
axes[0].semilogy(iters_hhhh_4,hhhh_4, 'r-', alpha=0.3)
axes[0].semilogy(iters_hhhh_5,hhhh_5, 'r-', alpha=0.3)
axes[0].semilogy(iters_hhhh_6,hhhh_6, 'r-', alpha=0.3)
axes[0].semilogy(iters_hhhh_7,hhhh_7, 'r-', alpha=0.3)
axes[0].semilogy(iters_hhhh_8,hhhh_8, 'r-', alpha=0.3)
axes[0].semilogy(iters_hhhh_9,hhhh_9, 'r-', alpha=0.3)
axes[0].semilogy(iters_hhhh_10,hhhh_10, 'r-', alpha=0.3)
axes[0].semilogy(iters_hhhh_12,hhhh_12, 'r-', alpha=0.3, label='qubit ADAPT 1/16 pool')
axes[0].set_xlabel('iterations',  fontsize=14)
axes[0].set_ylabel('E(iterations)-E(FCI) Hartree',  fontsize=14)
axes[0].set_ylim((1e-14,1))
axes[0].annotate('(a)',  fontsize=14, xy=(axes[0].get_xlim()[1]*0.85,axes[0].get_ylim()[1]*0.02))
axes[0].legend(loc='lower center',  fontsize=14)

axes[1].semilogy(iters_trim40,qubits_trim40, 'k.-', label='qubit ADAPT HF pool')
axes[1].semilogy(iters_trim40_1,trim40_rand_1, 'm-', alpha=0.3)
axes[1].semilogy(iters_trim40_2,trim40_rand_2, 'm-', alpha=0.3)
axes[1].semilogy(iters_trim40_3,trim40_rand_3, 'm-', alpha=0.3)
axes[1].semilogy(iters_trim40_4,trim40_rand_4, 'm-', alpha=0.3)
axes[1].semilogy(iters_trim40_5,trim40_rand_5, 'm-', alpha=0.3)
axes[1].semilogy(iters_trim40_6,trim40_rand_6, 'm-', alpha=0.3)
axes[1].semilogy(iters_trim40_7,trim40_rand_7, 'm-', alpha=0.3)
axes[1].semilogy(iters_trim40_8,trim40_rand_8, 'm-', alpha=0.3)
axes[1].semilogy(iters_trim40_9,trim40_rand_9, 'm-', alpha=0.3)
axes[1].semilogy(iters_trim40_10,trim40_rand_10, 'm-', alpha=0.3)
axes[1].semilogy(iters_trim40_11,trim40_rand_11, 'm-', alpha=0.3)
axes[1].semilogy(iters_trim40_12,trim40_rand_12, 'm-', alpha=0.3, label='HF pool random ordering')
axes[1].set_xlabel('iterations',  fontsize=14)
# axes[1].set_yticklabels([])
# axes[1].set_ylim((1e-14,1))
axes[1].annotate('(b)',  fontsize=14, xy=(axes[1].get_xlim()[1]*0.85,axes[1].get_ylim()[1]*0.5))
axes[1].legend(loc='center right',  fontsize=14)

# plt.subplot(1, 2, 1)
# plt.semilogy(x1,y1, '.-', label='qadapt with Zs 1e-3')
# plt.semilogy(x2,y2, '.-', label='qadapt without Zs 1e-3')
# plt.semilogy(x11,y11, '.-', label='qadapt group without Zs 1e-3')
# plt.semilogy(x14,y14, '.-', label='qadapt 2 without Zs 1e-4')
# plt.semilogy(x7,y7, '.-', label='GSD 1e-3')
# plt.semilogy(x5,y5, '.-', label='SD 1e-3')
# plt.semilogy(x6,y6, '.-', label='SD 1e-4')
# plt.semilogy(x8,y8, '.-', label='ADAPT old')
# plt.semilogy(x36,y36, '.-', label='qubit ADAPT')
# plt.semilogy(x37,y37, '.-', label='ADAPT')
# plt.semilogy(x38,y38, '.-', label='ADAPT energy step')
# plt.semilogy(x9,y9, '.-', label='GSD w/o Z 1e-9')
# plt.semilogy(x4,y4, '.-', label='qubit ADAPT old')
# plt.semilogy(iters_trim30,qubits_trim30, '.-', label='qubit ADAPT max 30 ops')
# plt.semilogy(iters_trim40,qubits_trim40, '.-', label='qubit ADAPT 40 operator HF pool')
# plt.semilogy(iters_trim50,qubits_trim50, '.-', label='qubit ADAPT max 50 ops')
# plt.semilogy(iters_trim60,qubits_trim60, '.-', label='qubit ADAPT max 60 ops')
# plt.semilogy(iters_trim_nonzero,qubits_trim_nonzero, '.-', label='qubit ADAPT non zero ops')
# plt.semilogy(x15[0:30],y15[0:30], '.-', label='Qubits w/ spin adapt 1e-9')
# plt.semilogy(x13,y13, '.-', label='Qubits w/ factorize  1e-4')
# plt.semilogy(x3,y3, '.-', label='Qubits w/ Zs 1e-4')
# plt.semilogy(x12,y12, 'k.-', label='GSD w/o spin adapt 1e-4')
# plt.semilogy(x26,y26, '.-', label='Qubits random')
# plt.semilogy(x27,y27, 'c-', alpha=0.3)
# plt.semilogy(x28,y28, 'c-', alpha=0.3)
# plt.semilogy(x29,y29, 'c-', alpha=0.3)
# plt.semilogy(x30,y30, 'c-', alpha=0.3)
# plt.semilogy(x31,y31, 'c-', alpha=0.3)
# plt.semilogy(x32,y32, 'c-', alpha=0.3)
# plt.semilogy(x33,y33, 'c-', alpha=0.3)
# plt.semilogy(x34,y34, 'c-', alpha=0.3, label='qubit ADAPT random ordering')
# plt.semilogy(x35,y35, '.-', label='qubit ADAPT energy step')
# plt.semilogy(iters_half,half_1, 'c-', alpha=0.3)
# plt.semilogy(iters_half,half_2, 'c-', alpha=0.3)
# plt.semilogy(iters_half,half_3, 'c-', alpha=0.3)
# plt.semilogy(iters_half,half_4, 'c-', alpha=0.3)
# plt.semilogy(iters_half,half_5, 'c-', alpha=0.3)
# plt.semilogy(iters_half,half_6, 'c-', alpha=0.3)
# plt.semilogy(iters_half,half_7, 'c-', alpha=0.3)
# plt.semilogy(iters_half,half_8, 'c-', alpha=0.3)
# plt.semilogy(iters_half,half_9, 'c-', alpha=0.3)
# plt.semilogy(iters_half,half_10, 'c-', alpha=0.3)
# plt.semilogy(iters_half,half_11, 'c-', alpha=0.3)
# plt.semilogy(iters_half,half_12, 'c-', alpha=0.3, label='qubit ADAPT random Half Pool (12 samples)')
# plt.semilogy(iters_hh_1,hh_1, 'g-', alpha=0.3)
# plt.semilogy(iters_hh_2,hh_2, 'g-', alpha=0.3)
# plt.semilogy(iters_hh_3,hh_3, 'g-', alpha=0.3)
# plt.semilogy(iters_hh_4,hh_4, 'g-', alpha=0.3)
# plt.semilogy(iters_hh_5,hh_5, 'g-', alpha=0.3)
# plt.semilogy(iters_hh_6,hh_6, 'g-', alpha=0.3)
# plt.semilogy(iters_hh_7,hh_7, 'g-', alpha=0.3)
# plt.semilogy(iters_hh_8,hh_8, 'g-', alpha=0.3)
# plt.semilogy(iters_hh_9,hh_9, 'g-', alpha=0.3)
# plt.semilogy(iters_hh_12,hh_12, 'g-', alpha=0.3, label='qubit ADAPT 1/4 Pool')
# plt.semilogy(iters_trim40_1,trim40_rand_1, 'g-', alpha=0.3)
# plt.semilogy(iters_trim40_2,trim40_rand_2, 'g-', alpha=0.3)
# plt.semilogy(iters_trim40_3,trim40_rand_3, 'g-', alpha=0.3)
# plt.semilogy(iters_trim40_4,trim40_rand_4, 'g-', alpha=0.3)
# plt.semilogy(iters_trim40_5,trim40_rand_5, 'g-', alpha=0.3)
# plt.semilogy(iters_trim40_6,trim40_rand_6, 'g-', alpha=0.3)
# plt.semilogy(iters_trim40_7,trim40_rand_7, 'g-', alpha=0.3)
# plt.semilogy(iters_trim40_8,trim40_rand_8, 'g-', alpha=0.3)
# plt.semilogy(iters_trim40_9,trim40_rand_9, 'g-', alpha=0.3)
# plt.semilogy(iters_trim40_10,trim40_rand_10, 'g-', alpha=0.3)
# plt.semilogy(iters_trim40_11,trim40_rand_11, 'g-', alpha=0.3)
# plt.semilogy(iters_trim40_12,trim40_rand_12, 'g-', alpha=0.3, label='40 operator HF pool random ordering')
# plt.semilogy(iters_hhh_1,hhh_1, 'r-', alpha=0.3)
# plt.semilogy(iters_hhh_2,hhh_2, 'r-', alpha=0.3)
# plt.semilogy(iters_hhh_3,hhh_3, 'r-', alpha=0.3)
# plt.semilogy(iters_hhh_4,hhh_4, 'r-', alpha=0.3)
# plt.semilogy(iters_hhh_5,hhh_5, 'r-', alpha=0.3)
# plt.semilogy(iters_hhh_6,hhh_6, 'r-', alpha=0.3)
# plt.semilogy(iters_hhh_7,hhh_7, 'r-', alpha=0.3)
# plt.semilogy(iters_hhh_8,hhh_8, 'r-', alpha=0.3)
# plt.semilogy(iters_hhh_9,hhh_9, 'r-', alpha=0.3)
# plt.semilogy(iters_hhh_10,hhh_10, 'r-', alpha=0.3)
# plt.semilogy(iters_hhh_11,hhh_11, 'r-', alpha=0.3)
# plt.semilogy(iters_hhh_12,hhh_12, 'r-', alpha=0.3, label='qubit ADAPT random 1/8 pool (11 samples)')
# plt.semilogy(iters_hhhh_1,hhhh_1, 'r-', alpha=0.3)
# plt.semilogy(iters_hhhh_2,hhhh_2, 'r-', alpha=0.3)
# plt.semilogy(iters_hhhh_4,hhhh_4, 'r-', alpha=0.3)
# plt.semilogy(iters_hhhh_5,hhhh_5, 'r-', alpha=0.3)
# plt.semilogy(iters_hhhh_6,hhhh_6, 'r-', alpha=0.3)
# plt.semilogy(iters_hhhh_7,hhhh_7, 'r-', alpha=0.3)
# plt.semilogy(iters_hhhh_8,hhhh_8, 'r-', alpha=0.3)
# plt.semilogy(iters_hhhh_9,hhhh_9, 'r-', alpha=0.3)
# plt.semilogy(iters_hhhh_10,hhhh_10, 'r-', alpha=0.3)
# plt.semilogy(iters_hhhh_12,hhhh_12, 'r-', alpha=0.3, label='qubit ADAPT 1/16 pool')
# plt.semilogy(iters_remove,remove_1, '.-', label='qubit ADAPT remove 1')
# plt.semilogy(iters_remove,remove_2, '.-', label='qubit ADAPT remove 2')
# plt.semilogy(iters_remove,remove_3, '.-', label='qubit ADAPT remove 3')
# plt.semilogy(iters_ham, qubits_ham, '.-', label='qubits ADAPT hamiltonian terms')
# plt.semilogy(iters_init, qubits_init, ',-', label='qubits ADAPT initial zero' )
# plt.semilogy(x10,y10, '.-', label='Qubits w/ multi select 1e-4')
# plt.semilogy(x16,y16, '.-', label='Qubits w/ fewer ops')
# plt.semilogy(x17,y17, '.-', label='hessian gp')
# plt.semilogy(x18,y18, '.-', label='hess -')
# plt.semilogy(x19,y19, '.-', label='GSD_trot_diff_1e-4')
# plt.semilogy(x21,y21, '.-', label='GSD_trot_same_new_1e-4')
# plt.semilogy(x20,y20, '.-', label='GSD_trot_same_old_1e-4')
# plt.semilogy(x24,y24, '.-', label='GSD_trot_same_old_1e-9')
# plt.semilogy(x22,y22, '.-', label='hessian qubits 1e-4')
# plt.semilogy(x23,y23, '.-', label='hessian GSD 1e-4')
# plt.semilogy(x25,y25, '.-', label='Qubits_A_1e-4')
# plt.title(r'$H_4$',  fontsize=14)
# plt.xlabel('iterations',  fontsize=14)
# plt.ylabel('E(iterations)-E(FCI) Hartree',  fontsize=14)
# # plt.xticks((0,2,4,6,8,10,12,14,16,18,20))
# plt.ylim((1e-14,1))
# plt.annotate('(a)',  fontsize=14, xy=(plt.gca().get_xlim()[1]*0.9,plt.gca().get_ylim()[1]*0.05))
# plt.legend(loc='center',  fontsize=14)

# plt.subplot(2, 1, 2)
# plt.semilogy(x36,z36, '.-', label='qubit ADAPT')
# plt.semilogy(iters_trim40,overlap_trim40, '.-', label='qubit ADAPT max 40 ops')
# plt.semilogy(iters_trim50,overlap_trim50, '.-', label='qubit ADAPT max 50 ops')
# plt.xlabel('iterations')
# plt.ylabel(r'1-$<GS|\theta>^2$')
# plt.legend()

# plt.subplot(2, 1, 2)
# # plt.semilogy(x1,z1, '.-', label='qadapt with Zs 1e-3')
# # plt.semilogy(x2,z2, '.-', label='qadapt without Zs 1e-3')
# # plt.semilogy(x11,z11, '.-', label='qadapt group without Zs 1e-3')
# # plt.semilogy(x14,z14, '.-', label='qadapt 2 without Zs 1e-4')
# # plt.semilogy(x7,z7, '.-', label='GSD 1e-3')
# # plt.semilogy(x5,z5, '.-', label='SD 1e-3')
# # plt.semilogy(x6,z6, '.-', label='SD 1e-4')
# # plt.semilogy(x8,z8, '.-', label='GSD 1e-4')
# # plt.semilogy(x12,z12, 'k.-', label='GSD w/o spin adapt 1e-4')
# # plt.semilogy(x9,z9, '.-', label='GSD w/o Z 1e-9')
# plt.semilogy(x4,z4, '.-', label='Qubits 1e-4')
# # plt.semilogy(x3,z3, '.-', label='Qubits w/ Zs 1e-4')
# # plt.semilogy(x15[0:30],z15[0:30], '.-', label='Qubits w/ spin adapt 1e-9')
# # plt.semilogy(x13,z13, '.-', label='Qubits w/ factorize  1e-4')
# # plt.semilogy(x10,z10, '.-', label='Qubits w/ multi select 1e-4')
# # plt.semilogy(x16,z16, '.-', label='Qubits w/ fewer ops')
# # plt.semilogy(x17,z17, '.-', label='hessian gp')
# # plt.semilogy(x18,z18, '.-', label='hess -')
# # plt.semilogy(x19,z19, '.-', label='GSD_trot_diff_1e-4')
# # plt.semilogy(x21,z21, '.-', label='GSD_trot_same_new_1e-4')
# # plt.semilogy(x20,z20, '.-', label='GSD_trot_same_old_1e-4')
# # plt.semilogy(x24,z24, '.-', label='GSD_trot_same_old_1e-9')
# # plt.semilogy(x22,z22, '.-', label='hessian qubits 1e-4')
# # plt.semilogy(x23,z23, '.-', label='hessian GSD 1e-4')
# plt.semilogy(x25,z25, '.-', label='Qubits_A_1e-4')
# plt.xlabel('iterations')
# plt.ylabel('Norm of Grad')
# plt.legend()

# plt.show()

# plt.subplot(2, 3, 4)
# plt.semilogy(CXq,y4, '.-', label='qubit ADAPT')
# plt.semilogy(CXGSD,y8, '.-', label='ADAPT')
# # # plt.semilogy(CXqf,y13, '.-', label='Qubits w/ factorize  1e-4')
# # # plt.semilogy(CXnS,y12, 'k.-', label='GSD w/o spin adapt 1e-4')
# # # plt.semilogy(CXqZ,y3, '.-', label='Qubits w/ Zs 1e-4')
# # # plt.semilogy(CXrand_average,y26, '.-', label='Qubits random')
# # # plt.title('H4')
# # plt.semilogy(CXrand0,y27, 'c-', alpha=0.3)
# # plt.semilogy(CXrand1,y28, 'c-', alpha=0.3)
# # plt.semilogy(CXrand2,y29, 'c-', alpha=0.3)
# # plt.semilogy(CXrand3,y30, 'c-', alpha=0.3)
# # plt.semilogy(CXrand4,y31, 'c-', alpha=0.3)
# # plt.semilogy(CXrand5,y32, 'c-', alpha=0.3)
# # plt.semilogy(CXrand6,y33, 'c-', alpha=0.3)
# # plt.semilogy(CXrand7,y34, 'c-', alpha=0.3, label='qubit-ADAPT Random Order')
# plt.ylabel('E(iterations)-E(FCI) Hartree')
# plt.xlabel('number of CNOTs')
# # plt.xticks(np.arange(0, 2400, step=200))
# plt.xticks((0,1000,2000),(r'$0$',r'$10^{3}$',r'$2\times 10^{3}$'))
# plt.ylim((1e-14,1))
# plt.annotate('(d)',  fontsize=12, xy=(plt.gca().get_xlim()[1]*0.9,plt.gca().get_ylim()[1]*0.05))
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.semilogy(lih_q_x,lih_q_y, '.-', label='qubit ADAPT')
# # # plt.semilogy(lih_f_x,lih_f_y, '.-', label='fermionic ADAPT')
# plt.semilogy(iter_lih_rand_1 ,lih_rand_1 , 'c-', alpha=0.3)
# plt.semilogy(iter_lih_rand_2 ,lih_rand_2 , 'c-', alpha=0.3)
# plt.semilogy(iter_lih_rand_3 ,lih_rand_3 , 'c-', alpha=0.3)
# plt.semilogy(iter_lih_rand_4 ,lih_rand_4 , 'c-', alpha=0.3)
# plt.semilogy(iter_lih_rand_5 ,lih_rand_5 , 'c-', alpha=0.3)
# plt.semilogy(iter_lih_rand_6 ,lih_rand_6 , 'c-', alpha=0.3)
# plt.semilogy(iter_lih_rand_7 ,lih_rand_7 , 'c-', alpha=0.3)
# plt.semilogy(iter_lih_rand_8 ,lih_rand_8 , 'c-', alpha=0.3)
# plt.semilogy(iter_lih_rand_9 ,lih_rand_9 , 'c-', alpha=0.3)
# plt.semilogy(iter_lih_rand_10,lih_rand_10, 'c-', alpha=0.3)
# plt.semilogy(iter_lih_rand_11,lih_rand_11, 'c-', alpha=0.3)
# plt.semilogy(iter_lih_rand_12,lih_rand_12, 'c-', alpha=0.3, label='qubit ADAPT random ordering')
# plt.title(r'$LiH$',  fontsize=14)
# plt.xlabel('iterations',  fontsize=14)
# # # frame1 = plt.gca()
# # # for ylabel_i in frame1.axes.get_yticklabels():
# # #     ylabel_i.set_fontsize(0.0)
# # #     ylabel_i.set_visible(False)
# plt.ylabel('E(iterations)-E(FCI) Hartree',  fontsize=14)
# # plt.ylim((1e-14,1))
# plt.annotate('(b)',  fontsize=14, xy=(plt.gca().get_xlim()[1]*0.9,plt.gca().get_ylim()[1]*0.2))
# plt.legend(loc='lower right',  fontsize=14)

# plt.subplot(2, 3, 5)
# plt.semilogy(n_gate, lih_q_y, '.-', label='qubit ADAPT')
# plt.semilogy(CX3, lih_f_y, '.-', label='ADAPT')
# plt.xlabel('number of CNOTs')
# frame1 = plt.gca()
# for ylabel_i in frame1.axes.get_yticklabels():
#     ylabel_i.set_fontsize(0.0)
#     ylabel_i.set_visible(False)
# # plt.ylabel('E(iterations)-E(FCI) Hartree')
# plt.xticks((0,3000,6000),(r'$0$',r'$3\times 10^{3}$',r'$6\times 10^{3}$'))
# plt.ylim((1e-14,1))
# plt.annotate('(e)',  fontsize=12, xy=(plt.gca().get_xlim()[1]*0.9,plt.gca().get_ylim()[1]*0.05))
# # # plt.legend()

# plt.subplot(2, 3, 3)
# plt.semilogy(h6_q_z,h6_q_y, '.-', label='qubit ADAPT')
# plt.semilogy(h6_f_z,h6_f_y, '.-', label='ADAPT')
# plt.title(r'$H_6$')
# plt.xlabel('iterations')
# frame1 = plt.gca()
# for ylabel_i in frame1.axes.get_yticklabels():
#     ylabel_i.set_fontsize(0.0)
#     ylabel_i.set_visible(False)
# # plt.ylabel('E(iterations)-E(FCI) Hartree')
# plt.ylim((1e-14,1))
# plt.annotate('(c)',  fontsize=12, xy=(plt.gca().get_xlim()[1]*0.9,plt.gca().get_ylim()[1]*0.05))
# # # plt.legend()

# plt.subplot(2, 3, 6)
# plt.semilogy(n_gate_h6, h6_q_y, '.-', label='qubit ADAPT')
# plt.semilogy(CX_h6, h6_f_y, '.-', label='ADAPT')
# plt.xlabel('number of CNOTs')
# frame1 = plt.gca()
# for ylabel_i in frame1.axes.get_yticklabels():
#     ylabel_i.set_fontsize(0.0)
#     ylabel_i.set_visible(False)
# # plt.ylabel('E(iterations)-E(FCI) Hartree')
# plt.xticks((0,10000,20000,30000),(r'$0$',r'$10^{4}$',r'$2\times 10^{4}$',r'$3\times 10^{4}$'))
# plt.ylim((1e-14,1))
# plt.annotate('(f)',  fontsize=12, xy=(plt.gca().get_xlim()[1]*0.9,plt.gca().get_ylim()[1]*0.05))
# # plt.legend()

plt.show()

# plt.subplot(2, 2, 1)
# # plt.semilogy(x1,y1, '.-', label='qadapt with Zs 1e-3')
# # plt.semilogy(x2,y2, '.-', label='qadapt without Zs 1e-3')
# # plt.semilogy(x11,y11, '.-', label='qadapt group without Zs 1e-3')
# # plt.semilogy(x14,y14, '.-', label='qadapt 2 without Zs 1e-4')
# # plt.semilogy(x7,y7, '.-', label='GSD 1e-3')
# # plt.semilogy(x5,y5, '.-', label='SD 1e-3')
# plt.semilogy(x6,y6, '.-', label='SD 1e-4')
# plt.semilogy(x8,y8, '.-', label='GSD 1e-4')
# # plt.semilogy(x12,y12, 'k.-', label='GSD w/o spin adapt 1e-4')
# # plt.semilogy(x9,y9, '.-', label='GSD w/o Z 1e-9')
# plt.semilogy(x4,y4, '.-', label='Qubits 1e-4')
# # plt.semilogy(x3,y3, '.-', label='Qubits w/ Zs 1e-4')
# plt.semilogy(x15[0:30],y15[0:30], '.-', label='Qubits w/ spin adapt 1e-9')
# # plt.semilogy(x13,y13, '.-', label='Qubits w/ factorize  1e-4')
# # plt.semilogy(x10,y10, '.-', label='Qubits w/ multi select 1e-4')
# plt.title('H4')
# plt.ylabel('E(iterations)-E(FCI) Hartree')
# plt.legend()

# plt.subplot(2, 2, 3)
# # plt.semilogy(x1,z1, '.-', label='qadapt with Zs 1e-3')
# # plt.semilogy(x2,z2, '.-', label='qadapt without Zs 1e-3')
# # plt.semilogy(x11,z11, '.-', label='qadapt group without Zs 1e-3')
# # plt.semilogy(x14,z14, '.-', label='qadapt 2 without Zs 1e-4')
# # plt.semilogy(x7,z7, '.-', label='GSD 1e-3')
# # plt.semilogy(x5,z5, '.-', label='SD 1e-3')
# plt.semilogy(x6,z6, '.-', label='SD 1e-4')
# plt.semilogy(x8,z8, '.-', label='GSD 1e-4')
# # plt.semilogy(x12,z12, 'k.-', label='GSD w/o spin adapt 1e-4')
# # plt.semilogy(x9,z9, '.-', label='GSD w/o Z 1e-9')
# plt.semilogy(x4,z4, '.-', label='Qubits 1e-4')
# # plt.semilogy(x3,z3, '.-', label='Qubits w/ Zs 1e-4')
# plt.semilogy(x15[0:30],z15[0:30], '.-', label='Qubits w/ spin adapt 1e-9')
# # plt.semilogy(x13,z13, '.-', label='Qubits w/ factorize  1e-4')
# # plt.semilogy(x10,z10, '.-', label='Qubits w/ multi select 1e-4')
# plt.xlabel('iterations')
# plt.ylabel('Norm of Grad')
# plt.legend()

# plt.subplot(2, 2, 2)
# # plt.semilogy(x1,y1, '.-', label='qadapt with Zs 1e-3')
# # plt.semilogy(x2,y2, '.-', label='qadapt without Zs 1e-3')
# # plt.semilogy(x11,y11, '.-', label='qadapt group without Zs 1e-3')
# # plt.semilogy(x14,y14, '.-', label='qadapt 2 without Zs 1e-4')
# # plt.semilogy(x7,y7, '.-', label='GSD 1e-3')
# # plt.semilogy(x5,y5, '.-', label='SD 1e-3')
# plt.semilogy(x6,y6, '.-', label='SD 1e-4')
# plt.semilogy(x8,y8, '.-', label='GSD 1e-4')
# # plt.semilogy(x12,y12, 'k.-', label='GSD w/o spin adapt 1e-4')
# plt.semilogy(x9,y9, '.-', label='GSD w/o Z 1e-9')
# # plt.semilogy(x4,y4, '.-', label='Qubits 1e-4')
# # plt.semilogy(x3,y3, '.-', label='Qubits w/ Zs 1e-4')
# # plt.semilogy(x15[0:30],y15[0:30], '.-', label='Qubits w/ spin adapt 1e-9')
# # plt.semilogy(x13,y13, '.-', label='Qubits w/ factorize  1e-4')
# # plt.semilogy(x10,y10, '.-', label='Qubits w/ multi select 1e-4')
# plt.title('H4')
# plt.ylabel('E(iterations)-E(FCI) Hartree')
# plt.legend()

# plt.subplot(2, 2, 4)
# # plt.semilogy(x1,z1, '.-', label='qadapt with Zs 1e-3')
# # plt.semilogy(x2,z2, '.-', label='qadapt without Zs 1e-3')
# # plt.semilogy(x11,z11, '.-', label='qadapt group without Zs 1e-3')
# # plt.semilogy(x14,z14, '.-', label='qadapt 2 without Zs 1e-4')
# # plt.semilogy(x7,z7, '.-', label='GSD 1e-3')
# # plt.semilogy(x5,z5, '.-', label='SD 1e-3')
# plt.semilogy(x6,z6, '.-', label='SD 1e-4')
# plt.semilogy(x8,z8, '.-', label='GSD 1e-4')
# # plt.semilogy(x12,z12, 'k.-', label='GSD w/o spin adapt 1e-4')
# plt.semilogy(x9,z9, '.-', label='GSD w/o Z 1e-9')
# # plt.semilogy(x4,z4, '.-', label='Qubits 1e-4')
# # plt.semilogy(x3,z3, '.-', label='Qubits w/ Zs 1e-4')
# # plt.semilogy(x15[0:30],z15[0:30], '.-', label='Qubits w/ spin adapt 1e-9')
# # plt.semilogy(x13,z13, '.-', label='Qubits w/ factorize  1e-4')
# # plt.semilogy(x10,z10, '.-', label='Qubits w/ multi select 1e-4')
# plt.xlabel('iterations')
# plt.ylabel('Norm of Grad')
# plt.legend()

# plt.show()

# plt.plot(numpars,pars0,label='par1')
# plt.plot(numpars[1:31],pars1[0:-1], label='par2')
# plt.plot(numpars[2:31],pars2[0:-2], label='par3')
# plt.plot(numpars[3:31],pars3[0:-3], label='par4')
# plt.plot(numpars[4:31],pars4[0:-4], label='par5')
# plt.title('H4 parameters')
# plt.xlabel('iterations')
# plt.legend()
# plt.show()