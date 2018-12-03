from sys import argv
import os
script, first = argv

file = open(str(first)+'.log', 'w')
file.close()
for i in range(10, 15):
    os.system('python OF_VQE.py -l '+str(first)+'3.log -m INFO -rw a -p increasing_unexp_comms -sys '+str(first)+' -d '+str(i/10))
    os.system('python OF_VQE.py -l '+str(first)+'3.log -m INFO -rw a -p decreasing_unexp_comms -sys '+str(first)+' -d '+str(i/10))
    os.system('python OF_VQE.py -l '+str(first)+'3.log -m INFO -rw a -p increasing_comms -sys '+str(first)+' -d '+str(i/10))
    os.system('python OF_VQE.py -l '+str(first)+'3.log -m INFO -rw a -p decreasing_comms -sys '+str(first)+' -d '+str(i/10))
    for j in range(0, 100):
        os.system('python OF_VQE.py -l '+str(first)+'3.log -m INFO -rw a -s '+str(j)+' -p random -sys '+str(first)+' -d '+str(i))
