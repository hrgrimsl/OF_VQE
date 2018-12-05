from sys import argv
import os
script, first = argv

file = open(str(first)+'2.log', 'w')
file.close()
for i in range(5, 10):
    os.system('python OF_VQE.py -l '+str(first)+'2.log -m INFO -rw a -p increasing_comms -sys '+str(first)+' -d '+str(i))
    os.system('python OF_VQE.py -l '+str(first)+'2.log -m INFO -rw a -p decreasing_comms -sys '+str(first)+' -d '+str(i))
    for j in range(0, 100):
        os.system('python OF_VQE.py -l '+str(first)+'2.log -m INFO -rw a -s '+str(j)+' -p random -sys '+str(first)+' -d '+str(i/10))
