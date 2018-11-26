import os
import logging

logging.basicConfig(filename='water.log', filemode='w', format='%(message)s')
logging.getLogger().setLevel(logging.INFO)
r1 = .763239
r2 = -.477047
for mesh in range(0, 1):
    r1+=.1*.763239
    r2-=.1*.477047
    for j in range(0, 100):
        os.system('python OF_VQE.py '+str(r1)+' '+str(r2))
