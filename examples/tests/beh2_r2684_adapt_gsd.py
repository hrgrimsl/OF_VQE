import vqe_methods 
import operator_pools

import sys

r = 2.684
geometry = [('H',   (0,0,-r)), 
            ('Be',  (0,0,0)), 
            ('H',   (0,0,r))]


filename = "beh2_r2684_adapt_gsd.out"

sys.stdout = open(filename, 'w')

vqe_methods.adapt_vqe(geometry,
        pool = operator_pools.singlet_GSD(),
        adapt_thresh = 1e-5)
