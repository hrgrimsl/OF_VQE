import vqe_methods 
import operator_pools

r = 1.5
geometry = [('H', (0,0,1*r)), ('H', (0,0,2*r)), ('H', (0,0,3*r)), ('H', (0,0,4*r))]


model = vqe_methods.adapt_vqe(geometry,adapt_thresh = 1e-1, pool = operator_pools.singlet_SD())
vqe_methods.compute_pt2(model)
