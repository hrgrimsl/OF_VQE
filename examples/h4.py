import vqe_methods 
import operator_pools

r = 1.5
geometry = [('H', (0,0,1*r)), ('H', (0,0,2*r)), ('H', (0,0,3*r)), ('H', (0,0,4*r))]

vqe_methods.adapt_vqe(geometry,
	                  adapt_thresh    = 1e-7,                        #gradient threshold
                      theta_thresh    = 1e-10,                     #optimization threshold
                      adapt_maxiter   = 400,                       #maximum number of ops
                      selection       = 'grad',                    #way of selecting ops: grad or random
                      rand_ham        = False,                     #random hamiltonian
                      mapping         = 'jw',                      #mapping, jw or bk
                      n               = 2,                         #number of qubits, only used when rand_ham == True
                      pool            = operator_pools.singlet_GSD(),       #choice of pool
                      random_prod     = False,                     #random product initial state or not
                      sec_deriv       = False,                     #use second derivative and single parameter vqe to select ops 
                      analy_grad      = True,                      #use analytic gradient for BFGS 
                      init_para       = 1,
                      rand_Z          = False)                         #initial value of theta for single parameter vqe
