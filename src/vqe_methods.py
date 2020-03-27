import scipy
import openfermion
import openfermionpsi4
import os
import numpy as np
import copy
import random 
import sys
import csv
# import cirq
# import openfermioncirq
# from openfermioncirq import trotter
import scipy.sparse.linalg

import operator_pools
import vqe_methods
from tVQE import *
import pickle

# from  openfermionprojectq  import  uccsd_trotter_engine, TimeEvolution
# from  projectq.backends  import CommandPrinter

from openfermion import *


def adapt_vqe(geometry,
        basis           = "sto-3g",
        multiplicity    = 1,
        charge          = 0,
        adapt_conver    = 'norm',
        adapt_thresh    = 1e-7,
        theta_thresh    = 1e-10,
        adapt_maxiter   = 400,
        pool            = operator_pools.singlet_GSD(),
        spin_adapt      = True,
        selection       = 'grad',
        rand_ham        = False,
        mapping         = 'jw'
        ):
# {{{
       
    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
    molecule = openfermionpsi4.run_psi4(molecule, 
                run_scf = 1, 
                run_mp2=1, 
                run_cisd=0, 
                run_ccsd = 0, 
                run_fci=1, 
                delete_input=1)

    print(" adapt threshold:", adapt_thresh)
    print(" theta threshold:", theta_thresh)
    print(" operator selection:", selection)
    print(" mapping:", mapping)
    
    pool.init(molecule)
    print(" Basis: ", basis)

    print(' HF energy      %20.16f au' %(molecule.hf_energy))
    print(' MP2 energy     %20.16f au' %(molecule.mp2_energy))
    #print(' CISD energy    %20.16f au' %(molecule.cisd_energy))
    #print(' CCSD energy    %20.16f au' %(molecule.ccsd_energy))
    print(' FCI energy     %20.16f au' %(molecule.fci_energy))

    #JW transform Hamiltonian computed classically with OFPsi4
    hamiltonian_op = molecule.get_molecular_hamiltonian()
    if mapping == 'jw':
        hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)
    if mapping == 'bk':
        hamiltonian_f = openfermion.transforms.get_fermion_operator(hamiltonian_op)
        hamiltonian_bk = openfermion.transforms.bravyi_kitaev(hamiltonian_f)
        hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_bk)
        # for i in range(2 ** molecule.n_qubits):
        #     if abs(hamiltonian[i, i]-molecule.hf_energy) < 1e-3:
        #         print(i)
        #         print(hamiltonian[i, i])

    #Thetas
    parameters = []
    # parameters_index = []
    # parameters_mult = []
    # Sign = []

    pool.generate_SparseMatrix()

    # over_mat = np.zeros(shape=(pool.n_ops, pool.n_ops))
    # vec = np.random.rand(2 ** pool.n_spin_orb, 1)
    # # print(vec)
    # norm = 0

    # for i in vec:
    #     norm += i * i

    # vec = np.true_divide(vec, np.sqrt(norm))
    # vec = scipy.sparse.csc_matrix(vec)

    # for i in range(pool.n_ops):
    #     for j in range(pool.n_ops):
    #         over_mat[i, j] = abs(vec.transpose().conjugate().dot(pool.spmat_ops[i].dot(pool.spmat_ops[j].dot(vec)))[0,0])

    # rank = np.linalg.matrix_rank(over_mat)

    # print("rank =", rank)
   
    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz
    # ansatz_ops_t = []
    # ansatz_mat_t = []
    # ansatz_sup_index = []
    # ansatz_left = []
    # ansatz_right = []
    # ansatz_scal = []
    
    #Build p-h reference and map it to JW transform
    if mapping == 'jw':
        reference_ket = scipy.sparse.csc_matrix(
        openfermion.jw_configuration_state(
        list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()
    if mapping == 'bk':
        jw_state = np.zeros((1, molecule.n_qubits))
        for i in range(molecule.n_electrons):
            jw_state[0][molecule.n_qubits-i-1] = 1
        print(jw_state)
        jw_to_bk = []
        M = [1]
        for i in range(1, 20):
            M = np.kron(np.eye(2), M)
            M[0][:] = 1
            # M[0:2 ** i] = np.ones((1, 2 ** i))
            print(M)
            if 2 ** i == molecule.n_qubits:
                jw_to_bk = M
                break
            if 2 ** i > molecule.n_qubits:
                jw_to_bk_1 = M[0:molecule.n_qubits]
                for k in jw_to_bk_1:
                    jw_to_bk.append(k[0:molecule.n_qubits])
    
        print(jw_to_bk)
    
        bk_state = []
    
        for i in range(len(jw_state[0])):
            element = sum(jw_state[0][k] * jw_to_bk[i][k] for k in range(len(jw_state[0]))) % 2
            bk_state.insert(0, element)
    
        print(bk_state)
    
        bk_vec = np.zeros((1, 2 ** molecule.n_qubits))
        bk_vec[0][int("".join(str(int(x)) for x in bk_state), 2)] = 1
        bk = bk_vec.tolist()
    
        print(bk)
    
        # Build p-h reference and map it to JW transform
        reference_ket = scipy.sparse.csc_matrix(bk).transpose()


    # for n in range(molecule.n_electrons, molecule.n_qubits):
    # 	reference_ket = 1/np.sqrt(2)*(np.exp(1j*0.5)*-1j*pool.spmat_ops[n].dot(reference_ket) + reference_ket)
    # 	print(pool.fermi_ops[n])
    
    # reference_ket = 1/np.sqrt(2)*(np.exp(1j*0.5)*-1j*pool.spmat_ops[3].dot(reference_ket) + reference_ket)

    print(reference_ket)

    reference_bra = reference_ket.transpose().conj()

    print(" Start ADAPT-VQE algorithm")
    op_indices = list(range(pool.n_ops))
    parameters = []

    curr_state = 1.0*reference_ket

    # for n in range(16,len(pool.spmat_ops)):
    # 	curr_state = 1/np.sqrt(2)*(pool.spmat_ops[n].dot(curr_state) + curr_state)
    # 	print(pool.fermi_ops[n])

    # curr_state = 1/np.sqrt(2)*(pool.spmat_ops[49].dot(curr_state) + curr_state)

    fermi_ops = pool.fermi_ops
    spmat_ops = pool.spmat_ops
    n_ops = pool.n_ops

    # Num = QubitOperator('Z1', 0)

    # for p in range(0,2*pool.n_orb):
    #     Num += QubitOperator('Z%d' %p)

    # Num = openfermion.transforms.get_sparse_operator(Num, n_qubits = pool.n_spin_orb)

    print(" Now start to grow the ansatz")
    for n_iter in range(0,adapt_maxiter):
    
        print("\n\n\n")
        print(" --------------------------------------------------------------------------")
        print("                         ADAPT-VQE iteration: ", n_iter)                 
        print(" --------------------------------------------------------------------------")
        next_index = None
        next_deriv = 0
        curr_norm = 0
        
        print(" Check each new operator for coupling")
        next_term = []
        group = []
        print(" Measure commutators:")
        sig = hamiltonian.dot(curr_state)

        # N_el = curr_state.transpose().conj().dot(Num.dot(curr_state))
        # print('Number of electron:', N_el)

        for op_trial in range(pool.n_ops):

            opA = pool.spmat_ops[op_trial]
            com = 2*(curr_state.transpose().conj().dot(opA.dot(sig))).real
            assert(com.shape == (1,1))
            com = com[0,0]
            assert(np.isclose(com.imag,0))
            com = com.real

            opstring = ""
            for t in pool.fermi_ops[op_trial].terms:
                opstring += str(t)
                break
        
            if abs(com) > adapt_thresh:
                print(" %4i %40s %12.8f" %(op_trial, opstring, com) )

            curr_norm += com*com

            if selection == 'grad':
                if abs(com) > abs(next_deriv) + 1e-9:
                    next_deriv = com
                    next_index = op_trial
            if selection == 'rand':
                next_index = random.choice(op_indices)
    
        curr_norm = np.sqrt(curr_norm)

        min_options = {'gtol': theta_thresh, 'disp':False}

        max_of_com = next_deriv
        print(" Norm of <[A,H]> = %12.8f" %curr_norm)
        print(" Max  of <[A,H]> = %12.8f" %max_of_com)

        converged = False
        if adapt_conver == "norm":
            if curr_norm < adapt_thresh:
                converged = True
        else:
            print(" FAIL: Convergence criterion not defined")
            exit()

        if converged:
            print(" Ansatz Growth Converged!")
            print(" Number of operators in ansatz: ", len(ansatz_ops))
            print(" *Finished: %20.12f" % trial_model.curr_energy)
            print(" -----------Final ansatz----------- ")
            print(" %4s %30s %12s" %("Term","Coeff","#"))
            for si in range(len(ansatz_ops)):
                s = ansatz_ops[si]
                print(" %4s %20f %10s" %(s, parameters[si], si) )
                print(" ")
            new_state = reference_ket
            E_step = []
            for k in reversed(range(0, len(parameters))):
                new_state = scipy.sparse.linalg.expm_multiply((parameters[k]*ansatz_mat[k]), new_state)
                E_step.append(new_state.transpose().conj().dot(hamiltonian.dot(new_state))[0,0].real)
                print(len(parameters))
                print(k)
                print('Energy step', float(E_step[len(parameters)-k-1]))
                print("")

                # compiler_engine = uccsd_trotter_engine( compiler_backend=CommandPrinter() )
                # wavefunction = compiler_engine.allocate_qureg(molecule.n_qubits)
        
                # H = openfermion.transforms.jordan_wigner(1j*s)  # fermionic pool
        
                # # Trotter step parameters.
                # time = parameters[si]
        
                # evolution_operator = TimeEvolution(time,H)
        
                # evolution_operator | wavefunction
        
                # compiler_engine.flush()

            break

            break

        new_op = pool.fermi_ops[next_index]
        new_mat = pool.spmat_ops[next_index]

        # for n in range(len(group)):
        #     new_op += Sign[n]*pool.fermi_ops[group[n]]
        #     new_mat += Sign[n]*pool.spmat_ops[group[n]]

        print(" Add operator %4i" %next_index)

        # for n in range(n_iter):
        # 	parameters[n] = 0

        # for n in group:
        #     print(" Add operator %4i " %n)

        parameters.insert(0,0)
        # parameters_index.append(n_iter)
        ansatz_ops.insert(0,new_op)
        ansatz_mat.insert(0,new_mat)
        # parameters_mult.insert(0,1)
        
        trial_model = tUCCSD(hamiltonian, ansatz_mat, ansatz_ops, reference_ket, parameters)
        

        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient, 
                options = min_options, method = 'BFGS', callback=trial_model.callback)

        # print(ansatz_ops)
    
        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        # print(" new state ",curr_state)
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(" -----------New ansatz----------- ")
        # print(curr_state)
        print(" %4s %30s %12s" %("Term","Coeff","#"))
        for si in range(len(ansatz_ops)):
                s = ansatz_ops[si]
                print(" %4s %20f %10s" %(s, parameters[si], si) )
                print(" ")