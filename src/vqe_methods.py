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
import re
from qiskit.aqua.operators import WeightedPauliOperator
import dill as pickle

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
        mapping         = 'jw',
        n               = 2,
        random_prod     = True,
        sec_deriv       = True,
        analy_grad      = True,
        init_para       = 0,
        imp_ham         = None,
        imp_n           = 8,
        hubbard         = False,
        t               = 1,
        U               = 1,
        pre_opt_initial = False,
        rand_Z          = False
        ):
# {{{

    print(" selection :", selection)
    print(" mapping:", mapping)
       
    if rand_ham:
        ham = np.random.rand(2 ** n, 2 ** n)
        ham_sym = ham + ham.transpose()
        hamiltonian = scipy.sparse.csc_matrix(ham_sym)

        pool.init_rand(n)

        reference_ket = pool.vec

    elif imp_ham != None:
        data_ben = pickle.load(open(imp_ham+'.p','rb'))
        exact_energy = {}
        exact_energy[None] = {}
        exact_energy[None]['energy'] = -227.94495772514333
        qubitOp = WeightedPauliOperator.from_dict(data_ben['hamiltonian'][None])
        s_op_qb = WeightedPauliOperator.from_dict(data_ben['aux_ops']['aux_ops'][0])
        sz_op_qb = WeightedPauliOperator.from_dict(data_ben['aux_ops']['aux_ops'][1])
        n_op_qb = WeightedPauliOperator.from_dict(data_ben['aux_ops']['aux_ops'][2])
        
        pauli_list = qubitOp.paulis
        ham_term = np.zeros((4 ** imp_n,))
        for i in pauli_list:
            Bin = np.zeros((2 * imp_n,), dtype=int)
            I = 0
            for j in range(imp_n):
                if str(i[1])[j] == 'Y':
                    Bin[j] = 1
                    Bin[j + imp_n] = 1
                if str(i[1])[j] == 'Z':
                    Bin[j] = 1
                if str(i[1])[j] == 'X':
                    Bin[j + imp_n] = 1

            index = int("".join(str(x) for x in Bin), 2)

            ham_term[index] = i[0].real

        nz = np.nonzero(ham_term)[0]

        m = 2 * imp_n

        hamiltonian_op = QubitOperator('X1', 0)

        for i in nz:
            p = int(i)
            bi = bin(p)
            b_string = [int(j) for j in bi[2:].zfill(m)]
            pauli_string = ''
            for k in range(imp_n):
                if b_string[k] == 0:
                    if b_string[k + imp_n] == 1:
                        pauli_string += 'X%d ' % k
                if b_string[k] == 1:
                    if b_string[k + imp_n] == 1:
                        pauli_string += 'Y%d ' % k
                if b_string[k] == 1:
                    if b_string[k + imp_n] == 0:
                        pauli_string += 'Z%d ' % k
            hamiltonian_op += QubitOperator(pauli_string, ham_term[p])

        hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)
        print(hamiltonian)
        HF = 0
        for i in range(2 ** imp_n):
            if hamiltonian[i, i] < HF:
                HF = hamiltonian[i, i]
                index = i
    
        vec = np.zeros((1, 2 ** imp_n))
        vec[0][index] = 1
        state = vec.tolist()
        # Build p-h reference and map it to JW transform
        reference_ket = scipy.sparse.csc_matrix(state).transpose()

        pool.init_rand(imp_n)

    elif hubbard:

        print('adapt hubbard')
        print('parameter:t=',t,', U=',U)
        print('number of sites:', n)

        hamiltonian_op = FermionOperator(((1,1),(0,0)), 0)
        ham_one = FermionOperator(((1,1),(0,0)), 0)
        ham_two = FermionOperator(((1,1),(0,0)), 0)

        for p in range(0,n):
            pa = 2*p
            pb = 2*p+1

            termB  = FermionOperator(((pa,1),(pa,0)), -1/2)
            termB += FermionOperator(((pb,1),(pb,0)), -1/2)
            termB += FermionOperator(((pa,1),(pa,0),(pb,1),(pb,0)), 1)

            termB = U * termB

            hamiltonian_op += termB
 
            if p != n-1:

                termA =  FermionOperator(((pa,1),(pa+2,0)))
                termA += FermionOperator(((pb,1),(pb+2,0)))
 
                termA += hermitian_conjugated(termA)
               
                termA = -t * termA

                hamiltonian_op += termA

        print(hamiltonian_op)
        hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)
        hamiltonian_jw = openfermion.transforms.jordan_wigner(hamiltonian_op)
        print(hamiltonian_jw)

        HF = 100
        for i in range(4 ** n):
            if hamiltonian[i, i].real < HF:
                HF = hamiltonian[i, i].real
                index = i

        group = []
        group.append(i)

        for i in range(4 ** n):
            if abs(hamiltonian[i, i]-HF) < 1e-9:
                if i != index:
                    group.append(i)

        vec = np.zeros((1, 4 ** n))
        for i in group:
            vec[0][i] = 1/np.sqrt(len(group))
        state = vec.tolist()
        # Build p-h reference and map it to JW transform
        reference_ket = scipy.sparse.csc_matrix(state).transpose()

        pool.init_hubb(n)

    else:
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
        if random_prod:
            print(" initial state: random product state")
        else:
            print(" initial state: HF state")
        if rand_ham:
            print(" random hamiltonian")
        
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
            hamiltonian_f = openfermion.transforms.get_fermion_operator(hamiltonian_op)
            hamiltonian_jw = openfermion.transforms.jordan_wigner(hamiltonian_f)
            print(hamiltonian_jw)
            hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)
            reference_ket = scipy.sparse.csc_matrix(
            openfermion.jw_configuration_state(
            list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()
        if mapping == 'bk':
            hamiltonian_f = openfermion.transforms.get_fermion_operator(hamiltonian_op)
            hamiltonian_bk = openfermion.transforms.bravyi_kitaev(hamiltonian_f)
            print(hamiltonian_bk)
            hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_bk)
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
                    break
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

    if random_prod:
        state = []
        th = random.uniform(0, 2 * math.pi)
        state.append(math.cos(th))
        state.append(math.sin(th))
        for i in range(pool.num-1):
            state_1 = []
            th_1 = random.uniform(0, 2 * math.pi)
            state_1.append(math.cos(th_1))
            state_1.append(math.sin(th_1))

            state = np.kron(state, state_1)

        reference_ket = scipy.sparse.csc_matrix(state).transpose()

    # if sing_diag:
    #     hamil

    if pre_opt_initial:
        vec = np.zeros((1, 4 ** n))
        state = vec.tolist()
        reference_ket = scipy.sparse.csc_matrix(state).transpose()

        pre_opt_mat = []
        pre_opt_ops = []

        pre_opt_para = []

        for i in range(2 * n):
            A = QubitOperator('Y%d' %i, 1j)
            pre_opt_ops.append(A)
            A_mat = openfermion.transforms.get_sparse_operator(A, n_qubits = 2 * n)
            pre_opt_mat.append(A_mat)

            pre_opt_para.append(0)

        trial_model = tUCCSD(hamiltonian, pre_opt_mat, pre_opt_ops, reference_ket, pre_opt_para)
        opt_result = scipy.optimize.minimize(trial_model.energy, pre_opt_para,
                                                             method='Nelder-Mead')

        pre_opt_para = list(opt_result['x'])
        print(pre_opt_para)
        reference_ket = trial_model.prepare_state(pre_opt_para)

    print('reference state:', reference_ket)

    w, v = scipy.sparse.linalg.eigs(hamiltonian, which='SR')
    GS = scipy.sparse.csc_matrix(v[:,w.argmin()]).transpose().conj()
    GSE = min(w).real
    print('Ground state energy:', GSE)

    #Thetas
    parameters = []

    pool.generate_SparseMatrix()
   
    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz
    
    # print(reference_ket)

    if rand_Z:
        for n in range(molecule.n_qubits):
            A = QubitOperator('Z%d' %n, 1j)
            A_mat = openfermion.transforms.get_sparse_operator(A, n_qubits = molecule.n_qubits)
            ansatz_ops.append(A)
            ansatz_mat.append(A_mat)
            parameters.append(random.uniform(0, 2 * math.pi))

            trial_model = tUCCSD(hamiltonian, ansatz_mat, ansatz_ops, reference_ket, parameters)
            reference_ket = trial_model.prepare_state(parameters)

    reference_bra = reference_ket.transpose().conj()
    E = reference_bra.dot(hamiltonian.dot(reference_ket))[0,0].real
    print('initial energy', E)

    print(" Start ADAPT-VQE algorithm")
    op_indices = list(range(pool.n_ops))

    curr_state = 1.0*reference_ket

    fermi_ops = pool.fermi_ops
    spmat_ops = pool.spmat_ops
    n_ops = pool.n_ops

    print(" Now start to grow the ansatz")
    for n_iter in range(0,adapt_maxiter):
    
        print("\n\n\n")
        print(" --------------------------------------------------------------------------")
        print("                         ADAPT-VQE iteration: ", n_iter)                 
        print(" --------------------------------------------------------------------------")
        next_index = None
        next_deriv = 0
        next_sec = 0
        curr_norm = 0
        curr_sec = 0
        
        print(" Check each new operator for coupling")
        next_term = []
        group = []
        print(" Measure commutators:")
        sig = hamiltonian.dot(curr_state)

        for op_trial in range(pool.n_ops):

            opA = pool.spmat_ops[op_trial]
            com = 2*(curr_state.transpose().conj().dot(opA.dot(sig))).real
            assert(com.shape == (1,1))
            com = com[0,0]
            assert(np.isclose(com.imag,0))
            com = com.real

            print(" %4i %40s %12.8f" %(op_trial, pool.fermi_ops[op_trial], com) )

            curr_norm += com*com

            if selection == 'grad':
                if abs(com) > abs(next_deriv) + 1e-9:
                    next_deriv = com
                    next_index = op_trial
            if selection == 'rand':
                next_index = random.choice(op_indices)
    
        curr_norm = np.sqrt(curr_norm)

        if curr_norm < adapt_thresh:
            if sec_deriv == True:
                print("------------------------------------------------------------")
                print("Second derivatives:")
                for op_trial in range(pool.n_ops):
                    opA = pool.spmat_ops[op_trial]
                    sig_1 = opA.dot(curr_state)
                    sec = -2*(sig_1.transpose().conj().dot(opA.dot(sig))).real
                    sec += 2*(sig_1.transpose().conj().dot(hamiltonian.dot(sig_1))).real
                    assert(sec.shape == (1,1))
                    sec = sec[0,0]
                    assert(np.isclose(sec.imag,0))
                    sec = sec.real

                    trial_model = tUCCSD(hamiltonian, ansatz_mat, ansatz_ops, reference_ket, parameters)
                    curr_state = trial_model.prepare_state(parameters)
    
                    ansatz_ops_trial = []
                    ansatz_mat_trial = []
                    parameters_trial = []
        
                    ansatz_ops_trial.insert(0, pool.fermi_ops[op_trial])
                    ansatz_mat_trial.insert(0, pool.spmat_ops[op_trial])
                    parameters_trial.insert(0, init_para)
                    trial_model_1 = tUCCSD(hamiltonian, ansatz_mat_trial, ansatz_ops_trial, curr_state, parameters_trial)
    
                    opt_result = scipy.optimize.minimize(trial_model_1.energy, parameters_trial,
                                                             method='Cobyla')
        
                    parameters_trial = list(opt_result['x'])
        
                    dE = E-trial_model_1.curr_energy
        
                    # if abs(com) > adapt_thresh:
                    print(" %4i %40s %12.8f %12.8f" % (op_trial, pool.fermi_ops[op_trial], sec, dE))
        
                    if dE > abs(next_deriv) + 1e-9:
                        next_deriv = dE
                        next_index = op_trial
                        parameters_cand = parameters_trial.copy()

                    curr_sec += sec*sec

                    # if sec < next_sec - 1e-9:
                    #     next_sec = sec
                    #     next_index = op_trial

                curr_sec = np.sqrt(curr_sec)

        min_options = {'gtol': theta_thresh, 'disp':False}

        max_of_com = next_deriv
        print(" Norm of <[A,H]> = %12.8f" %curr_norm)
        print(" Max  of <[A,H]> = %12.8f" %max_of_com)

        converged = False
        if sec_deriv:
            if adapt_conver == "norm":
                if curr_norm < adapt_thresh:
                    if curr_sec < adapt_thresh:
                        converged = True
        elif sec_deriv == False:
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

            break

            break

        new_op = pool.fermi_ops[next_index]
        new_mat = pool.spmat_ops[next_index]

        # for n in range(len(group)):
        #     new_op += Sign[n]*pool.fermi_ops[group[n]]
        #     new_mat += Sign[n]*pool.spmat_ops[group[n]]

        print(" Add operator %4i" %next_index)

        word_length = 0

        line = str(new_op)
        print(line)
        X_1 = re.findall('X', line)
        if X_1:
            word_length += len(X_1)
        X_1 = re.findall('Y', line)
        if X_1:
            word_length += len(X_1)
        X_1 = re.findall('Z', line)
        if X_1:
            word_length += len(X_1)
        # print(Bin)

        print("pauli word length %4i" %word_length)

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
        
        if curr_norm > adapt_thresh:
            opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient, 
                    options = min_options, method = 'BFGS', callback=trial_model.callback)
            print('BFGS')
        else:
            opt_result = scipy.optimize.minimize(trial_model.energy, parameters,
                   method = 'Nelder-Mead', callback=trial_model.callback)
            print('Nelder-Mead')

        # print(ansatz_ops)
    
        parameters = list(opt_result['x'])
        print(parameters)
        curr_state = trial_model.prepare_state(parameters)
        # print(" new state ",curr_state)
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(" Error: %20.12f" % abs(trial_model.curr_energy-GSE))
        print(" -----------New ansatz----------- ")
        # print(curr_state)
        print(" %4s %30s %12s" %("Term","Coeff","#"))
        for si in range(len(ansatz_ops)):
                s = ansatz_ops[si]
                print(" %4s %20f %10s" %(s, parameters[si], si) )
                print(" ")