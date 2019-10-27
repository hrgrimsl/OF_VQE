import scipy
import openfermion
import openfermionpsi4
import os
import numpy as np
import copy
import random 
import sys
import csv
import cirq
import openfermioncirq
from openfermioncirq import trotter
import scipy.sparse.linalg

import operator_pools
import vqe_methods
from tVQE import *
import pickle

from  openfermionprojectq  import  uccsd_trotter_engine, TimeEvolution
from  projectq.backends  import CommandPrinter

from openfermion import *


def adapt_vqe(geometry,
        basis           = "sto-3g",
        multiplicity    = 1,
        charge          = 0,
        adapt_conver    = 'norm',
        adapt_thresh    = 1e-7,
        theta_thresh    = 1e-15,
        adapt_maxiter   = 400,
        pool            = operator_pools.singlet_GSD(),
        spin_adapt      = True,
        psi4_filename   = "psi4_%12.12f"%random.random()
        ):
# {{{
       
    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
    molecule.filename = psi4_filename
    molecule = openfermionpsi4.run_psi4(molecule, 
                run_scf = 1, 
                run_mp2=1, 
                run_cisd=0, 
                run_ccsd = 0, 
                run_fci=1, 
                delete_input=1)
    
    pool.init(molecule)
    print(" Basis: ", basis)

    print(' HF energy      %20.16f au' %(molecule.hf_energy))
    print(' MP2 energy     %20.16f au' %(molecule.mp2_energy))
    #print(' CISD energy    %20.16f au' %(molecule.cisd_energy))
    #print(' CCSD energy    %20.16f au' %(molecule.ccsd_energy))
    print(' FCI energy     %20.16f au' %(molecule.fci_energy))

    #JW transform Hamiltonian computed classically with OFPsi4
    hamiltonian_op = molecule.get_molecular_hamiltonian()
    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)
    # print("H",hamiltonian)
  #  print(hamiltonian[:,240])
    HJW = openfermion.transforms.jordan_wigner(hamiltonian_op)
    pickle.dump(HJW, open('./h4_hamiltonian.p','wb'))

    print(HJW)

    #Thetas
    parameters = []
    parameters_index = []
    parameters_mult = []
    Sign = []

    pool.generate_SparseMatrix()

    over_mat = np.zeros(shape=(pool.n_ops, pool.n_ops))
    vec = np.random.rand(2 ** pool.n_spin_orb, 1)
    print(vec)
    norm = 0

    for i in vec:
        norm += i * i

    vec = np.true_divide(vec, np.sqrt(norm))
    vec = scipy.sparse.csc_matrix(vec)

    for i in range(pool.n_ops):
        for j in range(pool.n_ops):
            over_mat[i, j] = vec.transpose().conjugate().dot(pool.spmat_ops[i].dot(pool.spmat_ops[j].dot(vec)))[0,0]

    rank = np.linalg.matrix_rank(over_mat)

    print("rank =", rank)
   
    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz
    ansatz_ops_t = []
    ansatz_mat_t = []
    ansatz_sup_index = []
    ansatz_left = []
    ansatz_right = []
    ansatz_scal = []
    
    #Build p-h reference and map it to JW transform
    reference_ket = scipy.sparse.csc_matrix(
            openfermion.jw_configuration_state(
                list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()

    # for n in range(molecule.n_electrons, molecule.n_qubits):
    # 	reference_ket = 1/np.sqrt(2)*(np.exp(1j*0.5)*-1j*pool.spmat_ops[n].dot(reference_ket) + reference_ket)
    # 	print(pool.fermi_ops[n])
    
    # reference_ket = 1/np.sqrt(2)*(np.exp(1j*0.5)*-1j*pool.spmat_ops[3].dot(reference_ket) + reference_ket)

    print(reference_ket)

    reference_bra = reference_ket.transpose().conj()

    print(" Start ADAPT-VQE algorithm")
    op_indices = []
    parameters = []

    curr_state = 1.0*reference_ket

    # for n in range(16,len(pool.spmat_ops)):
    # 	curr_state = 1/np.sqrt(2)*(pool.spmat_ops[n].dot(curr_state) + curr_state)
    # 	print(pool.fermi_ops[n])

    # curr_state = 1/np.sqrt(2)*(pool.spmat_ops[49].dot(curr_state) + curr_state)

    fermi_ops = pool.fermi_ops
    spmat_ops = pool.spmat_ops
    n_ops = pool.n_ops

    Num = QubitOperator('Z1', 0)

    for p in range(0,2*pool.n_orb):
        Num += QubitOperator('Z%d' %p)

    Num = openfermion.transforms.get_sparse_operator(Num, n_qubits = pool.n_spin_orb)

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

        N_el = curr_state.transpose().conj().dot(Num.dot(curr_state))
        print('Number of electron:', N_el)

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

            if abs(com) > abs(next_deriv) + 1e-9:
                next_deriv = com
                next_index = op_trial
        
        print(Sign)

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
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
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

                compiler_engine = uccsd_trotter_engine( compiler_backend=CommandPrinter() )
                wavefunction = compiler_engine.allocate_qureg(molecule.n_qubits)
        
                H = openfermion.transforms.jordan_wigner(1j*s)  # fermionic pool
        
                # Trotter step parameters.
                time = parameters[si]
        
                evolution_operator = TimeEvolution(time,H)
        
                evolution_operator | wavefunction
        
                compiler_engine.flush()

            break

            break

        new_op = pool.fermi_ops[next_index]
        new_mat = pool.spmat_ops[next_index]

        for n in range(len(group)):
            new_op += Sign[n]*pool.fermi_ops[group[n]]
            new_mat += Sign[n]*pool.spmat_ops[group[n]]

        print(" Add operator %4i" %next_index)

        # for n in range(n_iter):
        # 	parameters[n] = 0

        for n in group:
            print(" Add operator %4i " %n)

        parameters.insert(0,0)
        parameters_index.append(n_iter)
        ansatz_ops.insert(0,new_op)
        ansatz_mat.insert(0,new_mat)
        parameters_mult.insert(0,1)
        
        trial_model = tUCCSD(hamiltonian, ansatz_mat, ansatz_ops, ansatz_mat_t, ansatz_ops_t, reference_ket, parameters, parameters_index, parameters_mult, ansatz_sup_index, ansatz_left, ansatz_right, ansatz_scal)
        

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
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4s %20f %10s" %(opstring, parameters[si], si) )
                print(" ")

def random_Ham(geometry,
        basis           = "sto-3g",
        multiplicity    = 1,
        charge          = 0,
        adapt_conver    = 'norm',
        adapt_thresh    = 1e-4,
        theta_thresh    = 1e-9,
        adapt_maxiter   = 400,
        pool            = operator_pools.qubits(),
        spin_adapt      = True,
        psi4_filename   = "psi4_%12.12f"%random.random()
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
    
    pool.init(molecule)   

    ham = np.random.rand(8, 8)
    ham_sym = ham + ham.transpose()
    hamiltonian = scipy.sparse.csc_matrix(ham_sym)
    # print(hamiltonian)

    w, v = scipy.sparse.linalg.eigs(hamiltonian, which='SR')
    GS = scipy.sparse.csc_matrix(v[:,w.argmin()]).transpose().conj()
    # print(GS)
    print('Ground state energy:',min(w))
    # GS_energy = GS.transpose().conj().dot(hamiltonian.dot(GS))
    # print('GS energy', GS_energy )

    #Thetas
    parameters = []
    parameters_index = []
    parameters_mult = []
    Sign = []

    pool.generate_SparseMatrix()

    over_mat = np.zeros(shape=(pool.n_ops, pool.n_ops))
    vec = np.random.rand(2 ** 3, 1)
    print(vec)
    norm = 0

    for i in vec:
        norm += i * i

    vec = np.true_divide(vec, np.sqrt(norm))
    vec = scipy.sparse.csc_matrix(vec)

    for i in range(pool.n_ops):
        for j in range(pool.n_ops):
            over_mat[i, j] = vec.transpose().conjugate().dot(pool.spmat_ops[i].dot(pool.spmat_ops[j].dot(vec)))[0,0]

    rank = np.linalg.matrix_rank(over_mat)

    print("rank =", rank)
   
    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz
    ansatz_ops_t = []
    ansatz_mat_t = []
    ansatz_sup_index = []
    ansatz_left = []
    ansatz_right = []
    ansatz_scal = []
    
    #Build p-h reference and map it to JW transform
    reference_ket = vec

    # for n in range(molecule.n_electrons, molecule.n_qubits):
    # 	reference_ket = 1/np.sqrt(2)*(np.exp(1j*0.5)*-1j*pool.spmat_ops[n].dot(reference_ket) + reference_ket)
    # 	print(pool.fermi_ops[n])
    
    # reference_ket = 1/np.sqrt(2)*(np.exp(1j*0.5)*-1j*pool.spmat_ops[3].dot(reference_ket) + reference_ket)

    print(reference_ket)

    reference_bra = reference_ket.transpose().conj()

    print(" Start ADAPT-VQE algorithm")
    op_indices = []
    parameters = []

    curr_state = 1.0*reference_ket
    
    # for n in range(16,len(pool.spmat_ops)):
    # 	curr_state = 1/np.sqrt(2)*(pool.spmat_ops[n].dot(curr_state) + curr_state)
    # 	print(pool.fermi_ops[n])

    # curr_state = 1/np.sqrt(2)*(pool.spmat_ops[49].dot(curr_state) + curr_state)

    fermi_ops = pool.fermi_ops
    spmat_ops = pool.spmat_ops
    n_ops = pool.n_ops

    Num = QubitOperator('Z1', 0)

    for p in range(0,3):
        Num += QubitOperator('Z%d' %p)

    Num = openfermion.transforms.get_sparse_operator(Num, n_qubits = 3)

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

        N_el = curr_state.transpose().conj().dot(Num.dot(curr_state))
        print('Number of electron:', N_el)


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
       
            # if abs(com) > adapt_thresh:
            print(" %4i %40s %12.8f" %(op_trial, pool.fermi_ops[op_trial], com) )

            curr_norm += com*com

            if abs(com) > abs(next_deriv) + 1e-9:
                next_deriv = com
                next_index = op_trial
        
        print(Sign)

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
            pickle.dump(ansatz_ops, open('./h4_ansatz.p','wb'))
            print(" Ansatz Growth Converged!")
            print(" Number of operators in ansatz: ", len(ansatz_ops))
            print(" *Finished: %20.12f" % trial_model.curr_energy)
            print(" -----------Final ansatz----------- ")
            print(" %4s %30s %12s" %("Term","Coeff","#"))
            new_state = reference_ket
            E_step = []
            for k in reversed(range(0, len(parameters))):
                new_state = scipy.sparse.linalg.expm_multiply((parameters[k]*ansatz_mat[k]), new_state)
                E_step.append(new_state.transpose().conj().dot(hamiltonian.dot(new_state))[0,0].real)
                print(len(parameters))
                print(k)
                print('Energy step', float(E_step[len(parameters)-k-1]))
                print("")
            for si in range(len(ansatz_ops)):
                s = ansatz_ops[si]
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4s %20f %10s" %(s, parameters[si], si) )
                print(" ")

                compiler_engine = uccsd_trotter_engine( compiler_backend=CommandPrinter() )
                wavefunction = compiler_engine.allocate_qureg(molecule.n_qubits)
        
                H = 1j*s  # Qubits -pool
        
                # Trotter step parameters.
                time = parameters[si]
        
                evolution_operator = TimeEvolution(time,H)
        
                evolution_operator | wavefunction
        
                compiler_engine.flush()

            break

            break

        new_op = pool.fermi_ops[next_index]
        new_mat = pool.spmat_ops[next_index]

        for n in range(len(group)):
            new_op += Sign[n]*pool.fermi_ops[group[n]]
            new_mat += Sign[n]*pool.spmat_ops[group[n]]

        print(" Add operator %4i" %next_index)

        # for n in range(n_iter):
        #     parameters[n] = 0

        for n in group:
            print(" Add operator %4i " %n)

        parameters.insert(0,0)
        parameters_index.append(n_iter)
        ansatz_ops.insert(0,new_op)
        ansatz_mat.insert(0,new_mat)
        parameters_mult.insert(0,1)
        
        trial_model = tUCCSD(hamiltonian, ansatz_mat, ansatz_ops, ansatz_mat_t, ansatz_ops_t, reference_ket, parameters, parameters_index, parameters_mult, ansatz_sup_index, ansatz_left, ansatz_right, ansatz_scal)
        

        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient, 
                options = min_options, method = 'BFGS', callback=trial_model.callback)

        # print(ansatz_ops)
    
        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        overlap = GS.transpose().conj().dot(curr_state)[0,0]
        overlap = overlap.real
        overlap = overlap*overlap
        # print(" new state ",curr_state)
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(" error: %20.12f" % (trial_model.curr_energy-min(w)))
        print(" Overlap: %20.12f" % overlap)
        print(" Variance: %20.12f" % trial_model.variance(parameters))
        print(" -----------New ansatz----------- ")
        # print(curr_state)
        print(" %4s %30s %12s" %("Term","Coeff","#"))
        for si in range(len(ansatz_ops)):
                s = ansatz_ops[si]
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4s %20f %10s" %(s, parameters[si], si) )
                print(" ")


def q_adapt_vqe(geometry,
        basis           = "sto-3g",
        multiplicity    = 1,
        charge          = 0,
        adapt_conver    = 'norm',
        adapt_thresh    = 1e-4,
        theta_thresh    = 1e-9,
        adapt_maxiter   = 400,
        pool            = operator_pools.qubits(),
        spin_adapt      = True,
        psi4_filename   = "psi4_%12.12f"%random.random()
        ):
# {{{
       
    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
    molecule.filename = psi4_filename
    molecule = openfermionpsi4.run_psi4(molecule, 
                run_scf = 1, 
                run_mp2=1, 
                run_cisd=0, 
                run_ccsd = 0, 
                run_fci=1, 
                delete_input=1)
    
    pool.init(molecule)

    print(" Basis: ", basis)

    print(' HF energy      %20.16f au' %(molecule.hf_energy))
    print(' MP2 energy     %20.16f au' %(molecule.mp2_energy))
    #print(' CISD energy    %20.16f au' %(molecule.cisd_energy))
    #print(' CCSD energy    %20.16f au' %(molecule.ccsd_energy))
    print(' FCI energy     %20.16f au' %(molecule.fci_energy))

    #JW transform Hamiltonian computed classically with OFPsi4
    hamiltonian_op = molecule.get_molecular_hamiltonian()
    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)
    # print("H",hamiltonian)
    #  print(hamiltonian[:,240])
    HJW = openfermion.transforms.jordan_wigner(hamiltonian_op)
    # print(HJW)
    # pickle.dump(HJW, open('./h4_hamiltonian.p','wb'))

    w, v = scipy.sparse.linalg.eigs(hamiltonian, which='SR')
    GS = scipy.sparse.csc_matrix(v[:,w.argmin()]).transpose().conj()
    # print(GS)
    # print(min(w))
    # GS_energy = GS.transpose().conj().dot(hamiltonian.dot(GS))
    # print('GS energy', GS_energy )

    

    #Thetas
    parameters = []
    parameters_index = []
    parameters_mult = []
    Sign = []

    pool.generate_SparseMatrix()

    over_mat = np.zeros(shape=(pool.n_ops, pool.n_ops))
    vec = np.random.rand(2 ** pool.n_spin_orb, 1)
    print(vec)
    norm = 0

    for i in vec:
        norm += i * i

    vec = np.true_divide(vec, np.sqrt(norm))
    vec = scipy.sparse.csc_matrix(vec)

    for i in range(pool.n_ops):
        for j in range(pool.n_ops):
            over_mat[i, j] = vec.transpose().conjugate().dot(pool.spmat_ops[i].dot(pool.spmat_ops[j].dot(vec)))[0,0]

    rank = np.linalg.matrix_rank(over_mat)

    print("rank =", rank)
   
    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz
    ansatz_ops_t = []
    ansatz_mat_t = []
    ansatz_sup_index = []
    ansatz_left = []
    ansatz_right = []
    ansatz_scal = []
    
    #Build p-h reference and map it to JW transform
    reference_ket = scipy.sparse.csc_matrix(
            openfermion.jw_configuration_state(
                list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()

    # for n in range(molecule.n_electrons, molecule.n_qubits):
    # 	reference_ket = 1/np.sqrt(2)*(np.exp(1j*0.5)*-1j*pool.spmat_ops[n].dot(reference_ket) + reference_ket)
    # 	print(pool.fermi_ops[n])
    
    # reference_ket = 1/np.sqrt(2)*(np.exp(1j*0.5)*-1j*pool.spmat_ops[3].dot(reference_ket) + reference_ket)

    print(reference_ket)

    reference_bra = reference_ket.transpose().conj()

    print(" Start ADAPT-VQE algorithm")
    op_indices = []
    parameters = []

    curr_state = 1.0*reference_ket
    
    # for n in range(16,len(pool.spmat_ops)):
    # 	curr_state = 1/np.sqrt(2)*(pool.spmat_ops[n].dot(curr_state) + curr_state)
    # 	print(pool.fermi_ops[n])

    # curr_state = 1/np.sqrt(2)*(pool.spmat_ops[49].dot(curr_state) + curr_state)

    fermi_ops = pool.fermi_ops
    spmat_ops = pool.spmat_ops
    n_ops = pool.n_ops

    Num = QubitOperator('Z1', 0)

    for p in range(0,2*pool.n_orb):
        Num += QubitOperator('Z%d' %p)

    Num = openfermion.transforms.get_sparse_operator(Num, n_qubits = pool.n_spin_orb)

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

        N_el = curr_state.transpose().conj().dot(Num.dot(curr_state))
        print('Number of electron:', N_el)


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
       
            # if abs(com) > adapt_thresh:
            print(" %4i %40s %12.8f" %(op_trial, pool.fermi_ops[op_trial], com) )

            curr_norm += com*com

            if abs(com) > abs(next_deriv) + 1e-9:
                next_deriv = com
                next_index = op_trial
        
        print(Sign)

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
            pickle.dump(ansatz_ops, open('./h4_ansatz.p','wb'))
            print(" Ansatz Growth Converged!")
            print(" Number of operators in ansatz: ", len(ansatz_ops))
            print(" *Finished: %20.12f" % trial_model.curr_energy)
            print(" -----------Final ansatz----------- ")
            print(" %4s %30s %12s" %("Term","Coeff","#"))
            new_state = reference_ket
            E_step = []
            for k in reversed(range(0, len(parameters))):
                new_state = scipy.sparse.linalg.expm_multiply((parameters[k]*ansatz_mat[k]), new_state)
                E_step.append(new_state.transpose().conj().dot(hamiltonian.dot(new_state))[0,0].real)
                print(len(parameters))
                print(k)
                print('Energy step', float(E_step[len(parameters)-k-1]))
                print("")
            for si in range(len(ansatz_ops)):
                s = ansatz_ops[si]
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4s %20f %10s" %(s, parameters[si], si) )
                print(" ")

                compiler_engine = uccsd_trotter_engine( compiler_backend=CommandPrinter() )
                wavefunction = compiler_engine.allocate_qureg(molecule.n_qubits)
        
                H = 1j*s  # Qubits -pool
        
                # Trotter step parameters.
                time = parameters[si]
        
                evolution_operator = TimeEvolution(time,H)
        
                evolution_operator | wavefunction
        
                compiler_engine.flush()

            break

            break

        new_op = pool.fermi_ops[next_index]
        new_mat = pool.spmat_ops[next_index]

        for n in range(len(group)):
            new_op += Sign[n]*pool.fermi_ops[group[n]]
            new_mat += Sign[n]*pool.spmat_ops[group[n]]

        print(" Add operator %4i" %next_index)

        # for n in range(n_iter):
        #     parameters[n] = 0

        for n in group:
            print(" Add operator %4i " %n)

        parameters.insert(0,0)
        parameters_index.append(n_iter)
        ansatz_ops.insert(0,new_op)
        ansatz_mat.insert(0,new_mat)
        parameters_mult.insert(0,1)
        
        trial_model = tUCCSD(hamiltonian, ansatz_mat, ansatz_ops, ansatz_mat_t, ansatz_ops_t, reference_ket, parameters, parameters_index, parameters_mult, ansatz_sup_index, ansatz_left, ansatz_right, ansatz_scal)
        

        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient, 
                options = min_options, method = 'BFGS', callback=trial_model.callback)

        # print(ansatz_ops)
    
        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        overlap = GS.transpose().conj().dot(curr_state)[0,0]
        overlap = overlap.real
        overlap = overlap*overlap
        # print(" new state ",curr_state)
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(" Overlap: %20.12f" % overlap)
        print(" Variance: %20.12f" % trial_model.variance(parameters))
        print(" -----------New ansatz----------- ")
        # print(curr_state)
        print(" %4s %30s %12s" %("Term","Coeff","#"))
        for si in range(len(ansatz_ops)):
                s = ansatz_ops[si]
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4s %20f %10s" %(s, parameters[si], si) )
                print(" ")

def q_adapt_vqe_min(geometry,
        basis           = "sto-3g",
        multiplicity    = 1,
        charge          = 0,
        adapt_conver    = 'norm',
        adapt_thresh    = 1e-12,
        theta_thresh    = 1e-12,
        adapt_maxiter   = 400,
        pool            = operator_pools.qubits(),
        spin_adapt      = True,
        psi4_filename   = "psi4_%12.12f"%random.random()
        ):
# {{{
       
    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
    molecule.filename = psi4_filename
    molecule = openfermionpsi4.run_psi4(molecule, 
                run_scf = 1, 
                run_mp2=1, 
                run_cisd=0, 
                run_ccsd = 0, 
                run_fci=1, 
                delete_input=1)
    
    pool.init(molecule)
    print(" Basis: ", basis)

    print(' HF energy      %20.16f au' %(molecule.hf_energy))
    print(' MP2 energy     %20.16f au' %(molecule.mp2_energy))
    #print(' CISD energy    %20.16f au' %(molecule.cisd_energy))
    #print(' CCSD energy    %20.16f au' %(molecule.ccsd_energy))
    print(' FCI energy     %20.16f au' %(molecule.fci_energy))

    #JW transform Hamiltonian computed classically with OFPsi4
    hamiltonian_op = molecule.get_molecular_hamiltonian()
    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)
    # print("H",hamiltonian)
  #  print(hamiltonian[:,240])
    HJW = openfermion.transforms.jordan_wigner(hamiltonian_op)

    w, v = scipy.sparse.linalg.eigs(hamiltonian, which='SR')
    GS = scipy.sparse.csc_matrix(v[:,w.argmin()]).transpose().conj()
    # print(GS)
    # print(min(w))
    # GS_energy = GS.transpose().conj().dot(hamiltonian.dot(GS))
    # print('GS energy', GS_energy )

    # print(HJW)

    #Thetas
    parameters = []
    parameters_index = []
    parameters_mult = []
    Sign = []

    pool.generate_SparseMatrix()
   
    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz
    ansatz_ops_t = []
    ansatz_mat_t = []
    ansatz_sup_index = []
    ansatz_left = []
    ansatz_right = []
    ansatz_scal = []
    del_E = []
    nonzero = 0
    
    #Build p-h reference and map it to JW transform
    reference_ket = scipy.sparse.csc_matrix(
            openfermion.jw_configuration_state(
                list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()

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

    Num = QubitOperator('Z1', 0)

    for p in range(0,2*pool.n_orb):
        Num += QubitOperator('Z%d' %p)

    Num = openfermion.transforms.get_sparse_operator(Num, n_qubits = pool.n_spin_orb)

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

        N_el = curr_state.transpose().conj().dot(Num.dot(curr_state))
        print('Number of electron:', N_el)

        min_options = {'gtol': theta_thresh, 'disp':False}

        for op_trial in op_indices:

            ansatz_ops_trial = []
            ansatz_mat_trial = []
            parameters_trial = []
            ansatz_ops_trial.append(pool.fermi_ops[op_trial])
            ansatz_mat_trial.append(pool.spmat_ops[op_trial])
            parameters_trial.insert(0,0.1)
            trial_model = tUCCSD(hamiltonian, ansatz_mat_trial, ansatz_ops_trial, ansatz_mat_t, ansatz_ops_t, curr_state, parameters_trial, parameters_index, parameters_mult, ansatz_sup_index, ansatz_left, ansatz_right, ansatz_scal)       
            opt_result = scipy.optimize.minimize(trial_model.energy, parameters_trial, jac=trial_model.gradient, options = min_options, method = 'BFGS', callback=trial_model.callback)
            dE = molecule.hf_energy-trial_model.curr_energy
            dE = abs(dE)

            opstring = ""
            for t in pool.fermi_ops[op_trial].terms:
                opstring += str(t)
                break
       
            # if abs(com) > adapt_thresh:
            print(" %4i %40s %12.8f" %(op_trial, opstring, dE) )

            if abs(dE) > abs(next_deriv) + 1e-9:
                next_deriv = dE
                next_index = op_trial
        
        print(Sign)

        min_options = {'gtol': theta_thresh, 'disp':False}     

        max_of_com = next_deriv
        print(" delta E = %12.8f" %next_deriv)

        converged = False
        if adapt_conver == "norm":
            if next_deriv < adapt_thresh:
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
            new_state = reference_ket
            E_step = []
            for k in reversed(range(0, len(parameters))):
                new_state = scipy.sparse.linalg.expm_multiply((parameters[k]*ansatz_mat[k]), new_state)
                E_step.append(new_state.transpose().conj().dot(hamiltonian.dot(new_state))[0,0].real)
                print(len(parameters))
                print(k)
                print('Energy step', float(E_step[len(parameters)-k-1]))
                print("")
            for si in range(len(ansatz_ops)):
                s = ansatz_ops[si]
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4s %20f %10s" %(s, parameters[si], si) )
                print(" ")

                compiler_engine = uccsd_trotter_engine( compiler_backend=CommandPrinter() )
                wavefunction = compiler_engine.allocate_qureg(molecule.n_qubits)
        
                H = 1j*s  # Qubits -pool
        
                # Trotter step parameters.
                time = parameters[si]
        
                evolution_operator = TimeEvolution(time,H)
        
                evolution_operator | wavefunction
        
                compiler_engine.flush()

            break

            break

        new_op = pool.fermi_ops[next_index]
        new_mat = pool.spmat_ops[next_index]

        for n in range(len(group)):
            new_op += Sign[n]*pool.fermi_ops[group[n]]
            new_mat += Sign[n]*pool.spmat_ops[group[n]]

        print(" Add operator %4i" %next_index)

        # for n in range(n_iter):
        #     parameters[n] = 0

        for n in group:
            print(" Add operator %4i " %n)

        parameters.insert(0,0)
        parameters_index.append(n_iter)
        ansatz_ops.insert(0,new_op)
        ansatz_mat.insert(0,new_mat)
        parameters_mult.insert(0,1)
        
        trial_model = tUCCSD(hamiltonian, ansatz_mat, ansatz_ops, ansatz_mat_t, ansatz_ops_t, reference_ket, parameters, parameters_index, parameters_mult, ansatz_sup_index, ansatz_left, ansatz_right, ansatz_scal)
        

        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient, 
                options = min_options, method = 'BFGS', callback=trial_model.callback)

        # print(ansatz_ops)
    
        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        overlap = GS.transpose().conj().dot(curr_state)[0,0]
        overlap = overlap.real
        overlap = overlap*overlap
        # print(" new state ",curr_state)
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(" Overlap: %20.12f" % overlap)
        print(" Variance: %20.12f" % trial_model.variance(parameters))
        print(" -----------New ansatz----------- ")
        # print(curr_state)
        print(" %4s %30s %12s" %("Term","Coeff","#"))
        for si in range(len(ansatz_ops)):
                s = ansatz_ops[si]
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4s %20f %10s" %(s, parameters[si], si) )
                print(" ")                


def q_adapt_vqe_sort_mini(geometry,
        basis           = "sto-3g",
        multiplicity    = 1,
        charge          = 0,
        adapt_conver    = 'norm',
        adapt_thresh    = 1e-7,
        theta_thresh    = 1e-12,
        adapt_maxiter   = 400,
        pool            = operator_pools.qubits(),
        spin_adapt      = True,
        psi4_filename   = "psi4_%12.12f"%random.random()
        ):
# {{{
       
    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
    molecule.filename = psi4_filename
    molecule = openfermionpsi4.run_psi4(molecule, 
                run_scf = 1, 
                run_mp2=1, 
                run_cisd=0, 
                run_ccsd = 0, 
                run_fci=1, 
                delete_input=1)
    
    pool.init(molecule)
    print(" Basis: ", basis)

    print(' HF energy      %20.16f au' %(molecule.hf_energy))
    print(' MP2 energy     %20.16f au' %(molecule.mp2_energy))
    #print(' CISD energy    %20.16f au' %(molecule.cisd_energy))
    #print(' CCSD energy    %20.16f au' %(molecule.ccsd_energy))
    print(' FCI energy     %20.16f au' %(molecule.fci_energy))

    #JW transform Hamiltonian computed classically with OFPsi4
    hamiltonian_op = molecule.get_molecular_hamiltonian()
    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)
    # print("H",hamiltonian)
  #  print(hamiltonian[:,240])
    HJW = openfermion.transforms.jordan_wigner(hamiltonian_op)

    w, v = scipy.sparse.linalg.eigs(hamiltonian, which='SR')
    GS = scipy.sparse.csc_matrix(v[:,w.argmin()]).transpose().conj()
    # print(GS)
    # print(min(w))
    # GS_energy = GS.transpose().conj().dot(hamiltonian.dot(GS))
    # print('GS energy', GS_energy )

    # print(HJW)

    #Thetas
    parameters = []
    parameters_index = []
    parameters_mult = []
    Sign = []

    pool.generate_SparseMatrix()
   
    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz
    ansatz_ops_t = []
    ansatz_mat_t = []
    ansatz_sup_index = []
    ansatz_left = []
    ansatz_right = []
    ansatz_scal = []
    del_E = []
    nonzero = 0
    
    #Build p-h reference and map it to JW transform
    reference_ket = scipy.sparse.csc_matrix(
            openfermion.jw_configuration_state(
                list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()

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

    Num = QubitOperator('Z1', 0)

    for p in range(0,2*pool.n_orb):
        Num += QubitOperator('Z%d' %p)

    Num = openfermion.transforms.get_sparse_operator(Num, n_qubits = pool.n_spin_orb)

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

        N_el = curr_state.transpose().conj().dot(Num.dot(curr_state))
        print('Number of electron:', N_el)

        min_options = {'gtol': theta_thresh, 'disp':False}

        for op_trial in op_indices:

            if n_iter == 0:
                ansatz_ops.append(pool.fermi_ops[op_trial])
                ansatz_mat.append(pool.spmat_ops[op_trial])
                parameters.insert(0,0.1)
                trial_model = tUCCSD(hamiltonian, ansatz_mat, ansatz_ops, ansatz_mat_t, ansatz_ops_t, reference_ket, parameters, parameters_index, parameters_mult, ansatz_sup_index, ansatz_left, ansatz_right, ansatz_scal)       
                opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient, options = min_options, method = 'BFGS', callback=trial_model.callback)
                dE = molecule.hf_energy-trial_model.curr_energy
                del_E.append(dE)
                if abs(dE) > 1e-6:
                    nonzero += 1
                ansatz_ops = []
                ansatz_mat = []
                parameters = []

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
       
            # if abs(com) > adapt_thresh:
            print(" %4i %40s %12.8f" %(op_trial, opstring, com) )

            curr_norm += com*com

            if abs(com) > abs(next_deriv) + 1e-9:
                next_deriv = com
                next_index = op_trial
        
        print(Sign)

        curr_norm = np.sqrt(curr_norm)

        min_options = {'gtol': theta_thresh, 'disp':False}

        if n_iter == 0:
            op_indices = [ op_indices for _, op_indices in sorted(zip(del_E,op_indices),reverse=True)]
            for x in op_indices:
                print(fermi_ops[x],del_E[x])
            op_indices = op_indices[:nonzero]     

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
            new_state = reference_ket
            E_step = []
            for k in reversed(range(0, len(parameters))):
                new_state = scipy.sparse.linalg.expm_multiply((parameters[k]*ansatz_mat[k]), new_state)
                E_step.append(new_state.transpose().conj().dot(hamiltonian.dot(new_state))[0,0].real)
                print(len(parameters))
                print(k)
                print('Energy step', float(E_step[len(parameters)-k-1]))
                print("")
            for si in range(len(ansatz_ops)):
                s = ansatz_ops[si]
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4s %20f %10s" %(opstring, parameters[si], si) )
                print(" ")

                compiler_engine = uccsd_trotter_engine( compiler_backend=CommandPrinter() )
                wavefunction = compiler_engine.allocate_qureg(molecule.n_qubits)
        
                H = 1j*s  # Qubits -pool
        
                # Trotter step parameters.
                time = parameters[si]
        
                evolution_operator = TimeEvolution(time,H)
        
                evolution_operator | wavefunction
        
                compiler_engine.flush()

            break

            break

        new_op = pool.fermi_ops[next_index]
        new_mat = pool.spmat_ops[next_index]

        for n in range(len(group)):
            new_op += Sign[n]*pool.fermi_ops[group[n]]
            new_mat += Sign[n]*pool.spmat_ops[group[n]]

        print(" Add operator %4i" %next_index)

        # for n in range(n_iter):
        #     parameters[n] = 0

        for n in group:
            print(" Add operator %4i " %n)

        parameters.insert(0,0)
        parameters_index.append(n_iter)
        ansatz_ops.insert(0,new_op)
        ansatz_mat.insert(0,new_mat)
        parameters_mult.insert(0,1)
        
        trial_model = tUCCSD(hamiltonian, ansatz_mat, ansatz_ops, ansatz_mat_t, ansatz_ops_t, reference_ket, parameters, parameters_index, parameters_mult, ansatz_sup_index, ansatz_left, ansatz_right, ansatz_scal)
        

        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient, 
                options = min_options, method = 'BFGS', callback=trial_model.callback)

        # print(ansatz_ops)
    
        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        overlap = GS.transpose().conj().dot(curr_state)[0,0]
        overlap = overlap.real
        overlap = overlap*overlap
        # print(" new state ",curr_state)
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(" Overlap: %20.12f" % overlap)
        print(" Variance: %20.12f" % trial_model.variance(parameters))
        print(" -----------New ansatz----------- ")
        # print(curr_state)
        print(" %4s %30s %12s" %("Term","Coeff","#"))
        for si in range(len(ansatz_ops)):
                s = ansatz_ops[si]
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4s %20f %10s" %(opstring, parameters[si], si) )
                print(" ")                

def q_adapt_vqe_sort(geometry,
        basis           = "sto-3g",
        multiplicity    = 1,
        charge          = 0,
        adapt_conver    = 'norm',
        adapt_thresh    = 1e-7,
        theta_thresh    = 1e-12,
        adapt_maxiter   = 400,
        pool            = operator_pools.qubits(),
        spin_adapt      = True,
        psi4_filename   = "psi4_%12.12f"%random.random(),
        cut_threshold   = 50
        ):
# {{{
       
    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
    molecule.filename = psi4_filename
    molecule = openfermionpsi4.run_psi4(molecule, 
                run_scf = 1, 
                run_mp2=1, 
                run_cisd=0, 
                run_ccsd = 0, 
                run_fci=1, 
                delete_input=1)
    
    pool.init(molecule)
    print(" Basis: ", basis)

    print(' HF energy      %20.16f au' %(molecule.hf_energy))
    print(' MP2 energy     %20.16f au' %(molecule.mp2_energy))
    #print(' CISD energy    %20.16f au' %(molecule.cisd_energy))
    #print(' CCSD energy    %20.16f au' %(molecule.ccsd_energy))
    print(' FCI energy     %20.16f au' %(molecule.fci_energy))

    #JW transform Hamiltonian computed classically with OFPsi4
    hamiltonian_op = molecule.get_molecular_hamiltonian()
    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)
    # print("H",hamiltonian)
  #  print(hamiltonian[:,240])
    HJW = openfermion.transforms.jordan_wigner(hamiltonian_op)

    w, v = scipy.sparse.linalg.eigs(hamiltonian, which='SR')
    GS = scipy.sparse.csc_matrix(v[:,w.argmin()]).transpose().conj()
    # print(GS)
    # print(min(w))
    # GS_energy = GS.transpose().conj().dot(hamiltonian.dot(GS))
    # print('GS energy', GS_energy )

    # print(HJW)

    #Thetas
    parameters = []
    parameters_index = []
    parameters_mult = []
    Sign = []

    pool.generate_SparseMatrix()
   
    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz
    ansatz_ops_t = []
    ansatz_mat_t = []
    ansatz_sup_index = []
    ansatz_left = []
    ansatz_right = []
    ansatz_scal = []
    
    #Build p-h reference and map it to JW transform
    reference_ket = scipy.sparse.csc_matrix(
            openfermion.jw_configuration_state(
                list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()

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

    Num = QubitOperator('Z1', 0)

    for p in range(0,2*pool.n_orb):
        Num += QubitOperator('Z%d' %p)

    Num = openfermion.transforms.get_sparse_operator(Num, n_qubits = pool.n_spin_orb)

    print(" Now start to grow the ansatz")
    print('adapt_thresh = %20.5f' %(adapt_thresh))
    print('theta_thresh = %20.5f' %(theta_thresh))
    print('cut_threshold = %20.5f' %(cut_threshold))
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
        com_list = []
        nonzero_start = 0
        print(" Measure commutators:")
        sig = hamiltonian.dot(curr_state)

        N_el = curr_state.transpose().conj().dot(Num.dot(curr_state))
        print('Number of electron:', N_el)

        for op_trial in op_indices:

            opA = pool.spmat_ops[op_trial]
            com = 2*(curr_state.transpose().conj().dot(opA.dot(sig))).real
            assert(com.shape == (1,1))
            com = com[0,0]
            assert(np.isclose(com.imag,0))
            com = com.real
            if abs(com) > 1e-6:
            	nonzero_start += 1
            com_list.append(abs(com))

            opstring = ""
            for t in pool.fermi_ops[op_trial].terms:
                opstring += str(t)
                break
       
            # if abs(com) > adapt_thresh:
            print(" %4i %40s %12.8f" %(op_trial, opstring, com) )

            curr_norm += com*com

            if abs(com) > abs(next_deriv) + 1e-9:
                next_deriv = com
                next_index = op_trial
        
        print(Sign)

        curr_norm = np.sqrt(curr_norm)

        min_options = {'gtol': theta_thresh, 'disp':False}
        

        if n_iter == 0:
            op_indices = [ op_indices for _, op_indices in sorted(zip(com_list,op_indices),reverse=True)]
            for x in op_indices:
                print(fermi_ops[x],com_list[x])
            op_indices = op_indices[:cut_threshold]

        max_of_com = next_deriv
        print(" Norm of <[A,H]> = %12.8f" %curr_norm)
        print(" Max  of <[A,H]> = %12.8f" %max_of_com)
        print(" nonzero grad: %12.5i" %nonzero_start)

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
            new_state = reference_ket
            E_step = []
            for k in reversed(range(0, len(parameters))):
                new_state = scipy.sparse.linalg.expm_multiply((parameters[k]*ansatz_mat[k]), new_state)
                E_step.append(new_state.transpose().conj().dot(hamiltonian.dot(new_state))[0,0].real)
                print(len(parameters))
                print(k)
                print('Energy step', float(E_step[len(parameters)-k-1]))
                print("")
            for si in range(len(ansatz_ops)):
                s = ansatz_ops[si]
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4s %20f %10s" %(opstring, parameters[si], si) )
                print(" ")

                compiler_engine = uccsd_trotter_engine( compiler_backend=CommandPrinter() )
                wavefunction = compiler_engine.allocate_qureg(molecule.n_qubits)
        
                H = 1j*s  # Qubits -pool
        
                # Trotter step parameters.
                time = parameters[si]
        
                evolution_operator = TimeEvolution(time,H)
        
                evolution_operator | wavefunction
        
                compiler_engine.flush()

            break

            break

        new_op = pool.fermi_ops[next_index]
        new_mat = pool.spmat_ops[next_index]

        for n in range(len(group)):
            new_op += Sign[n]*pool.fermi_ops[group[n]]
            new_mat += Sign[n]*pool.spmat_ops[group[n]]

        print(" Add operator %4i" %next_index)

        # for n in range(n_iter):
        #     parameters[n] = 0

        for n in group:
            print(" Add operator %4i " %n)

        parameters.insert(0,0)
        parameters_index.append(n_iter)
        ansatz_ops.insert(0,new_op)
        ansatz_mat.insert(0,new_mat)
        parameters_mult.insert(0,1)
        
        trial_model = tUCCSD(hamiltonian, ansatz_mat, ansatz_ops, ansatz_mat_t, ansatz_ops_t, reference_ket, parameters, parameters_index, parameters_mult, ansatz_sup_index, ansatz_left, ansatz_right, ansatz_scal)
        

        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient, 
                options = min_options, method = 'BFGS', callback=trial_model.callback)

        # print(ansatz_ops)
    
        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        overlap = GS.transpose().conj().dot(curr_state)[0,0]
        overlap = overlap.real
        overlap = overlap*overlap
        # print(" new state ",curr_state)
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(" Overlap: %20.12f" % overlap)
        print(" Variance: %20.12f" % trial_model.variance(parameters))
        print(" -----------New ansatz----------- ")
        # print(curr_state)
        print(" %4s %30s %12s" %("Term","Coeff","#"))
        for si in range(len(ansatz_ops)):
                s = ansatz_ops[si]
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4s %20f %10s" %(opstring, parameters[si], si) )
                print(" ")

def q_adapt_vqe_remove(geometry,
        basis           = "sto-3g",
        multiplicity    = 1,
        charge          = 0,
        adapt_conver    = 'norm',
        adapt_thresh    = 1e-9,
        theta_thresh    = 1e-20,
        adapt_maxiter   = 400,
        pool            = operator_pools.qubits(),
        spin_adapt      = True,
        psi4_filename   = "psi4_%12.12f"%random.random()
        ):
# {{{
       
    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
    molecule.filename = psi4_filename
    molecule = openfermionpsi4.run_psi4(molecule, 
                run_scf = 1, 
                run_mp2=1, 
                run_cisd=0, 
                run_ccsd = 0, 
                run_fci=1, 
                delete_input=1)
    
    pool.init(molecule)
    print(" Basis: ", basis)

    print(' HF energy      %20.16f au' %(molecule.hf_energy))
    print(' MP2 energy     %20.16f au' %(molecule.mp2_energy))
    #print(' CISD energy    %20.16f au' %(molecule.cisd_energy))
    #print(' CCSD energy    %20.16f au' %(molecule.ccsd_energy))
    print(' FCI energy     %20.16f au' %(molecule.fci_energy))

    #JW transform Hamiltonian computed classically with OFPsi4
    hamiltonian_op = molecule.get_molecular_hamiltonian()
    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)
    # print("H",hamiltonian)
  #  print(hamiltonian[:,240])
    HJW = openfermion.transforms.jordan_wigner(hamiltonian_op)

    # print(HJW)

    #Thetas
    parameters = []
    parameters_index = []
    parameters_mult = []
    Sign = []

    pool.generate_SparseMatrix()

    with open('Results/H4/qubits_remove_ansatz.csv','r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            pool.spmat_ops[int(row[2])] = openfermion.transforms.get_sparse_operator(QubitOperator('X%d X%d'% (1, 2), 0), n_qubits = pool.n_spin_orb)
   
    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz
    ansatz_ops_t = []
    ansatz_mat_t = []
    ansatz_sup_index = []
    ansatz_left = []
    ansatz_right = []
    ansatz_scal = []
    
    #Build p-h reference and map it to JW transform
    reference_ket = scipy.sparse.csc_matrix(
            openfermion.jw_configuration_state(
                list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()

    # for n in range(molecule.n_electrons, molecule.n_qubits):
    # 	reference_ket = 1/np.sqrt(2)*(np.exp(1j*0.5)*-1j*pool.spmat_ops[n].dot(reference_ket) + reference_ket)
    # 	print(pool.fermi_ops[n])
    
    # reference_ket = 1/np.sqrt(2)*(np.exp(1j*0.5)*-1j*pool.spmat_ops[3].dot(reference_ket) + reference_ket)

    print(reference_ket)

    reference_bra = reference_ket.transpose().conj()

    print(" Start ADAPT-VQE algorithm")
    op_indices = []
    parameters = []

    curr_state = 1.0*reference_ket
    
    # for n in range(16,len(pool.spmat_ops)):
    # 	curr_state = 1/np.sqrt(2)*(pool.spmat_ops[n].dot(curr_state) + curr_state)
    # 	print(pool.fermi_ops[n])

    # curr_state = 1/np.sqrt(2)*(pool.spmat_ops[49].dot(curr_state) + curr_state)

    fermi_ops = pool.fermi_ops
    spmat_ops = pool.spmat_ops
    n_ops = pool.n_ops

    Num = QubitOperator('Z1', 0)

    for p in range(0,2*pool.n_orb):
        Num += QubitOperator('Z%d' %p)

    Num = openfermion.transforms.get_sparse_operator(Num, n_qubits = pool.n_spin_orb)

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

        N_el = curr_state.transpose().conj().dot(Num.dot(curr_state))
        print('Number of electron:', N_el)


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
       
            # if abs(com) > adapt_thresh:
            print(" %4i %40s %12.8f" %(op_trial, opstring, com) )

            curr_norm += com*com

            if abs(com) > abs(next_deriv) + 1e-9:
                next_deriv = com
                next_index = op_trial
        
        print(Sign)

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
            new_state = reference_ket
            E_step = []
            for k in reversed(range(0, len(parameters))):
                new_state = scipy.sparse.linalg.expm_multiply((parameters[k]*ansatz_mat[k]), new_state)
                E_step.append(new_state.transpose().conj().dot(hamiltonian.dot(new_state))[0,0].real)
                print(len(parameters))
                print(k)
                print('Energy step', float(E_step[len(parameters)-k-1]))
                print("")
            for si in range(len(ansatz_ops)):
                s = ansatz_ops[si]
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4s %20f %10s" %(opstring, parameters[si], si) )
                print(" ")

                compiler_engine = uccsd_trotter_engine( compiler_backend=CommandPrinter() )
                wavefunction = compiler_engine.allocate_qureg(molecule.n_qubits)
        
                H = 1j*s  # Qubits -pool
        
                # Trotter step parameters.
                time = parameters[si]
        
                evolution_operator = TimeEvolution(time,H)
        
                evolution_operator | wavefunction
        
                compiler_engine.flush()

            break

            break

        new_op = pool.fermi_ops[next_index]
        new_mat = pool.spmat_ops[next_index]

        for n in range(len(group)):
            new_op += Sign[n]*pool.fermi_ops[group[n]]
            new_mat += Sign[n]*pool.spmat_ops[group[n]]

        print(" Add operator %4i" %next_index)

        # for n in range(n_iter):
        # 	parameters[n] = 0

        for n in group:
            print(" Add operator %4i " %n)

        parameters.insert(0,0)
        parameters_index.append(n_iter)
        ansatz_ops.insert(0,new_op)
        ansatz_mat.insert(0,new_mat)
        parameters_mult.insert(0,1)
        
        trial_model = tUCCSD(hamiltonian, ansatz_mat, ansatz_ops, ansatz_mat_t, ansatz_ops_t, reference_ket, parameters, parameters_index, parameters_mult, ansatz_sup_index, ansatz_left, ansatz_right, ansatz_scal)
        

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
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4s %20f %10s" %(opstring, parameters[si], si) )
                print(" ")

def q_adapt_vqe_sort_rand(geometry,
        basis           = "sto-3g",
        multiplicity    = 1,
        charge          = 0,
        adapt_conver    = 'norm',
        adapt_thresh    = 1e-7,
        theta_thresh    = 1e-12,
        adapt_maxiter   = 400,
        pool            = operator_pools.qubits(),
        spin_adapt      = True,
        psi4_filename   = "psi4_%12.12f"%random.random(),
        cut_threshold   = 50
        ):
# {{{
       
    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
    molecule.filename = psi4_filename
    molecule = openfermionpsi4.run_psi4(molecule, 
                run_scf = 1, 
                run_mp2=1, 
                run_cisd=0, 
                run_ccsd = 0, 
                run_fci=1, 
                delete_input=1)
    
    pool.init(molecule)
    print(" Basis: ", basis)

    print(' HF energy      %20.16f au' %(molecule.hf_energy))
    print(' MP2 energy     %20.16f au' %(molecule.mp2_energy))
    #print(' CISD energy    %20.16f au' %(molecule.cisd_energy))
    #print(' CCSD energy    %20.16f au' %(molecule.ccsd_energy))
    print(' FCI energy     %20.16f au' %(molecule.fci_energy))

    #JW transform Hamiltonian computed classically with OFPsi4
    hamiltonian_op = molecule.get_molecular_hamiltonian()
    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)
    # print("H",hamiltonian)
  #  print(hamiltonian[:,240])
    HJW = openfermion.transforms.jordan_wigner(hamiltonian_op)

    # print(HJW)

    #Thetas
    parameters = []
    parameters_index = []
    parameters_mult = []
    Sign = []

    pool.generate_SparseMatrix()
   
    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz
    ansatz_ops_t = []
    ansatz_mat_t = []
    ansatz_sup_index = []
    ansatz_left = []
    ansatz_right = []
    ansatz_scal = []
    
    #Build p-h reference and map it to JW transform
    reference_ket = scipy.sparse.csc_matrix(
            openfermion.jw_configuration_state(
                list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()

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

    Num = QubitOperator('Z1', 0)

    for p in range(0,2*pool.n_orb):
        Num += QubitOperator('Z%d' %p)

    Num = openfermion.transforms.get_sparse_operator(Num, n_qubits = pool.n_spin_orb)

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
        com_list = []
        nonzero_start = 0
        print(" Measure commutators:")
        sig = hamiltonian.dot(curr_state)

        N_el = curr_state.transpose().conj().dot(Num.dot(curr_state))
        print('Number of electron:', N_el)


        for op_trial in op_indices:

            opA = pool.spmat_ops[op_trial]
            com = 2*(curr_state.transpose().conj().dot(opA.dot(sig))).real
            assert(com.shape == (1,1))
            com = com[0,0]
            assert(np.isclose(com.imag,0))
            com = com.real
            if abs(com) > 1e-6:
            	nonzero_start += 1
            com_list.append(abs(com))

            curr_norm += com*com
        
        print(Sign)

        if n_iter == 0:
            op_indices = [ op_indices for _, op_indices in sorted(zip(com_list,op_indices),reverse=True)]
            for x in op_indices:
                print(fermi_ops[x],com_list[x])
            op_indices = op_indices[:cut_threshold]

        next_index = random.choice(op_indices)

        curr_norm = np.sqrt(curr_norm)

        min_options = {'gtol': theta_thresh, 'disp':False}
     

        max_of_com = next_deriv
        print(" Norm of <[A,H]> = %12.8f" %curr_norm)
        print(" Max  of <[A,H]> = %12.8f" %max_of_com)
        print(" Nonzero gradient: %12.5i" %nonzero_start)

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
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4s %20f %10s" %(opstring, parameters[si], si) )
                print(" ")

                compiler_engine = uccsd_trotter_engine( compiler_backend=CommandPrinter() )
                wavefunction = compiler_engine.allocate_qureg(molecule.n_qubits)
        
                H = 1j*s  # Qubits -pool
        
                # Trotter step parameters.
                time = parameters[si]
        
                evolution_operator = TimeEvolution(time,H)
        
                evolution_operator | wavefunction
        
                compiler_engine.flush()

            break

            break

        new_op = pool.fermi_ops[next_index]
        new_mat = pool.spmat_ops[next_index]

        for n in range(len(group)):
            new_op += Sign[n]*pool.fermi_ops[group[n]]
            new_mat += Sign[n]*pool.spmat_ops[group[n]]

        print(" Add operator %4i" %next_index)

        # for n in range(n_iter):
        # 	parameters[n] = 0

        for n in group:
            print(" Add operator %4i " %n)

        parameters.insert(0,0)
        parameters_index.append(n_iter)
        ansatz_ops.insert(0,new_op)
        ansatz_mat.insert(0,new_mat)
        parameters_mult.insert(0,1)
        
        trial_model = tUCCSD(hamiltonian, ansatz_mat, ansatz_ops, ansatz_mat_t, ansatz_ops_t, reference_ket, parameters, parameters_index, parameters_mult, ansatz_sup_index, ansatz_left, ansatz_right, ansatz_scal)
        

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
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4s %20f %10s" %(opstring, parameters[si], si) )
                print(" ")

def q_adapt_vqe_rand(geometry,
        basis           = "sto-3g",
        multiplicity    = 1,
        charge          = 0,
        adapt_conver    = 'norm',
        adapt_thresh    = 1e-7,
        theta_thresh    = 1e-12,
        adapt_maxiter   = 400,
        pool            = operator_pools.qubits(),
        spin_adapt      = True,
        psi4_filename   = "psi4_%12.12f"%random.random()
        ):
# {{{
       
    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
    molecule.filename = psi4_filename
    molecule = openfermionpsi4.run_psi4(molecule, 
                run_scf = 1, 
                run_mp2=1, 
                run_cisd=0, 
                run_ccsd = 0, 
                run_fci=1, 
                delete_input=1)
    
    pool.init(molecule)
    print(" Basis: ", basis)

    print(' HF energy      %20.16f au' %(molecule.hf_energy))
    print(' MP2 energy     %20.16f au' %(molecule.mp2_energy))
    #print(' CISD energy    %20.16f au' %(molecule.cisd_energy))
    #print(' CCSD energy    %20.16f au' %(molecule.ccsd_energy))
    print(' FCI energy     %20.16f au' %(molecule.fci_energy))

    #JW transform Hamiltonian computed classically with OFPsi4
    hamiltonian_op = molecule.get_molecular_hamiltonian()
    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)
    # print("H",hamiltonian)
  #  print(hamiltonian[:,240])
    HJW = openfermion.transforms.jordan_wigner(hamiltonian_op)

    # print(HJW)

    #Thetas
    parameters = []
    parameters_index = []
    parameters_mult = []
    Sign = []

    pool.generate_SparseMatrix()
   
    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz
    ansatz_ops_t = []
    ansatz_mat_t = []
    ansatz_sup_index = []
    ansatz_left = []
    ansatz_right = []
    ansatz_scal = []
    
    #Build p-h reference and map it to JW transform
    reference_ket = scipy.sparse.csc_matrix(
            openfermion.jw_configuration_state(
                list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()

    # for n in range(molecule.n_electrons, molecule.n_qubits):
    # 	reference_ket = 1/np.sqrt(2)*(np.exp(1j*0.5)*-1j*pool.spmat_ops[n].dot(reference_ket) + reference_ket)
    # 	print(pool.fermi_ops[n])
    
    # reference_ket = 1/np.sqrt(2)*(np.exp(1j*0.5)*-1j*pool.spmat_ops[3].dot(reference_ket) + reference_ket)

    print(reference_ket)

    reference_bra = reference_ket.transpose().conj()

    print(" Start ADAPT-VQE algorithm")
    op_indices = []
    parameters = []

    curr_state = 1.0*reference_ket
    
    # for n in range(16,len(pool.spmat_ops)):
    # 	curr_state = 1/np.sqrt(2)*(pool.spmat_ops[n].dot(curr_state) + curr_state)
    # 	print(pool.fermi_ops[n])

    # curr_state = 1/np.sqrt(2)*(pool.spmat_ops[49].dot(curr_state) + curr_state)

    fermi_ops = pool.fermi_ops
    spmat_ops = pool.spmat_ops
    n_ops = pool.n_ops

    Num = QubitOperator('Z1', 0)

    for p in range(0,2*pool.n_orb):
        Num += QubitOperator('Z%d' %p)

    Num = openfermion.transforms.get_sparse_operator(Num, n_qubits = pool.n_spin_orb)

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

        N_el = curr_state.transpose().conj().dot(Num.dot(curr_state))
        print('Number of electron:', N_el)


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
        
        print(Sign)

        next_index = random.choice(list(range(pool.n_ops)))

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
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4s %20f %10s" %(opstring, parameters[si], si) )
                print(" ")

                compiler_engine = uccsd_trotter_engine( compiler_backend=CommandPrinter() )
                wavefunction = compiler_engine.allocate_qureg(molecule.n_qubits)
        
                H = 1j*s  # Qubits -pool
        
                # Trotter step parameters.
                time = parameters[si]
        
                evolution_operator = TimeEvolution(time,H)
        
                evolution_operator | wavefunction
        
                compiler_engine.flush()

            break

            break

        new_op = pool.fermi_ops[next_index]
        new_mat = pool.spmat_ops[next_index]

        for n in range(len(group)):
            new_op += Sign[n]*pool.fermi_ops[group[n]]
            new_mat += Sign[n]*pool.spmat_ops[group[n]]

        print(" Add operator %4i" %next_index)

        # for n in range(n_iter):
        # 	parameters[n] = 0

        for n in group:
            print(" Add operator %4i " %n)

        parameters.insert(0,0)
        parameters_index.append(n_iter)
        ansatz_ops.insert(0,new_op)
        ansatz_mat.insert(0,new_mat)
        parameters_mult.insert(0,1)
        
        trial_model = tUCCSD(hamiltonian, ansatz_mat, ansatz_ops, ansatz_mat_t, ansatz_ops_t, reference_ket, parameters, parameters_index, parameters_mult, ansatz_sup_index, ansatz_left, ansatz_right, ansatz_scal)
        

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
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4s %20f %10s" %(opstring, parameters[si], si) )
                print(" ")

def adapt_vqe_trot_same(geometry,
        basis           = "sto-3g",
        multiplicity    = 1,
        charge          = 1,
        adapt_conver    = 'norm',
        adapt_thresh    = 1e-7,
        theta_thresh    = 1e-12,
        adapt_maxiter   = 400,
        pool            = operator_pools.singlet_GSD(),
        spin_adapt      = True,
        psi4_filename   = "psi4_%12.12f"%random.random()
        ):
# {{{
       
    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
    molecule.filename = psi4_filename
    molecule = openfermionpsi4.run_psi4(molecule, 
                run_scf = 1, 
                run_mp2=1, 
                run_cisd=0, 
                run_ccsd = 0, 
                run_fci=1, 
                delete_input=1)
    
    pool.init(molecule)
    print(" Basis: ", basis)

    print(' HF energy      %20.16f au' %(molecule.hf_energy))
    print(' MP2 energy     %20.16f au' %(molecule.mp2_energy))
    #print(' CISD energy    %20.16f au' %(molecule.cisd_energy))
    #print(' CCSD energy    %20.16f au' %(molecule.ccsd_energy))
    print(' FCI energy     %20.16f au' %(molecule.fci_energy))

    #JW transform Hamiltonian computed classically with OFPsi4
    hamiltonian_op = molecule.get_molecular_hamiltonian()
    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)
    # print("H",hamiltonian)
  #  print(hamiltonian[:,240])
    HJW = openfermion.transforms.jordan_wigner(hamiltonian_op)

    # print(HJW)

    #Thetas
    parameters = []
    parameters_index = []
    parameters_mult = []
    Sign = []

    pool.generate_SparseMatrix()
   
    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz
    ansatz_ops_t = []
    ansatz_mat_t = []
    ansatz_sup_index = []
    ansatz_left = []
    ansatz_right = []
    ansatz_scal = []
    
    #Build p-h reference and map it to JW transform
    reference_ket = scipy.sparse.csc_matrix(
            openfermion.jw_configuration_state(
                list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()

    # for n in range(molecule.n_electrons, molecule.n_qubits):
    # 	reference_ket = 1/np.sqrt(2)*(np.exp(1j*0.5)*-1j*pool.spmat_ops[n].dot(reference_ket) + reference_ket)
    # 	print(pool.fermi_ops[n])
    
    # reference_ket = 1/np.sqrt(2)*(np.exp(1j*0.5)*-1j*pool.spmat_ops[3].dot(reference_ket) + reference_ket)

    print(reference_ket)

    reference_bra = reference_ket.transpose().conj()

    print(" Start ADAPT-VQE algorithm")
    op_indices = []
    parameters = []

    curr_state = 1.0*reference_ket
    
    fermi_ops = pool.fermi_ops
    spmat_ops = pool.spmat_ops
    n_ops = pool.n_ops

    Num = QubitOperator('Z1', 0)

    for p in range(0,2*pool.n_orb):
        Num += QubitOperator('Z%d' %p)

    Num = openfermion.transforms.get_sparse_operator(Num, n_qubits = pool.n_spin_orb)

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

        N_el = curr_state.transpose().conj().dot(Num.dot(curr_state))
        print('Number of electron:', N_el)

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
       
            # if abs(com) > adapt_thresh:
            print(" %4i %40s %12.8f" %(op_trial, opstring, com) )

            curr_norm += com*com

            if abs(com) > abs(next_deriv) + 1e-9:
                next_deriv = com
                next_index = op_trial
        
        print(Sign)

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

            for si in range(len(ansatz_ops_t)):  # trottered fermi ops
                s = ansatz_ops_t[si]
                opstring = ""
                for t in s.terms:
                    tt = openfermion.FermionOperator(t,s.terms[t])
                    tt -= openfermion.hermitian_conjugated(tt)
                    print(" %4s %20f %10s" %(tt, parameters[parameters_index[si]], parameters_index[si]) )
                    print(" ")

                    compiler_engine = uccsd_trotter_engine( compiler_backend=CommandPrinter() )
                    wavefunction = compiler_engine.allocate_qureg(molecule.n_qubits)
            
                    H = openfermion.transforms.jordan_wigner(1j*tt)  # fermionic pool
            
                    # Trotter step parameters.
                    time = parameters[parameters_index[si]]
            
                    evolution_operator = TimeEvolution(time,H)
            
                    evolution_operator | wavefunction
            
                    compiler_engine.flush()

            break

            break

        new_op = pool.fermi_ops[next_index]
        new_mat = pool.spmat_ops[next_index]

        for n in range(len(group)):
            new_op += Sign[n]*pool.fermi_ops[group[n]]
            new_mat += Sign[n]*pool.spmat_ops[group[n]]

        print(" Add operator %4i" %next_index)

        # for n in range(n_iter):
        # 	parameters[n] = 0

        for n in group:
            print(" Add operator %4i " %n)

        ansatz_ops.insert(0,new_op)
        ansatz_mat.insert(0,new_mat)

        # trotter fermi ops
        
        NN = 0
        MM = 0
        LL = 0

        if n_iter > 0:
        	for k in range(0, len(parameters_index)):
        		parameters_index[k] += 1

        for tt in new_op.terms:
            NN += 1

        for t in new_op.terms:
            MM += 1
            tF = openfermion.FermionOperator(t,new_op.terms[t])
            tQ = openfermion.transforms.get_sparse_operator(tF - openfermion.hermitian_conjugated(tF), n_qubits = pool.n_spin_orb)
            ansatz_ops_t.insert(0,tF)
            ansatz_mat_t.insert(0,tQ)
            #================================================
            parameters_index.insert(0,0)                   # same parameters for different terms in new fermi op
            if MM == int(NN/2):
                parameters_mult.insert(0,MM)
                break

        parameters.insert(0,0)
        # #====================================================
            
        #     # if LL > 0:
        #     #     for k in range(0, len(parameters_index)):
        #     #         parameters_index[k] += 1
        #     # parameters_index.insert(0,0)                   # different parameters
        #     # parameters.insert(0,0)
        #     # parameters_mult.insert(0,1)
        #     # LL += 1
        #     # if MM == int(NN/2):
        #     # 	break

        # #================================================================================

        # # print(openfermion.transforms.jordan_wigner(new_op))

        
        trial_model = ttUCCSD(hamiltonian, ansatz_mat, ansatz_ops, ansatz_mat_t, ansatz_ops_t, reference_ket, parameters, parameters_index, parameters_mult, ansatz_sup_index, ansatz_left, ansatz_right, ansatz_scal)
        

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
        for si in range(0,len(ansatz_ops_t)):
            s = ansatz_ops_t[si]
            opstring = ""
            for t in s.terms:
                tt = openfermion.FermionOperator(t,s.terms[t])
                tt -= openfermion.hermitian_conjugated(tt)
                print(" %4s %20f %10s" %(tt, parameters[parameters_index[si]], parameters_index[si]) )
                print(" ")

def adapt_vqe_trot_same(geometry,
        basis           = "sto-3g",
        multiplicity    = 1,
        charge          = 1,
        adapt_conver    = 'norm',
        adapt_thresh    = 1e-7,
        theta_thresh    = 1e-12,
        adapt_maxiter   = 400,
        pool            = operator_pools.singlet_GSD(),
        spin_adapt      = True,
        psi4_filename   = "psi4_%12.12f"%random.random()
        ):
# {{{
       
    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
    molecule.filename = psi4_filename
    molecule = openfermionpsi4.run_psi4(molecule, 
                run_scf = 1, 
                run_mp2=1, 
                run_cisd=0, 
                run_ccsd = 0, 
                run_fci=1, 
                delete_input=1)
    
    pool.init(molecule)
    print(" Basis: ", basis)

    print(' HF energy      %20.16f au' %(molecule.hf_energy))
    print(' MP2 energy     %20.16f au' %(molecule.mp2_energy))
    #print(' CISD energy    %20.16f au' %(molecule.cisd_energy))
    #print(' CCSD energy    %20.16f au' %(molecule.ccsd_energy))
    print(' FCI energy     %20.16f au' %(molecule.fci_energy))

    #JW transform Hamiltonian computed classically with OFPsi4
    hamiltonian_op = molecule.get_molecular_hamiltonian()
    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)
    # print("H",hamiltonian)
  #  print(hamiltonian[:,240])
    HJW = openfermion.transforms.jordan_wigner(hamiltonian_op)

    # print(HJW)

    #Thetas
    parameters = []
    parameters_index = []
    parameters_mult = []
    Sign = []

    pool.generate_SparseMatrix()
   
    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz
    ansatz_ops_t = []
    ansatz_mat_t = []
    ansatz_sup_index = []
    ansatz_left = []
    ansatz_right = []
    ansatz_scal = []
    
    #Build p-h reference and map it to JW transform
    reference_ket = scipy.sparse.csc_matrix(
            openfermion.jw_configuration_state(
                list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()

    # for n in range(molecule.n_electrons, molecule.n_qubits):
    # 	reference_ket = 1/np.sqrt(2)*(np.exp(1j*0.5)*-1j*pool.spmat_ops[n].dot(reference_ket) + reference_ket)
    # 	print(pool.fermi_ops[n])
    
    # reference_ket = 1/np.sqrt(2)*(np.exp(1j*0.5)*-1j*pool.spmat_ops[3].dot(reference_ket) + reference_ket)

    print(reference_ket)

    reference_bra = reference_ket.transpose().conj()

    print(" Start ADAPT-VQE algorithm")
    op_indices = []
    parameters = []

    curr_state = 1.0*reference_ket
    
    fermi_ops = pool.fermi_ops
    spmat_ops = pool.spmat_ops
    n_ops = pool.n_ops

    Num = QubitOperator('Z1', 0)

    for p in range(0,2*pool.n_orb):
        Num += QubitOperator('Z%d' %p)

    Num = openfermion.transforms.get_sparse_operator(Num, n_qubits = pool.n_spin_orb)

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

        N_el = curr_state.transpose().conj().dot(Num.dot(curr_state))
        print('Number of electron:', N_el)

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
       
            # if abs(com) > adapt_thresh:
            print(" %4i %40s %12.8f" %(op_trial, opstring, com) )

            curr_norm += com*com

            if abs(com) > abs(next_deriv) + 1e-9:
                next_deriv = com
                next_index = op_trial
        
        print(Sign)

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

            for si in range(len(ansatz_ops_t)):  # trottered fermi ops
                s = ansatz_ops_t[si]
                opstring = ""
                for t in s.terms:
                    tt = openfermion.FermionOperator(t,s.terms[t])
                    tt -= openfermion.hermitian_conjugated(tt)
                    print(" %4s %20f %10s" %(tt, parameters[parameters_index[si]], parameters_index[si]) )
                    print(" ")

                    compiler_engine = uccsd_trotter_engine( compiler_backend=CommandPrinter() )
                    wavefunction = compiler_engine.allocate_qureg(molecule.n_qubits)
            
                    H = openfermion.transforms.jordan_wigner(1j*tt)  # fermionic pool
            
                    # Trotter step parameters.
                    time = parameters[parameters_index[si]]
            
                    evolution_operator = TimeEvolution(time,H)
            
                    evolution_operator | wavefunction
            
                    compiler_engine.flush()

            break

            break

        new_op = pool.fermi_ops[next_index]
        new_mat = pool.spmat_ops[next_index]

        for n in range(len(group)):
            new_op += Sign[n]*pool.fermi_ops[group[n]]
            new_mat += Sign[n]*pool.spmat_ops[group[n]]

        print(" Add operator %4i" %next_index)

        # for n in range(n_iter):
        # 	parameters[n] = 0

        for n in group:
            print(" Add operator %4i " %n)

        ansatz_ops.insert(0,new_op)
        ansatz_mat.insert(0,new_mat)

        # trotter fermi ops
        
        NN = 0
        MM = 0
        LL = 0

        if n_iter > 0:
        	for k in range(0, len(parameters_index)):
        		parameters_index[k] += 1

        for tt in new_op.terms:
            NN += 1

        for t in new_op.terms:
            MM += 1
            tF = openfermion.FermionOperator(t,new_op.terms[t])
            tQ = openfermion.transforms.get_sparse_operator(tF - openfermion.hermitian_conjugated(tF), n_qubits = pool.n_spin_orb)
            ansatz_ops_t.insert(0,tF)
            ansatz_mat_t.insert(0,tQ)            
            if LL > 0:
                for k in range(0, len(parameters_index)):
                    parameters_index[k] += 1
            parameters_index.insert(0,0)                   # different parameters
            parameters.insert(0,0)
            parameters_mult.insert(0,1)
            LL += 1
            if MM == int(NN/2):
            	break

        # #================================================================================

        # # print(openfermion.transforms.jordan_wigner(new_op))

        
        trial_model = ttUCCSD(hamiltonian, ansatz_mat, ansatz_ops, ansatz_mat_t, ansatz_ops_t, reference_ket, parameters, parameters_index, parameters_mult, ansatz_sup_index, ansatz_left, ansatz_right, ansatz_scal)
        

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
        for si in range(0,len(ansatz_ops_t)):
            s = ansatz_ops_t[si]
            opstring = ""
            for t in s.terms:
                tt = openfermion.FermionOperator(t,s.terms[t])
                tt -= openfermion.hermitian_conjugated(tt)
                print(" %4s %20f %10s" %(tt, parameters[parameters_index[si]], parameters_index[si]) )
                print(" ")

def q_adapt_vqe_sup(geometry,
        basis           = "sto-3g",
        multiplicity    = 1,
        charge          = 1,
        adapt_conver    = 'norm',
        adapt_thresh    = 1e-7,
        theta_thresh    = 1e-12,
        adapt_maxiter   = 400,
        pool            = operator_pools.singlet_GSD(),
        spin_adapt      = True,
        psi4_filename   = "psi4_%12.12f"%random.random()
        ):
# {{{
       
    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
    molecule.filename = psi4_filename
    molecule = openfermionpsi4.run_psi4(molecule, 
                run_scf = 1, 
                run_mp2=1, 
                run_cisd=0, 
                run_ccsd = 0, 
                run_fci=1, 
                delete_input=1)
    
    pool.init(molecule)
    print(" Basis: ", basis)

    print(' HF energy      %20.16f au' %(molecule.hf_energy))
    print(' MP2 energy     %20.16f au' %(molecule.mp2_energy))
    #print(' CISD energy    %20.16f au' %(molecule.cisd_energy))
    #print(' CCSD energy    %20.16f au' %(molecule.ccsd_energy))
    print(' FCI energy     %20.16f au' %(molecule.fci_energy))

    #JW transform Hamiltonian computed classically with OFPsi4
    hamiltonian_op = molecule.get_molecular_hamiltonian()
    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)
    # print("H",hamiltonian)
  #  print(hamiltonian[:,240])
    HJW = openfermion.transforms.jordan_wigner(hamiltonian_op)

    # print(HJW)

    #Thetas
    parameters = []
    parameters_index = []
    parameters_mult = []
    Sign = []

    pool.generate_SparseMatrix()
   
    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz
    ansatz_ops_t = []
    ansatz_mat_t = []
    ansatz_sup_index = []
    ansatz_left = []
    ansatz_right = []
    ansatz_scal = []
    
    #Build p-h reference and map it to JW transform
    reference_ket = scipy.sparse.csc_matrix(
            openfermion.jw_configuration_state(
                list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()

    # for n in range(molecule.n_electrons, molecule.n_qubits):
    # 	reference_ket = 1/np.sqrt(2)*(np.exp(1j*0.5)*-1j*pool.spmat_ops[n].dot(reference_ket) + reference_ket)
    # 	print(pool.fermi_ops[n])
    
    # reference_ket = 1/np.sqrt(2)*(np.exp(1j*0.5)*-1j*pool.spmat_ops[3].dot(reference_ket) + reference_ket)

    print(reference_ket)

    reference_bra = reference_ket.transpose().conj()

    print(" Start ADAPT-VQE algorithm")
    op_indices = []
    parameters = []

    curr_state = 1.0*reference_ket
    
    fermi_ops = pool.fermi_ops
    spmat_ops = pool.spmat_ops
    n_ops = pool.n_ops

    Num = QubitOperator('Z1', 0)

    for p in range(0,2*pool.n_orb):
        Num += QubitOperator('Z%d' %p)

    Num = openfermion.transforms.get_sparse_operator(Num, n_qubits = pool.n_spin_orb)

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

        N_el = curr_state.transpose().conj().dot(Num.dot(curr_state))
        print('Number of electron:', N_el)

        for op_trial in range(pool.n_ops):

            if pool.sup_index[op_trial] == 0:
                opA = pool.spmat_ops[op_trial]
                com = 2*(curr_state.transpose().conj().dot(opA.dot(sig))).real
                assert(com.shape == (1,1))
                com = com[0,0]
                assert(np.isclose(com.imag,0))
                com = com.real
            else:
                opA = pool.spmat_ops[op_trial]
                left = openfermion.transforms.get_sparse_operator(pool.left_ops[pool.sup_index[op_trial]-1], n_qubits = pool.n_spin_orb)
                right = openfermion.transforms.get_sparse_operator(pool.right_ops[pool.sup_index[op_trial]-1], n_qubits = pool.n_spin_orb)
                ket = scipy.sparse.linalg.expm_multiply(right,curr_state)
                com = 2*(ket.transpose().conj().dot(scipy.sparse.linalg.expm_multiply(-1*left,hamiltonian.dot(scipy.sparse.linalg.expm_multiply(left, opA.dot(ket))))))
                assert(com.shape == (1,1))
                com = com[0,0]
                assert(np.isclose(com.imag,0))
                com = com.real

            opstring = ""
            for t in pool.fermi_ops[op_trial].terms:
                opstring += str(t)
                break
       
            # if abs(com) > adapt_thresh:
            print(" %4i %40s %12.8f" %(op_trial, opstring, com) )

            curr_norm += com*com

            if abs(com) > abs(next_deriv) + 1e-9:
                next_deriv = com
                next_index = op_trial
        
        print(Sign)

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
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4s %20f %10s" %(opstring, parameters[si], si) )
                print(" ")

                compiler_engine = uccsd_trotter_engine( compiler_backend=CommandPrinter() )
                wavefunction = compiler_engine.allocate_qureg(molecule.n_qubits)
        
                H = 1j*s  # Qubits pool
                # H = openfermion.transforms.jordan_wigner(1j*s)  # fermionic pool
        
                # Trotter step parameters.
                time = parameters[si]
        
                evolution_operator = TimeEvolution(time,H)
        
                evolution_operator | wavefunction
        
                compiler_engine.flush()

            break

            break

        new_op = pool.fermi_ops[next_index]
        new_mat = pool.spmat_ops[next_index]

        for n in range(len(group)):
            new_op += Sign[n]*pool.fermi_ops[group[n]]
            new_mat += Sign[n]*pool.spmat_ops[group[n]]

        print(" Add operator %4i" %next_index)

        # for n in range(n_iter):
        # 	parameters[n] = 0

        for n in group:
            print(" Add operator %4i " %n)

        #===========================================
        parameters.insert(0,0)
        parameters_index.append(n_iter)
        ansatz_ops.insert(0,new_op)
        ansatz_mat.insert(0,new_mat)
        parameters_mult.insert(0,1)
        #================================================== A gate
        ansatz_sup_index.insert(0,pool.sup_index[next_index])
        if pool.sup_index[next_index] > 0:
            ansatz_left.append(openfermion.transforms.get_sparse_operator(pool.left_ops[pool.sup_index[next_index]-1], n_qubits = pool.n_spin_orb))
            ansatz_right.append(openfermion.transforms.get_sparse_operator(pool.right_ops[pool.sup_index[next_index]-1], n_qubits = pool.n_spin_orb))
            ansatz_scal.append(pool.scalar[pool.sup_index[next_index]-1])
        #===========================================
        
        trial_model = A(hamiltonian, ansatz_mat, ansatz_ops, ansatz_mat_t, ansatz_ops_t, reference_ket, parameters, parameters_index, parameters_mult, ansatz_sup_index, ansatz_left, ansatz_right, ansatz_scal)
        

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
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4s %20f %10s" %(opstring, parameters[si], si) )
                print(" ")
       
def q_adapt_vqe_hess(geometry,
        basis           = "sto-3g",
        multiplicity    = 1,
        charge          = 1,
        adapt_conver    = 'norm',
        adapt_thresh    = 1e-7,
        theta_thresh    = 1e-12,
        adapt_maxiter   = 400,
        pool            = operator_pools.singlet_GSD(),
        spin_adapt      = True,
        psi4_filename   = "psi4_%12.12f"%random.random()
        ):
# {{{
       
    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
    molecule.filename = psi4_filename
    molecule = openfermionpsi4.run_psi4(molecule, 
                run_scf = 1, 
                run_mp2=1, 
                run_cisd=0, 
                run_ccsd = 0, 
                run_fci=1, 
                delete_input=1)
    
    pool.init(molecule)
    print(" Basis: ", basis)

    print(' HF energy      %20.16f au' %(molecule.hf_energy))
    print(' MP2 energy     %20.16f au' %(molecule.mp2_energy))
    #print(' CISD energy    %20.16f au' %(molecule.cisd_energy))
    #print(' CCSD energy    %20.16f au' %(molecule.ccsd_energy))
    print(' FCI energy     %20.16f au' %(molecule.fci_energy))

    #JW transform Hamiltonian computed classically with OFPsi4
    hamiltonian_op = molecule.get_molecular_hamiltonian()
    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)
    # print("H",hamiltonian)
  #  print(hamiltonian[:,240])
    HJW = openfermion.transforms.jordan_wigner(hamiltonian_op)

    # print(HJW)

    #Thetas
    parameters = []
    parameters_index = []
    parameters_mult = []
    Sign = []

    pool.generate_SparseMatrix()
   
    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz
    ansatz_ops_t = []
    ansatz_mat_t = []
    ansatz_sup_index = []
    ansatz_left = []
    ansatz_right = []
    ansatz_scal = []
    
    #Build p-h reference and map it to JW transform
    reference_ket = scipy.sparse.csc_matrix(
            openfermion.jw_configuration_state(
                list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()

    # for n in range(molecule.n_electrons, molecule.n_qubits):
    # 	reference_ket = 1/np.sqrt(2)*(np.exp(1j*0.5)*-1j*pool.spmat_ops[n].dot(reference_ket) + reference_ket)
    # 	print(pool.fermi_ops[n])
    
    # reference_ket = 1/np.sqrt(2)*(np.exp(1j*0.5)*-1j*pool.spmat_ops[3].dot(reference_ket) + reference_ket)

    print(reference_ket)

    reference_bra = reference_ket.transpose().conj()

    print(" Start ADAPT-VQE algorithm")
    op_indices = []
    parameters = []

    curr_state = 1.0*reference_ket
    
    # for n in range(16,len(pool.spmat_ops)):
    # 	curr_state = 1/np.sqrt(2)*(pool.spmat_ops[n].dot(curr_state) + curr_state)
    # 	print(pool.fermi_ops[n])

    # curr_state = 1/np.sqrt(2)*(pool.spmat_ops[49].dot(curr_state) + curr_state)

    fermi_ops = pool.fermi_ops
    spmat_ops = pool.spmat_ops
    n_ops = pool.n_ops

    Num = QubitOperator('Z1', 0)

    for p in range(0,2*pool.n_orb):
        Num += QubitOperator('Z%d' %p)

    Num = openfermion.transforms.get_sparse_operator(Num, n_qubits = pool.n_spin_orb)

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

        N_el = curr_state.transpose().conj().dot(Num.dot(curr_state))
        print('Number of electron:', N_el)

        hess = np.zeros((n_ops,n_ops))
    
        for ai in range(n_ops):
            opA = spmat_ops[ai]
            pA = opA.dot(curr_state)
        
            for bi in range(n_ops):
                opB = spmat_ops[bi]
                pB = opB.dot(curr_state)
               
                term1 = 2*(sig.transpose().conj().dot(opA.dot(pB))).real 
                term2 = 2*(pA.transpose().conj().dot(hamiltonian.dot(pB))).real
             
                term = term1 + term2
                assert(term.shape == (1,1))
                term = term[0,0]
                hess[ai,bi] = term

        lin_dep_thresh = 1e-4
    
        U,s,V = np.linalg.svd(hess)
        n_vecs = 0
    
        hess_pool = []
    
        for j in  range(0,n_ops):
            # if s[j] > lin_dep_thresh:
            left_vec = FermionOperator(((1,1),(2,0)), 0)
            # left_vec = QubitOperator('X1', 0)
            n_vecs += 1
            for i in range(0,n_ops):
                left_vec += U[i,j]*fermi_ops[i]
                # print('right vectors :' v[:,n_vecs])
            # print('left vectors :', left_vec)
            # print('')

            hess_pool.append(left_vec)
    
        hess_mat_pool = []
    
        hess_n_ops = 0
    
        for op in hess_pool:
            hess_mat_pool.append(transforms.get_sparse_operator(op, n_qubits = pool.n_spin_orb))
        hess_n_ops = len(hess_pool)

        print('number of effecive ops:', hess_n_ops)
        
        pool.n_ops = hess_n_ops
        pool.fermi_ops = hess_pool
        pool.spmat_ops = hess_mat_pool

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
       
            # if abs(com) > adapt_thresh:
            print(" %4i %40s %12.8f" %(op_trial, opstring, com) )

            curr_norm += com*com

            #====================================================

            if abs(com) > abs(next_deriv) + 1e-9:
                next_deriv = com
                next_index = op_trial
        
        print(Sign)

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
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4s %20f %10s" %(opstring, parameters[si], si) )
                print(" ")

                compiler_engine = uccsd_trotter_engine( compiler_backend=CommandPrinter() )
                wavefunction = compiler_engine.allocate_qureg(molecule.n_qubits)
        
                # H = openfermion.transforms.get_interaction_operator(1j*T)
                H = 1j*s  # Qubits pool
                # H = openfermion.transforms.jordan_wigner(1j*s)  # fermionic pool
        
                # Trotter step parameters.
                time = parameters[si]
        
                evolution_operator = TimeEvolution(time,H)
        
                evolution_operator | wavefunction
        
                compiler_engine.flush()

            break

            break

        new_op = pool.fermi_ops[next_index]
        new_mat = pool.spmat_ops[next_index]

        for n in range(len(group)):
            new_op += Sign[n]*pool.fermi_ops[group[n]]
            new_mat += Sign[n]*pool.spmat_ops[group[n]]

        print(" Add operator %4i" %next_index)

        # for n in range(n_iter):
        # 	parameters[n] = 0

        for n in group:
            print(" Add operator %4i " %n)

        #===========================================
        parameters.insert(0,0)
        parameters_index.append(n_iter)
        ansatz_ops.insert(0,new_op)
        ansatz_mat.insert(0,new_mat)
        parameters_mult.insert(0,1)
        
        trial_model = tUCCSD(hamiltonian, ansatz_mat, ansatz_ops, ansatz_mat_t, ansatz_ops_t, reference_ket, parameters, parameters_index, parameters_mult, ansatz_sup_index, ansatz_left, ansatz_right, ansatz_scal)
        

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
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4s %20f %10s" %(opstring, parameters[si], si) )
                print(" ")

def adapt_vqe_multiselect(geometry,
        basis           = "sto-3g",
        multiplicity    = 1,
        charge          = 1,
        adapt_conver    = 'norm',
        adapt_thresh    = 1e-7,
        theta_thresh    = 1e-12,
        adapt_maxiter   = 400,
        pool            = operator_pools.singlet_GSD(),
        spin_adapt      = True,
        psi4_filename   = "psi4_%12.12f"%random.random()
        ):
# {{{
       
    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
    molecule.filename = psi4_filename
    molecule = openfermionpsi4.run_psi4(molecule, 
                run_scf = 1, 
                run_mp2=1, 
                run_cisd=0, 
                run_ccsd = 0, 
                run_fci=1, 
                delete_input=1)
    
    pool.init(molecule)
    print(" Basis: ", basis)

    print(' HF energy      %20.16f au' %(molecule.hf_energy))
    print(' MP2 energy     %20.16f au' %(molecule.mp2_energy))
    #print(' CISD energy    %20.16f au' %(molecule.cisd_energy))
    #print(' CCSD energy    %20.16f au' %(molecule.ccsd_energy))
    print(' FCI energy     %20.16f au' %(molecule.fci_energy))

    #JW transform Hamiltonian computed classically with OFPsi4
    hamiltonian_op = molecule.get_molecular_hamiltonian()
    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)
    # print("H",hamiltonian)
  #  print(hamiltonian[:,240])
    HJW = openfermion.transforms.jordan_wigner(hamiltonian_op)

    # print(HJW)

    #Thetas
    parameters = []
    parameters_index = []
    parameters_mult = []
    Sign = []

    pool.generate_SparseMatrix()
   
    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz
    ansatz_ops_t = []
    ansatz_mat_t = []
    ansatz_sup_index = []
    ansatz_left = []
    ansatz_right = []
    ansatz_scal = []
    
    #Build p-h reference and map it to JW transform
    reference_ket = scipy.sparse.csc_matrix(
            openfermion.jw_configuration_state(
                list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()

    # for n in range(molecule.n_electrons, molecule.n_qubits):
    # 	reference_ket = 1/np.sqrt(2)*(np.exp(1j*0.5)*-1j*pool.spmat_ops[n].dot(reference_ket) + reference_ket)
    # 	print(pool.fermi_ops[n])
    
    # reference_ket = 1/np.sqrt(2)*(np.exp(1j*0.5)*-1j*pool.spmat_ops[3].dot(reference_ket) + reference_ket)

    print(reference_ket)

    reference_bra = reference_ket.transpose().conj()

    print(" Start ADAPT-VQE algorithm")
    op_indices = []
    parameters = []

    curr_state = 1.0*reference_ket
    
    fermi_ops = pool.fermi_ops
    spmat_ops = pool.spmat_ops
    n_ops = pool.n_ops

    Num = QubitOperator('Z1', 0)

    for p in range(0,2*pool.n_orb):
        Num += QubitOperator('Z%d' %p)

    Num = openfermion.transforms.get_sparse_operator(Num, n_qubits = pool.n_spin_orb)

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

        N_el = curr_state.transpose().conj().dot(Num.dot(curr_state))
        print('Number of electron:', N_el)

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
       
            # if abs(com) > adapt_thresh:
            print(" %4i %40s %12.8f" %(op_trial, opstring, com) )

            curr_norm += com*com

            if abs(com) > abs(next_deriv) + 1e-6:
                group = []
                next_deriv = com
                next_index = op_trial
                Sign = []
            elif (next_deriv) > 1e-6:
                if abs(abs(com)-abs(next_deriv)) < 1e-6:
                    group.append(op_trial)
                    sign = com*next_deriv/abs(com*next_deriv)
                    Sign.append(sign)
        
        print(Sign)

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
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4s %20f %10s" %(opstring, parameters[si], si) )
                print(" ")

                compiler_engine = uccsd_trotter_engine( compiler_backend=CommandPrinter() )
                wavefunction = compiler_engine.allocate_qureg(molecule.n_qubits)
        
                # H = openfermion.transforms.get_interaction_operator(1j*T)
                H = 1j*s  # Qubits pool
                # H = openfermion.transforms.jordan_wigner(1j*s)  # fermionic pool
        
                # Trotter step parameters.
                time = parameters[si]
        
                evolution_operator = TimeEvolution(time,H)
        
                evolution_operator | wavefunction
        
                compiler_engine.flush()

            break

            break

        new_op = pool.fermi_ops[next_index]
        new_mat = pool.spmat_ops[next_index]

        for n in range(len(group)):
            new_op += Sign[n]*pool.fermi_ops[group[n]]
            new_mat += Sign[n]*pool.spmat_ops[group[n]]

        print(" Add operator %4i" %next_index)

        # for n in range(n_iter):
        # 	parameters[n] = 0

        for n in group:
            print(" Add operator %4i " %n)

        #===========================================
        parameters.insert(0,0)
        parameters_index.append(n_iter)
        ansatz_ops.insert(0,new_op)
        ansatz_mat.insert(0,new_mat)
        parameters_mult.insert(0,1)
        
        trial_model = tUCCSD(hamiltonian, ansatz_mat, ansatz_ops, ansatz_mat_t, ansatz_ops_t, reference_ket, parameters, parameters_index, parameters_mult, ansatz_sup_index, ansatz_left, ansatz_right, ansatz_scal)
        

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
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4s %20f %10s" %(opstring, parameters[si], si) )
                print(" ")
        
# }}}

def ucc(geometry,
        basis           = "sto-3g",
        multiplicity    = 1,
        charge          = 1,
        theta_thresh    = 1e-7,
        pool            = operator_pools.singlet_GSD(),
        spin_adapt      = True,
        psi4_filename   = "psi4_%12.12f"%random.random()
        ):
# {{{

    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
    molecule.filename = psi4_filename
    molecule = openfermionpsi4.run_psi4(molecule, 
                run_scf = 1, 
                run_mp2=1, 
                run_cisd=0, 
                run_ccsd = 0, 
                run_fci=1, 
                delete_input=1)
    pool.init(molecule)
    print(" Basis: ", basis)

    print(' HF energy      %20.16f au' %(molecule.hf_energy))
    print(' MP2 energy     %20.16f au' %(molecule.mp2_energy))
    #print(' CISD energy    %20.16f au' %(molecule.cisd_energy))
    #print(' CCSD energy    %20.16f au' %(molecule.ccsd_energy))
    print(' FCI energy     %20.16f au' %(molecule.fci_energy))

    #Build p-h reference and map it to JW transform
    reference_ket = scipy.sparse.csc_matrix(
            openfermion.jw_configuration_state(
                list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()
    reference_bra = reference_ket.transpose().conj()

    #JW transform Hamiltonian computed classically with OFPsi4
    hamiltonian_op = molecule.get_molecular_hamiltonian()
    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)

    #Thetas
    parameters = [0]*pool.n_ops 

    pool.generate_SparseMatrix()
    
    ucc = UCC(hamiltonian, pool.spmat_ops, reference_ket, parameters)
    
    opt_result = scipy.optimize.minimize(ucc.energy, 
                parameters, options = {'gtol': 1e-6, 'disp':True}, 
                method = 'BFGS', callback=ucc.callback)
    print(" Finished: %20.12f" % ucc.curr_energy)
    parameters = opt_result['x']
    for p in parameters:
        print(p)

# }}}

def test_random(geometry,
        basis           = "sto-3g",
        multiplicity    = 1,
        charge          = 1,
        adapt_conver    = 'norm',
        adapt_thresh    = 1e-3,
        theta_thresh    = 1e-7,
        adapt_maxiter   = 200,
        pool            = operator_pools.singlet_GSD(),
        spin_adapt      = True,
        psi4_filename   = "psi4_%12.12f"%random.random(),
        seed            = 1
        ):

    # {{{
    random.seed(seed)

    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
    molecule.filename = psi4_filename
    molecule = openfermionpsi4.run_psi4(molecule, 
                run_scf = 1, 
                run_mp2=1, 
                run_cisd=0, 
                run_ccsd = 0, 
                run_fci=1, 
                delete_input=1)
    pool.init(molecule)
    print(" Basis: ", basis)

    print(' HF energy      %20.16f au' %(molecule.hf_energy))
    print(' MP2 energy     %20.16f au' %(molecule.mp2_energy))
    #print(' CISD energy    %20.16f au' %(molecule.cisd_energy))
    #print(' CCSD energy    %20.16f au' %(molecule.ccsd_energy))
    print(' FCI energy     %20.16f au' %(molecule.fci_energy))

    #Build p-h reference and map it to JW transform
    reference_ket = scipy.sparse.csc_matrix(
            openfermion.jw_configuration_state(
                list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()
    reference_bra = reference_ket.transpose().conj()

    #JW transform Hamiltonian computed classically with OFPsi4
    hamiltonian_op = molecule.get_molecular_hamiltonian()
    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)

    #Thetas
    parameters = []

    pool.generate_SparseMatrix()
   
    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz
    
    print(" Start ADAPT-VQE algorithm")
    op_indices = []
    parameters = []
    curr_state = 1.0*reference_ket

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
        print(" Measure commutators:")
        sig = hamiltonian.dot(curr_state)
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
            if abs(com) > abs(next_deriv):
                next_deriv = com
                next_index = op_trial

      
        next_index = random.choice(list(range(pool.n_ops)))
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
            print(" %4s %40s %12s" %("#","Term","Coeff"))
            for si in range(len(ansatz_ops)):
                s = ansatz_ops[si]
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4i %40s %12.8f" %(si, opstring, parameters[si]) )
            break
        
        print(" Add operator %4i" %next_index)
        parameters.insert(0,0)
        ansatz_ops.insert(0,pool.fermi_ops[next_index])
        ansatz_mat.insert(0,pool.spmat_ops[next_index])
        
        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)
        

        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient, 
                options = min_options, method = 'BFGS', callback=trial_model.callback)
    
        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(" -----------New ansatz----------- ")
        print(" %4s %40s %12s" %("#","Term","Coeff"))
        for si in range(len(ansatz_ops)):
            s = ansatz_ops[si]
            opstring = ""
            for t in s.terms:
                opstring += str(t)
                break
            print(" %4i %40s %12.8f" %(si, opstring, parameters[si]) )

        # if n_iter == 0:

        #     with open('h4_q_adapt_rand.csv', mode='w') as h4:
        #         h4 = csv.writer(h4, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #         h4.writerow([molecule.fci_energy-trial_model.curr_energy, n_iter])
        # else:
        #     with open('h4_q_adapt_rand.csv', mode='a') as h4:
        #         h4 = csv.writer(h4, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #         h4.writerow([molecule.fci_energy-trial_model.curr_energy, n_iter])

    return
# }}}

def test_lexical(geometry,
        basis           = "sto-3g",
        multiplicity    = 1,
        charge          = 1,
        adapt_conver    = 'norm',
        adapt_thresh    = 1e-3,
        theta_thresh    = 1e-7,
        adapt_maxiter   = 200,
        pool            = operator_pools.singlet_GSD(),
        spin_adapt      = True,
        psi4_filename   = "psi4_%12.12f"%random.random()
        ):
# {{{

    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
    molecule.filename = psi4_filename
    molecule = openfermionpsi4.run_psi4(molecule, 
                run_scf = 1, 
                run_mp2=1, 
                run_cisd=0, 
                run_ccsd = 0, 
                run_fci=1, 
                delete_input=1)
    pool.init(molecule)
    print(" Basis: ", basis)

    print(' HF energy      %20.16f au' %(molecule.hf_energy))
    print(' MP2 energy     %20.16f au' %(molecule.mp2_energy))
    #print(' CISD energy    %20.16f au' %(molecule.cisd_energy))
    #print(' CCSD energy    %20.16f au' %(molecule.ccsd_energy))
    print(' FCI energy     %20.16f au' %(molecule.fci_energy))

    #Build p-h reference and map it to JW transform
    reference_ket = scipy.sparse.csc_matrix(
            openfermion.jw_configuration_state(
                list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()
    reference_bra = reference_ket.transpose().conj()

    #JW transform Hamiltonian computed classically with OFPsi4
    hamiltonian_op = molecule.get_molecular_hamiltonian()
    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)

    #Thetas
    parameters = []

    pool.generate_SparseMatrix()
   
    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz
    
    print(" Start ADAPT-VQE algorithm")
    op_indices = []
    parameters = []
    curr_state = 1.0*reference_ket

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
        print(" Measure commutators:")
        sig = hamiltonian.dot(curr_state)
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
            if abs(com) > abs(next_deriv):
                next_deriv = com
                next_index = op_trial

       
        next_index = n_iter % pool.n_ops
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
            print(" %4s %40s %12s" %("#","Term","Coeff"))
            for si in range(len(ansatz_ops)):
                s = ansatz_ops[si]
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4i %40s %12.8f" %(si, opstring, parameters[si]) )
            break
        
        print(" Add operator %4i" %next_index)
        parameters.insert(0,0)
        ansatz_ops.insert(0,pool.fermi_ops[next_index])
        ansatz_mat.insert(0,pool.spmat_ops[next_index])
        
        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)
        

        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient, 
                options = min_options, method = 'BFGS', callback=trial_model.callback)
    
        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(" -----------New ansatz----------- ")
        print(" %4s %40s %12s" %("#","Term","Coeff"))
        for si in range(len(ansatz_ops)):
            s = ansatz_ops[si]
            opstring = ""
            for t in s.terms:
                opstring += str(t)
                break
            print(" %4i %40s %12.8f" %(si, opstring, parameters[si]) )

    return
# }}}



if __name__== "__main__":
    r = 1.5
    geometry = [('H', (0,0,1*r)), ('H', (0,0,2*r)), ('H', (0,0,3*r)), ('H', (0,0,4*r))]

    vqe_methods.ucc(geometry,pool = operator_pools.singlet_SD())
    #vqe_methods.adapt_vqe(geometry,pool = operator_pools.singlet_SD())
    #vqe_methods.adapt_vqe(geometry,pool = operator_pools.singlet_GSD())
