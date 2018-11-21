import scipy
import openfermion
import openfermionpsi4
import os
import numpy as np
import copy
import random 
import sys

import hdvv
import qubit

from lib import Hamiltonian
from lib import ci_string 

from qubit import *

from openfermion import *

def update_ivqe_ref(parameters):
    new_state = reference_ket
    for k in reversed(range(0, len(parameters))):
        new_state = scipy.sparse.linalg.expm_multiply((parameters[k]*JW_CC_ops[k]), new_state)
    new_bra = new_state.transpose().conj()
    assert(new_bra.dot(new_state).toarray()[0][0]-1<0.0000001)
    return new_state 

def get_neel(n_qubits):
    """
    get 010101... reference
    """
    flip = scipy.sparse.csc_matrix(np.array([[0,1],[1,0]]))
    site_state = scipy.sparse.csc_matrix(np.array([[1],[0]]))
 
    state = site_state*1.0
    for j in range(n_qubits-1):
        site_state = flip.dot(site_state)
        state = scipy.sparse.kron(state,site_state)

    print(" Norm of neel state: ", scipy.sparse.linalg.norm(state))
    return state

def fermionized_hf(system, ham):
    print(" Do HF on spin lattice")
    print(" Number of qubits = ", system["n_qubits"])
    lattice = QubitLattice(system["n_qubits"])
    print(lattice)

    n_qubits = system["n_qubits"]
    hamiltonian = []      #   list of operator strings
    ucc_generator = []      #   list of operator strings


    for t in ham:
        coeff = t[1]
        if abs(coeff) < 1e-12:
            continue
        opstr = OperatorString(lattice)
        opstr.coeff = coeff*1.0
        for ti in t[0]:
            opstr.update_operator(ti[0],ti[1])
        hamiltonian.append(opstr)

    J12 = system["j12"]
    if False:
        print(" This doesn't work because of the linear n terms in the transformed part")
        for i in range(n_qubits):
            for j in range(i,n_qubits):
                J12[i,j] *= np.random.rand()
                J12[j,i] = J12[i,j] 

    # Make isotropic
    K12 = J12*2.0
  
    print(" J12 and K12:")
    print(" J12: ")
    print(J12)
    print(" K12: ")
    print(K12)
    do_hdvv_diag = 1
    if do_hdvv_diag:
        print(" Compute exact energies")
        Hmat, tmp, S2, Sz = hdvv.form_aniso_hdvv_H([1]*n_qubits,J12,K12)
        print(" Exact eigenvalues:")
        E,C = np.linalg.eigh(Hmat)
        sort_ind = np.argsort(E)
        E = E[sort_ind]
        C = C[:,sort_ind]
        
        s2 = C.T.dot(S2).dot(C)
        sz = C.T.dot(Sz).dot(C)
        
        for i in range(min(len(E),12)):
            print("     State %4i: Energy %12.8f <S2> = %12.8f <Sz> = %12.8f"%(i, E[i], s2[i,i], sz[i,i]))

    """
    H = - sum_i J(i,i+1) (p(i)m(i+1) + m(i)p(i+1)) - sum_i K(i,i+1) 2 z(i)z(i+1)
           |
        JW |
           v
    H = - sum_i J(i,i+1) (i'(i+1) + i(i+1)') - 2 sum_i K(i,i+1) (n(i)-1/2)(n(i+1)-1/2)
      = - sum_i J(i,i+1) (i'(i+1) + i(i+1)') - 2 sum_i K(i,i+1) (n(i)n(i+1) - n(i)/2 - n(i+1)/2 + 1/4)
      = - sum_i J(i,i+1) (i'(i+1) + i(i+1)') + 2 sum_i K(i,i+1) n(i) - 2 sum_i K(i,i+1) n(i)n(i+1)  - k
                    where k = -.5*sum_i K(i,i+1)
      
      = - sum_i (J(i,i+1) (i'(i+1) + i(i+1)') + 2 K(i,i+1) i'i) - 2 sum_i K(i,i+1) i'i(i+1)'(i+1)  - k
      = - sum_i (J(i,i+1) (i'(i+1) + i(i+1)') + 2 K(i,i+1) i'i) - 2 sum_i K(i,i+1) i'(i+1)'(i+1)i  - k

      Convert this to normal form of ab initio hamiltonian
      
        h(i,i) = 2*K
        h(i,i+1) = 2*J(i,i+1)
    """

    n_fermions = int(n_qubits/2)
    e_core = 0
    for i in range(n_qubits):
        e_core -= .5*K12[i,(i+1)%n_qubits]
    v = np.zeros((n_qubits,n_qubits,n_qubits,n_qubits))
    h = np.zeros((n_qubits,n_qubits))
    for i in range(n_qubits):
        j = (i+1)%n_qubits
        sign = int(i>j)
        odd_e = 2*((n_qubits/2)%2)-1
        
        h[i,j] = -J12[i,j] * (odd_e)**sign # phase only needed for PBC
        h[j,i] =  h[i,j]
    for i in range(n_qubits):
        j = (i+1)%n_qubits
        h[i,i] = 2*K12[i,j]
        v[i,i,j,j] = -2*K12[i,j]
        v[j,j,i,i] = -2*K12[i,j]

  
    H = Hamiltonian.Hamiltonian()
    H.S = np.eye(n_qubits,n_qubits)
    H.C = np.eye(n_qubits,n_qubits)
    H.e_core = e_core 
    H.t = h 
    H.V = v 

    hf_state = []
    for i in range(n_fermions):
        hf_state.append(i)
    
    print(" Core energy for fermionized hamiltonian: %12.8f" %H.e_core)
   
    Ehf = H.compute_determinant_energy(hf_state,[]) 
    print(" Energy of Determinant: %12.8f" %Ehf)


    orb_e = np.zeros((n_qubits))
    Cnew = np.eye(n_qubits) 
    hf_iter = 0
    e_last = 0 
    e_curr = 1000
    print()
    print(" ========== SCF ========== ")
    print(" %-12s %-16s" %("Iteration", "Energy"))
    while abs(e_last-e_curr) > 1e-12 and hf_iter < 100: 
        e_last = e_curr
        # build Fock operator
        D = Cnew[:,0:n_fermions]
        D = D.dot(D.T)
        G = np.einsum("pqrs,rs->pq",H.V,D) - np.einsum("psrq,rs->pq",H.V,D)
        F = H.t + G 
        if hf_iter==0:
            F = H.t 
            #F = H.t + .0001*G
        
        orb_e, Cnew = np.linalg.eigh(F)
        
        e_curr = np.trace( D.dot(H.t + .5*G))  + H.e_core
        print(" %-12i %-+16.12f" %(hf_iter, e_curr))
        hf_iter += 1
  
    H.transform_orbs(Cnew)
    for i in range(n_qubits):
        print(" Orbital Energy: %4i %12.8f" %(i,orb_e[i]))
    Ehf = H.compute_determinant_energy(hf_state,[]) 
    print(" Energy of Determinant: %16.12f" %Ehf)

    molecular_hamiltonian = InteractionOperator(H.e_core, H.t, .5*H.V)

#    for p in range(0,n_qubits):
#        for q in range(p,n_qubits):
#            for r in range(q,n_qubits):
#                for s in range(r,n_qubits):
#                    print(p,q,r,s,H.V[p,q,r,s])

    print(" Molecular Orbitals:")
    print("    ", end='')
    for i in range(n_qubits):
        print("%12i " %i, end='')
    print()
    for i in range(n_qubits):
        print("%4i" %i, end='')
        for j in range(n_qubits):
            print("%12.8f " %H.C[i,j], end='')
        print()
    
    f_ham = FermionOperator("",H.e_core)
    # Create some ladder operators
    for i in range(n_qubits):
        for j in range(n_qubits):
            if abs(H.t[i,j]) > 1e-8:
                #f_ops.append(jordan_wigner(FermionOperator('%s^ %s'%(i,j), H.t[i,j])))
                f_ham += FermionOperator('%s^ %s'%(i,j), H.t[i,j])
                molecular_hamiltonian.one_body_tensor[i,j] = H.t[i,j]

    
    for i in range(n_qubits):
        for j in range(n_qubits):
            for k in range(n_qubits):
                for l in range(n_qubits):
                    #print(i,j,k,l,"%12.8f" %(H.V[i,j,k,l]))
                    if abs(H.V[i,l,k,j]) > 1e-6:
                        #f_ops.append(jordan_wigner(FermionOperator('%s^ %s^ %s %s'%(i,j,k,l), H.V[i,j,k,l])))
                        f_ham += FermionOperator('%s^ %s^ %s %s'%(i,j,k,l), .5* H.V[i,l,k,j])
                        molecular_hamiltonian.two_body_tensor[i,j,k,l] = .5*H.V[i,l,k,j]
    return molecular_hamiltonian
 
    f_ham.compress()
    s_ham = jordan_wigner(f_ham)
    jw_spectrum = eigenspectrum(s_ham)
    print(" Eigspectrum of operator:")
    for i in range(10):
        print("     State %4i: Energy %12.8f "%(i, jw_spectrum[i]))
    ham = []
    system = {}
    system['n_qubits'] = n_qubits
    system['n_alpha'] = int(n_fermions/2) + n_fermions%2
    system['n_beta'] = int(n_fermions/2) 
    system['fci_energy'] = jw_spectrum[0] 

    for t in s_ham.terms:
        assert(np.isclose(s_ham.terms[t].imag,0))
        s_ham.terms[t] = s_ham.terms[t].real
        ham.append((t,s_ham.terms[t]))


    #
    #   CI solver
    print()
    print(" ========== CI ========== ")
    ci = ci_string.ci_solver()
    ci.algorithm = "direct"
    ci.init(H,n_fermions,0,10)
    print(ci)
    ci.run()
    E_ci = ci.results_e[0] 
    print(" *Energy of CI:       %12.8f" %(E_ci + H.e_core))
    print("  Energy/site of CI:  %12.8f" %((E_ci + H.e_core)/n_qubits))
    v0 = ci.results_v[:,0]
    print("  Weight of HF state: %12.8f" %(v0[0]*v0[0]))


    return molecular_hamiltonian


n_sites = 10 
j12 = np.zeros((n_sites,n_sites))

lat_type = "1d chain"

if lat_type == "1d chain":
    for i in range(n_sites-1):
        j12[i,i+1] -= 1
        j12[i+1,i] -= 1

    j12[0,n_sites-1] -= 1
    j12[n_sites-1,0] -= 1
print(j12)

lattice = np.ones((n_sites))
lattice_guess = np.ones((n_sites))
j12_guess = j12[0:4][0:4] 

np.random.seed(2)

do_hdvv_diag = 0
if do_hdvv_diag:
    #
    #    Get full H
    #
    H, tmp, S2, Sz = hdvv.form_hdvv_H(lattice,j12)
    
    
    #
    #    Diagonalize full H
    #
    l_full, v_full = np.linalg.eigh(H)
    s2 = v_full.T.dot(S2).dot(v_full)
    print(" Exact Energies:")
    for s in range(min(10,l_full.shape[0])):
        print("   %12.8f %12.8f " %(l_full[s],s2[s,s]))




system = {
        "n_qubits" : n_sites,
        "n_alpha": 1,
        "n_beta": 1,
        #"reference_energy": l_full[0],
        "j12": j12,
        }

ham = []
for hi in range(n_sites):
    for hj in range(hi+1, n_sites):
        hij = j12[hi,hj]
        #ham.append( [ [[hi,"P"],[hj,"M"]], -1*hij ]  )
        #ham.append( [ [[hi,"M"],[hj,"P"]], -1*hij ]  )
        #ham.append( [ [[hi,"Z"],[hj,"Z"]], -.5*hij ]  )
        ham.append( [ [[hi,"X"],[hj,"X"]], -.5*hij ]  )
        ham.append( [ [[hi,"Y"],[hj,"Y"]], -.5*hij ]  )
        ham.append( [ [[hi,"Z"],[hj,"Z"]], -.5*hij ]  )



#JW transform Hamiltonian computed classically with OFPsi4
hamiltonian_op = fermionized_hf(system, ham)
hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)

global global_der 
global global_energy  
global global_iter  
global_der = np.array([])
global_energy = 0.0 
global_iter = 0 

print(system)
n_qubits = system['n_qubits']
n_alpha  = system['n_alpha']
n_beta   = system['n_beta']
n_spinorbitals = n_qubits

basis = 'sto-3g'
multiplicity = 1
geometry = [("He",(0,0,0))]
molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
molecule.n_electrons = int(n_qubits/2) 
molecule.n_qubits = n_qubits

#Build p-h reference and map it to JW transform
print(" n_alpha, n_beta, n_qubits ", n_alpha, n_beta, n_qubits)
reference_ket = scipy.sparse.csc_matrix(openfermion.jw_configuration_state(list(range(molecule.n_electrons)), n_qubits)).transpose()
#reference_ket = get_neel(n_qubits)
reference_bra = reference_ket.transpose().conj()

#print(" Reference energy: %12.8f" %reference_bra.dot(hamiltonian).dot(reference_ket)[0,0].real)

#Thetas
parameters = []

#Second_quantized operations (not Jordan-Wignered)
SQ_CC_ops = []


'''
Count t2 second-quantized operations, add a parameter for each one, and add each one to the list
***CURRENTLY DOES NOT DISCRIMINATE AGAINST SPIN-FLIPS***
'''
for p in range(0, molecule.n_electrons):
    for q in range(p+1, molecule.n_electrons):
        for a in range(molecule.n_electrons, n_spinorbitals):
            for b in range(a+1, n_spinorbitals):
                if abs(hamiltonian_op.two_body_tensor[p,a,b,q]) < 1e-8:
                    print(" Dropping term %4i %4i %4i %4i" %(p,a,b,q), " V= %+6.1e" %hamiltonian_op.two_body_tensor[p,a,b,q])
                    continue
                two_elec = openfermion.FermionOperator(((a,1),(p,0),(b,1),(q,0)))-openfermion.FermionOperator(((q,1),(b,0),(p,1),(a,0)))
                parameters.append(0)
                SQ_CC_ops.append(two_elec)

order = list(range(len(SQ_CC_ops)))
random.shuffle(order)

#n_doubles = 10
#order = order[0:n_doubles]
#parameters = parameters[0:n_doubles]

print(" Order: ", order)
SQ_CC_ops = [ SQ_CC_ops[i] for i in order]
#for p in range(0, n_spinorbitals-1):
#    q = p+1
#    for r in range(p, n_spinorbitals-1):
#        s = r+1
#        two_elec = openfermion.FermionOperator(((p,1),(r,0),(q,1),(s,0)))-openfermion.FermionOperator(((s,1),(q,0),(r,1),(p,0)))
#        parameters.append(0)
#        SQ_CC_ops.append(two_elec)

'''
Count t1 second-quantized operations, add a parameter for each one, and add each one to the list
***CURRENTLY DOES NOT DISCRIMINATE AGAINST SPIN-FLIPS***
'''
singles = []
for p in range(0, molecule.n_electrons):
    for q in range(molecule.n_electrons, n_spinorbitals):
        if abs(hamiltonian_op.one_body_tensor[p,q]) < 1e-8:
            print(" Dropping term %4i %4i" %(p,q), " V= %+6.1e" %hamiltonian_op.one_body_tensor[p,q])
            continue
        one_elec = openfermion.FermionOperator(((q,1),(p,0)))-openfermion.FermionOperator(((p,1),(q,0)))
        parameters.append(0)
        singles.append(one_elec)
        #SQ_CC_ops.append(one_elec)

order = list(range(len(singles)))
random.shuffle(order)
print(" Order: ", order)
singles = [ singles[i] for i in order]
SQ_CC_ops.extend(singles)

print(" Number of parameters: ", len(parameters))
#Jordan_Wigners into the Pauli Matrices, then computes their products as sparse matrices.
JW_CC_ops = []
for classical_op in SQ_CC_ops:
    JW_CC_ops.append(openfermion.transforms.get_sparse_operator(classical_op, n_qubits = molecule.n_qubits))
'''
SPE based on a traditional, untrotterized ansatz
v'=exp(a+b+...+n)v
'''
def SPE(parameters):
    generator = scipy.sparse.csc_matrix((2**(molecule.n_qubits), 2**(molecule.n_qubits)), dtype = complex)
    for mat_op in range(0,len(JW_CC_ops)):
        generator = generator+parameters[mat_op]*JW_CC_ops[mat_op]
    new_state = scipy.sparse.linalg.expm_multiply(generator, reference_ket)
    new_bra = new_state.transpose().conj()
    assert(new_bra.dot(new_state).toarray()[0][0]-1<0.0000001)
    energy = new_bra.dot(hamiltonian.dot(new_state))
    return energy.toarray()[0][0].real

'''
SPE based on full, 1st-order Trotter decomposition
v'=exp(a)exp(b)...exp(n)v
'''
def Trotter_SPE(parameters):
    global global_energy  
    new_state = reference_ket
    for k in reversed(range(0, len(parameters))):
        new_state = scipy.sparse.linalg.expm_multiply((parameters[k]*JW_CC_ops[k]), new_state)
    new_bra = new_state.transpose().conj()
    assert(new_bra.dot(new_state).toarray()[0][0]-1<0.0000001)
    energy = new_bra.dot(hamiltonian.dot(new_state))
    global_energy = energy.toarray()[0][0].real
    assert(global_energy.imag <  1e-14)
    global_energy = global_energy.real
    return global_energy 


                 
                
#Numerical trotterized gradient
def Numerical_Trot_Grad(parameters):
    global global_der
    step_size = 1e-6
    grad = []
    for k in reversed(range(0, len(parameters))):
        para = copy.copy(parameters)
        para[k]+=step_size
        diff = Trotter_SPE(para)
        para[k]-=2*step_size
        diff -= Trotter_SPE(para)
        grad.append(diff/(step_size*2))
    global_der = np.asarray(grad)
    return np.asarray(grad)

def Five_Point_Grad(parameters):
    grad = []
    for k in reversed(range(0, len(parameters))):
        forw = copy.copy(parameters)
        forw2 = copy.copy(parameters)
        reve = copy.copy(parameters)
        reve2 = copy.copy(parameters)
        forw[k]+=1e-7
        forw2[k]+=2e-7
        reve[k]-=1e-7
        reve2[k]-=2e-7
        f2 = Trotter_SPE(forw2)
        f1 = Trotter_SPE(forw)
        r1 = Trotter_SPE(reve)
        r2 = Trotter_SPE(reve2)
        diff = (-f2+8*f1-8*r1+r2)/(1.2e-6)
        grad.append(diff)
    return np.asarray(grad)

#Analytical trotter gradient
def Trotter_Gradient(parameters):
    global global_der 
    grad = []
    new_state = copy.copy(reference_ket)
    for k in reversed(range(0, len(parameters))):
        new_state = scipy.sparse.linalg.expm_multiply((parameters[k]*JW_CC_ops[k]), new_state)
    new_bra = new_state.transpose().conj()
    hbra = new_bra.dot(hamiltonian)
    #H_l = hamiltonian
    #H_r = hamiltonian
    term = 0
    ket = copy.copy(new_state)
    grad = Recurse(parameters, grad, hbra, ket, term)
    global_der = grad
    return np.asarray(grad)

#Recursive component of analytical trotter gradient
def Recurse(parameters, grad, hbra, ket, term):
    if term == 0:
        hbra = hbra
        ket = ket
    else:
        hbra = (scipy.sparse.linalg.expm_multiply(-JW_CC_ops[term-1]*parameters[term-1], hbra.transpose().conj())).transpose().conj()
        ket = scipy.sparse.linalg.expm_multiply(-JW_CC_ops[term-1]*parameters[term-1], ket)
    grad.append((2*hbra.dot(JW_CC_ops[term]).dot(ket).toarray()[0][0].real))
    #grad.insert(0,(2*hbra.dot(JW_CC_ops[term]).dot(ket).toarray()[0][0].real))
    if term<len(parameters)-1:
        term += 1
        Recurse(parameters, grad, hbra, ket, term)
    return np.asarray(grad)

def callback(parameters):
    global global_der 
    global global_energy  
    global global_iter 
    err = np.sqrt(np.vdot(global_der, global_der))
    print(" Iter:%4i Current Energy = %20.16f Gradient Norm %10.1e Gradient Max %10.1e" %(global_iter,
        global_energy, err, np.max(np.abs(global_der))))
    global_iter += 1
    sys.stdout.flush()


#for p in range(len(parameters)):
#    parameters[p] = (random.random()-.5)*.001

#der_num = Numerical_Trot_Grad(parameters)
#der_ana = Trotter_Gradient(parameters)
#print(" Numerical: ")
#print(der_num)
#print("\n Analytical: ")
#print(der_ana)
#print("\n Error: ")
#print(np.linalg.norm(der_num-der_ana))

print(" Start optimization. Starting energy: %12.8f" %SPE(parameters))

opt_result = scipy.optimize.minimize(Trotter_SPE, parameters, jac=Trotter_Gradient, options = {'gtol': 1e-6, 'disp': True}, method = 'BFGS', callback=callback)

print(" Finished: %20.12f" % global_energy)

