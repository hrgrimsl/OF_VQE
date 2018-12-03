import scipy
import openfermion
import openfermionpsi4
import os
import numpy as np
import copy
import random
import logging
import argparse
import math

#Parse Initial Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--log', type=str, default='app.log', help='Output File for logging.')
parser.add_argument('-m', '--mode', type=str, default='DEBUG', help='Mode for logging.')
parser.add_argument('-rw', '--readwrite', type=str, default='w', help='Obliterate existing log file?')
parser.add_argument('-s', '--seed', type=int, default=111596, help='Random seed.')
parser.add_argument('-p', '--protocol', type=str, default='random', help='Protocol for ordering terms in Trotter approximation.')
parser.add_argument('-sys', '--system', type=str, default='H2O', help='Which chemical system listed do you want to solve?')
parser.add_argument('-d', '--dis', type=float, default=0.0, help='Dissociation parameter in angstroms')
args = parser.parse_args()

#Seed RNG
random.seed(args.seed)

#Logging Preferences
logging.basicConfig(filename=args.log, filemode=args.readwrite, format='%(message)s')
logging.getLogger().setLevel(eval('logging.%s' %args.mode))
logging.debug(args)

#Manually initialize state
basis = 'sto-3g'
r = args.dis
if args.system == 'H2O':
    multiplicity = 1
    ry = .763239
    rz = .477047
    angle = math.atan(rz/ry)
    y = abs(r*math.sin(angle))
    z = abs(r*math.cos(angle))
    geometry = [('H', (0, ry+y, -rz-z)), ('H', (0, -ry-y, -rz-z)), ('O', (0, 0, 0.119262))]
elif args.system == 'BeH2':
    multiplicity = 1
    rz = 1.4276
    geometry = [('H', (0, 0, -rz-r)), ('H', (0, 0, rz+r)), ('Be', (0, 0, 0))]
elif args.system == 'LiH':
    multiplicity = 1
    rz = 1.64
    geometry = [('H', (0, 0, rz+r)), ('Li', (0, 0, 0))]
elif args.system == 'H8':
    multiplicity = 1
    geometry = [('H', (0,0,0)), ('H', (.74+r,0,0)), ('H', (4.48,0,0)), ('H', (5.22+r,0,0)), ('H', (8.96,0,0)), ('H', (9.7+r,0,0)), ('H', (13.44, 0, 0)), ('H', (14.18+r, 0,0))]
else:
   logging.critical('Unsupported system.')
molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
molecule = openfermionpsi4.run_psi4(molecule, run_scf = 1, run_ccsd = 1, run_fci=1)

logging.debug('Molecule: '+str(geometry))
logging.debug('Qubits: '+str(molecule.n_qubits))
logging.debug('Spin-Orbitals: '+str(molecule.n_orbitals*2))
n_spinorbitals = int(molecule.n_orbitals*2)

#Build p-h reference and map it to JW transform
reference_ket = scipy.sparse.csc_matrix(openfermion.jw_configuration_state(list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()
reference_bra = reference_ket.transpose().conj()

#JW transform Hamiltonian computed classically with OFPsi4
hamiltonian = openfermion.transforms.get_sparse_operator(molecule.get_molecular_hamiltonian())

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
                 two_elec = openfermion.FermionOperator(((a,1),(p,0),(b,1),(q,0)))-openfermion.FermionOperator(((q,1),(b,0),(p,1),(a,0)))
                 parameters.append(0)
                 SQ_CC_ops.append(two_elec)

'''
Count t1 second-quantized operations, add a parameter for each one, and add each one to the list
***CURRENTLY DOES NOT DISCRIMINATE AGAINST SPIN-FLIPS***
'''
for p in range(0, molecule.n_electrons):
    for q in range(molecule.n_electrons, n_spinorbitals):
        one_elec = openfermion.FermionOperator(((q,1),(p,0)))-openfermion.FermionOperator(((p,1),(q,0)))
        parameters.append(0)
        SQ_CC_ops.append(one_elec)

#Jordan_Wigners into the Pauli Matrices, then computes their products as sparse matrices.
JW_CC_ops = []
for classical_op in SQ_CC_ops:
    JW_CC_ops.append(openfermion.transforms.get_sparse_operator(classical_op, n_qubits = molecule.n_qubits))

#Use some sorting method for the operators.

#Random ordering based on a seed input
if args.protocol == 'random':
    op_indices = []
    for i in range(0, len(JW_CC_ops)):
        op_indices.append(i)
    random.shuffle(op_indices)
    new_ops = []
    for i in op_indices:
        new_ops.append(JW_CC_ops[i])
    logging.debug(op_indices)
    JW_CC_ops=new_ops

#Commutator Evaluations:
ham_ket = hamiltonian.dot(reference_ket)
def Commutator(op_no):
    op = JW_CC_ops[op_no]
    bra = ham_ket.transpose()
    ket = scipy.sparse.linalg.expm_multiply(op, reference_ket)
    comm = bra.dot(ket)    
    bra = (scipy.sparse.linalg.expm_multiply(-op, reference_ket)).transpose().conj()
    ket = ham_ket
    comm = comm-bra.dot(ket)
    return abs(comm.toarray()[0][0])

def Commutator2(op_no):
    op = JW_CC_ops[op_no]
    bra = ham_ket.transpose()
    ket = op.dot(reference_ket)
    comm = bra.dot(ket)    
    bra = reference_bra.dot(op)
    ket = ham_ket
    comm = comm-bra.dot(ket)
    return abs(comm.toarray()[0][0])

#Special Orders of Interest:
if args.protocol == 'increasing_comms':
    op_indices = []
    for i in range(0, len(JW_CC_ops)):
        op_indices.append(i)
    op_indices = sorted(op_indices, key=Commutator)         
    new_ops = []
    for i in op_indices:
        new_ops.append(JW_CC_ops[i])
    logging.info('Increasing Commutators:')
    logging.debug(op_indices)
    JW_CC_ops=new_ops
    
if args.protocol == 'decreasing_comms':
    op_indices = []
    for i in range(0, len(JW_CC_ops)):
        op_indices.append(i)
    op_indices = sorted(op_indices, key=Commutator, reverse=True)         
    new_ops = []
    for i in op_indices:
        new_ops.append(JW_CC_ops[i])
    logging.info('Decreasing Commutators:')
    logging.debug(op_indices)
    JW_CC_ops=new_ops

if args.protocol == 'increasing_unexp_comms': 
    logging.info('Correlation: '+str(molecule.fci_energy-molecule.hf_energy))
    op_indices = []
    for i in range(0, len(JW_CC_ops)):
        op_indices.append(i)
    op_indices = sorted(op_indices, key=Commutator2)         
    new_ops = []
    for i in op_indices:
        new_ops.append(JW_CC_ops[i])
    logging.info('Increasing Unexponentiated Commutators:')
    logging.debug(op_indices)
    JW_CC_ops=new_ops

if args.protocol == 'decreasing_unexp_comms':
    op_indices = []
    for i in range(0, len(JW_CC_ops)):
        op_indices.append(i)
    op_indices = sorted(op_indices, key=Commutator2, reverse=True)         
    new_ops = []
    for i in op_indices:
        new_ops.append(JW_CC_ops[i])
    logging.info('Decreasing Unexponentiated Commutators:')
    logging.debug(op_indices)
    JW_CC_ops=new_ops

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
v'=exp(n)exp(n-1)...exp(a)v
'''
def Trotter_SPE(parameters):
    new_state = reference_ket
    for k in reversed(range(0, len(parameters))):
        new_state = scipy.sparse.linalg.expm_multiply((parameters[k]*JW_CC_ops[k]), new_state)
    new_bra = new_state.transpose().conj()
    assert(new_bra.dot(new_state).toarray()[0][0]-1<0.0000001)
    energy = new_bra.dot(hamiltonian.dot(new_state))
    return energy.toarray()[0][0].real                       

#Analytical trotter gradient
def Trotter_Gradient(parameters):
    grad = []
    new_state = copy.copy(reference_ket)
    for k in reversed(range(0, len(parameters))):
        new_state = scipy.sparse.linalg.expm_multiply((parameters[k]*JW_CC_ops[k]), new_state)
    new_bra = new_state.transpose().conj()
    hbra = new_bra.dot(hamiltonian)
    term = 0
    ket = copy.copy(new_state)
    grad = Recurse(parameters, grad, hbra, ket, term)
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
    if term<len(parameters)-1:
        term += 1
        Recurse(parameters, grad, hbra, ket, term)
    return np.asarray(grad)

#Callback Function
def callback(parameters):
    global iterations
    logging.debug(Trotter_SPE(parameters))
    iterations+= 1

global iterations
iterations = 1

logging.debug('HF = '+str(molecule.hf_energy))
logging.debug('CCSD = '+str(molecule.ccsd_energy))
logging.debug('FCI = '+str(molecule.fci_energy))
logging.debug('Optimizing:')
optimization = scipy.optimize.minimize(Trotter_SPE, parameters, jac=Trotter_Gradient, options = {'gtol': 1e-5, 'disp': True}, method = 'BFGS', callback=callback)
logging.debug('Converged in '+str(iterations)+ ' iterations.')
logging.info(Trotter_SPE(optimization.x)-molecule.fci_energy)
