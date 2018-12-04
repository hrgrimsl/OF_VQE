from openfermionpsi4._psi4_conversion_functions import *

def Harvest_CCSD_Amps(molecule):
    psi_filename = molecule.filename+'.out'
    single_cc_amplitudes, double_cc_amplitudes = (
        parse_psi4_ccsd_amplitudes(
        2 * molecule.n_orbitals,
        molecule.get_n_alpha_electrons(),
        molecule.get_n_beta_electrons(),
        psi_filename))
    molecule.ccsd_single_amps = single_cc_amplitudes
    molecule.ccsd_double_amps = double_cc_amplitudes
    return (molecule)

def Harvest_MP2_Amps(molecule):
    pass
    #We would like mp2 amplitudes at some point.
