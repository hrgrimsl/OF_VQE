import openfermion
import numpy as np
import copy as cp

from openfermion import *

def singlet_GSD(n_orbs):
    """
    n_orb is number of spatial orbitals assuming that spin orbitals are labelled
    0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
    """

    print(" Form singlet GSD operators")
    terms = []
    orbs = range(n_orbs)

    """
    rs
    pq  -> 
    """

    #q' p r' p -> 1/sqrt(2) ( qa pa rb pb  + qb pb ra pa)
    pq = 0
    for p in range(0,n_orbs):
        pa = 2*p
        pb = 2*p+1

        for q in range(p,n_orbs):
            qa = 2*q
            qb = 2*q+1
    
            pq += 1
    
            rs = 0 
            for r in range(0,n_orbs):
                ra = 2*r
                rb = 2*r+1
                
                for s in range(r,n_orbs):
                    sa = 2*s
                    sb = 2*s+1
                
                    rs += 1
                
                    if(pq >= rs):
                        continue

                    termA =  FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)), 2/np.sqrt(12))
                    termA += FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)), 2/np.sqrt(12))
                    termA += FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)), 1/np.sqrt(12))
                    termA += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)), 1/np.sqrt(12))
                    termA += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), 1/np.sqrt(12))
                    termA += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), 1/np.sqrt(12))

                    termB =  FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)), 1/2.0)
                    termB += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)), 1/2.0)
                    termB += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), -1/2.0)
                    termB += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), -1/2.0)

                    termA -= hermitian_conjugated(termA)
                    termB -= hermitian_conjugated(termB)
           
                    termA = normal_ordered(termA)
                    termB = normal_ordered(termB)
        
                    #print("A:")
                    #print(termA)
                    #print("B:")
                    #print(termB)
                    
                    if termA.many_body_order() > 0:
                        terms.append(termA)
                    
                    if termB.many_body_order() > 0:
                        terms.append(termB)

                


    return terms





def singlet_SD(n_occ,n_vir):
    """
    n_occ = number of doubly occupieds
    n_vir = number of doubly virtual
    0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
    """

    n_orb = n_occ + n_vir
    print(" Form singlet SD operators")
    terms = []
    
    
    for i in range(0,n_occ):
        ia = 2*i
        ib = 2*i+1

        for a in range(0,n_vir):
            aa = 2*n_occ + 2*a
            ab = 2*n_occ + 2*a+1
                
            term0 =  FermionOperator(((aa,1),(ia,0)), 1/np.sqrt(2))
            term0 += FermionOperator(((ab,1),(ib,0)), 1/np.sqrt(2))
            
            term0 -= hermitian_conjugated(term0)
                    
            term0 = normal_ordered(term0)
           
            if term0.many_body_order() > 0:
                terms.append(term0)
    
    for i in range(0,n_occ):
        ia = 2*i
        ib = 2*i+1

        for j in range(i,n_occ):
            ja = 2*j
            jb = 2*j+1
    
            for a in range(0,n_vir):
                aa = 2*n_occ + 2*a
                ab = 2*n_occ + 2*a+1

                for b in range(a,n_vir):
                    ba = 2*n_occ + 2*b
                    bb = 2*n_occ + 2*b+1

                    term0 =  FermionOperator(((aa,1),(ia,0),(ba,1),(ja,0)), 2/np.sqrt(12))
                    term0 += FermionOperator(((ab,1),(ib,0),(bb,1),(jb,0)), 2/np.sqrt(12))
                    term0 += FermionOperator(((aa,1),(ia,0),(bb,1),(jb,0)), 1/np.sqrt(12))
                    term0 += FermionOperator(((ab,1),(ib,0),(ba,1),(ja,0)), 1/np.sqrt(12))
                    term0 += FermionOperator(((aa,1),(ib,0),(bb,1),(ja,0)), 1/np.sqrt(12))
                    term0 += FermionOperator(((ab,1),(ia,0),(ba,1),(jb,0)), 1/np.sqrt(12))
                                                                  
                    term1  = FermionOperator(((aa,1),(ia,0),(bb,1),(jb,0)), 1/2)
                    term1 += FermionOperator(((ab,1),(ib,0),(ba,1),(ja,0)), 1/2)
                    term1 += FermionOperator(((aa,1),(ib,0),(bb,1),(ja,0)), -1/2)
                    term1 += FermionOperator(((ab,1),(ia,0),(ba,1),(jb,0)), -1/2)
            
                    term0 -= hermitian_conjugated(term0)
                    term1 -= hermitian_conjugated(term1)
           
                    term0 = normal_ordered(term0)
                    term1 = normal_ordered(term1)
                    
                    #print(i,j,a,b)
                    #print(" A:\n",term0)
                    #print(" B:\n",term1)
                    if term0.many_body_order() > 0:
                        terms.append(term0)
                    
                    if term1.many_body_order() > 0:
                        terms.append(term1)

    return terms



def unrestricted_SD(n_occ_a, n_occ_b, n_vir_a, n_vir_b):
    """
    """

    n_orb = n_occ_a + n_vir_a
    assert(n_occ_a+n_vir_a == n_occ_b+n_vir_b)
    print(" Form unrestricted SD operators")
    terms = []
   
    #
    #   Singles
    #
    # a
    for ii in range(0,n_occ_a):
        i = 2*ii
        for aa in range(0,n_vir_a):
            a = 2*aa
            
            term =  FermionOperator(((a,1),(i,0)))
                
            term -= hermitian_conjugated(term)
           
            if term0.many_body_order() > 0:
                terms.append(term0)
   
#    singles = cp.deepcopy(terms)
#    for ei in range(len(singles)):
#        for ej in range(ei,len(singles)):
#            term0 = singles[ei] * singles[ej] 
#            term0 -= hermitian_conjugated(term0)
#            terms.append(term0)
#   
#    for t in terms:
#        print(t)
#    return terms
    #doubles
    
    for i in range(0,n_occ):
        ia = 2*i
        ib = 2*i+1

        for j in range(i,n_occ):
            ja = 2*j
            jb = 2*j+1
    
            for a in range(0,n_vir):
                aa = n_occ + 2*a
                ab = n_occ + 2*a+1

                for b in range(a,n_vir):
                    ba = n_occ + 2*b
                    bb = n_occ + 2*b+1

               
                    term0 =  FermionOperator(((aa,1),(ia,0),(ba,1),(ja,0)), 2/np.sqrt(12))
                    term0 += FermionOperator(((ab,1),(ib,0),(bb,1),(jb,0)), 2/np.sqrt(12))
                    term0 += FermionOperator(((aa,1),(ia,0),(bb,1),(jb,0)), 1/np.sqrt(12))
                    term0 += FermionOperator(((ab,1),(ib,0),(ba,1),(ja,0)), 1/np.sqrt(12))
                    term0 += FermionOperator(((aa,1),(ib,0),(bb,1),(ja,0)), -1/np.sqrt(12))
                    term0 += FermionOperator(((ab,1),(ia,0),(ba,1),(jb,0)), -1/np.sqrt(12))
            
                    term1  = FermionOperator(((aa,1),(ia,0),(bb,1),(jb,0)), 1/2)
                    term1 += FermionOperator(((ab,1),(ib,0),(ba,1),(ja,0)), 1/2)
                    term1 += FermionOperator(((aa,1),(ib,0),(bb,1),(ja,0)), 1/2)
                    term1 += FermionOperator(((ab,1),(ia,0),(ba,1),(jb,0)), 1/2)
            
                    term0 -= hermitian_conjugated(term0)
                    term1 -= hermitian_conjugated(term1)
           
                    print("A: ", term0, i, j, a, b)
                    print("B: ", term0)
                    if term0.many_body_order() > 0:
                        terms.append(term0)
                    
                    if term1.many_body_order() > 0:
                        terms.append(term1)

                
    return terms



