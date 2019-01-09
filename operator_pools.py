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

    #p' p q' q
    for p in range(0,n_orbs):
        pa = 2*p
        pb = 2*p+1
        for q in range(p,n_orbs):
            qa = 2*q
            qb = 2*q+1
            
            #term0 = FermionOperator(((qa,1),(qb,1),(pa,0),(pb,0)))
            #term0 = FermionOperator(((pa,1),(qb,1),(pa,0),(qb,0)))
            term0 = FermionOperator(((pa,1),(pa,0),(qb,1),(qb,0)))
            term0 -= hermitian_conjugated(term0)
            
            if term0.many_body_order() > 0:
                terms.append(term0)

    #q' p r' p -> 1/sqrt(2) ( qa pa rb pb  + qb pb ra pa)
    for p in range(0,n_orbs):
        pa = 2*p
        pb = 2*p+1

        qr = -1 
        for q in range(0,n_orbs):
            qa = 2*q
            qb = 2*q+1
            for r in range(0,n_orbs):
                ra = 2*r
                rb = 2*r+1
                qr += 1
                
                print(q,r,p,p)
                if p < qr: 
                    continue
               
                term0 =  FermionOperator(((qa,1),(pa,0),(rb,1),(pb,0)), 1)
                term0 += FermionOperator(((qb,1),(pb,0),(ra,1),(pa,0)), 1)
            
                term0 -= hermitian_conjugated(term0)
            
                if term0.many_body_order() > 0:
                    terms.append(term0)
                

    print(len(terms))






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
            aa = n_occ + 2*a
            ab = n_occ + 2*a+1
                
            #term0 =  FermionOperator(((aa,1),(ia,0)), 1/np.sqrt(2))
            #term0 += FermionOperator(((ab,1),(ib,0)), 1/np.sqrt(2))
            term0 =  FermionOperator(((aa,1),(ia,0)), 1/np.sqrt(2))
            term0 += FermionOperator(((ab,1),(ib,0)), 1/np.sqrt(2))
            
            term0 -= hermitian_conjugated(term0)
           
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



