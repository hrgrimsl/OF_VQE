import openfermion
import numpy as np
import copy as cp

from openfermion import *



class OperatorPool:
    def __init__(self):
        self.n_orb = 0
        self.n_occ_a = 0
        self.n_occ_b = 0
        self.n_vir_a = 0
        self.n_vir_b = 0

        self.n_spin_orb = 0

    def init(self,molecule):

        self.n_orb = molecule.n_orbitals
        self.n_spin_orb = 2*self.n_orb 
        self.n_occ_a = molecule.get_n_alpha_electrons()
        self.n_occ_b = molecule.get_n_beta_electrons()
    
        self.n_vir_a = self.n_orb - self.n_occ_a
        self.n_vir_b = self.n_orb - self.n_occ_b
        
        self.n_occ = self.n_occ_a
        self.n_vir = self.n_vir_a
        self.n_ops = 0

        self.generate_SQ_Operators()

    def generate_SQ_Operators(self):
        print("Virtual: Reimplement")
        exit()

    def generate_SparseMatrix(self):
        self.spmat_ops = []
        print(" Generate Sparse Matrices for operators in pool")
        for op in self.fermi_ops:
            self.spmat_ops.append(transforms.get_sparse_operator(op, n_qubits = self.n_spin_orb))
        assert(len(self.spmat_ops) == self.n_ops)
 #       print(self.spmat_ops[1])
        return


class singlet_GSD(OperatorPool):
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """
        
        print(" Form singlet GSD operators")
        
        self.fermi_ops = []
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1
 
            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1
        
                termA =  FermionOperator(((pa,1),(qa,0)))
                termA += FermionOperator(((pb,1),(qb,0)))
 
                termA -= hermitian_conjugated(termA)
               
                termA = normal_ordered(termA)
                
                #Normalize
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t
            
                if termA.many_body_order() > 0:
                    termA = termA/np.sqrt(coeffA)
                    self.fermi_ops.append(termA)
                       
      
        pq = -1 
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1
 
            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1
        
                pq += 1
        
                rs = -1 
                for r in range(0,self.n_orb):
                    ra = 2*r
                    rb = 2*r+1
                    
                    for s in range(r,self.n_orb):
                        sa = 2*s
                        sb = 2*s+1
                    
                        rs += 1
                    
                        if(pq > rs):
                            continue

                        termA =  FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), 1/np.sqrt(12))
                                                                      
                        termB =  FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)),  1/2.0)
                        termB += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)),  1/2.0)
                        termB += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), -1/2.0)
                        termB += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), -1/2.0)
 
                        termA -= hermitian_conjugated(termA)
                        termB -= hermitian_conjugated(termB)
               
                        termA = normal_ordered(termA)
                        termB = normal_ordered(termB)
                        
                        #Normalize
                        coeffA = 0
                        coeffB = 0
                        for t in termA.terms:
                            coeff_t = termA.terms[t]
                            coeffA += coeff_t * coeff_t
                        for t in termB.terms:
                            coeff_t = termB.terms[t]
                            coeffB += coeff_t * coeff_t

                        
                        if termA.many_body_order() > 0:
                            termA = termA/np.sqrt(coeffA)
                            self.fermi_ops.append(termA)
                        
                        if termB.many_body_order() > 0:
                            termB = termB/np.sqrt(coeffB)
                            self.fermi_ops.append(termB)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return 





class singlet_SD(OperatorPool):
    def generate_SQ_Operators(self):
        """
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """

        print(" Form singlet SD operators")
        self.fermi_ops = [] 
       
        n_occ = self.n_occ
        n_vir = self.n_vir
       
        for i in range(0,n_occ):
            ia = 2*i
            ib = 2*i+1

            for a in range(0,n_vir):
                aa = 2*n_occ + 2*a
                ab = 2*n_occ + 2*a+1
                    
                termA =  FermionOperator(((aa,1),(ia,0)), 1/np.sqrt(2))
                termA += FermionOperator(((ab,1),(ib,0)), 1/np.sqrt(2))
                
                termA -= hermitian_conjugated(termA)
                        
                termA = normal_ordered(termA)
               
                #Normalize
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t
                
                if termA.many_body_order() > 0:
                    termA = termA/np.sqrt(coeffA)
                    self.fermi_ops.append(termA)
       

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

                        termA =  FermionOperator(((aa,1),(ba,1),(ia,0),(ja,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((ab,1),(bb,1),(ib,0),(jb,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((aa,1),(bb,1),(ia,0),(jb,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((ab,1),(ba,1),(ib,0),(ja,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((aa,1),(bb,1),(ib,0),(ja,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((ab,1),(ba,1),(ia,0),(jb,0)), 1/np.sqrt(12))
                                                                      
                        termB  = FermionOperator(((aa,1),(bb,1),(ia,0),(jb,0)), 1/2)
                        termB += FermionOperator(((ab,1),(ba,1),(ib,0),(ja,0)), 1/2)
                        termB += FermionOperator(((aa,1),(bb,1),(ib,0),(ja,0)), -1/2)
                        termB += FermionOperator(((ab,1),(ba,1),(ia,0),(jb,0)), -1/2)
                
                        termA -= hermitian_conjugated(termA)
                        termB -= hermitian_conjugated(termB)
               
                        termA = normal_ordered(termA)
                        termB = normal_ordered(termB)
                        
                        #Normalize
                        coeffA = 0
                        coeffB = 0
                        for t in termA.terms:
                            coeff_t = termA.terms[t]
                            coeffA += coeff_t * coeff_t
                        for t in termB.terms:
                            coeff_t = termB.terms[t]
                            coeffB += coeff_t * coeff_t

                        
                        if termA.many_body_order() > 0:
                            termA = termA/np.sqrt(coeffA)
                            self.fermi_ops.append(termA)
                        
                        if termB.many_body_order() > 0:
                            termB = termB/np.sqrt(coeffB)
                            self.fermi_ops.append(termB)
        
        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return 


class qubits(OperatorPool):
    def generate_SQ_Operators(self):

        self.fermi_ops = []
        
        n_occ = self.n_occ
        n_vir = self.n_vir

                
        for p in range(0,2*self.n_orb):
            X = QubitOperator('X%d'% p, 0+1j)

            self.fermi_ops.append(X)
        for p in range(0,2*self.n_orb):
            Z = QubitOperator('Z%d'% p, 0+1j)

            self.fermi_ops.append(Z)
        for p in range(0,2*self.n_orb):

            for q in range(p+1,2*self.n_orb):

                ZZ = QubitOperator('Z%d Z%d'% (p, q), 0+1j)
                self.fermi_ops.append(ZZ)

        for p in range(0,2*self.n_orb):
            for q in range(p+1,2*self.n_orb):

                XX = QubitOperator('X%d X%d'% (p, q), 0+1j)
                YX = QubitOperator('Y%d X%d'% (p, q), 0+1j)
                XZ = QubitOperator('X%d Z%d'% (p, q), 0+1j)
                YZ = QubitOperator('Y%d Z%d'% (p, q), 0+1j)

                self.fermi_ops.append(XX)
                self.fermi_ops.append(YX)
                self.fermi_ops.append(XZ)
                self.fermi_ops.append(YZ)
        for p in range(0,2*self.n_orb):

            for q in range(p+1,2*self.n_orb):

                XY = QubitOperator('X%d Y%d'% (p, q), 0+1j)
                self.fermi_ops.append(XY)


  #      for i in range(0,2*self.n_orb):
  #          for j in range(i+1, 2*self.n_orb):
  #              for k in range(j+1, 2*self.n_orb):
  #
  #                  XXY = QubitOperator('X%d X%d Y%d'% (i, j, k), 1j)
  #                  XYZ = QubitOperator('X%d Y%d Z%d'% (i, j, k), 1j)
  #                  YYZ = QubitOperator('Y%d Y%d Z%d'% (i, j, k), 1j)
  #
  #                  self.fermi_ops.append(XXY)
  #                  self.fermi_ops.append(XYZ)
  #                  self.fermi_ops.append(YYZ)
  #      for i in range(0,n_occ):
  #          ia = 2*i
  #          ib = 2*i+1

  #          for j in range(i,n_occ):
  #              ja = 2*j
  #              jb = 2*j+1
        
  #              for a in range(0,n_vir):
  #                  aa = 2*n_occ + 2*a
  #                  ab = 2*n_occ + 2*a+1

  #                  for b in range(a,n_vir):
  #                      ba = 2*n_occ + 2*b
  #                      bb = 2*n_occ + 2*b+1

  #                      XYYY1 = QubitOperator('X%d Y%d Y%d Y%d'% (ia,ja,aa,ba), 1j)
  #                      XYYY2 = QubitOperator('X%d Y%d Y%d Y%d'% (ib,jb,ab,bb), 1j)   
  #                      XYYY3 = QubitOperator('X%d Y%d Y%d Y%d'% (ia,jb,aa,bb), 1j) 
  #                      XYYY4 = QubitOperator('X%d Y%d Y%d Y%d'% (ib,ja,aa,bb), 1j)
  #                      XYYY5 = QubitOperator('X%d Y%d Y%d Y%d'% (ib,ja,ab,ba), 1j)
  #                      XYYY6 = QubitOperator('X%d Y%d Y%d Y%d'% (ia,jb,ab,ba), 1j)

  #                      YXYY1 = QubitOperator('Y%d X%d Y%d Y%d'% (ia,ja,aa,ba), 1j)
  #                      YXYY2 = QubitOperator('Y%d X%d Y%d Y%d'% (ib,jb,ab,bb), 1j)
  #                      YXYY3 = QubitOperator('Y%d X%d Y%d Y%d'% (ia,jb,aa,bb), 1j)
  #                      YXYY4 = QubitOperator('Y%d X%d Y%d Y%d'% (ib,ja,aa,bb), 1j)
  #                      YXYY5 = QubitOperator('Y%d X%d Y%d Y%d'% (ia,jb,ab,ba), 1j)
  #                      YXYY6 = QubitOperator('Y%d X%d Y%d Y%d'% (ib,ja,ab,ba), 1j)

  #                      YYXY1 = QubitOperator('Y%d Y%d X%d Y%d'% (ia,ja,aa,ba), 1j)
  #                      YYXY2 = QubitOperator('Y%d Y%d X%d Y%d'% (ib,jb,ab,bb), 1j)
  #                      YYXY3 = QubitOperator('Y%d Y%d X%d Y%d'% (ia,jb,aa,bb), 1j)
  #                      YYXY4 = QubitOperator('Y%d Y%d X%d Y%d'% (ib,ja,aa,bb), 1j)
  #                      YYXY5 = QubitOperator('Y%d Y%d X%d Y%d'% (ia,jb,ab,ba), 1j)
  #                      YYXY6 = QubitOperator('Y%d Y%d X%d Y%d'% (ib,ja,ab,ba), 1j)

  #                      YYYX1 = QubitOperator('Y%d Y%d Y%d X%d'% (ia,ja,aa,ba), 1j)                       
  #                      YYYX2 = QubitOperator('Y%d Y%d Y%d X%d'% (ib,jb,ab,bb), 1j)
  #                      YYYX3 = QubitOperator('Y%d Y%d Y%d X%d'% (ia,jb,aa,bb), 1j)
  #                      YYYX4 = QubitOperator('Y%d Y%d Y%d X%d'% (ib,ja,aa,bb), 1j)
  #                      YYYX5 = QubitOperator('Y%d Y%d Y%d X%d'% (ia,jb,ab,ba), 1j)
  #                      YYYX6 = QubitOperator('Y%d Y%d Y%d X%d'% (ib,ja,ab,ba), 1j)

  #                      if (i != j) and (a != b):
  #                          self.fermi_ops.append(XYYY1)
  #                          self.fermi_ops.append(XYYY2)
  #                          self.fermi_ops.append(YXYY1)
  #                          self.fermi_ops.append(YXYY2)
  #                          self.fermi_ops.append(YYXY1)
  #                          self.fermi_ops.append(YYXY2)
  #                          self.fermi_ops.append(YYYX1)
  #                          self.fermi_ops.append(YYYX2)

  #                      self.fermi_ops.append(XYYY3)
  #                      self.fermi_ops.append(XYYY4)
  #                      self.fermi_ops.append(XYYY5)
  #                      self.fermi_ops.append(XYYY6)

  #                      self.fermi_ops.append(YXYY3)
  #                      self.fermi_ops.append(YXYY4)
  #                      self.fermi_ops.append(YXYY5)
  #                      self.fermi_ops.append(YXYY6)

  #                      self.fermi_ops.append(YYXY3)
  #                      self.fermi_ops.append(YYXY4)
  #                      self.fermi_ops.append(YYXY5)
  #                      self.fermi_ops.append(YYXY6)

  #                      self.fermi_ops.append(YYYX3)
  #                      self.fermi_ops.append(YYYX4)
  #                      self.fermi_ops.append(YYYX5)
  #                      self.fermi_ops.append(YYYX6)


        for p in range(0,2*n_occ):
            for q in range(p+1,2*n_occ):
                for r in range(q+1,2*self.n_orb):
                    for s in range(r+1,2*self.n_orb):
                        XYYY = QubitOperator('X%d Y%d Y%d Y%d'% (p, q, r, s), 1j)
                      #  YXYY = QubitOperator('Y%d X%d Y%d Y%d'% (p, q, r, s), 1j)
                      #  YYXY = QubitOperator('Y%d Y%d X%d Y%d'% (p, q, r, s), 1j)
                      #  YYYX = QubitOperator('Y%d Y%d Y%d X%d'% (p, q, r, s), 1j)
                       # XXXY = QubitOperator('X%d X%d X%d Y%d'% (p, q, r, s), 1j)
                       # YXXX = QubitOperator('Y%d X%d X%d X%d'% (p, q, r, s), 1j)
                       # XYXX = QubitOperator('X%d Y%d X%d X%d'% (p, q, r, s), 1j)
                       # XXYX = QubitOperator('X%d X%d Y%d X%d'% (p, q, r, s), 1j)
                        self.fermi_ops.append(XYYY)
                      #  self.fermi_ops.append(YXYY)
                      #  self.fermi_ops.append(YYXY)
                      #  self.fermi_ops.append(YYYX)
                      #  self.fermi_ops.append(XXXY)
                      #  self.fermi_ops.append(XXYX)
                      #  self.fermi_ops.append(XYXX)
                      #  self.fermi_ops.append(YXXX)

                
        for op in self.fermi_ops:
            print(op)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)        
        return





