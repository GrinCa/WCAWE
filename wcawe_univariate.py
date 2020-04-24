#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:15:51 2020

@author: julien
"""

#----------------------------------
from time import time
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import factorized
from numpy import linalg as LA
#----------------------------------

class FFS():

    def __init__(self,mat_manager,param,deriv):
        # Initialization of data required for calculation. This part is unecessary for
        # the algorithm to work but it enables to clear the code and make it more 
        # consise
        self.ndof = mat_manager.ndof
        self.nmat = mat_manager.nmat
        self.LHS = mat_manager.matrices
        self.RHS = mat_manager.RHS
        
        self.param = param
        
        self.deriv = deriv

    def build_basis_UNIVARIATE(self,coeff_deriv_LHS,coeff_deriv_RHS,nvecfreq):
        print("***************************************")
        print("*Calculating WCAWE basis ({0} vectors)*".format(nvecfreq))
        print("***************************************")
        
        t0 = time()
        
        # The algorithm from Slone (See the doc) begins here. Vtilde, V, U are defined 
        # according to the Gram-Schmidt process by V = Vtilde*U^-1. U is the ortho-
        # gonalization matrix. 
        Vtilde = np.zeros((self.ndof,nvecfreq),dtype=np.cfloat)
        V = np.zeros((self.ndof,nvecfreq),dtype=np.cfloat)
        U = np.zeros((nvecfreq,nvecfreq),dtype=np.cfloat)
        
        #here is the definition of the correction terms
        def Pu1(n,m,U):
            Pu = np.eye(n-m,dtype=np.cfloat)
            for t in range(0,m):
                Pu = np.dot(Pu, LA.inv(U[t:n-m+t,t:n-m+t]))
            return Pu
        def Pu2(n,m,U):
            Pu = np.eye(n-m,dtype=np.cfloat)
            for t in range(1,m):
                Pu = np.dot(Pu, LA.inv(U[t:n-m+t,t:n-m+t]))
            return Pu
        
        # Aglob generally stands for a temporary LHS. Note that we use csc_matrix 
        # which is equivalent to sparse() function of Matlab. Furthermore, we have 
        # to specify the type of data of the matrix, here complex. Note that for 
        # every change that you can make on the algorithm, specify dtype=np.cfloat. 
        # Otherwise if there one non-complex array, all calculation will be cast into
        # float, discarding the complex part
        Aglob = csc_matrix((self.ndof,self.ndof),dtype=np.cfloat)
        for k in range(self.nmat):
            Aglob += coeff_deriv_LHS[k,0]*self.LHS[k]
        
        # here we factorized the LHS, with the LU method (see scipy.org doc)
        solve = factorized(Aglob)
        Vtilde[:,0] = solve(coeff_deriv_RHS[0]*self.RHS)
        U[0,0] = LA.norm(Vtilde[:,0])
        V[:,0] = Vtilde[:,0]/U[0,0]
        
        for n in range(1,nvecfreq):
            term1 = 0
            term2 = 0
            for m in range(n-1):
                # Pu1 and Pu2 are correction terms
                PU1 = Pu1(n,m,U)
                term1 += coeff_deriv_RHS[m+1]*self.RHS*PU1[0,n-m-1]
            for m in range(1,n-1):
                PU2 = Pu2(n,m,U)
                A_m = csc_matrix((self.ndof,self.ndof),dtype=np.cfloat)
                for k in range(self.nmat):
                    A_m += coeff_deriv_LHS[k,m+1]*self.LHS[k]
                    term2 += np.dot(A_m.dot(V[:,0:n-m]), PU2[:,n-m-1])
            A_1 = csc_matrix((self.ndof,self.ndof),dtype=np.cfloat)
            for k in range(self.nmat):
                A_1 += coeff_deriv_LHS[k,1]*self.LHS[k]
            
            Vtilde[:,n] = solve(term1-term2-A_1.dot(V[:,n-1]))
            

            # here we orthogonalize the vectors of the basis
            for alpha in range(n):
                U[alpha,n] = np.dot(np.matrix.getH(V[:,alpha]), Vtilde[:,n])
                Vtilde[:,n] -= U[alpha,n]*V[:,alpha]
            U[n,n] = LA.norm(Vtilde[:,n])
            V[:,n] = Vtilde[:,n]/U[n,n]
        
        print("CPU time for building of WCAWE basis [UNIVARIATE] : {0}".format(time()-t0))    
        
        # Therefore, V stands for the WCAWE basis
        return V


    def solve_wcawe_UNIVARIATE(self):
        # reduced_matrices results as the projection of the different matrices on
        # the WCAWE basis. The size of the matrices is now (nvecfreq, nvecfreq)
        reduced_matrices = []

        # sub_range contains the sub intervals. For i Padé point (f=f0), sub_range 
        # contains i sub intervals.
        n_sub_range = len(self.param['sub_range'])
        nvecfreq_sub = int(self.param['nvecfreq']/n_sub_range)
        # here WCAWE_BASIS will be the concatenation of the sub basis, this will be
        # done with an svd recombination. The svd can delete a few vectors so that 
        # the size of WCAWE_BASIS before and after the svd can change a little, see 
        # log info during computation to know.
        WCAWE_BASIS = np.zeros((self.ndof,n_sub_range*nvecfreq_sub),
                               dtype=np.cfloat)
        
        for l in range(n_sub_range):
            (coeff_deriv_LHS_fi,coeff_deriv_RHS_fi) = \
                                    self.deriv.evaluate([self.param['freqref'][l]])
            WCAWE_BASIS_tmp = self.build_basis_UNIVARIATE(coeff_deriv_LHS_fi,
                                                          coeff_deriv_RHS_fi,
                                                          nvecfreq_sub)
            WCAWE_BASIS[:,l*nvecfreq_sub:(l+1)*nvecfreq_sub] = WCAWE_BASIS_tmp[:,:]
            
        ########### SVD computation ##########
        WCAWE_BASIS = compute_svd(WCAWE_BASIS)
        ######################################    
        
        # projection of the LHS matrices onto the WCAWE basis
        for i in range(self.nmat):
            matrix_tmp = csc_matrix((self.ndof,self.ndof))
            matrix_tmp = self.LHS[i].dot(WCAWE_BASIS)
            matrix_tmp = (np.matrix.getH(WCAWE_BASIS)).dot(matrix_tmp)
            reduced_matrices.append(np.squeeze(np.asarray(matrix_tmp)))
         
        # projection of RHS onto the WCAWE basis
        RHS_VEC = np.dot(np.matrix.getH(WCAWE_BASIS), self.RHS)
        
        Pout_WCAWE = np.zeros((self.ndof,self.param['nfreq']),dtype=np.cfloat)
        
        # frequency loop
        for i,freq_i in enumerate(self.param['freq']):
            # both value in the tuple below stand for the LHS and RHS array of cross
            # derivatives @ freq
            (coeff_deriv_LHS,coeff_deriv_RHS) = self.deriv.evaluate([freq_i])
            Aglob_red = np.zeros(reduced_matrices[0].shape,dtype=np.cfloat)
            for k in range(self.nmat):
                Aglob_red += coeff_deriv_LHS[k,0]*reduced_matrices[k]
            RHS_tmp = coeff_deriv_RHS[0]*RHS_VEC
            sol_P = LA.inv(Aglob_red).dot(RHS_tmp)
            resp_P = WCAWE_BASIS.dot(sol_P)
            Pout_WCAWE[:,i] = resp_P
    
        return Pout_WCAWE

    # here we compute the fast frequency sweep with the iterative finite element 
    # solution. Note that this calculation is costly. Use for small problems(ndof<20000) 
    # to compare WCAWE and FE solution.
    def compute_FFS_FE(self):
        FE_solution = np.zeros((self.ndof,self.param['nfreq']),dtype=np.cfloat)
        for i in range(self.param['nfreq']):
            Aglob = csc_matrix(self.LHS[0].shape)
            Aglob = self.LHS[1]-(2*np.pi*self.param['freq'][i]/340)**2*self.LHS[0]                           
            solve = factorized(Aglob)                                  
            RHS_tmp = 1j*2*np.pi*self.param['freq'][i]*self.RHS
            FE_solution[:,i] = solve(RHS_tmp)
        return FE_solution


# computation of the singular value decomposition. The goal is to only keep
# significant vectors of the basis.That is made by considering that a very small
# singular value stands for a collinear vector
def compute_svd(matrix):
    u,s,v = LA.svd(matrix)
    max_sv = max(s)
    index = []
    for i in range(len(s)):
        if s[i] > max_sv*1e-15:
            index.append(i)
    print("[SVD:Info] Number of selected vector : {0}/{1}".format(len(index),len(s)))
    return u[:,index]










