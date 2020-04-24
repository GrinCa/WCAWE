#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:15:51 2020

@author: julien
"""

from time import time
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import factorized
from numpy import linalg as LA


def build_basis_UNIVARIATE(mat_manager,coeff_deriv_LHS,coeff_deriv_RHS,nvecfreq):
    print("***************************************")
    print("*Calculating WCAWE basis ({0} vectors)*".format(nvecfreq))
    print("***************************************")
    
    t0 = time()
    
    # Initialization of data needed for calculation. This part is unecessary for
    # the algorithm to work but it enables to clear the code and make it more 
    # consice
    ndof = mat_manager.ndof
    LHS = mat_manager.matrices
    nmat = mat_manager.nmat
    RHS = mat_manager.RHS
    
    # The algorithm from Slone (See the doc) begins here. Vtilde, V, U are defined 
    # according to the Gram-Schmidt process by V = Vtilde*U^-1. U is the ortho-
    # gonalization matrix. 
    Vtilde = np.zeros((ndof,nvecfreq),dtype=np.cfloat)
    V = np.zeros((ndof,nvecfreq),dtype=np.cfloat)
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
    Aglob = csc_matrix((ndof,ndof),dtype=np.cfloat)
    for k in range(nmat):
        Aglob += coeff_deriv_LHS[k,0]*LHS[k]
    
    # here we factorized the LHS, with the LU method (see scipy.org doc)
    solve = factorized(Aglob)
    Vtilde[:,0] = solve(coeff_deriv_RHS[0]*RHS)
    U[0,0] = LA.norm(Vtilde[:,0])
    V[:,0] = Vtilde[:,0]/U[0,0]
    
    for n in range(1,nvecfreq):
        term1 = 0
        term2 = 0
        for m in range(n-1):
            # Pu1 and Pu2 are correction terms
            PU1 = Pu1(n,m,U)
            term1 += coeff_deriv_RHS[m+1]*RHS*PU1[0,n-m-1]
        for m in range(1,n-1):
            PU2 = Pu2(n,m,U)
            A_m = csc_matrix((ndof,ndof),dtype=np.cfloat)
            for k in range(nmat):
                A_m += coeff_deriv_LHS[k,m+1]*LHS[k]
                term2 += np.dot(A_m.dot(V[:,0:n-m]), PU2[:,n-m-1])
        A_1 = csc_matrix((ndof,ndof),dtype=np.cfloat)
        for k in range(nmat):
            A_1 += coeff_deriv_LHS[k,1]*LHS[k]
        try:
            Vtilde[:,n] = solve(term1-term2-A_1.dot(V[:,n-1]))
        except:
            print('')
        # here we orthogonalize the vectors of the basis
        for alpha in range(n):
            U[alpha,n] = np.dot(np.matrix.getH(V[:,alpha]), Vtilde[:,n])
            Vtilde[:,n] -= U[alpha,n]*V[:,alpha]
        U[n,n] = LA.norm(Vtilde[:,n])
        V[:,n] = Vtilde[:,n]/U[n,n]
    
    print("CPU time for building of WCAWE basis [UNIVARIATE] : {0}".format(time()-t0))    
    
    # Therefore, V stands for the WCAWE basis
    return V


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


def solve_wcawe_UNIVARIATE(param,mat_manager,deriv_obj):
    # reduced_matrices results as the projection of the different matrices on
    # the WCAWE basis. The size of the matrices is now (nvecfreq, nvecfreq)
    reduced_matrices = []
    ndof = mat_manager.ndof
    nmat = mat_manager.nmat
    # sub_range contains the sub intervals. For i Padé point (f=f0), sub_range 
    # contains i sub intervals.
    n_sub_range = len(param['sub_range'])
    nvecfreq_sub = int(param['nvecfreq']/n_sub_range)
    # here WCAWE_BASIS will be the concatenation of the sub basis, this will be
    # done with an svd recombination. The svd can delete a few vectors so that 
    # the size of WCAWE_BASIS before and after the svd can change a little, see 
    # log info during computation to know.
    WCAWE_BASIS = np.zeros((ndof,n_sub_range*nvecfreq_sub),
                           dtype=np.cfloat)
    
    for l in range(n_sub_range):
        (coeff_deriv_LHS_f0,coeff_deriv_RHS_f0) = deriv_obj.evaluate([param['freqref'][l]])
        WCAWE_BASIS_tmp = build_basis_UNIVARIATE(mat_manager,
                                                coeff_deriv_LHS_f0,
                                                coeff_deriv_RHS_f0,
                                                nvecfreq_sub)
        WCAWE_BASIS[:,l*nvecfreq_sub:(l+1)*nvecfreq_sub] = WCAWE_BASIS_tmp[:,:]
        
    ########### SVD computation ##########
    WCAWE_BASIS = compute_svd(WCAWE_BASIS)
    ######################################    
    
    for i in range(nmat):
        matrix_tmp = csc_matrix((ndof,ndof))
        matrix_tmp = mat_manager.matrices[i].dot(WCAWE_BASIS)
        matrix_tmp = (np.matrix.getH(WCAWE_BASIS)).dot(matrix_tmp)
        reduced_matrices.append(np.squeeze(np.asarray(matrix_tmp)))
        
    RHS_VEC = np.dot(np.matrix.getH(WCAWE_BASIS), mat_manager.RHS)
    
    Pout_WCAWE = np.zeros((ndof,param['nfreq']),dtype=np.cfloat)
    
    # frequency loop
    for i,freq in enumerate(param['freq']):
        # both value in the tuple below stand for the LHS and RHS array of cross
        # derivatives @ freq
        (coeff_deriv_LHS,coeff_deriv_RHS) = deriv_obj.evaluate([freq])
        Aglob_red = np.zeros(reduced_matrices[0].shape,dtype=np.cfloat)
        for k in range(nmat):
            Aglob_red += coeff_deriv_LHS[k,0]*reduced_matrices[k]
        RHS = coeff_deriv_RHS[0]*RHS_VEC
        sol_P = LA.inv(Aglob_red).dot(RHS)
        resp_P = WCAWE_BASIS.dot(sol_P)
        Pout_WCAWE[:,i] = resp_P

    return Pout_WCAWE

# here we compute the fast frequency sweep with the iterative finite element 
# solution. Note that this calculation is costly. Use for small problems(ndof<20000) 
# to compare WCAWE and FE solution.
def compute_FFS_FE(param,mat_manager):
    FE_solution = np.zeros((mat_manager.ndof,param['nfreq']),dtype=np.cfloat)
    for i in range(param['nfreq']):
        Aglob = csc_matrix(mat_manager.matrices[0].shape)
        Aglob = mat_manager.matrices[1]-(2*np.pi*param['freq'][i]/340)**2*mat_manager.matrices[0]                           
        sol = factorized(Aglob)                                  
        RHS = 1j*2*np.pi*param['freq'][i]*mat_manager.RHS
        FE_solution[:,i] = sol(RHS)
    return FE_solution













