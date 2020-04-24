#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:58:54 2020

@author: julien
"""


###############################################################################
#                           Test case using WCAWE                             #
#                                                                             #
#                               March 2020                                    #
###############################################################################


#------------------------------------------------------------------------------
# Init main program
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# begin import
#------------------------------------------------------------------------------
import numpy as np
import os
import contextlib
from matrices_manager import*
from sympy import*
from derivatives import*
from wcawe_univariate import*
#------------------------------------------------------------------------------
# end import
#------------------------------------------------------------------------------


# import pickle
# from sympy.parsing.sympy_parser import parse_expr


# str_func = 'x'
# sym = {'x':Symbol('x'),'y':Symbol('y')}
# func = parse_expr(str_func,sym)

# func = diff(func,sym['x'],0)
# #print(type(func))
# func = diff(func,sym['y'],0)
# #print(func)
# # a = a.subs(sym['x'],2)
# # a = a.subs(sym['y'],2)
# #print(type(func) != mul.Mul)


flag = dict()
flag['rerun'] = 0
flag['recalculated'] = 0
flag['calculateFE'] = 0
flag['calculateWCAWE'] = 0

flag['plotcomparison'] = 0
flag['plotcomparisonMULTI'] = 0
flag['convert2VTK'] = 0



#Source
param = dict()
param['source'] = np.array([0,0,0])
param['S_amp'] = 0.1

#Material parameters

param['rho'] = 1.213
param['c0'] = 342.2
param['P0'] = 2e-5

#Frequency range
param['fmin'] = 200
param['fmax'] = 600
param['f_range'] = np.array([param.get('fmin'),
                             param.get('fmax')])
param['freqincr'] = 4
param['freq'] = np.array(range(param.get('fmin'),
                          param.get('fmax'),
                          param.get('freqincr')))
param['nfreq'] = param.get('freq').size
param['freqref'] = np.array([300,400,500])
param['nfreqref'] = param.get('freqref').size

# Input data for the loop over expansion orders. Note that for each
# frequency sweep the number of vectors from the WCAWE basis will depend on
# the number of point for Padé expension. For instance, if we have 2 points
# for expansion, and nvecfreq=5 (order of expansion), we will have 15
# vectors in the basis, 5 by intervals.
param['nvecfreqmin'] = 20 
param['nvecfreqmax'] = 20
param['incrvec'] = 20
param['vecfreqrange'] = np.array(range(param.get('nvecfreqmin'),
                                  param.get('nvecfreqmax'),
                                  param.get('incrvec'))) 

# build sub_range with the different freqref
index_freqref = [np.argmin(np.abs(param.get('freq')-param.get('freqref')[i]))
                 for i in range(param.get('nfreqref'))]

caracteristic_index = [0]
for i in range(len(index_freqref)):
    caracteristic_index.append(index_freqref[i])
caracteristic_index.append(param.get('nfreq')) #equivalent to end in MATLAB

sub_range = []
if param.get('nfreqref')==1:
    sub_range.append(param.get('freq'))
else:
    for i in range(param.get('nfreqref')):
        if i==0:
            left_edge = 0
            right_edge = int((caracteristic_index[i+1]+caracteristic_index[i+2])/2)
            sub_range.append(param.get('freq')[left_edge:right_edge])
        elif i==param.get('nfreqref')-1:
            left_edge = int((caracteristic_index[i]+caracteristic_index[i+1])/2)
            right_edge = caracteristic_index[-1]
            sub_range.append(param.get('freq')[left_edge:right_edge])
        else:
            left_edge = int((caracteristic_index[i]+caracteristic_index[i+1])/2)
            right_edge = int((caracteristic_index[i+1]+caracteristic_index[i+2])/2)
            sub_range.append(param.get('freq')[left_edge:right_edge])

param['sub_range'] = sub_range

param['filename'] = "Diapason"

#---------------------------------------------------------------------------------
# Create lists for systematic dealing with "linear" combination" for system matrix
#---------------------------------------------------------------------------------

# Cell array of global matrices for LHS

#--------------------------------------------------------------------------
#matrices calculated with Freefem++
param['matrices_names'] = ["ReM.txt","ImM.txt","ReK.txt","ImK.txt"]
param['Nodes'] = "Nodes_"+param.get('filename')+".txt"
#--------------------------------------------------------------------------



m = MatManager(param,flag)
m.run_freefem()
m.load_matrices()
m.merge_matrices([[0,1],[2,3]], [[1,1j],[1,1j]])



m.coeff_LHS = ['-(2*3.1415*f/340)**2','1']
m.coeff_RHS = '1j*2*3.1415*f'
m.derivative_orders = [30]
m.variables = ['f']
m.build_RHS(0.1,[0,0,0])

deriv = Derivative(m)
t0 = time()
deriv.create_cross_derivatives()
deriv.load_cross_derivatives()



param['nvecfreq'] = 60

Pout_WCAWE = solve_wcawe_UNIVARIATE(param,m,deriv)

FE_solution = compute_FFS_FE(param,m)

import matplotlib.pyplot as plt

plt.plot(param['freq'],np.absolute(Pout_WCAWE[m.id_source,:]))
plt.plot(param['freq'],np.absolute(FE_solution[m.id_source,:]),'+',color='red')
plt.show()












            
            
            