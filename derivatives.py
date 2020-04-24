#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:24:53 2020

@author: julien
"""

import pickle
from sympy import*
from sympy.parsing.sympy_parser import parse_expr
from os import path
import numpy as np

class Derivative():
    
    def __init__(self,mat):
        self.tuples = []
        self.symbol = {}
        self.cross_derivatives_LHS = {}
        self.cross_derivatives_RHS = {}
        self.mat = mat
        self.folder = 'Derivatives/'
        
        """ Initialization of symbol dictionnary """
        
        for i in range(len(self.mat.variables)):
            self.symbol[self.mat.variables[i]] = symbols(self.mat.variables[i])


    def create_cross_derivatives(self):
        print("*******************************")
        print("*Calculating cross derivatives*")
        print("*******************************")
        
        # both dictionnary below are build so that for a tuple (i1,i2,...,in), it 
        # calculates the cross derivatives with respectively i1, the order of the
        # first variable and so on. Note that the tuple(i1,i2,...,in) is converted
        # into a string 'i1i2i3..in' to match with the input argument of a
        # dictionnary. Until this version of this code, this is the best way I found
        # to get the work done.
        derivatives_LHS = {}
        derivatives_RHS = {}
        
        # the following function creates all the tuples given a specific range limit
        # (derivative_orders). This will be useful for cross derivatives culculation
        # It fill self.tuple array. For instance, if derivative_orders is [2,3],
        # self.tuple will be :
        #[0 0;
        # 0 1
        # ...
        # 1 2
        # 1 3]

        # Note that it is a recursive function
        def get_array_index(array):
            if len(array)==len(self.mat.variables):
                self.tuples.append(array)
            else:
                for i in range(0,self.mat.derivative_orders[len(array)]):
                    tmp = array.copy()
                    tmp.append(i)
                    get_array_index(tmp)
                    
        get_array_index([])
        
        # the following loop fill derivative_LHS. Note that for LHS we iterate 
        # over the number of matrices
        for i in range(self.mat.nmat):
            for j in range(len(self.tuples)):
                indexlist = [str(val) for val in self.tuples[j]]
                indexlist.insert(0,str(i))
                func = parse_expr(self.mat.coeff_LHS[i],self.symbol)
                for k in range(len(self.symbol)):
                    func = diff(func,self.symbol[self.mat.variables[k]],self.tuples[j][k])
                    # the following line cnvert tuple into string 
                    #as said in the firstparagraph
                    derivatives_LHS[','.join(indexlist)] = func
                    
        # the following loop fill derivative_RHS
        for j in range(len(self.tuples)):
            indexlist = [str(val) for val in self.tuples[j]]
            func = parse_expr(self.mat.coeff_RHS,self.symbol)
            for k in range(len(self.symbol)):
                func = diff(func,self.symbol[self.mat.variables[k]],self.tuples[j][k])
                derivatives_RHS[','.join(indexlist)] = func   
                
        # here we use the module pickle, which enables us to serialize object,
        # or more spcefically all king of data from python, into a binary file
        with open(self.folder + 'coeff_LHS.p','wb') as coeff_LHS_file:
            pickle.dump(derivatives_LHS,coeff_LHS_file)
        with open(self.folder + 'coeff_RHS.p','wb') as coeff_RHS_file:
            pickle.dump(derivatives_RHS,coeff_RHS_file)
        # we save the derivative orders, in order not to recalculate every time
        # Then we are able to know when reading the files the highest order
        # already calculated.
        with open(self.folder + 'orders.p','wb') as orders_file:
            pickle.dump(self.mat.derivative_orders, orders_file)
               
    
    # load both coeff_LHS and coeff_RHS, and check if the orders are below those 
    # already calculated. If not, it recalculate all cross derivatives. For 
    # instance (regarding the frequency) if derivative_orders = [20], and the value
    # stored in the file orders.p = [100], we don't need to recalculate the solution
    def load_cross_derivatives(self):
        if not(path.exists(self.folder+'coeff_LHS.p')) or \
           not(path.exists(self.folder+'coeff_RHS.p') or \
           not(path.exists(self.folder+'orders.p'))):
            print('[derivatives.py] recalculation of cross derivatives')
            self.create_cross_derivatives()
            self.load_cross_derivatives()
        else:
            with open(self.folder + 'orders.p','rb') as orders_file:
                tmp_orders = pickle.load(orders_file)
                update = False
                if len(tmp_orders) != len(self.mat.derivative_orders):
                    update = True
                else:
                    for i in range(len(tmp_orders)):
                        if self.mat.derivative_orders[i] > tmp_orders[i]:
                            update = True
                if update:
                    print('[derivatives.py] recalculation of cross derivatives')
                    self.create_cross_derivatives()
                    self.load_cross_derivatives()
            with open(self.folder + 'coeff_LHS.p','rb') as coeff_LHS_file:
                self.cross_derivatives_LHS = pickle.load(coeff_LHS_file)
            with open(self.folder + 'coeff_RHS.p','rb') as coeff_RHS_file:
                self.cross_derivatives_RHS = pickle.load(coeff_RHS_file)
    
    # this function will return 2 arrays (LHS & RHS), of the cross derivatives
    # @ values
    def evaluate(self,values):
        if len(self.cross_derivatives_LHS)==0 or len(self.cross_derivatives_RHS)==0:
            print('[derivatives.py] need to load coeff matrices')
            return 0
        else:
            # Note that for LHS the size will be (nmat,ord1,...ordn), while 
            # the size for RHS will be (ord1,...,ordn)
            size_LHS = self.mat.derivative_orders.copy()
            size_LHS.insert(0,self.mat.nmat)
            
            size_RHS = self.mat.derivative_orders.copy()
            # initialization of arrays
            coeff_deriv_LHS = np.zeros(size_LHS,dtype=np.cfloat)
            coeff_deriv_RHS = np.zeros(size_RHS,dtype=np.cfloat)
            
            # here we get the values and the key of both dictionnary. The Keys
            # are strings 'nmat,ord1,...,ordn' for LHS and 'ord1,...,ordn' for RHS
            # The values are functions from sympy (symbolic module of python)
            values_LHS = self.cross_derivatives_LHS.values()
            values_RHS = self.cross_derivatives_RHS.values()
            keys_LHS = self.cross_derivatives_LHS.keys()
            keys_RHS = self.cross_derivatives_RHS.keys()
            
            # here we transform strings into tuple
            index_LHS = [tuple(map(int,keys.split(','))) 
                         for keys in keys_LHS]
            index_RHS = [tuple(map(int,keys.split(','))) 
                         for keys in keys_RHS]
            
            # this step is required to structurate data for .subs() method. 
            # It contains for each variable (for instance f) the values that it
            # takes
            data_eval = [(key_symbol,values[i]) for i,key_symbol in enumerate(self.symbol)]
            
            for i,val_LHS in enumerate(values_LHS):
                coeff_deriv_LHS[index_LHS[i]] = val_LHS.subs(data_eval)
            for i,val_RHS in enumerate(values_RHS):
                coeff_deriv_RHS[index_RHS[i]] = val_RHS.subs(data_eval)
                
        return (coeff_deriv_LHS,coeff_deriv_RHS)
      


            


    







        
            
