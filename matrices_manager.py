#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:58:54 2020

@author: julien
"""

import os
import contextlib
from time import time
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu,factorized

class MatManager():
    def __init__(self,param,flag):
        self.param = param
        self.flag = flag
        self.matrices = []
        self.RHS = None
        self.nmat = 0
        self.nodes = []
        self.ndof = 0
        self.coeff_LHS = None 
        self.coeff_RHS = None
        self.derivative_orders = None
        self.variables = None
        self.path = "Matrices/"+param.get('filename')+"/"
        self.id_source = 0
        
        
    def run_freefem(self):
        updated = False
        for matrix_name in self.param['matrices_names']:
            if not(os.path.exists(self.path+matrix_name)):
                updated = 1
        if not(os.path.exists(self.path+self.param.get('Nodes'))):
                updated = 1

                
        if updated or self.flag.get('rerun'):
            t0 = time()
            print("***********************************")
            print("*      Rerun FreeFem++ script     *")
            print("***********************************")
            edpcommand = "FreeFem++ "+self.param.get('filename')+".edp"
            os.system(edpcommand)
            print("**********************************************************")
            print("[get_matrices:infos] CPUtime for building of matrices %d s",time()-t0)
            print("**********************************************************")
            self.flag['recalculated'] = 1
        else:
            print("***********************************")
            print("*No need to rerun FreeFem++ script*")
            print("***********************************")
        
        

    
    def load_matrices(self):
        for matrix_name in self.param['matrices_names']:
            self.build_matrix(matrix_name)
        self.nmat = len(self.matrices)
        self.build_Nodes(self.param['Nodes'])
        

    def build_matrix(self,file_name):
        header_lines,counter = 3,0
        rows,colums,values = [],[],[]
        with open(self.path+file_name) as mat_file:
            for line in mat_file:
                counter+=1
                if counter > header_lines:
                    data = line.split()
                    if len(data)!=0:
                        rows.append(int(data[0]))
                        colums.append(int(data[1]))
                        values.append(float(data[2]))
        sparse_matrix = csc_matrix((values,(rows,colums)),dtype=np.cfloat)
        self.matrices.append(sparse_matrix)

            
    
    def build_Nodes(self,file_name):
        with open(self.path+file_name,'r') as matrix_file:
            for line in matrix_file:
                data = line.split()
                if len(data)!=0:
                    self.nodes.append([float(data[0]),float(data[1]),float(data[2])])
        self.nodes = np.array(self.nodes)
        self.ndof = len(self.nodes)
        
    def build_RHS(self,S_amp,loc_source):
        self.RHS = np.zeros((self.ndof,),dtype=np.cfloat)
        array_dist = np.abs(self.nodes[:,0]-loc_source[0]) + \
                     np.abs(self.nodes[:,1]-loc_source[1]) + \
                     np.abs(self.nodes[:,2]-loc_source[2])
        self.id_source = np.argmin(array_dist)
        self.RHS[self.id_source] = S_amp
    
    def merge_matrices(self,merge_list,coeff_merge):
        new_matrices = []
        for i in range(len(merge_list)):
            tmpmat = csc_matrix((self.ndof,self.ndof))
            for k in range(len(merge_list[i])):
                tmpmat += coeff_merge[i][k]*self.matrices[merge_list[i][k]]
            new_matrices.append(tmpmat)
        self.matrices = new_matrices
        self.nmat = len(self.matrices)
        
        
        
        
        
        
        
            