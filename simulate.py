"""
functions for generating binary variables, non-sequential data
"""
import numpy as np
import function as ft

#=========================================================================================
# 2018.12.08: generate coupling matrix w0: wji from j to i
def generate_interactions(n,g):
    w = np.random.normal(0.0,g/np.sqrt(n),size=(n,n))
    
    # no self-interactions
    for i in range(n):
        w[i,i] = 0.   # no self-interactions

    # symmetry
    for i in range(n):
        for j in range(n):
            if j > i: w[i,j] = w[j,i]       
        
    return w
#=========================================================================================    
""" 2018.12.08: generate non-sequential data composed of binary variables 
    input: interaction matrix w[n,n], interaction variance g, data length l
    output: non-sequential s[l,n]
"""     
# generate non-sequential data
def generate_data(w,h0,l):
    n = w.shape[0]    
    s = np.random.randint(2,size=(l,n))
    s = 2.*s - 1.

    nrepeat = 50*n
    for irepeat in range(nrepeat):
        for i in range(n):   
            h = s.dot(w[:,i])+h0[i]  # Wji from j to i
            p = 1/(1+np.exp(-2*h))
            s[:,i]= ft.sign_vec(p-np.random.rand(l))
    return s
