##========================================================================================
import numpy as np
from scipy import linalg

# 2018.12.08: inferring network interactions from non-sequential data, no self interactions
# input: non-sequential data s[l,n] 
# output: interaction w_infer[n,n], and extenal local field (bias) h0_infer[n]

def fit_interaction(s):
    l,n = s.shape
    
    nloop = 100

    wini = np.random.normal(0.0,1/np.sqrt(n),size=(n-1,n))
    w_infer = np.zeros((n,n))
    h0_infer = np.zeros(n)
    
    h0ini = np.random.normal(0.0,1/np.sqrt(n),size=n)
    for i in range(n):
        x = np.delete(s,i,axis=1)  # 
        y = s[:,i].copy()          # target
        w = wini[:,i].copy()
        h0 = h0ini[i].copy()

        m = x.mean(axis=0)
        dx = x - m[np.newaxis,:]
        c = np.cov(dx,rowvar=False,bias=True)
        c_inv = linalg.inv(c)

        cost = np.full(nloop,100.)
        for iloop in range(nloop):
            h = x.dot(w) + h0

            y_model = np.tanh(h)
            cost[iloop] = ((y - y_model)**2).mean()    
            if iloop > 0 and cost[iloop] >= cost[iloop-1]: break

            which_non_zero = h!=0
            h[which_non_zero] *=  y[which_non_zero]/y_model[which_non_zero]

            h_av = np.mean(h)
            dhdx = dx*((h - h_av)[:,np.newaxis])
            dhdx_av = dhdx.mean(axis=0)

            w = dhdx_av.dot(c_inv)
            h0 = h_av-np.sum(w*m)

        w_infer[:i,i] = w[:i]   
        w_infer[i+1:,i] = w[i:]
        h0_infer[i] = h0
        
    return w_infer,h0_infer

