# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 03:34:53 2024

@author: poloz
"""

### Glicko Rating System Algorithm

import numpy as np

tau = 0.4

def gphi(phi):
    g = 1/(np.sqrt(1+((3*phi**2)/np.pi**2)))
    return g

def E(mu1, mu2, phi2):
    g = gphi(phi2)
    e = 1/(1+np.exp((-g)*(mu1-mu2)))
    return e

def f(x, delta, phi, v, a):
    y = (np.exp(x)*((delta**2)-(phi**2)-v-np.exp(x)))/((2*((phi**2)+v+np.exp(x)))**2) - ((x-a)/tau**2)
    return y

def newvol(delta, phi, v, vol):
    ### Iteration used to determine new volitility rating
    a = np.log(vol**2)
    A = a
    
    if delta**2 > ((phi**2)+v):
        B = np.log((delta**2)-(phi**2)-v)
    elif delta**2 <=((phi**2)+v):
        k = 0
        fval = -1
        
        while fval < k:
            k += 1
            fval = f((a-(k*tau)), delta, phi, v, a)
            
            if k >= 1000:
                print("ERROR")
                quit()
            
        B = a - (k*tau)
        
    fA = f(A, delta, phi, v, a)
    fB = f(B, delta, phi, v, a)
    #Convergent tolerance
    eta = 0.000001
    count = 0
    while np.abs(B-A) > eta:
        count += 1
        C = A+((A-B)*fA/(fB-fA))
        fC = f(C, delta, phi, v, a)
        
        if fC*fB <= 0:
            A = B
            fA = fB
        else:
            fA = fA/2
            
        B = C
        fB = fC
        if count >= 1000:
            print("ERROR")
            quit()
    
    newsig = np.exp(A/2)
        
    return newsig

def glicko2(r, rd, vol, r2, rd2, s):
    ### r, rd, vol are all initial player values for rating, rating deviation, and volatility
    ### r2 and rd2 are the same as above but for the other player
    ### s is the outcome of the game, 0 for a loss, .5 for a draw, 1 for a win
    
    ### Convert r, r2, rd, rd2 to Glicko-2 Scale
    mu1 = (r-1500)/173.7178
    phi1 = rd/173.7178
    mu2 = (r2-1500)/173.7178
    phi2 = rd2/173.7178
    
    ### Solve for g function and E funtion valeus
    g = gphi(phi2)
    e = E(mu1, mu2, phi2)
    
    ###Solve for variance
    v = 1/((g**2)*e*(1-e))
    
    ### Solve for Improvement in Rating
    delta = v*(g)*(s-e)
    
    ### Solve for new volatility
    nv = newvol(delta, phi1, v, vol)
    
    ### Update to new pre-period value
    newpreval = np.sqrt((phi1**2)+(nv**2))
    
    ### Update to new rating and rd
    newphi = 1/np.sqrt(((1/(newpreval**2))+(1/v)))
    newmu = mu1 + (newphi**2)*g*(s-e)
    
    newr = round(173.7178*newmu+1500)
    newrd = round(173.7178*newphi, 10)
    
    return (newr, newrd, nv)