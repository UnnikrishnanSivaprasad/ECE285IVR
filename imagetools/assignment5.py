""" Assignment 5

COMPLETE THIS FILE

Your name here:

"""

from .assignment4 import *
import numpy.fft as npf

def kernel2fft(nu, n1, n2, separable=None):
    kernel=np.zeros((n1,n2))
    lbd = 0
    if not separable:
        r,c=np.shape(nu)
        s1= int((r-1)/2)
        s2= int((c-1)/2)
        
        kernel[:s1+1, :s2+1] = nu[s1:2*s1+1, s2:2*s2+1] #blue
        kernel[-s1:,-s2:] = nu[:s1,:s2] #red
        kernel[-s1:,:s2+1] = nu[:s1,s2:] #yellow
        kernel[:s1+1,-s2:] = nu[s1:,:s2] #green
        
        lbd = npf.fft2(kernel, axes=(0, 1))
    elif separable is "product":
        
        nu1=nu[0]
        nu2=nu[1]
        
        r=nu1.shape[0]
        c=nu2.shape[1]
        
        s1= int((r-1)/2)
        s2= int((c-1)/2)
        
        kernel1=np.zeros((n1,1))
        kernel1[-s1:]=nu1[:s1]
        kernel1[:s1+1]=nu1[s1:]
        
        kernel2=np.zeros((1,n2))
        kernel2[0,-s2:]=nu2[0,:s2]
        kernel2[0,:s2+1]=nu2[0,s2:]
        
        lbd1 = npf.fft(kernel1,axis=0)
        lbd2 = npf.fft(kernel2,axis=1)
        
        lbd = np.outer(lbd1,lbd2)
        
    elif separable is "sum":
        nu1=nu[0]
        nu2=nu[1]
        
        r=nu1.shape[0]
        c=nu2.shape[1]
        
        s1= int((r-1)/2)
        s2= int((c-1)/2)

        kernel1 = np.zeros((n1,1))
        kernel1[-s1:] = nu1[:s1] 
        kernel1[:s1+1] = nu1[s1:] 
        
        kernel2 = np.zeros((1,n2))
        kernel2[0,-s2:] = nu2[0,:s2] 
        kernel2[0,:s2+1] = nu2[0,s2:] 
        
        nu1 = nf.fft(kernel1,axis=0)
        nu2 = nf.fft(kernel2,axis=1)
        nu1 = np.repeat(nu1,nu2.shape[1],axis=1)
        nu2 = np.repeat(nu2,nu1.shape[0],axis=0)
        lbd = nu1+nu2
        
    
    return lbd

def convolvefft(x, lbd):
    ##Fourier domain
    if x.ndim==3:
        lbd=lbd.reshape(x.shape[0],x.shape[1],1)
    tf=npf.fft2(x,axes=(0,1))
    a=np.abs(tf)
    phi=np.angle(tf)
    ap=a*lbd
    ans_fourier=ap*np.exp(1j*phi)
    ##Time domain
    ans_time=np.abs(npf.ifft2(ans_fourier,axes=(0,1)))
    return ans_time