import numpy as np 
import numpy.fft as nf 

def kernel2fft(nu, n1, n2, separable=None):
    '''
    Converts kernel to frequency domain for convolution operation
    Args:
    -----
        nu:kernel for 
        n1:number of rows in lbd
        n2:numf of columns in lbd
    Kwargs:
    -------
        separable:
        default:None
    Returns:
    -------
        lbd: n1 × n2 complex valued array lambda corresponding to the frequential response of the convolution kernel function 
    '''
    if not separable:
        kernel = np.zeros((n1,n2))
        s1,s2 = nu.shape
        s1,s2 = int((s1-1)/2),int((s2-1)/2)
        try:
            kernel[:s1+1, :s2+1] = nu[s1:2*s1+1, s2:2*s2+1]#4
            kernel[-s1:,-s2:] = nu[:s1,:s2]#1
            kernel[-s1:,:s2+1] = nu[:s1,s2:]#2
            kernel[:s1+1,-s2:] = nu[s1:,:s2]#3
        except:
            kernel[0,0] = 1
        lbd = nf.fft2(kernel, axes=(0, 1))
    if separable is "product":
        nu1,nu2 = nu
        s1,s2 = nu1.shape[0],nu2.shape[1]
        s1,s2 = int((s1-1)/2),int((s2-1)/2)

        kernel1 = np.zeros((n1,1))
        if s1!=0:
            kernel1[-s1:] = nu1[:s1] 
            kernel1[:s1+1] = nu1[s1:] 
        else:
            kernel1[0,0] = 1
        
        kernel2 = np.zeros((1,n2))
        if s2!=0:
            kernel2[0,-s2:] = nu2[0,:s2] 
            kernel2[0,:s2+1] = nu2[0,s2:] 
        else:
            kernel2[0,0] = 1
            
        nu1 = nf.fft(kernel1,axis=0)
        nu2 = nf.fft(kernel2,axis=1)
        lbd = np.outer(nu1,nu2)
    elif separable is "sum":
        nu1,nu2 = nu
        s1,s2 = nu1.shape[0],nu2.shape[1]
        s1,s2 = int((s1-1)/2),int((s2-1)/2)

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
    '''
    Computes convolution of x with a kernel of frequency response lbd
    Args:
        x : input image
        lbd : frequential response of kernel
    Returns:
        y : output of convolution
    '''
    if len(x.shape)==3:
        temp = nf.fft2(x,axes=(0,1))*np.expand_dims(lbd,axis=-1)
    elif len(x.shape)==2:
        temp = nf.fft2(x,axes=(0,1))*lbd
    res = np.abs(nf.ifft2(temp,axes=(0,1)))
    return res

def adjoint(nu):
    '''
    Computes adjoint of convolution kernel nu
    Args:
        nu: kernel to compute adjoint
    Returns:
        mu: adjoint of kernel nu
    '''
    mu = np.flip(np.flip(nu,axis=0),axis=1)
    return mu