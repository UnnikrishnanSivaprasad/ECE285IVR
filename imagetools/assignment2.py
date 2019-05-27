""" Assignment 2

COMPLETE THIS FILE

Your name here: Unnikrishnan Sivaprasad

"""

from .provided import *

def shift(x, k, l, boundary='periodical'):
    n1, n2 = x.shape[:2]
    xshifted = np.zeros(x.shape)
    # Main part
    #for i in range(max(-k, 0), min(n1-k, n1)):
     #   for j in range(max(-l, 0), min(n2-l, n2)):
      #      xshifted[i, j] = x[i + k, j + l]
    if boundary is 'periodical':
        irange = np.mod(np.arange(n1) + k, n1)
        jrange = np.mod(np.arange(n2) + l, n2)
        xshifted = x[irange, :][:, jrange]
    
    if boundary is 'zero_padding':
        xshifted = np.zeros(x.shape)
        # Main part
        for i in range(max(-k, 0), min(n1-k, n1)):
            for j in range(max(-l, 0), min(n2-l, n2)):
                xshifted[i, j] = x[i + k, j + l]
                
    if boundary is 'Extension':
        irange = np.mod(np.arange(n1) + k, n1)
        jrange = np.mod(np.arange(n2) + l, n2)
        xshifted = x[irange, :][:, jrange]
        for i in range(n1 - k, n1):
            for j in range(-l,n2):
                xshifted[i, j] =xshifted[n1-k-1, j]
        for i in range(0, n1):
            for j in range(0,-l):
                xshifted[i, j] =xshifted[i,-l+1]
    elif boundary is "mirror":
        xshifted[max(-k,0):min(n1-k,n1),max(-l,0):min(n2-l,n2)] = x[max(-k,0)+k:min(n1-k,n1)+k,max(-l,0)+l:min(n2-l,n2)+l]
        if k > 0:
            num_samples = n1-min(n1-k,n1)
            rows = np.arange(min(n1-k,n1)-1,min(n1-k,n1)-num_samples-1,-1)
            xshifted[min(n1-k,n1):,max(-l,0):min(n2-l,n2)]=xshifted[rows,max(-l,0):min(n2-l,n2)]
        else:
            num_samples = max(-k,0)
            rows = np.arange(2*max(-k,0),max(-k,0),-1)
            xshifted[:max(-k,0),max(-l,0):min(n2-l,n2)] = xshifted[rows,max(-l,0):min(n2-l,n2)]
        if l > 0:
            index = 1
            for col in range(min(n2-l,n2),n2):
                xshifted[:,col] = xshifted[:,min(n2-l,n2)-index]
                index = index +1
        else:
            index = 1
            for col in range(max(-l,0),2*max(-l,0)):
                xshifted[:,max(-l,0)-index] = xshifted[:,col]
                index = index + 1
                
    if boundary is 'Mirror':
        irange = np.mod(np.arange(n1) + k, n1)
        jrange = np.mod(np.arange(n2) + l, n2)
        xshifted = x[irange, :][:, jrange]
        for i in range(n1 - k, n1):
            for j in range(-l,n2):
                xshifted[i, j] =xshifted[2*(n1-k)-i-1, j]
        for i in range(0, n1):
            for j in range(0,-l):
                xshifted[i, j] =xshifted[i,2*(-l+1)-j]
        
       
    return xshifted


def kernel(name,tau=1,eps=1e-3): 
    
    if name is "gaussian":
        s1 = np.floor(np.sqrt(-2*np.square(tau)*np.log(eps)))
        s2 = np.floor(np.sqrt(-2*np.square(tau)*np.log(eps)))
        x_coord,y_coord = np.meshgrid(np.arange(-s1,s1+1),np.arange(-s2,s2+1),indexing="ij")
        kernel = np.exp((-np.square(x_coord) - np.square(y_coord))/(2*np.square(tau)))
    elif name is "exponential":
        s1 = np.floor(-tau*np.log(eps))
        s2 = np.floor(-tau*np.log(eps))
        x_coord,y_coord = np.meshgrid(np.arange(-s1,s1+1),np.arange(-s2,s2+1),indexing="ij")
        kernel = np.exp(-np.sqrt(np.square(x_coord)+np.square(y_coord))/tau)
    elif name is "box":
        s1 = tau
        s2 = tau
        kernel = np.ones(((2*s1) + 1,(2*s2) + 1))
    kernel = kernel/np.sum(kernel)
    
    return kernel


def convolve_naive(x, nu):
    n1, n2 = x.shape[:2]
    s1 = int((nu.shape[0] - 1) / 2)
    s2 = int((nu.shape[1] - 1) / 2)
    xconv = np.zeros(x.shape)
    for i in range(s1, n1-s1):
        for j in range(s2, n2-s2):
            for k in range(-s1, s1+1):
                for l in range(-s2, s2+1):
                    xconv[i,j,:]+= x[i-k,j-l,:]*nu[k+s1,l+s2]
    return xconv

def convolve1(x, nu, boundary='periodical_naive'):
    xconv = np.zeros(x.shape)
    s1 = int((nu.shape[0] - 1) / 2)
    s2 = int((nu.shape[1] - 1) / 2)
    for k in range(-s1, s1+1):
        for l in range(-s2, s2+1):
            xconv+= shift(x,k,l,boundary=boundary)*nu[k,l]
    return xconv
           
    
