""" Assignment 6

COMPLETE THIS FILE

Your name here:

"""

from .assignment5 import *

def average_power_spectral_density(x):
    x1=np.mean(x[0],axis=-1)
    y=(abs((npf.fft2(x1, axes=(0, 1))))**2)
    for x1 in x[1:]:
        x2=np.mean(x1,axis=-1)
        y1=(abs((npf.fft2(x2, axes=(0, 1))))**2)
        y+=y1
    return y/len(x)
    
def mean_power_spectrum_density(apsd):
    
    n1,n2 = apsd.shape
    s = np.log(apsd)-np.log(n1)-np.log(n2)
    
    u,v = im.fftgrid(n1,n2)
    t = np.log(np.sqrt(np.square(u/n1) + np.square(v/n2)))
    
    A = np.zeros((2,2))
    z = t.reshape(-1,1)[1:]
    A[0,0] = np.sum(np.square(z))
    A[1,0] = np.sum(z)
    A[0,1] = np.sum(z)
    A[1,1] = z.shape[0]
    
    s = s.reshape(-1,1)[1:]
    b = np.array([np.sum(s*z),np.sum(s)])
    ##  Ax=b  ##
    x = np.dot(np.linalg.inv(A),b)
    alpha=x[0]
    beta=x[1]
    mpsd = (t*alpha) + beta
    mpsd = n1*n2*np.exp(mpsd)
    mpsd[0,0] = np.inf
    
    
    return mpsd, alpha, beta
