import numpy as np
from .assignment2 import shift,kernel,convolve

def bilateral_naive(y, sig, s1=2, s2=2, h=1):
    '''
    Bilateral naive denoising. Densoing approach where pixel intensities in neighbourhood are reweighted based on
    how similar they are to the pixel under consideration
    Args:
        y:Input image
        sig:std deviation of noise assumed in the image
    Kwargs:
        s1:height of filter/ region of interest
        s2:width of filter/ region of interest
        h:scaling for variance
    Returns:
        x:denoised image
    ''' 
    assert isinstance(s1,int),"Incorrect input for s1"
    assert isinstance(s2,int),"Incorrect input for s2"
    assert isinstance(h,int),"Incorrect input for h"

    n1, n2 = y.shape[:2] 
    c = y.shape[2] if y.ndim == 3 else 1 
    x = np.zeros(y.shape) 
    Z = np.zeros((n1, n2, *[1] * (y.ndim - 2))) 
    var = np.square(sig)
    for i in range(s1, n1-s1): 
        for j in range(s2, n2-s2): 
            for k in range(-s1, s1 + 1): 
                for l in range(-s2, s2 + 1): 
                    alpha = ((y[i + k, j + l] - y[i, j])**2).mean()
                    phi = np.exp(-np.sqrt(c)*max(alpha - 2*h*var,0)/(2*np.sqrt(2)*h*var))
                    x[i,j] = x[i,j] + phi*y[i + k, j + l]
                    Z[i,j] = Z[i,j] + phi
    Z[Z==0] = 1
    x = x/Z
    return x

def bilateral(y, sig, s1=10, s2=10, h=1,boundry="mirror"): 
    '''
    Bilateral naive denoising. Densoing approach where pixel intensities in neighbourhood are reweighted based on
    how similar they are to the pixel under consideration
    Args:
        y:Input image
        sig:std deviation of noise assumed in the image

    Kwargs:
        s1:height of filter/ region of interest
        s2:width of filter/ region of interest
        h:scaling for variance
    Returns:
        x:denoised image
    '''
    assert isinstance(s1,int),"Incorrect input for s1"
    assert isinstance(s2,int),"Incorrect input for s2"
    assert isinstance(h,int),"Incorrect input for h"
    assert isinstance(boundry,str),"Incorrect input for boundry"

    n1, n2 = y.shape[:2] 
    c = y.shape[2] if y.ndim == 3 else 1 
    x = np.zeros(y.shape) 
    Z = np.zeros((n1, n2, *[1] * (y.ndim - 2))) 
    var = np.square(sig)
    for k in range(-s1, s1 + 1): 
        for l in range(-s2, s2 + 1): 
            y_shift = shift(y,k,l,boundry=boundry)
            alpha = np.mean(np.square(y_shift - y),axis=-1)
            phi = np.expand_dims(np.exp(-np.maximum(alpha - (2*h*var),np.zeros_like(alpha))/(2*np.sqrt(2/c)*h*var)),axis=-1)
            x = x + (phi*y_shift)
            Z = Z + phi
            
    Z[Z==0] = 1
    x = x/Z
    return x

def nlmeans_naive(y, sig, s1=2, s2=2, p1=1, p2=1, h=1):
    '''
    Uses non local means to compute restored image. Compares patches and computes patch as a linear combination of 
    similar patches in its vicinity
    Args:
        y:input image
        sig:standard deviation of noise in conisderation
    Kwargs:
        s1:height of region of observation
        s2:width of region of observation
        p1:height of patch
        p2:width of patch
        h:scaling of variance
    Returns:
        x:denoised image
    '''

    assert isinstance(s1,int),"Incorrect input for s1"
    assert isinstance(s2,int),"Incorrect input for s2"
    assert isinstance(h,int),"Incorrect input for h"

    n1, n2 = y.shape[:2]
    P = (2*p1 + 1)*(2*p2 + 1)
    c = y.shape[2] if y.ndim == 3 else 1 
    x = np.zeros(y.shape) 
    Z = np.zeros((n1, n2, *[1] * (y.ndim - 2))) 
    var = np.square(sig)
    for i in range(s1, n1-s1-p1): 
        for j in range(s2, n2-s2-p2): 
            for k in range(-s1, s1 + 1): 
                for l in range(-s2, s2 + 1): 
                    dist2 = 0 
                    for u in range(-p1, p1 + 1): 
                        for v in range(-p2, p2 + 1): 
                            dist2 = dist2 + np.mean(np.square(y[i+k+u,j+l+v] - y[i+u,j+v]),axis=-1)
                    dist2 = dist2/P
                    phi = kernel(dist2,var,C=c,P=P,h=h)
                    x[i,j] = x[i,j] + phi*y[i+k,j+l]
                    Z[i,j] = Z[i,j] + phi
    Z[Z==0]=1
    x = x/Z         
    return x     

def nlmeans(y, sig, s1=7, s2=7, p1=None, p2=None, h=1, boundary='mirror'):
    '''
    Uses non local means to compute restored image. Compares patches and computes patch as a linear combination of 
    similar patches in its vicinity
    Args:
        y:input image
        sig:standard deviation of noise in conisderation
    Kwargs:
        s1:height of region of observation
        s2:width of region of observation
        p1:height of patch
        p2:width of patch
        h:scaling of variance
        boundary:nature of bindary used
    Returns:
        x:denoised image
    '''
    assert isinstance(s1,int),"Incorrect input for s1"
    assert isinstance(s2,int),"Incorrect input for s2"
    assert isinstance(h,int),"Incorrect input for h"
    assert isinstance(boundary,str),"Incorrect input"
    
    p1 = (1 if y.ndim == 3 else 2) if p1 is None else p1
    p2 = (1 if y.ndim == 3 else 2) if p2 is None else p2
    n1, n2 = y.shape[:2]
    x = np.zeros(y.shape) 
    Z = np.zeros((n1, n2, *[1] * (y.ndim - 2))) 
    c = y.shape[2] if y.ndim == 3 else 1 
    var = sig**2
    P = ((2*p1)+1)*((2*p2)+1)
    for k in range(-s1,s1+1):
        for l in range(-s2,s2+1):
            box1 = kernel("box1",tau=p1)
            box2 = kernel("box2",tau=p2)
            shift = shift(y,k,l)
            d = np.expand_dims(np.mean(np.square(y - shift),axis=-1),axis=-1)
            distance = convolve(d,(box1,box2),boundry=boundary,separable="product")
            phi = kernel(distance,var,C=c,P=P,h=h)
            x = x + phi*shift
            Z = Z + phi
    Z[Z==0]=1
    x = x/Z
    return x

def psnr(x,x0):
    '''
    Calculates PSNR from input image x and predicted image x0
    Args:
        x:input image
        x0:predicted image
    Returns:
        snr: peak signal to noise ration
    '''
    R = np.max(x) - np.min(x)
    snr = 10*np.log10(np.square(R)/np.mean(np.square(x - x0)))
    return snr
