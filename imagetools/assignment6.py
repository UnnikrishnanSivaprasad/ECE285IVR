from .assignment2 import kernel,shift,convolve
import numpy as np 
import numpy.fft as nf 
from .provided import fftgrid

channel_mean = lambda x:np.mean(x,axis=-1)

def average_power_spectral_density(x):
    '''
    Computes average power spectral density
    Args:
        x:list of images
    Returns:
        apsd: average power density
    '''
    data = np.square(np.abs(np.dstack(list(map(nf.fft2,map(channel_mean,x))))))
    return np.mean(data,axis=-1)

def mesh(n1,n2):
    '''
    computes omega matrix for fourier transform
    Args:
        n1:number of rows in image
        n2:number of columns in image
    Returns:
        t:meshgrid of omega
    '''
    u,v = fftgrid(n1,n2)
    t = np.log(np.sqrt(np.square(u/n1) + np.square(v/n2)))
    return t
    
def coefficients(t):
    '''
    Computes coefficients of alpha and beta 
    Args:
        t:mesh of omega
    Returns:
        A:coefficients of variables
    '''
    
    A = np.zeros((2,2))
    temp = t.reshape(-1,1)[1:]
    A[0,0] = np.sum(np.square(temp))
    A[1,0] = np.sum(temp)
    A[0,1] = A[1,0]
    A[1,1] = temp.shape[0]
    return A

def constants(s,t):
    '''
    Computes constants in equations used for alpha and beta
    Args:
        apsd: average power spectral density
        t:mesh of omega values
    Returns:
        b:constant in linear equation between the log frequency and log psd
    '''
    t = t.reshape(-1,1)[1:]
    s = s.reshape(-1,1)[1:]
    b = np.zeros((2,1))
    b[0] = np.sum(s*t)
    b[1] = np.sum(s)
    return b

def mean_power_spectrum_density(apsd):
    '''
    Computes the mean power spectral density from the average power spectral density. Assumes linear relationship between log
    of power spectral density and log of frequency. Estimates the linear transformation which is used to compute the mpsd
    Args:
        apsd: Average power spectral density
    Returns:
        mpsd: Mean power spectral density
    '''
    import numpy.linalg
    n1,n2 = apsd.shape
    s = np.log(apsd)-np.log(n1)-np.log(n2)
    t = mesh(n1,n2)
    A = coefficients(t)
    b = constants(s,t)
    x = np.dot(np.linalg.inv(A),b)
    mpsd = (t*x[0]) + x[1]
    mpsd = n1*n2*np.exp(mpsd)
    mpsd[0,0] = np.inf
    return mpsd,x[0],x[1]


def deconvolve_naive(y, lbd, return_transfer=False):
    '''
    Computes naive deconvolution of noise in the image assuming blurring function is known
    Args:
        y: input image
        lbd: kernel used for blurring of image
    Kwargs:
        return_transfer: Option to return inverse of transform function
    Returns:
        xdeconv: deconvolution of image
        hhat:inverse of kernel
    '''
    y_fft = nf.fft2(y,axes=(0,1))
    hhat = 1/lbd
    n1,n2 = hhat.shape
    xdec = np.real(nf.ifft2(y_fft*hhat.reshape(n1,n2,1),axes=(0,1)))
    if return_transfer:
        return xdec, hhat
    else:
        return xdec
    
def condition_number(kernel):
    '''
    Computes the condition number of kernel
    Args:
        kernel: kernel to compute condition number of
    '''
    import numpy.linalg as linalg
    eigvals = np.abs(linalg.eigvals(kernel))
    return np.max(eigvals[eigvals!=0])/np.min(eigvals[eigvals!=0])

def deconvolve_wiener(x, lbd, sig, mpsd, return_transfer=False):
    '''
    Computes weiner deconvolution of noise in the image assuming blurring function is known
    Args:
        y: input image
        lbd: kernel used for blurring of image
        sig: variance of noise present besides the blurring
        mpsd: mean psd of images
    Kwargs:
        return_transfer: Option to return inverse of transform function
    Returns:
        xdeconv: deconvolution of image
        h:inverse of kernel
    '''
    n1,n2 = x.shape[:2]
    den = np.square(np.absolute(lbd)) + (n1*n2*np.square(sig)/mpsd) 
    h = np.conjugate(lbd)/(den)
    x_fft = nf.fft2(x,axes=(0,1))
    xdec = np.absolute(nf.ifft2(x_fft*h.reshape(n1,n2,1),axes=(0,1)))
    if return_transfer:
        return xdec, h
    else:
        return xdec   
