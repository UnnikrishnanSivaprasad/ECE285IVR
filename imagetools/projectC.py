""" Project C

COMPLETE THIS FILE

Your names here:

"""
import numpy as np 
import numpy.fft as nf
from .provided import *
from .assignment5 import kernel2fft,convolvefft,adjoint
from .assignment2 import *
class Identity(LinearOperator):
    '''
    Implements the identity operator encapsulated as a class
    '''
    def __call__(self, x):
        '''
        Calls Idenetity class as operator
        Args:
            x:input image
        Returns:
            copy of x
        '''
        assert x.shape == self.ishape
        return x.copy()
        
    def adjoint(self,x):
        '''
        Returns operation of adjoint of identity operation on input x
        Args:
            x: input to the operator
        Returns:
            copy of x
        '''
        assert x.shape == self.ishape
        return x.copy()
    
    def gram(self,x):
        '''
        Returns operation of gram operation on input x
        Args:
            x: input to the operator
        Returns:
            copy of x
        '''
        assert x.shape == self.ishape
        return x.copy()

    def gram_resolvent(self,x,tau):
        '''
         
        Returns operation of  of idegramntity operation on input x
        Args:
            x: input to the operator
            tau: scaling offered to matrix
        Returns:
            copy of x      
        '''
        res = x.copy()/(1 + tau)
        return res 

class Convolution(LinearOperator):
    '''
    Implements the convolution operator encapsulated as a class
    '''
    def __init__(self,ishape, nu, separable=None):
        '''
        Initializes the convolution as an operator encapsulated as a class
        Args:
            self: pointer to current instance of the class
            shape: shape of the input 
            nu: kernel to convolve input with
        Kwargs:
            separable: indicates nature of separabiity
        '''
        #self.ishape = shape
        if separable:
            assert isinstance(separable,str)
            assert separable in ("product","sum")
        self.n1,self.n2 = ishape[:2]
        self.lbd = kernel2fft(nu,self.n1,self.n2,separable=separable)
        self.adj_lbd = np.conjugate(self.lbd)

    def __call__(self,x):
        '''
        Computes convolution of x with a kernel of frequency response lbd
        Args:
            x : input image
        Returns:
            y : output of convolution
        '''
        #assert x.shape == self.ishape
        return convolvefft(x,self.lbd)

    def adjoint(self,x):
        '''
        Computes the adjoint of the convolution with kernel nu
        Args:
            x: input to convolution operator
        Returns:
            y : output of convolution
        '''
        return convolvefft(x,self.adj_lbd)
    
    def gram(self,x):
        '''
        Returns gram of convolution operation
        Args:
            x: input to operation
        Returns:
            gram operation of kernel
        '''
        #assert x.shape == self.ishape
        return convolvefft(x,self.lbd*self.adj_lbd)

    def gram_resolvent(self,x,tau):
        '''
        Args:
            x: input to the operator
            tau: scaling offered to matrix
        Returns:
            copy of x      
        '''
        #assert x.shape == self.ishape
        return cg(lambda z: z + tau * self.gram(z), x)

class RandomMasking(LinearOperator):
    def __init__(self,ishape,p):
        '''
        Initializes matrix for Random Masking
        '''
        self.kernel = np.random.choice([0,1],size=ishape,p=[p,1-p])
    
    def __call__(self,x):
        '''
        '''
        #assert x.shape == self.ishape
        return x*self.kernel

    def adjoint(self,x):
        '''
        '''
        #assert x.shape == self.ishape
        return x*self.kernel 

    def gram(self,x):
        '''
        '''
        #assert x.shape == self.ishape
        return x*self.kernel 

    def gram_resolvent(self,x,tau):
        '''
        '''
        #assert x.shape == self.ishape
        return cg(lambda z: z + tau * self.gram(z), x)

def dwt(x,J,h,g):
    '''
    Computes the discrete wavelet transform
    Args:
        x: input image
        J: number of scales at which the wavelet transform is computed
        h: high pass filter used 
        g: lowpass filter used
    Returns:
        z : dwt coefficients
    '''
    if J == 0:
        return x
    n1, n2 = x.shape[:2]
    m1, m2 = (int(n1 / 2), int(n2 / 2))
    z = dwt1d(x, h, g)
    z = np.rot90(dwt1d(np.rot90(z,k=3), h, g),k=1)
    z[:m1, :m2] = dwt(z[:m1, :m2], J - 1, h, g)
    return z

def dwt1d(x, h, g): # 1d and 1scale
    '''
    compute 1D wavelet transform
    Args:
        x: input image
        h:high pass filter
        g:low pass filter
    Returns:
        z:dwt coefficients
    '''
    coarse = convolve(x, g)
    detail = convolve(x, h)
    z = np.concatenate((coarse[::2, :], detail[::2, :]), axis=0)
    return z
        
def idwt(z, J, h, g): # 2d and multi-scale
    if J == 0:
        return z
    n1, n2 = z.shape[:2]
    m1, m2 = (int(n1 / 2), int(n2 / 2))
    x = z.copy()
    x[:m1, :m2] = idwt(x[:m1, :m2], J - 1, h, g)
    x = np.rot90(idwt1d(np.rot90(x,k=3), h, g),k=1)
    x = idwt1d(x, h, g)
    return x

def idwt1d(z, h, g): # 1d and 1scale
    n1 = z.shape[0]
    m1 = int(n1 / 2)
    coarse, detail = np.zeros(z.shape), np.zeros(z.shape)
    coarse[::2, :], detail[::2, :] = z[:m1, :], z[m1:, :]
    x = convolve(coarse, g[::-1]) + convolve(detail, h[::-1])
    return x    

class DWT(LinearOperator):
    def __init__(self,ishape,J,name="db2"):
        '''
        Initializes op for DWT transform encalsulated as a class
        Args:
            shape : shape of input provided
            J: number of scales at which the op is carried out
        Kwargs:
            name:name of wavelet used
        '''
        self.shape = ishape
        self.h,self.g = wavelet(name) 
        self.J = J

    def __call__(self,x):
        return dwt(x,self.J,self.h,self.g)

    def adjoint(self,x):
        return idwt(x,self.J,self.h,self.g)

    def gram(self,x):
        return idwt(dwt(x,self.J,self.h,self.g),self.J,self.h,self.g)

    def gram_resolvent(self,x,tau):
        return cg(lambda z: z + tau * self.gram(z), x)

    def invert(self,x):
        return idwt(x,self.J,self.h,self.g)

    def power(self):
        return dwt_power(self.shape[0],self.shape[1],len(self.shape))
    
def softthresh(z,t):
    '''
    Implements soft thresholding of matrix z with threshold t
    Args:
        z: array
        t: threshold
    Returns:
        threshold: thresholded version of z
    '''
    
    threshold = np.maximum(z-t,np.zeros_like(z))+np.minimum(z+t,np.zeros_like(z))
    
    return threshold

def softthresh_denoise(y, sig, W, alpha=10/255):
    '''
    Removes noise by performing soft-thresholding on the wavelet coefficients
    Args:
        y: image
        sig: standard deviation
        W: wavelet transformation
    Kwargs:
        alpha: scaling factor
    Returns:
        Denoised image
    '''
    
    coeff = W(y)
    tau = np.sqrt(2)*np.square(sig)/(alpha*W.power())
    return softthresh(coeff,tau)


def interleave0(x):
    '''
    Upsample the filters h and g in udwt by injecting 2^(j − 1) zeros between each entries.
    Args:
        x: input vector
        j: interleaving factor
    Returns:
        x1: interleaved vector
    '''
    x1=np.zeros(((x.shape[0]-1)*2+1,1))
    x1[::2,:]=x
    return x1


def udwt(x, J, h, g):
    '''
    Implements the 2d Undecimated Discrete Wavelet Transform (UDWT) with J scales
    Args:
        x: image
        J: scales
        h: filter h
        g: filter g
    Returns:
        z: udwt transformation
    '''
    if J == 0:
        return x[:, :, None]
    tmph = np.rot90(convolve(np.rot90(x,k=3), h),k=1) / 2
    tmpg = np.rot90(convolve(np.rot90(x,k=3), g),k=1) / 2     
    z = np.stack((convolve(tmpg, h),convolve(tmph, g),convolve(tmph, h)), axis=2)
    coarse = convolve(tmpg, g)
    h2 = interleave0(h)
    g2 = interleave0(g)
    z = np.concatenate((udwt(coarse, J - 1, h2, g2), z),axis=2)
    return z

def iudwt(z, J, h, g):
    '''
    Implements the 2d Inverse UDWT
    Args:
        z: input img
        J: scales
        h: filter h
        g: filter g
    Returns:
        x: the 2D inverse UDWT
    '''
    if J == 0:
        return z[:, :, 0]
    h2 = interleave0(h)
    g2 = interleave0(g)
    coarse = iudwt(z[:, :, :-3], J - 1, h2, g2)
    tmpg = convolve(coarse, g[::-1]) + convolve(z[:, :, -3], h[::-1])
    tmph = convolve(z[:, :, -2], g[::-1]) + convolve(z[:, :, -1], h[::-1])
    x = (np.rot90(convolve(np.rot90(tmpg,k=3), g[::-1]),k=1) + np.rot90(convolve(np.rot90(tmph,k=3), h[::-1]),k=1)) / 2
    return x

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

def fftpad(signal,length):
    '''
    '''
    kernel = np.zeros((length,1))
    kernel[:signal.shape[0],:]=signal
    return kernel
    
def udwt_create_fb(n1, n2, J, h, g, ndim=3):
    '''
    Implements UDWT with filter bank
    Args:
        n1: kernel dim n
        n2: kernel dim m
        J: scales
        h: filter h
        g: filter g
    Kwargs:
        ndim: number of dimensions
    Returns:
        fb: filter bank
    '''
    
    if J == 0:
        return np.ones((n1, n2, 1, *[1] * (ndim - 2)))
    h2 = interleave0(h)
    g2 = interleave0(g)
    fbrec = udwt_create_fb(n1, n2, J - 1, h2, g2, ndim=ndim)
    gf1 = nf.fft(fftpad(g, n1), axis=0)
    hf1 = nf.fft(fftpad(h, n1), axis=0)
    gf2 = nf.fft(fftpad(g, n2), axis=0)
    hf2 = nf.fft(fftpad(h, n2), axis=0)
    fb = np.zeros((n1, n2, 4), dtype=np.complex128)
    fb[:, :, 0] = np.outer(gf1, gf2) / 2
    fb[:, :, 1] = np.outer(gf1, hf2) / 2
    fb[:, :, 2] = np.outer(hf1, gf2) / 2
    fb[:, :, 3] = np.outer(hf1, hf2) / 2
    fb = fb.reshape(n1, n2, 4, *[1] * (ndim - 2))
    fb = np.concatenate((fb[:, :, 0:1] * fbrec, fb[:, :, -3:]),axis=2)
    return fb

def fb_apply(x, fb):
    '''
    Application of filter bank
    Args:
        x: image 
        fb: filter bank
    Returns:
        z: applied filter bank
    '''
    
    x = nf.fft2(x, axes=(0, 1))
    z = fb * x[:, :, np.newaxis]
    z = np.real(nf.ifft2(z, axes=(0, 1)))
    return z

def fb_adjoint(z, fb):
    '''
    Application of adjoint filter bank
    Args:
        z: input img
        fb: filter bank
    Returns:
        x: adjoint filter bank
    '''
    
    z = nf.fft2(z, axes=(0, 1))
    x = (np.conj(fb) * z).sum(axis=2)
    x = np.real(nf.ifft2(x, axes=(0, 1)))
    return x

class UDWT(LinearOperator):
    def __init__(self,ishape,J,name="db2",using_fb=True):
        '''
        Initializes op for DWT transform encalsulated as a class
        Args:
            shape : shape of input provided
            J: number of scales at which the op is carried out
        Kwargs:
            name:name of wavelet used
        '''
        self.shape = ishape
        self.h,self.g = wavelet(name) 
        self.ah,self.ag = adjoint(self.h),adjoint(self.g)
        self.mode = using_fb
        self.J = J
        if self.mode:
            self.fb = udwt_create_fb(self.shape[0], self.shape[1], self.J, self.h, self.g, ndim=3)
        else:
            self.fb = None
            

    def __call__(self,x):
        if self.mode:
            return fb_apply(x,self.fb)
        else:
            return udwt(x,self.J,self.h,self.g)

    def adjoint(self,x):
        if self.mode:
            return fb_adjoint(x,self.fb)
        else:
            return iudwt(x,self.J,self.ah,self.ag)

    def gram(self,x):
        return self.adjoint(self.__call__(x))

    def gram_resolvent(self,x,tau):
        return cg(lambda z: z + tau * self.gram(z), x)

    def invert(self,x):
        return self.adjoint(x)

    def power(self):
        return udwt_power(self.J)
    