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
