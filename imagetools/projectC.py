""" Project C

COMPLETE THIS FILE

Your names here:

"""
import numpy as np 
import numpy.fft as nf
from .provided import *
from .assignment5 import kernel2fft,convolvefft,adjoint

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
    def __init__(self,shape, nu, separable=None):
        '''
        Initializes the convolution as an operator encapsulated as a class
        Args:
            self: pointer to current instance of the class
            shape: shape of the input 
            nu: kernel to convolve input with
        Kwargs:
            separable: indicates nature of separabiity
        '''
        self.ishape = shape
        if separable:
            assert isinstance(separable,str)
            assert separable in ("product","sum")
        self.n1,self.n2 = shape[:2]
        self.lbd = kernel2fft(nu,self.n1,self.n2,separable=separable)
        self.adj_lbd = kernel2fft(adjoint(nu),self.n1,self.n2,separable=separable)

    def __call__(self,x):
        '''
        Computes convolution of x with a kernel of frequency response lbd
        Args:
            x : input image
        Returns:
            y : output of convolution
        '''
        assert x.shape == self.ishape
        if len(x.shape)==3:
            temp = nf.fft2(x,axes=(0,1))*np.expand_dims(self.lbd,axis=-1)
        elif len(x.shape)==2:
            temp = nf.fft2(x,axes=(0,1))*self.lbd
        res = np.abs(nf.ifft2(temp,axes=(0,1)))
        return res

    def adjoint(self,x):
        '''
        Computes the adjoint of the convolution with kernel nu
        Args:
            x: input to convolution operator
        Returns:
            y : output of convolution
        '''
        assert x.shape == self.ishape
        if len(x.shape)==3:
            temp = nf.fft2(x,axes=(0,1))*np.expand_dims(self.adj_lbd,axis=-1)
        elif len(x.shape)==2:
            temp = nf.fft2(x,axes=(0,1))*self.adj_lbd
        res = np.abs(nf.ifft2(temp,axes=(0,1)))
        return res    

    def gram(self,x):
        '''
        Returns gram of convolution operation
        Args:
            x: input to operation
        Returns:
            gram operation of kernel
        '''
        assert x.shape == self.ishape
        return self.adjoint(self.__call__(x))

    def gram_resolvent(self,x,tau):
        '''
        Args:
            x: input to the operator
            tau: scaling offered to matrix
        Returns:
            copy of x      
        '''
        assert x.shape == self.ishape
        res_lbd = 1 / (1 - tau * self.lbd)
        return convolvefft(x, res_lbd)

class RandomMasking(LinearOperator):
    def __init__(self,shape,p):
        '''
        Initializes matrix for Random Masking
        '''
        self.ishape = shape
        self.kernel = np.random.choice([0,1],size=shape,p=[p,1-p])
    
    def __call__(self,x):
        '''
        '''
        assert x.shape == self.ishape
        return x*self.kernel

    def adjoint(self,x):
        '''
        '''
        assert x.shape == self.ishape
        return x*self.kernel 

    def gram(self,x):
        '''
        '''
        assert x.shape == self.ishape
        return x*self.kernel 

    def gram_resolvent(self,x,tau):
        '''
        '''
        assert x.shape == self.ishape
        return cg(lambda z: z + tau * self.gram(z), x)


        
    