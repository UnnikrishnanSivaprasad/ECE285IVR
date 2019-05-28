""" Project C

COMPLETE THIS FILE

Your names here:

"""

from .assignment6 import *
import numpy as np

def matrix_prod(x,y):
    return np.sum(np.multiply(x,y))

def adjoint(nu):
    mu = np.flip(np.flip(nu,axis=0),axis=1)
    return mu

class Identity:
    '''
    Initialises identity matrix of dimension shape
    '''
    def __call__(self, shape):
        '''
        n is an integer 
        '''
        m=np.ones((shape.shape[0],shape.shape[1],shape.shape[2]))
        return m*shape
        
    def adjoint(self, shape):
        self.m=np.ones((shape.shape[0],shape.shape[1],shape.shape[2]))
        return adjoint(self.m)*shape
            
class Convolution:
    def __call__(self,shape, nu, separable=None):
        '''
        shape is a matrix 
        nu is the kernel
        '''
        x=shape
        boundary="periodical"
        n1, n2 = x.shape[:2]
        xconv = np.zeros(x.shape)
        if not separable:
            s1 =int((nu.shape[0] - 1) / 2)
            s2 =int((nu.shape[1] - 1) / 2)
            for k in range(-s1,s1+1):
                for l in range(-s2,s2+1):
                    shifted_image = im.shift(x,-k,-l,boundary)
                    xconv = xconv + shifted_image*nu[s1+k,s2+l]

        elif separable is "product":
            nu1,nu2 = nu
            s1 =int((nu1.shape[0] - 1) / 2)
            s2 =int((nu2.shape[1] - 1) / 2)
            for k in range(-s1,s1+1):
                shifted_image = im.shift(x,-k,0,boundary)
                xconv = xconv + (shifted_image*nu1[s1+k,0])
            x = xconv.copy()
            xconv = np.zeros(x.shape)
            for l in range(-s2,s2+1):
                shifted_image = im.shift(x,0,-l,boundary)
                xconv = xconv + (shifted_image*nu2[0,s2+l])

        elif separable is "sum":
            nu1,nu2 = nu
            s1 =int((nu1.shape[0] - 1) / 2)
            s2 =int((nu2.shape[1] - 1) / 2)
            xconv1 = np.zeros(x.shape)
            xconv2 = np.zeros(x.shape)
            for k in range(-s1,s1+1):
                shifted_image = im.shift(x,-k,0,boundary)
                xconv1 = xconv1 + (shifted_image*nu1[s1+k,0])
            for l in range(-s2,s2+1):
                shifted_image = im.shift(x,0,-l,boundary)
                xconv2 = xconv2 + (shifted_image*nu2[0,s2+l])           
            xconv = xconv1 + xconv2
        return xconv
        
class RandomMasking:
    def __call__(self,shape,p1):
        x=np.random.choice([0, 1], size=(shape.shape[0],shape.shape[1]), p=[p1, 1-p1])
        m=np.dstack((np.dstack((x,x)),x))
        return m*shape
    def adjoint(self,shape,p1):
        x=np.random.choice([0, 1], size=(shape.shape[0],shape.shape[1]), p=[p1, 1-p1])
        self.m=np.dstack((np.dstack((x,x)),x))
        return adjoint(self.m)*shape
    