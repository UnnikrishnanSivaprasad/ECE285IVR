""" Project C

COMPLETE THIS FILE

Your names here:

"""

from .assignment6 import *
import numpy as np

class Identity:
    '''
    Initialises identity matrix of dimension nxn
    '''
    def __init__(self, n):
        '''
        n is an integer 
        '''
        self.m = np.zeros(shape=(n,n))
        for i in range(n):
            self.m[i][i]=1

            
class Convolution:
    def __init__(self,shape, nu, separable=None):
        '''
        shape is a matrix 
        nu is the kernel
        '''
        x=shape
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
        self.conv=xconv
    