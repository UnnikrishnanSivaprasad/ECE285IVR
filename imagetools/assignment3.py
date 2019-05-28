""" Assignment 3

COMPLETE THIS FILE

Your name here:

"""
import numpy as np
from .assignment2 import kernel,convolve,shift

def laplacian(x, boundary='periodical'):
    '''
    Computes the laplacian using kernel and covolve functions defined prior
    Args:
        x : input image
    Kwargs:
        boundary : type of boundry condition
    Returns:
        Lapplacian of the image
    '''
    nu1 = kernel("laplacian1")
    nu2 = kernel("laplacian2")
    l = convolve(x,(nu1,nu2),boundry=boundary,separable="sum")
    return l

def grad(x, boundary='periodical'):
    '''
    Calculates gradient along each axis and stacks the images
    Args:
        x : input to the gradient function
    Kwargs:
        boundary : nature of boundary condition during convolution
    '''
    grad1 = convolve(x,kernel("grad1_forward"),boundry=boundary)
    grad2 = convolve(x,kernel("grad2_forward"),boundry=boundary)
    soln = np.stack([grad1,grad2],axis=2)
    return soln

def div(x, boundary='periodical'):
    '''
    Calculates divergence from gradients 
    Args:
        f : gradients
    Kwargs
        boundary : nature of boundary condition
    '''
    grad1 = convolve(x[:,:,0,:],kernel("grad1_backward"),boundry=boundary)
    grad2 = convolve(x[:,:,1,:],kernel("grad2_backward"),boundry=boundary)
    return grad1 + grad2