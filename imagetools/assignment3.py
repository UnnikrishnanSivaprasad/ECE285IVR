""" Assignment 3

COMPLETE THIS FILE

Your name here:

"""

#from .assignment2 import *
import numpy as np
import imagetools.assignment2 as im


def kernel(name,tau=1,eps=1e-3): 
    
    if name is "gaussian":
        s1 = np.floor(np.sqrt(-2*np.square(tau)*np.log(eps)))
        s2 = np.floor(np.sqrt(-2*np.square(tau)*np.log(eps)))
        x_coord,y_coord = np.meshgrid(np.arange(-s1,s1+1),np.arange(-s2,s2+1),indexing="ij")
        kernel = np.exp((-np.square(x_coord) - np.square(y_coord))/(2*np.square(tau)))
    elif name is "gaussian1":
        s1 = np.floor(np.sqrt(-2*np.square(tau)*np.log(eps)))
        x_coord= np.meshgrid(np.arange(-s1,s1+1))
        kernel = np.exp((-np.square(x_coord))/(2*np.square(tau)))
        kernel=np.transpose(kernel)
    elif name is "gaussian2":
        s1 = np.floor(np.sqrt(-2*np.square(tau)*np.log(eps)))
        x_coord= np.meshgrid(np.arange(-s1,s1+1))
        kernel = np.exp((-np.square(x_coord))/(2*np.square(tau)))
    elif name is "exponential":
        s1 = np.floor(-tau*np.log(eps))
        s2 = np.floor(-tau*np.log(eps))
        x_coord,y_coord = np.meshgrid(np.arange(-s1,s1+1),np.arange(-s2,s2+1),indexing="ij")
        kernel = np.exp(-np.sqrt(np.square(x_coord)+np.square(y_coord))/tau)
    elif name is "exponential1":
        s1 = np.floor(-tau*np.log(eps))
        x_coord= np.meshgrid(np.arange(-s1,s1+1))
        kernel = np.exp(-np.sqrt(np.square(x_coord))/tau)        
        kernel = np.transpose(kernel)
    elif name is "exponential2":
        s1 = np.floor(-tau*np.log(eps))
        x_coord= np.meshgrid(np.arange(-s1,s1+1))
        kernel = np.exp(-np.sqrt(np.square(x_coord))/tau)
    elif name is "box":
        s1 = tau
        s2 = tau
        kernel = [np.ones(((2*s1) + 1,(2*s2) + 1))]
    elif name is "box1":
        s1 = tau
        kernel = [np.ones(((2*s1) + 1)),]
        kernel = np.transpose(kernel) 
        kernel = kernel/np.sum(kernel)
        return kernel
    elif name is "box2":
        s1 = tau
        kernel = np.array([np.ones(((2*s1) + 1)),])
        kernel = kernel/np.sum(kernel)
        return kernel
    elif name is 'grad1_forward':
        kernel = np.zeros((3, 1))
        kernel[1, 0] = -1
        kernel[2, 0] = 1
        kernel = kernel/np.sum(kernel)
        return kernel
    elif name is 'grad2_forward':
        kernel = np.zeros((3, 1))
        kernel[1, 0] = -1
        kernel[2, 0] = 1
        return np.transpose(kernel)
    elif name is 'grad1_backward':
        kernel = np.zeros((3, 1))
        kernel[0, 0] = -1
        kernel[1, 0] = 1
        return kernel
    elif name is 'grad2_backward':
        kernel = np.zeros((3, 1))
        kernel[0, 0] = -1
        kernel[1, 0] = 1
        return np.transpose(kernel)
    elif name is 'laplacian1':
        kernel = np.ones((3, 1))
        kernel[1, 0] = -2
        return kernel
    elif name is 'laplacian2':
        kernel = np.ones((3, 1))
        kernel[1, 0] = -2
        return np.transpose(kernel)
    
    kernel = kernel/np.sum(kernel)
    
    return kernel

def convolve(x,nu,boundary="periodical",separable=None):
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

def laplacian(x, boundary='periodical'):
    nu1=kernel("laplacian2")
    nu2=kernel("laplacian1")
    xout=convolve(x,(nu1,nu2),separable='sum')
    return xout

def grad(x, boundary="periodical"):
    g1=convolve(x,kernel("forward"),boundary=boundary)
    g2=convolve(x,kernel("backward"),boundary=boundary)
    g=np.stack(g1,g2)
    
    return g


