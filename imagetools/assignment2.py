import numpy as np

def shift(x,k,l,boundry="periodical"):
    '''
    Implements shift of input array x by k pixels along rows and l pixels along columns
    Args:
        x:input image
        k:shift introduced along rows
        l:shift along columns
    Kwargs:
        boundry: nature of boundry to implement shift
    Returns:
        xshifted: shifted array
    '''
    assert isinstance(k,int),"shift parameters must be positive integers"
    assert isinstance(l,int),"shift parameters must be positive integers"
    assert isinstance(x,np.ndarray),"Image must be provided as a numpy array"
    assert isinstance(boundry,str),"nature of boundry"
    assert boundry in("periodical","extension","zero_padding","mirror"),"boundry nature not in available"
    
    if k==0 and l==0:
        return x
    xshifted = np.zeros_like(x)
    n1,n2 = x.shape[:2]
    xshifted[max(-k,0):min(n1-k,n1),max(-l,0):min(n2-l,n2)] = x[max(-k,0)+k:min(n1-k,n1)+k,max(-l,0)+l:min(n2-l,n2)+l]
    if boundry is "periodical":
        irange = np.mod(np.arange(n1)+k,n1)
        jrange = np.mod(np.arange(n2)+l,n2)
        xshifted = x[irange,:][:,jrange]
    elif boundry is "extension":
        if k > 0:
            xshifted[min(n1-k,n1):,max(-l,0):min(n2-l,n2)] =xshifted[min(n1-k,n1)-1,max(-l,0):min(n2-l,n2)]
        else:
            xshifted[:max(-k,0),max(-l,0):min(n2-l,n2)]= xshifted[max(-k,0),max(-l,0):min(n2-l,n2)]
        if l > 0:
            for col in range(min(n2-l,n2),n2):
                xshifted[:,col] = xshifted[:,min(n2-l,n2)-1]
        else:
            for col in range(max(-l,0)):
                xshifted[:,col] = xshifted[:,max(-l,0)]
    elif boundry is "mirror":
        if k > 0:
            num_samples = n1-min(n1-k,n1)
            rows = np.arange(min(n1-k,n1)-1,min(n1-k,n1)-num_samples-1,-1)
            xshifted[min(n1-k,n1):,max(-l,0):min(n2-l,n2)]=xshifted[rows,max(-l,0):min(n2-l,n2)]
        else:
            num_samples = max(-k,0)
            rows = np.arange(2*max(-k,0),max(-k,0),-1)
            xshifted[:max(-k,0),max(-l,0):min(n2-l,n2)] = xshifted[rows,max(-l,0):min(n2-l,n2)]
        if l > 0:
            index = 1
            for col in range(min(n2-l,n2),n2):
                xshifted[:,col] = xshifted[:,min(n2-l,n2)-index]
                index = index +1
        else:
            index = 1
            for col in range(max(-l,0),2*max(-l,0)):
                xshifted[:,max(-l,0)-index] = xshifted[:,col]
                index = index + 1
    return xshifted
            
                
def kernel(name,tau=1,eps=1e-3):
    '''
    Return kernel used for model
    Args:
        name : type of kernel to return. Acceptable values include gaussian,exponential and box
    Kwargs:
        tau : variance of model(emponential/gaussian)/ receptive field of filter
        default : 1
        
        eps : maximum range of values taken by kernel
        default : 1e-3
    Returns:
        mu : array encoding a convolution kernel of finite support for given type
    '''
    assert isinstance(name,str)
    
    if name.startswith("gaussian"):
        s1 = np.floor(np.sqrt(-2*np.square(tau)*np.log(eps)))
        s2 = np.floor(np.sqrt(-2*np.square(tau)*np.log(eps)))
        if name.endswith("n"):
            x_coord,y_coord = np.meshgrid(np.arange(-s1,s1+1),np.arange(-s2,s2+1),indexing="ij")
            kernel = np.exp((-np.square(x_coord) - np.square(y_coord))/(2*np.square(tau)))
        elif name.endswith("1"):
            x_coord = np.arange(-s1,s1+1)
            kernel = np.exp(-np.square(x_coord)/(2*np.square(tau)))
            kernel = kernel.reshape(-1,1)
        elif name.endswith("2"):
            y_coord = np.arange(-s2,s2+1)
            kernel = np.exp(-np.square(y_coord)/(2*np.square(tau)))
            kernel = kernel.reshape(1,-1)
            
    elif name.startswith("exponential"):
        s1 = np.floor(-tau*np.log(eps))
        s2 = np.floor(-tau*np.log(eps))
        if name.endswith("l"):
            x_coord,y_coord = np.meshgrid(np.arange(-s1,s1+1),np.arange(-s2,s2+1),indexing="ij")
            kernel = np.exp(-np.sqrt(np.square(x_coord)+np.square(y_coord))/tau)
        elif name.endswith("1"):
            x_coord = np.arange(-s1,s1+1)
            kernel = np.exp(-np.abs(x_coord)/tau)
            kernel = kernel.reshape(-1,1)
        elif name.endswith("2"):
            y_coord = np.arange(-s2,s2+1)
            kernel = np.exp(-np.abs(y_coord)/tau)
            kernel = kernel.reshape(1,-1)
    elif name is "box":
        s1 = tau
        s2 = tau
        if name.endswith("x"):
            kernel = np.ones(((2*s1) + 1,(2*s2) + 1))
        elif name.endswith("1"):
            kernel = np.ones(((2*s1) + 1,1))
        elif name.endswith("2"):
            kernel = np.ones((1,(2*s2) + 1))
    elif name.startswith("grad1"):
        kernel = np.zeros((3,1))
        if name.endswith("forward"):
            kernel[1,0] = -1
            kernel[2,0] = 1
        elif name.endswith("backward"):
            kernel[0,0] = -1
            kernel[1,0] = 1
    elif name.startswith("grad2"):
        kernel = np.zeros((1,3))
        if name.endswith("forward"):
            kernel[0,1] = -1
            kernel[0,2] = 1
        elif name.endswith("backward"):
            kernel[0,0] = -1
            kernel[0,1] = 1 
    elif name.startswith("laplacian"):
        if name.endswith("1"):
            kernel = np.ones((3,1))
            kernel[1,0] = -2
        elif name.endswith("2"):
            kernel = np.ones((1,3))
            kernel[0,1] = -2
    if name.startswith("grad") or name.startswith("laplacian"):
        return kernel
    else:
        kernel = kernel/np.sum(kernel)
        return kernel    

def convolve(x,nu,boundry="periodical",separable=None):
    '''
    Carry out filtering of the image x with filter nu under boundry conditions dictated by boundry kwarg
    Args:
        x : input image
        nu : kernel to filter with
    Kwargs:
        boundry : nature of boundry conditions 
        default : periodical
        
        separable : nature of separable operation
        default : None
    Returns:
        xconv : result of convolution
    '''
    n1, n2 = x.shape[:2]
    xconv = np.zeros(x.shape)
    if not separable:
        s1 =int((nu.shape[0] - 1) / 2)
        s2 =int((nu.shape[1] - 1) / 2)
        for k in range(s1,-s1-1,-1):
            for l in range(s2,-s2-1,-1):
                shifted_image = shift(x,-k,-l,boundry)
                xconv = xconv + shifted_image*nu[s1+k,s2+l]
                
    elif separable is "product":
        nu1,nu2 = nu
        s1 =int((nu1.shape[0] - 1) / 2)
        s2 =int((nu2.shape[1] - 1) / 2)
        for k in range(s1,-s1-1,-1):
            shifted_image = shift(x,-k,0,boundry)
            xconv = xconv + (shifted_image*nu1[s1+k])
        x = xconv.copy()
        xconv = np.zeros(x.shape)
        for l in range(s2,-s2-1,-1):
            shifted_image = shift(x,0,-l,boundry)
            xconv = xconv + (shifted_image*nu2[0,s2+l])
            
    elif separable is "sum":
        nu1,nu2 = nu
        s1 =int((nu1.shape[0] - 1) / 2)
        s2 =int((nu2.shape[1] - 1) / 2)
        xconv1 = np.zeros(x.shape)
        xconv2 = np.zeros(x.shape)
        for k in range(s1,-s1-1,-1):
            shifted_image = shift(x,-k,0,boundry)
            xconv1 = xconv1 + (shifted_image*nu1[s1+k,0])
        for l in range(s2,-s2-1,-1):
            shifted_image = shift(x,0,-l,boundry)
            xconv2 = xconv2 + (shifted_image*nu2[0,s2+l])           
        xconv = xconv1 + xconv2
    return xconv
