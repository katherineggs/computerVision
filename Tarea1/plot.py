# Katherine Garcia 20190418

import matplotlib.pyplot as plt
import numpy as np

def imgView(img, title=None, filename=None, axis=False):
    """
    Diferenciar entre una imgagen a color y bw
    Si hay filename -> grabar la img
    Si hay axis ->

    Args:
        img (_type_): imagen a identificar
        title (_type_, optional): Defaults to None.
        filename (_type_, optional): Defaults to None.
        axis (bool, optional): Defaults to False.
    """
    dim = len(img.shape)
    fig = plt.figure(figsize=(10.0,10.0))
    if axis != None:
        ax = fig.add_subplot(111)
    else:
        ax = axis
    
    if dim == 3: # imagen a color
        ax.imshow(img)
    elif dim == 2: # imagen bw
        ax.imshow(img, vmin=0, vmax=255, cmap='gray')

    ax.axes.xaxis.set_visible(False) # quitar axis
    ax.axes.yaxis.set_visible(False) # quitar axis

    if title:
        ax.set_title(title)
    if filename:
        plt.savefig(filename,bbox_inches='tight',pad_inches=0)
    else:
        # # h = plt.savefig(filename,bbox_inches='tight',pad_inches=0)
        # return fig
        plt.show()

def imgFilter(img, kernel):
    """
    Filter img with a Kernel 
    Args:
        img (openCv img): Img loaded with OpenCV
        kernel (numpy array): squared matrix

    Returns:
        numpy matrix: new filtered matrix not img
    """
    rImg,cImg = img.shape
    rK,cK = kernel.shape
    assert(rK == cK)
    R = rK//2
    filtered = np.zeros((rImg,cImg), dtype=np.float32)
    for i in range(R, rImg-R):# i filas j columnas
        for j in range(R, cImg-R):
            w = img[i-R:i+R+1,j-R:j+R+1]
            mult = 0
            for iK in range(0, rK):# i filas j columnas
                for jK in range(0, cK):
                    mult += w[iK,jK] * kernel[iK,jK]
            filtered[i,j] = mult
    return filtered

def imgComp(img1, img2, title=None, filename=None):
    """Image comparison
    Args:
        img1: image 1
        img2: image 2
        title: titles for the comparison
        filename: name to save img
    """
    if len(title) == 2:
        t1 = title[0]
        t2 = title[1]
    elif len(title) == 1:
        t1,t2 = title[0],title[0]
    else: 
        t1,t2 = None,None

    dim1 = len(img1.shape)
    dim2 = len(img2.shape)
    
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(121)
    if dim1 == 3: # imagen a color
        ax1.imshow(img1)
    elif dim1 == 2: # imagen bw
        ax1.imshow(img1, vmin=0, vmax=255, cmap='gray')
    ax1.set_title(t1)
    plt.axis('off')

    ax2 = fig.add_subplot(122)
    if dim2 == 3: # imagen a color
        ax2.imshow(img2)
    elif dim2 == 2: # imagen bw
        ax2.imshow(img2, vmin=0, vmax=255, cmap='gray')
    ax2.set_title(t2)
    plt.axis('off')

    if filename:
        plt.savefig(filename,bbox_inches='tight',pad_inches=0, format='png')
    else:
        plt.show()

def imgSobel(img):
    """
    receives a black-and-white image and calculates
    the magnitude and direction of the Sobel gradient
    Args:
        img (cv img): bw image to calculate

    Returns:
        magnitude, direction: magnitude and direction of the Sobel grad
    """
    kernelX = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    kernelY = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    gx = imgFilter(img,kernelX)
    gy = imgFilter(img,kernelY)

    magnGrad = np.sqrt(((gx**2)+(gy**2)))
    dirGrad = np.arctan2(gy,gx)
    return magnGrad, dirGrad