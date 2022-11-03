# Katherine García 20190418
# https://github.com/katherineggs/computerVision#laboratorios

import sys ; sys.path.append("/Users/katherinegarcia/Desktop/computerVision/Tarea1/") # osx
from skimage.util import view_as_windows
from math import ceil
import numpy as np
import cv2 as cv
import os

kernelX = np.array([-1,0,1]).reshape(1,-1)
kernelY = np.array([-1,0,1]).reshape(-1,1)

def loadImg(path):
    """Load img from imgs folder

    Args:
        path (str): path of file

    Returns:
        im: image in grayscale
    """
    # color
    im = cv.imread(os.path.join(path), cv.IMREAD_COLOR)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)

    #BW
    # im = cv.imread(os.path.join(path,'p1.png'), cv.IMREAD_GRAYSCALE)
    return im

def pad(img, r, value, type=np.uint8):
    """Pad an image with value v with a border of r pixels.
    Args:
        img (numpy array): Image to pad.
        r (int): Number of pixels to pad around border.
        value (int): Value to use for padding.
        type (np.types): Type of new padded image.
 
    Returns:
        padded (numpy array): Padded image.
    """
    rows, cols = img.shape
    R = rows + 2*r
    C = cols + 2*r
     
    padded = np.full((R, C), value, dtype=type)
    padded[r:-r,r:-r] = img
    return padded

def filterImg(img, kernel, dtype = np.float64):
    """Convolve img

    Args:
        img (_type_): _description_
        kernel (_type_): _description_
        dtype (_type_): _description_

    Returns:
        _type_: _description_
    """
    # check that img is BW 
    assert len(img.shape) == 2

    # allow non square kernels with odd shape 
    # but validate that the mayor axis size is odd
    assert (max(kernel.shape)%2)!=0

    # create placeholder for result
    # crear una matriz como variable de salida
    convolution = np.zeros_like(img, dtype=dtype)
    rK, cK = kernel.shape

    R = max(rK, cK)//2

    # pnerle padding al img
    padded = pad(img, R ,-1)
    r, c = padded.shape

    for i in range(1, r-1):
        for j in range(1, c-1):
            imgN = padded[i-R:i+R+1, j-R:j+R+1]
            pValue = np.multiply(imgN, kernel).sum()
            convolution[i-1,j-1] = pValue
    
    return convolution

def grad2magn(gx, gy):
    """ Compute the magnitude corresponding to gx and gy gradient matrices.
    Args:
        gx (numpy.array): Gradient in the x direction.
        gy (numpy.array): Gradient in the y direction.
    Returns:
        magnitudes (numpy.array): Magnitudes computed.
    """
    
    magn = np.sqrt(((gx**2)+(gy**2)))
    return magn

def grad2angle(gx,gy):
    """ Compute the angle corresponding to gx and gy gradient matrices.
    Atan2 result is always between -pi and pi.
    
    Args:
        gx (numpy array): Gradient in the x direction.
        gy (numpy array): Gradient in the y direction.
        
    Returns:
        angles (numpy array): Angles computed by the np.arctan2 function.
    """
    
    dir = np.arctan2(gy,gx)
    return dir

def rad2deg(rad):
    """ Convert an angle measure from radians to degrees.
    Arg:
        - rad (float): Angle in radians
    Returs:
        deg (float): Angle in degrees
    """
    
    deg = (rad*180)/ np.pi
    return deg

def convolveRGB(img, kernelX, kernelY):
    """ Compute the gradient of an RGB image.
    Args:
        img (numpy array): Source Image.
        kx (numpy.array): Kernel for the x direction.
        ky (numpy.array): Kernel for the y direction.
        unsigned (bool): Select if gradient computation is signed or unsigned.
        
    Returns:
        gradients (list): List of numpy arrays containing the x and y convolution per channel.
    """   
    gradients = []

   # pasarle convolve
    for i in range(len(img.shape)):
        c = img[:,:,i]

        gradXc = filterImg(c, kernelX)
        gradYc = filterImg(c, kernelY)
        
        gradients.append([gradXc, gradYc])

    return gradients

def grad2vect(gradients, degrees=False, unsigned=False):
    """ Compute from a vector of horizontal 
    and vertical gradients, gx and gy, the corresponding rgb
    magnitudes and angle array.
    
    Args:
        gradients (list): Pairs of gx and gy gradients in a list. [[gx,gy],[gx,gy],[gx,gy]]
        degrees (bool): Output angles in degrees
        unsigned (bool): Discard angle sign information
        
    Returns:
        magnitude (np.array): 3 channel image of magnitudes.
        angle (np.array): 3 channel image of angles.
    """   
    magnitude = []
    angles = []
    for gx, gy in gradients:
        magnitude.append(grad2magn(gx,gy))
        angles.append(grad2angle(gx,gy))

    # returns (128, 64, 3) not (3, 64, 128)
    magnitude = cv.merge(magnitude)
    angles = cv.merge(angles)

    if degrees:
        angles = rad2deg(angles)
    
    if unsigned:
        angles = np.absolute(angles)
    
    return np.array(magnitude), np.array(angles)

def hogVector(magnitude, angles):
    """ Build magnitude and orientation matrices required by Hog.
        Max magnitude per channel is selected pixelwise and the corresponding orientation.

    Args:
        magnitude (list of np.array): List of 2D arrays with channel magnitudes.
        angles (list of np.array): List of 2D arrays with channel orientations. Must be expressed in degrees.
    Returns:
        mag, ang: (np.array): Maximum magnitude per channel (mag) and corresponding orientation (ang)
    """
    img_shape = magnitude.shape
    mag = np.zeros((img_shape[0], img_shape[1]))
    ang = np.zeros((img_shape[0], img_shape[1]))

    for row in range(magnitude.shape[0]):
        for col in range(magnitude.shape[1]):
            maxChannel = np.argmax(magnitude[row, col])
            maxMagn = magnitude[row, col, maxChannel]
            maxAng = angles[row, col, maxChannel]
            mag[row, col]=(maxMagn)
            ang[row, col]=(maxAng)
    return mag, ang

def hogHist(mag, ang):
    """Build Hog histogram
    
    Args:
        mag (np.array): 2D array of magnitudes. Size 8X8
        ang (np.array): 2D array of orientations. Must be expressed in degrees. Size 8X8
        
    Returns:
        hogHist (np.array): Histogram of 9 bins for the. [12,1234,31231,12312,213,23342,432,54,53]
    """
    # [0 | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 180] #  9 boxes of 20-degree increments
    hogHist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
    step = 20

    # print(np.max(ang))
    # print(np.min(ang))
    assert not np.max(ang) > 180 and not np.min(ang) < 0
    
    for row in range(mag.shape[0]): # mag -> 8x8
        for col in range(mag.shape[1]):
            angBin = ceil(ang[row, col] / step) # no. of bin
            angQty = angBin - ang[row, col] / step # amount
            actualMag = mag[row, col]
            
            hogHist[angBin] += actualMag * (1 - angQty)
            
            # if angBin == 0: # para tirar los datos a; 160 y no al 180 
            #     hogHist[angBin - 2] += actualMag * (angQty) # + por el floor 
            # else:
            hogHist[angBin - 1] += actualMag * (angQty) # + por el floor 
    
    # 180 merge with 0
    hogHist[0] += hogHist[-1]
    hogHist.pop(-1) # remove 180
    return hogHist

def normL2(hist):
    """normalize the histogram with l2 normalization

    Args:
        hist (array): histogram

    Returns:
        array: array normalized
    """
    norm = hist / np.sqrt( np.sum( hist ** 2 ) )
    return norm

def hog(imgPath):
    """Build HOG descriptor
    Convolution, gradients and image manipulation to calculate the HOG descriptor. 

    Args:
        imgPath (openCV img): colored img

    Returns:
        numpy array: final HOG descriptor
    """
    img = loadImg(imgPath)
    grads = convolveRGB(img, kernelX, kernelY)
    magnitude, angles = grad2vect(grads, degrees=True, unsigned=True)
    magnitude, angles = hogVector(magnitude, angles)

    histsImg = []
    windowsMag = view_as_windows(magnitude, (8,8), 8)
    windowsAng = view_as_windows(angles, (8,8), 8)

    for r in range(windowsMag.shape[0]): # Filas 
        for c in range(windowsMag.shape[1]): # Cols
            histsImg.append(hogHist(windowsMag[r][c],windowsAng[r][c]))

    histsImg = np.array(histsImg).reshape(magnitude.shape[0]//8, magnitude.shape[1]//8, 9)

    descriptorHOG = []
    winds = view_as_windows(histsImg, (2, 2, 9))
    for r in range(winds.shape[0]): # Filas 
        for c in range(winds.shape[1]): # Cols
            vect = np.reshape(winds[r][c], -1)
            vect = normL2(vect)
            descriptorHOG.append(vect)

    descriptorHOG = np.reshape(np.array(descriptorHOG), -1)

    return descriptorHOG