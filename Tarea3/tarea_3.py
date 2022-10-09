# Katherine Garcia 20190418
# https://github.com/katherineggs/computerVision
# 
import sys
from turtle import pen ; sys.path.append("../") 
import numpy as np
import cv2 as cv
import time
# camara de la compu = 0, otras = 1,2,3,...
camID = 0 
cap = cv.VideoCapture(camID, cv.CAP_AVFOUNDATION)

# verify that video handle is open
if (cap.isOpened() == False):
    print("Video capture failed to open")

# kernel
laplacian = np.array([[ -1,-1, -1],
                        [-1, 9.5,-1],
                        [ -1,-1, -1]])

def imgFilter(img, kernel): # in plot file
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

while True:
    now = time.time()
    ret, frame = cap.read()
    im = frame[:,:,:]

    if ret: # if frame is read correctly ret is True
        # create windows
        win0 = 'Original'
        win1 = 'SharpEffect'

        cv.namedWindow(win0, cv.WINDOW_NORMAL)
        cv.namedWindow(win1, cv.WINDOW_NORMAL)

        # change resolution
        percent = 60 # percent of original size
        width = int(im.shape[1] * percent / 100)
        height = int(im.shape[0] * percent / 100)
        dim = (width, height)
        
        # resize image
        im = cv.resize(im, dim, interpolation = cv.INTER_AREA)

        # resize windows
        rows, cols = im.shape[0:2]
        resizeFact = 2
        miniRows = int(rows//resizeFact)
        miniCols = int(cols//resizeFact)
        newSize = (miniCols, miniRows) 

        cv.resizeWindow(win0, newSize)
        cv.resizeWindow(win1, newSize)

        # vars
        channels = im.shape[2]
        filtered = []

        # apply operation
        for channel in range(channels):
            imChannel = im[:,:,channel]
            fx = imgFilter(imChannel, laplacian)

            #f = np.zeros_like(img, dtype=np.float32)
            f = cv.normalize(fx, 0, 255, cv.NORM_MINMAX)
            
            # para que no sea tan obscura
            # f = f.astype(np.uint8)
            f = (np.clip(f,0,1) * 255).astype(np.uint8)
            filtered.append(f)
            
        merged = cv.merge(filtered)
        # ---

        print("Time per frame", time.time() - now)
        # show windows
        cv.imshow(win0, im)
        cv.imshow(win1, merged)
	
        # align windows        
        cv.moveWindow(win1, 0, 0)
        cv.moveWindow(win0, miniCols, miniRows)
        
        # exit with q
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

#clean up before exit
cap.release()
cv.destroyAllWindows()
