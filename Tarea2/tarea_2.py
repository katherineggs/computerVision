# Katherine García 20190418
import sys
from turtle import pen ; sys.path.append("../") 
import numpy as np
import cv2 as cv

# camara de la compu = 0, otras = 1,2,3,...
camID = 0 
cap = cv.VideoCapture(camID, cv.CAP_AVFOUNDATION)

# verify that video handle is open
if (cap.isOpened() == False):
    print("Video capture failed to open")

def HDR(img):
    hdr = cv.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return  hdr

# def pencilSketch(img):
#     bw, color = cv.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
#     return  bw, color


while True:
    ret, frame = cap.read()
    im = frame[:,:,:]
    imgs = [] # HDR - capture multiple imgs 

    if ret: # if frame is read correctly ret is True
        # create windows
        win0 = 'Original'
        win1 = 'HDR'
        # win2 = 'PencilColor'
        # win3 = 'PencilBW'

        cv.namedWindow(win0, cv.WINDOW_NORMAL)
        cv.namedWindow(win1, cv.WINDOW_NORMAL)
        # cv.namedWindow(win2, cv.WINDOW_NORMAL)
        # cv.namedWindow(win3, cv.WINDOW_NORMAL)

        # resize windows
        rows, cols = im.shape[0:2]
        resizeFact = 2
        miniRows = int(rows//resizeFact)
        miniCols = int(cols//resizeFact)
        newSize = (miniCols, miniRows) 

        cv.resizeWindow(win0, newSize)
        cv.resizeWindow(win1, newSize)
        # cv.resizeWindow(win2, newSize)
        # cv.resizeWindow(win3, newSize)

        # apply operation
        # making the hdr img
        hdr = HDR(im)
        # pencilBW = pencilSketch(im)[0]
        # pencilC = pencilSketch(im)[1]

        # show windows
        cv.imshow(win0, im)
        cv.imshow(win1, hdr)
        # cv.imshow(win2, pencilC)
        # cv.imshow(win3, pencilBW)
	
        # align windows        
        cv.moveWindow(win1, 0, 0)
        # cv.moveWindow(win2, miniCols, 0)
        # cv.moveWindow(win3, 0, miniRows)
        cv.moveWindow(win0, miniCols, miniRows)
        
        # exit with q
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

#clean up before exit
cap.release()
cv.destroyAllWindows()
