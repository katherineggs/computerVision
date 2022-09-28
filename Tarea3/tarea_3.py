# Katherine Garcia 20190418
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

while True:
    ret, frame = cap.read()
    im = frame[:,:,:]

    if ret: # if frame is read correctly ret is True
        # create windows
        win0 = 'Original'
        win1 = 'SharpEffect'

        cv.namedWindow(win0, cv.WINDOW_NORMAL)
        cv.namedWindow(win1, cv.WINDOW_NORMAL)

        # resize windows
        rows, cols = im.shape[0:2]
        resizeFact = 2
        miniRows = int(rows//resizeFact)
        miniCols = int(cols//resizeFact)
        newSize = (miniCols, miniRows) 

        cv.resizeWindow(win0, newSize)
        cv.resizeWindow(win1, newSize)

        # apply operation
        # ---

        # show windows
        cv.imshow(win0, im)
        cv.imshow(win1)
	
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
