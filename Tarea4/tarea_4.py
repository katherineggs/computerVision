# Katherine García 20190418
# https://github.com/katherineggs/computerVision/tree/main/Tarea4#tarea-4---fast-sharp-effect

# ---------------------------------------
import sys ; sys.path.append("../")
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import time
import os
# ---- cython
from Cython.Distutils import build_ext
import pyximport
pyximport.install()
from imgFilterC import imgFilterC
# ---------------------------------------

# kernel
laplacian = np.array([[ -1, -1, -1],
                      [ -1,  9, -1],
                      [ -1, -1, -1]])

# ------------ 
def normalize(arr):
    valueMin = arr.min()
    valueMax = arr.max()
    delta = valueMax-valueMin
    
    norm_arr = []
    for i in arr:
        temp = 255*(i-valueMin)/delta
        norm_arr.append(temp) 
    return np.array(norm_arr, dtype=np.ubyte).reshape(arr.shape)

def process(vid, kernel):
    """
        Speed up de = 235.49
        Tiempo original = 1551.88
        Tiempo cython = 6.59
    """
    times = []
    while True:
        now = time.time()
        ret, frame = vid.read()

        if ret: # if frame is read correctly ret is True
            im = frame[:,:,:]
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
            # filter and normalize all channels
            for channel in range(channels):
                imChannel = im[:,:,channel]
                fx = imgFilterC(imChannel.astype(np.intc), kernel.astype(np.intc)) # pasa por la func 

                cFiltProcessed = np.array(fx, dtype=np.intc)
                
                # hacemos normalizacion y clip para que no sea tan obscura
                f = normalize(np.clip(cFiltProcessed,0,255))
                filtered.append(f)
                
            # merge a los canales de la img
            merged = cv.merge((filtered))
            # ---
            
            tt = time.time() - now # time taken per frame
            print("Time per frame", tt)
            times.append(tt)

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
    
    print("AVG Time", sum(times)/len(times))

    #clean up before exit
    vid.release()
    cv.destroyAllWindows()

def run(path):
    cap = cv.VideoCapture(path, cv.CAP_AVFOUNDATION)
    process(cap, laplacian)

if __name__ == '__main__':
    # import argparse # no funciona el default? 
    # parser = argparse.ArgumentParser()
    # parser.add_argument("input", default=pathVid, type=str)
    # vid = parser.parse_args()
    # print(vid)
    # final = run(vid.input)
    pathVid = '/Users/katherinegarcia/Desktop/computerVision/imgs/count.mp4'
    
    # TENER EL VIDEO EN LA MISMA CARPETA
    # pathVid = 'count.mp4'
    # pathVid = 'boat.mp4'
    final = run(pathVid)