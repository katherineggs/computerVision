# Katherine Garcia 20190418
import plot
import numpy as np
import os
import cv2 as cv
import random as rng
rng.seed(42)

def imgPad(img, r):
    """ Add padding to a BW image
    Args:
        img: BW image to pad
        r: lenght of pad
    Returns: padded: img padded
    """
    padded = np.pad(img,pad_width=r)
    return padded

def connected_c(img):
    """Tags components with two pass variant of the Connected Componet labeling algorithm
    Args: img (cv img): binarized img 
    Returns: numpy array: array with tags for each line
    """
    # binarizar la img - el blanco es el objeto de interes
    rows,cols = img.shape
    label = 1
    info = {}
    # new = np.zeros((rows,cols), dtype=np.uint8)

    # First Pass
    for i in range(1, rows-1):
        for j in range(1, cols-1):# i filas j columnas
            if img[i][j] != 0:
                n1 = img[i-1][j]
                n2 = img[i][j-1]
                if n1 == 0 and n2 == 0: # no hay vecinos
                    info[label] = [label]
                    img[i][j] = label
                    label += 1
                elif n1 != 0 or n2 != 0: 
                    nlabels = [img[i-1][j],img[i][j-1]] # neighbor labels
                    mini = min(i for i in nlabels if i != 0)
                    maxi = max(i for i in nlabels if i != 0)
                    img[i][j] = mini
                    if maxi not in info[mini]:
                        info[mini].append(maxi)

    # only relations, no singles
    list1 = [k for k,v in info.items() if len(v)<=1]
    for i in list1:
        info.pop(i)

    # Second Pass
    for i in range(0, rows):
        for j in range(0, cols):# i filas j columnas
            if img[i][j] != 0:
                for k,v in info.items():
                    if img[i][j] in v and img[i][j] != k : 
                        img[i][j] = k

    return img

def labelview(labels):
    """Assign color to each label and visualize
    Args:labels (numpy array uint): array returned by ccl algorithm
    """
    rows,cols = labels.shape
    new = np.zeros((rows,cols,3), dtype=np.uint8)
    colors = []
    labs = np.unique(labels)
    labs = labs[labs != 0] # no 0 
    for l in labs:
        colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))
    
    lenLabs = list(range(0, len(labs)))
    for i in range(0, rows):
        for j in range(0, cols):# i filas j columnas
            if labels[i][j] != 0:
                for c in range(len(labs)):
                    if labs[c] == labels[i][j]:
                        new[i][j] = colors[lenLabs[c]]

    plot.imgView(new)
    return new

def run(img):
    """Run Final secuence of tha laboratory
    Args:img (str): img name with its extention. Defaults to 'fprint3.pgm'.
    return : final img
    """
    imBW = cv.imread(img, cv.IMREAD_GRAYSCALE)

    imgBin = cv.threshold(imBW,128,255,cv.THRESH_BINARY_INV|cv.THRESH_OTSU)[1] # Binarizamos img
    imgPadd = imgPad(imgBin,1) # ponemos Pad
    ccl = connected_c(imgPadd)
    fin = labelview(ccl)
    return fin

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", default="fprint3.pgm", type=str)
    parser.add_argument("output", type=str)
    img = parser.parse_args()
    final = run(img.input)
    cv.imwrite(img.output, final)