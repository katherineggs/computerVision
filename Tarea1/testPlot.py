import os
import cv2 as cv
import matplotlib.pyplot as plt

import plot

path = '/Users/katherinegarcia/Desktop/Vision/imgs/'

# img a color
im = cv.imread(os.path.join(path,'baboon.tiff'), cv.IMREAD_COLOR)
# RGB -> BGR
img = cv.cvtColor(im, cv.COLOR_BGR2RGB)
# img bw
imBW = cv.imread(os.path.join(path,'cameraman.tif'), cv.IMREAD_GRAYSCALE)

plot.imgView(img,"Babboon",filename="IMGView")
plt.show()

plot.imgComp(img,imBW,title=["Hello"],filename="IMGComp")

magn, dirr = plot.imgSobel(imBW)
plot.imgComp(magn,dirr,title=["Magn","dirr"])

# # para ver con OpenCV
# cv.imshow("bab",im)
# cv.wait(0)