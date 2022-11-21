from skimage.transform import pyramid_gaussian
from skimage.draw import rectangle_perimeter
from skimage.util import view_as_windows
from skimage.io import imread, imshow
from skimage.feature import hog
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def loadModel(path="SVM"):
    """ Load persisted model from pkl
    """
    with open('modelPkl'+path , 'rb') as files:
        model = pickle.load(files)
    return model

def loadImgs(srcPath):
    """ Load images from specific folder
    Returns:
        - images (List): List of rgb images
    """
    images = []
    names = []
    for filename in os.listdir(srcPath):
        try:
            img = imread(os.path.join(srcPath, filename))
            if img is not None:
                images.append(img)
                head, sep, tail = filename.partition('.')
                names.append(head)
        except:
            pass
    # print(names)
    return images, names

def buildPyramid(img, scale=2):
    """ Get pyramid of image
    Returns:
        - imgs(generator): pyramid of images
    """
    pyramid = tuple(pyramid_gaussian(img, downscale=scale, channel_axis=-1))
    return pyramid

def getWindows(pymImg,shape=(128,64,3), step=(32,16,3)):
    """ Get image windows 

    Args:
        pymImg (_type_): image to get windows
        shape (tuple, optional): shape of the window. Defaults to (128,64,3).
        step (tuple, optional): step in between windows. Defaults to (32,16,3).

    Returns:
        array : image windows
    """
    windows = view_as_windows(pymImg, shape, step)
    return windows

def HOG(img):
    """ get HOg of image

    Args:
        img: image

    Returns:
        array: HOG array
    """
    hogImg = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
        block_norm='L2', feature_vector=True, channel_axis=-1) # feature vector= true, channel_Axis=2
    return hogImg

def predictPed(img, model="SVM"):
    """ Predict if the given image is a pedestrian
    Returns:
        - detection (Bool)
    """
    model = loadModel(model)
    ans = model.predict(img.reshape(1, -1))
    score = model.predict_proba(img.reshape(1, -1))
    
    return ans, score

def nmsScore(boxes, overlapThresh=0.5):
	"""Non maximum supression for all the detections

	Args:
		boxes (array): array of the coords of all detections and score
		overlapThresh (float, optional): Overlap treshold for boxes. Defaults to 0.75.

	Returns:
		list: list of all selected boxes
	"""
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes are integers, convert them to floats -- this
	# is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]
	# compute the area of the bounding boxes and grab the indexes to sort
	# (in the case that no probabilities are provided, simply sort on the
	# bottom-left y-coordinate)
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = y2

	# sort the indexes
	idxs = np.argsort(idxs)
	# keep looping while some indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the index value
		# to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of the bounding
		# box and the smallest (x, y) coordinates for the end of the bounding
		# box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have overlap greater
		# than the provided overlap threshold
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked
	return boxes[pick].astype("int")

def scaleCoords(x,y, level, f):
    """scale coordinates to the real size of the image

    Args:
        x (int): x coordinate
        y (int): y coordinate
        level (int): level of the pyramid
        f (int): factor of the pyramid

    Returns:
        int: x,y scaled coordinates
    """
    
    x = (int(x*(f**(level-1))))
    y = (int(y*(f**(level-1))))
    
    return x, y

def getCoords(row,col,level):
    """get coordinates and scale them

    Args:
        row (int): row of the window 
        col (int): column of the window
        level (int): level of the pyramid

    Returns:
        tuple: start, end coordinate tuple
    """
    if level != 0: 
        row,col = scaleCoords(row,col,level, 2)
    
    start = ((32 * row)* (2**level), (16 * col)* (2**level))
    end = (((128 + (32 * row))* (2**level)), ((64 + (16 * col))* (2**level)))
    return start, end

def drawBox(img, coords):
    """draw box in original image

    Args:
        img (image): image to draw on
        coords (array): coordinates of the box

    Returns:
        img: new image with draw boxes 
    """
    newImg = img.copy()

    start, end = (coords[0],coords[1]), (coords[2], coords[3])
    
    try:
        rr, cc = rectangle_perimeter(start, end, clip=True, shape=img.shape)
        newImg[rr,cc] = [0,255,0]
        # newImg[rr+1,cc+1] = [0,255,0]
        newImg[rr-1,cc-1] = [0,255,0]
    except:
        pass

    return newImg

def runDetector(path, model="SVM"):
    """run pedestrian detector

    Args:
        path (str): path of folder in wich images are

    Returns:
        list : list of images to be saved
    """

    imgs, names = loadImgs(path)

    for imgIndex in range(len(imgs)): 
        all1 = []
        finalImg = imgs[imgIndex].copy()
        pyramid = buildPyramid(imgs[imgIndex])
        for level, pymImg in enumerate(pyramid):
            # print(level)
            if pymImg.shape >= (128,64,3):
                window = getWindows(pymImg,(128,64,3))
                rows,cols = window.shape[0], window.shape[1]
                for row in range(rows):
                    for col in range(cols):
                        hogWindow = HOG(window[row,col,0])
                        ans, score = predictPed(hogWindow, model)
                        # print(ans, score[0][int(ans)],int(ans))
                        if ans == 1:
                            accScore = score[0][int(ans)]
                            start, end = getCoords(row,col,level)
                            all1.append([start[0],start[1],end[0],end[1],accScore])
            else: 
                break
    
        # applicar NMS
        # necesito coordenada start, end y score
        # print("all detected qty. ", len(all1))
        all1 = np.array(all1)

        selected = nmsScore(all1)
        # print("all selected from detected qty. ",len(selected))

        for i in selected:
            finalImg = drawBox(finalImg,i)
        
        fig, ax = plt.subplots(1,1, figsize=(20,20))
        ax.imshow(finalImg)
        ax.axes.xaxis.set_visible(False) # quitar axis
        ax.axes.yaxis.set_visible(False)
        plt.savefig((names[imgIndex]+"PedestriansDetector"),bbox_inches='tight',pad_inches=0)
    print("Images processed"+names[imgIndex])
    return "Images processed"+names[imgIndex]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="imgsEval", type=str, help="Path of the dataset")
    parser.add_argument("--model", default="SVM", type=str, help="Model election", choices=["ABC", "SVM"])
    args = parser.parse_args()
    print("\nHOG Clasiffy\n")
    print("If the images folder has the same images this might take 50 - 55 mins.")
    print("The images will be saved in the folder this file is in.\n")

    print("The model trained is SVM.\n")
    print("Metrics")
    print("-------------------------------")
    print("Precision score - 0.974324")
    print("Recall score - 0.952443")    
    print("")    

    runDetector(args.path, (args.model).upper)