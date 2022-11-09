# Katherine García 20190418

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage.io import imread
import numpy as np
import pickle
import os


def getHog(folderP,folderB):
    """ get HOG of all imgs

    Args:
        folderP (_type_): _description_

    Returns:
        _type_: _description_
    """
    lenFP = len(os.listdir(folderP))
    lenFB = len(os.listdir(folderB))
    total = lenFP+lenFB
    # print(total)
    X = np.zeros(shape=(total,3780))
    y = np.append(np.ones(shape=(1,lenFP)),np.zeros(shape=(1,lenFB)))
    count = 0

    for filename in os.listdir(folderP):
        img = imread(os.path.join(folderP,filename))
        if img is not None:
            hogImg = hog(img,orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
            block_norm='L2', feature_vector=True, channel_axis=2) # feature vector= true, channel_Axis=2
            X[count] = hogImg
        count += 1

    for filename in os.listdir(folderB):
        img = imread(os.path.join(folderB,filename))
        if img is not None:
            hogImg = hog(img,orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
            block_norm='L2', feature_vector=True, channel_axis=2) # feature vector= true, channel_Axis=2
            X[count] = hogImg
        count += 1
    return X,y

def model(X,y, modelType="ABC"):
    # Create adaboost classifer object
    if modelType == "GBC":
        mod = GradientBoostingClassifier(n_estimators=150, learning_rate=1)
    else:
        mod = AdaBoostClassifier(n_estimators=150, learning_rate=1)
    
    # Train Adaboost Classifer
    model = mod.fit(X, y)
    
    return model

def saveModel(model, path=""):
    with open(path+'modelPkl', 'wb') as files:
        pickle.dump(model, files)

def loadModel(path=""):
    with open(path+'modelPkl' , 'rb') as files:
        model = pickle.load(files)
    return model


# folderP = 'Pedestrians-Dataset-complete/Pedestrians-Dataset/Pedestrians'
# folderB = 'Pedestrians-Dataset-complete/Pedestrians-Dataset/Background'

def run(path, modelSelected="ABC"):
    folderP = 'Pedestrians'
    folderB = 'Background'

    # X y HOG
    X, y = getHog(os.path.join(path, folderP), os.path.join(path, folderB))
    
    # separar la data
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

    try: 
        modelT = loadModel()
    except:
        # Train the model
        modelT = model(xTrain,yTrain, modelSelected)

        #persist and load model
        saveModel(modelT)

    # predict
    yPred = modelT.predict(xTest)

    # metrics
    print("Metrics")
    print("-------------------------------")
    print("Precision score - ", precision_score(yTest, yPred))
    print("Recall score - ", recall_score(yTest, yPred))    
    print("")    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="Pedestrians-Dataset-complete/Pedestrians-Dataset", type=str, help="Path of the dataset")
    parser.add_argument("--model", default="ABC", type=str, help="Model election", choices=["ABC", "GBC"])
    args = parser.parse_args()
    print("\nProyecto 1\n")
    run(args.path, (args.model).upper)