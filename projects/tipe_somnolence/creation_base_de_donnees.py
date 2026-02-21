##################################################
#############        IMPORTS         #############
##################################################

import numpy as np
import cv2
import random as rd
import pickle          # Conversion python objects into binary
import os
from tqdm import tqdm

##################################################
#############    CREATION DATASET    #############
##################################################

DATADIR = "DATASET"
CATEGORIES = ["close", "open"]
IMG_SIZE = 100
training_data = []

def create_training_data():
    for category in CATEGORIES:  
        path      = os.path.join(DATADIR,category) 
        class_num = CATEGORIES.index(category)  
        for img in tqdm(os.listdir(path)):
            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])  
            
create_training_data()
rd.shuffle(training_data)

feature_list, label_list = zip(*training_data)
feature_list, label_list = list(feature_list), list(label_list)

feature_list = np.array(feature_list).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # -1 unknown number of images, IMG_SIZE for the shape of the image, 1 for the grayscale

##################################################
#############      SAVE DATASET      #############
##################################################

pickle_out = open("feature_list.pickle","wb") # wb for writing in binary mode
pickle.dump(feature_list, pickle_out)         # creates a binary copy of feature_list and puts it in pickle_out
pickle_out.close()

pickle_out = open("label_list.pickle","wb")
pickle.dump(label_list, pickle_out)
pickle_out.close()







