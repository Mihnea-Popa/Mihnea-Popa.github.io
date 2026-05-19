##################################################
#############        IMPORTS         #############
##################################################

import pickle 
import cv2
import numpy as np
import random as rd 
import os
import datetime
    # Import tensorflow for neural networks
from tensorflow.keras.models    import Sequential
from tensorflow.keras.layers    import Dense, Flatten
from tensorflow.keras.layers    import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard


##################################################
############  CREATION TEST DATASET  #############
##################################################
DATADIR = "DATASET2"
CATEGORIES = ["close", "open"]
IMG_SIZE = 100 
test_data = []

def create_test_data():
    for category in CATEGORIES:  
        path = os.path.join(DATADIR,category)  
        class_num = CATEGORIES.index(category)  
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  
            test_data.append([new_array, class_num])  
            
create_test_data()
rd.shuffle(test_data)

##################################################
############  CREATION TEST FUNCTION  ############
##################################################

def test_perf():
    c=0
    for (image,label) in test_data:
        image=image.reshape(-1, IMG_SIZE, IMG_SIZE, 1) 
        prediction = model.predict([image])
        if round(prediction[0][0])==label:
            c+=1
    return (c/len(test_data))

##################################################
##############    LOAD  DATASET    ###############
##################################################

feature_in = open("feature_list.pickle","rb")
feature_list = pickle.load(feature_in) 
feature_list = feature_list/255.0                 # Normalize between 0 and 1


label_in = open("label_list.pickle","rb")
label_list = pickle.load(label_in)
label_list=np.array(label_list)


##################################################
##############    CNN  STRUCTURE    ##############
##################################################

networks=[[256, 256, 128, 128]]
convolutions=[[32,32,64]] 
batch_sizes=[8]
epochs=[5]


def create_convolution_layers(convolution):
    for layer in convolution:
        model.add(Conv2D(layer, (3, 3), activation = 'relu', input_shape = feature_list.shape[1:]))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        
def create_fully_connected_layers(network):
    for layer in network:
        model.add(Dense(layer,activation = "relu"))

##################################################
##############    CNN  TRAININGS    ##############
##################################################

PERFORMANCE=[]

for network in networks:
    for convolution in convolutions:
        for batch_size_chosen in batch_sizes:
            for epoch_chosen in epochs: 
                
                model   = Sequential()
                create_convolution_layers(convolution)
                model.add(Flatten())                         # To convert our 3D feature to 1D feature
                create_fully_connected_layers(network)
                model.add(Dense(1, activation = "sigmoid"))  # Only one neuron: activated=open / inactivated= close
                model.compile(loss = 'binary_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])
                run_name = f"conv{convolution}_net{network}_bs{batch_size_chosen}_ep{epoch_chosen}"
                log_dir = "logs/" + run_name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                tensorboard = TensorBoard(log_dir=log_dir)
                history = model.fit(feature_list, label_list, batch_size=batch_size_chosen, epochs=epoch_chosen, validation_split=0.1, callbacks=[tensorboard])
                PERFORMANCE.append(['convolution'] + convolution + ['network'] + network + ['batch_size'] + [batch_size_chosen] + ['epoch'] + [epoch_chosen] + ['test'] + [test_perf()])
                model.save("model.keras")
             
print(PERFORMANCE) 
input("Press Enter to exit...")
