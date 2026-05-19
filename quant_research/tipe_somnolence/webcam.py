##################################################
#############        IMPORTS         #############
##################################################

import numpy as np
import pickle
import cv2
import time
import tensorflow as tf
import winsound
import keras

##################################################
#############       LOAD MODEL       #############
##################################################

CATEGORIES = ["Close", "Open"] 
IMG_SIZE = 100 

model = keras.models.load_model("model.keras")

##################################################
#############       WEBCAM USE       #############
##################################################

cap = cv2.VideoCapture(0)
compteur_yeux_fermes = 0
while True:
    ret, frame = cap.read()
    start      = time.time()
    image      = cv2.cvtColor(frame[150:350, 200:400], cv2.COLOR_BGR2GRAY)    # Extract a part of the image

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) 
    image = image.astype("float32") / 255.0 
    image = image.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    prediction = model.predict(image)
    cv2.putText(frame, CATEGORIES[round(prediction[0][0])], (500,30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    
    if round(prediction[0][0]) == 0:
        compteur_yeux_fermes+=1
    else:
        compteur_yeux_fermes = 0
    if compteur_yeux_fermes >= 10:
        cv2.putText(frame,"DANGER", (250,400), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 5)
        #winsound.Beep(350, 100)
        #winsound.MessageBeep(type=-1)
        
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
        
    fps= 1 / (time.time()-start)
    cv2.putText(frame, "FPS: {:4.1f}".format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.rectangle(frame, (200,150),(400,350),(255,255,255),2)
    
    cv2.imshow('Webcam image', frame)
    
cap.release()
cv2.destroyAllWindows()
