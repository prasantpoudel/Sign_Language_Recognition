import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
from PIL import Image
from keras.models import load_model


cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)
classifier=Classifier("model/model.h5")
model=load_model("model/model.h5")

offset=20
imgSize=180
labels=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']



while True:
    sucess,img=cap.read()
    imgOutput=img.copy()
    hands,img=detector.findHands(img)
    if hands:
        hand=hands[0]
        x,y,w,h=hand['bbox']

        imgwhite=np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop=img[y-offset:y+h+offset,x-offset:x+w+offset]
        imgCropShape=imgCrop.shape

        preprocessed_image = cv2.resize(img, (224, 224))  # Adjust the size according to your model's input
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        prediction = model.predict(preprocessed_image)
        index = np.argmax(prediction)
        

        cv2.putText(imgOutput,labels[index],(x,y-20),cv2.FONT_HERSHEY_DUPLEX,2,(255,0,255),2)
           

        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            cv2.imshow("ImageCrop", imgCrop)
            # cv2.imshow("ImageWhite", imgwhite)

    cv2.imshow("Image", imgOutput)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit the loop
        break


cap.release()
cv2.destroyAllWindows()