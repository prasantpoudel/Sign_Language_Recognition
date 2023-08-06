import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)

offset=20
imgSize=300

folder="Data/A" ## itterate every folder manually for different sign language
counter=0

while True:
    sucess,img=cap.read()
    hands,img=detector.findHands(img)
    if hands:
        hand=hands[0]
        x,y,w,h=hand['bbox']

        imgwhite=np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop=img[y-offset:y+h+offset,x-offset:x+w+offset]
        imgCropShape=imgCrop.shape

        aspectRatio=h/w
        if aspectRatio>1:
            k=imgSize/h
            wCal=math.ceil(k*w)
            imgResize=cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape=imgResize.shape
            wGap=math.ceil((imgSize-wCal)/2)
            imgwhite[:,wGap:wCal+wGap]=imgResize

        else:
            k=imgSize/w
            hCal=math.ceil(k*h)
            imgResize=cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape=imgResize.shape
            hGap=math.ceil((imgSize-hCal)/2)
            imgwhite[hGap:hCal+hGap,:]=imgResize


        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            # cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgwhite)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit the loop
        break

    key=cv2.waitKey(1)
    if key==ord("s"): # press s button for save the image
        counter+=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgwhite)
        print(counter)

cap.release()
cv2.destroyAllWindows()