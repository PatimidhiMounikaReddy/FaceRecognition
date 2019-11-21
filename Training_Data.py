# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:42:51 2019

@author: Mounika Reddy P
"""

import cv2
import os
import numpy as np
import faceRecognition as fr

test_img=cv2.imread("test_img.jpg")
faces_Detected,gray_img=fr.faceDetection(test_img)
print("faces_detected:",faces_Detected)

#for (x,y,w,h) in faces_detected:
    #cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),2)
    
    #resized_img=cv2.resize(test_img,(1000,700))
    #cv2.imshow("face dtecetion",resized_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows
    
faces,faceID=fr.labels_for_training_data("C://Users//Mounika Reddy P//Desktop//FACE//trainingimages")
face_recognizer= fr.train_classifier(faces,faceID)
face_recognizer.save('trainingData.yml')
name={0:"Priyanka",1:"Nick"}
    
for face in faces_Detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print("confidence:",confidence)
    print("label:",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    fr.put_text(test_img,predicted_name,x,y)
    
    test_img=cv2.resize(test_img,(1000,1000))
cv2.imshow("face dtecetion",test_img)
cv2.waitKey(0)
cv2.destroyAllWindows
    
        
                 