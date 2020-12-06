# Import module
import cv2 as cv
# Read image
Img = CV. Imread (' Lena. JPG)
# Display image
cv.imshow('read_img',img)
# Wait for keyboard input
cv.waitKey(3000)
# Free memory
cv.destroyAllWindows()
                 

                  
import cv2 as cv
img=cv.imread('lena.jpg')
cv.imshow('BGR_img',img)
# Grayscale the image
gray_img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray_img',gray_img)
# Save image
cv.imwrite('gray_lena.jpg',gray_img)
cv.waitKey(0)
cv.destroyAllWindows()
                  
                  
import cv2 as cv
img=cv.imread('lena.jpg')
cv.imshow('img',img)
print('The shape of the original picture',img.shape)
# resize_img=cv.resize(img,dsize=(200,240))
resize_img=cv.resize(img,dsize=(600,560))
print('The shape of the modified image：',resize_img.shape)
cv.imshow('resize_img',resize_img)

# cv.waitKey(0)
# Only enter Q, exit
while True:
    if ord('q')==cv.waitKey(0):
        break
cv.destroyAllWindows()  
                  
                  
import cv2 as cv
img=cv.imread('lena.jpg')
x,y,w,h=100,100,100,100
cv.rectangle(img,(x,y,x+w,y+h),color=(0,255,255),thickness=3) #BGR
x,y,r=200,200,100
cv.circle(img,center=(x,y),radius=r,color=(0,0,255),thickness=2)
cv.imshow('rectangle_img',img)
cv.waitKey(0)
cv.destroyAllWindows()                  
 
                  
import cv2 as cv
def face_detect_demo():
    # Convert image to grayscale image
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # Load characteristic data
    face_detector=cv.CascadeClassifier('E:/soft/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    faces=face_detector.detectMultiScale(gray)
    for x,y,w,h in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)
    cv.imshow('result',img)
# load pictures
img=cv.imread('lena.jpg')
face_detect_demo()
cv.waitKey(0)
cv.destroyAllWindows()                  

                  
                  
import cv2 as cv
def face_detect_demo():
    # Grayscale the image
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # Load characteristic data
    face_detector = cv.CascadeClassifier(
        'E:/soft/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    faces = face_detector.detectMultiScale(gray)
    for x,y,w,h in faces:
        print(x,y,w,h)
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
        cv.circle(img,center=(x+w//2,y+h//2),radius=w//2,color=(0,255,0),thickness=2)
    # Display pictures
    cv.imshow('result',img)

# load pictures
img=cv.imread('face3.jpg')
# Call the face detection method
face_detect_demo()
cv.waitKey(0)
cv.destroyAllWindows() 
                  
                  
import cv2 as cv
def face_detect_demo(img):
    # Grayscale the image
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # Load characteristic data
    face_detector = cv.CascadeClassifier(
        'E:/soft/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    faces = face_detector.detectMultiScale(gray)
    for x,y,w,h in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
        cv.circle(img,center=(x+w//2,y+h//2),radius=(w//2),color=(0,255,0),thickness=2)
    cv.imshow('result',img)
# Read videos
cap=cv.VideoCapture('video.mp4')
while True:
    flag,frame=cap.read()
    print('flag:',flag,'frame.shape:',frame.shape)
    if not flag:
        break
    face_detect_demo(frame)
    if ord('q') == cv.waitKey(10):
        break
cv.destroyAllWindows()
cap.release()                  

                  

import os
import cv2
import sys
from PIL import Image
import numpy as np
def getImageAndLabels(path):
    facesSamples=[]
    ids=[]
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    # Detection of human face
    face_detector = cv2.CascadeClassifier(
        'E:/soft/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')

    # Traverses the image in the list
    for imagePath in imagePaths:
        # open pictures
        PIL_img=Image.open(imagePath).convert('L')
        # Convert the image to an array
        img_numpy=np.array(PIL_img,'uint8')
        faces = face_detector.detectMultiScale(img_numpy)
        # Gets the ID of each image
        id=int(os.path.split(imagePath)[1].split('.')[0])
        for x,y,w,h in faces:
            facesSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return facesSamples,ids

if __name__ == '__main__':
    # Image path
    path='./data/jm/'
    # Get an array of images and an array of ID tags
    faces,ids=getImageAndLabels(path)
    # Acquisition of training objects
    recognizer=cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces,np.array(ids))
    # save files
    recognizer.write('trainer/trainer.yml')                 

                  
                  
                  
import cv2
import numpy as np
import os
# Load the training dataset file
recogizer=cv2.face.LBPHFaceRecognizer_create()
recogizer.read('trainer/trainer.yml')
img=cv2.imread('3.pgm')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
face_detector = cv2.CascadeClassifier(
    'E:/soft/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
faces = face_detector.detectMultiScale(gray)
for x,y,w,h in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    # Face Recognition
    id,confidence=recogizer.predict(gray[y:y+h,x:x+w])
    print('Label id:',id,'confidence score：',confidence)
cv2.imshow('result',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
