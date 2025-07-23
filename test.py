import cv2 as cv
import numpy as np 

haar_cascade = cv.CascadeClassifier(r'C:\Users\Manpreet\Desktop\Avnoor\opencv\haar_face.xml')

people = ['Dwayne Johnson','jacqueline fernandez','Janhvi Kapoor','mitchell starc','Tom holland']

#features = np.load('features.npy')
#lables = np.load('lables.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_t.yml')

img = cv.imread('opencv/person.jpg')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)


faces_rect = haar_cascade.detectMultiScale(gray,1.1,2)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+h]

    lable, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[lable]} with a confidence of {confidence}')

    cv.putText(img, str(people[lable]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), thickness=2)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

cv.imshow('Face', img)
cv.waitKey(0)