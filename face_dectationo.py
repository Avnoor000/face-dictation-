import cv2 as cv

img  = cv.imread('opencv/person.jpg')
cv.imshow('Person',img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray person',gray)

haar_cascade = cv.CascadeClassifier(r'C:\Users\Manpreet\Desktop\Avnoor\opencv\haar_face.xml')

face_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)

print(f'{len(face_rect)} face(s) found')

for (x,y,w,h) in face_rect:
    cv.rectangle(img,(x,y), (x+w,y+h), (0,0,255),thickness=2)

cv.imshow('Face dected',img)

cv.waitKey(0)