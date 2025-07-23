import cv2 as cv
import os 
import numpy as np

people = ['Dwayne Johnson','jacqueline fernandez','Janhvi Kapoor','mitchell starc','Tom holland']

DIR = r'C:\Users\Manpreet\Desktop\Avnoor\opencv\data'

haar_cascade = cv.CascadeClassifier(r'C:\Users\Manpreet\Desktop\Avnoor\opencv\haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)
    
        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()

print('Done -----------------------------------')


features = np.array(features,dtype='object')
labels = np.array(labels)

print(f'Length of features = {len(features)}')
print(f'Length of labels = {len(labels)}')



face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features,labels)

face_recognizer.save('face_t.yml')
np.save('features.npy',features)
np.save('lables.npy',labels)