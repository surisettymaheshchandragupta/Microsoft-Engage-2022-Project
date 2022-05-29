import numpy as np
import cv2 as cv
import os
import face_recognition
from datetime import datetime
import warnings


path = 'img'

img = []
image_names = []
my_list = os.listdir(path)
print(my_list)

for i in my_list:

    img_read = cv.imread(f'{path}/{i}')
    # print("reading images successful")
    img.append(img_read)

    # taking students names from img name
    image_names.append(os.path.splitext(i)[0])

print(image_names)
# print(img)

def encoding_images(img):

    encode_list = []
    for j in img:

        image = cv.cvtColor(j, cv.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0] 
        encode_list.append(encode)
    return encode_list

img_after_encoding = encoding_images(img)
print(len(img_after_encoding))

cap = cv.VideoCapture(0,cv.CAP_DSHOW)

def attendance(name):

    with open('attendance.csv', 'r+') as f:

        myDataList = f.readlines()
        nameList = []
        for line in myDataList:

            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:

            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tStr},{dStr}')
while True:

    ret, frame = cap.read()
    faces = cv.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv.cvtColor(faces, cv.COLOR_BGR2RGB)
    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):

        matches = face_recognition.compare_faces(img_after_encoding, encodeFace)
        faceDis = face_recognition.face_distance(img_after_encoding, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:

            name = image_names[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv.FILLED)
            cv.putText(frame, name, (x1 + 6, y2 - 6), cv.FONT_HERSHEY_COMPLEX, 1, ((0,0,0)), 2)
            attendance(name)
        
        else :

            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv.FILLED)
            cv.putText(frame, "Unknown", (x1 + 6, y2 - 6), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

    cv.imshow('Webcam', frame)
    if cv.waitKey(1) == 32:
        
        break
cap.release()
cv.destroyAllWindows()
