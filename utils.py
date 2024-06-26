import os
import random
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import date
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import to_categorical
from PIL import Image, ImageEnhance

#### Saving Date today in 2 different formats
def datetoday():
    return date.today().strftime("%m_%d_%y")
def datetoday2():
    return date.today().strftime("%d-%B-%Y")


#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)


#### If these directories don't exist, create them
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')


#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))

#### Increase the contrast of images
def increase_contrast(path, newusername):
    inpath, outpath = path, path
    for i in range(50):
        fname = newusername+str(i)+'.jpg'
        im = Image.open(inpath + '/' + fname)
        ImageEnhance.Contrast(im).enhance(2.5).save(outpath + '/' + fname)


#### extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points


#### Identify face using ML model
def identify_face(facearray):
    with open('face_recognition_model.json', 'r') as json_file:
        json_savedModel= json_file.read()

    face_recognition_model = tf.keras.models.model_from_json(json_savedModel)
    face_recognition_model.load_weights('FaceRecognition_weights.h5')
    return np.argmax(face_recognition_model.predict(facearray)[0])


#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []

    y = []
    i = 0
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (64, 64))
            faces.append(resized_face)
            labels.append(user)
            y.append(i)
        i+=1

    pairs = list(zip(faces, y))

    random.shuffle(pairs)

    faces_random, y_random = zip(*pairs)

    y = to_categorical(y_random, num_classes = i)
    faces = np.array(faces_random)
    y = np.array(y)

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(i, activation='relu')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    model.fit(faces, y, validation_split=0.2)

    json_model = model.to_json()


    with open('face_recognition_model.json', 'w') as json_file:
        json_file.write(json_model)

    model.save_weights('FaceRecognition_weights.h5')


#### Extract info from today's attendance file in attendance folder
def extract_attendance():

    from user.models import Attendance
    attendance = Attendance()
    attendanceList = attendance.get_attendance()

    if len(attendanceList) == 0:
        return (0,0,0,0)


    names = pd.Series()
    rolls = pd.Series()
    times = pd.Series()
    l = 0

    df = pd.DataFrame(attendanceList)
    names = df['student_name']
    rolls = df['student_id']
    times = df['time']
    l = len(df)

    return names,rolls,times,l


#### Add Attendance of a specific user
def add_attendance(name):
    student_name = name.split('_')[0]
    student_id = name.split('_')[1]

    attendee = {
        'student_name' : student_name,
        'student_id' : student_id,
    }

    from user.models import Attendance

    attendance = Attendance()
    return attendance.add_attendance(attendee)

    # df = pd.read_csv(f'Attendance/Attendance-{datetoday()}.csv')
    # if int(userid) not in list(df['Roll']):
    #     with open(f'Attendance/Attendance-{datetoday()}.csv','a') as f:
    #         f.write(f'\n{username},{userid},{current_time}')
