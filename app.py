import os
from datetime import timedelta
from functools import wraps
import cv2
import numpy as np
import pymongo
from flask import Flask, redirect, render_template, request, session
from utils import extract_attendance, totalreg, datetoday2, extract_faces, add_attendance, identify_face, increase_contrast, train_model
from user.models import User

numClasses = 10

#### Defining Flask App
app = Flask(__name__)
app.secret_key = 'LegitUser@2023'

@app.before_request
def make_session_permanent():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(minutes=5)

client = pymongo.MongoClient("localhost", 27017)
db = client.face_recognition_login_system

# Decorators
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            return redirect('/authenticate')

    return wrap

################## ROUTING FUNCTIONS #########################


from user import routes


#### Our main page
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/authenticate')
def authenticate():
    return render_template('authentication.html')

@app.route('/user/signup', methods=['POST'])
def signup():
    user = User()
    return user.signup()

@app.route('/user/login', methods=['POST'])
def login():
    return User().login()

@app.route('/attendance', methods=['GET'])
@login_required
def attendance():
    names,rolls,times,l = extract_attendance()
    if l == 0:
        return render_template('home.html',totalreg=totalreg(),datetoday2=datetoday2())
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2())

#### This function will run when we click on Take Attendance Button
@app.route('/attendance/start',methods=['GET','POST'])
@login_required
def start():
    if 'face_recognition_model.json' not in os.listdir('.'):
        return render_template('home.html',totalreg=totalreg(),datetoday2=datetoday2(),mess='There is no trained model in the static folder. Please add a new face to continue.')

    face_names = []

    userlist = os.listdir('static/faces')
    for user in userlist:
        face_names.append(user)

    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret,frame = cap.read()

        if extract_faces(frame) != ():
            (x,y,w,h) = extract_faces(frame)[0]
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y+h,x:x+w], (64, 64))
            faceArr = []
            faceArr.append(face)
            faceArr = np.array(faceArr)
            identified_person = face_names[identify_face(faceArr)]
            cv2.putText(frame,f'{identified_person}',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if not add_attendance(identified_person):
                continue
        cv2.imshow('Attendance',frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names,rolls,times,l = extract_attendance()
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2())


#### This function will run when we add a new attendee
@app.route('/add',methods=['GET','POST'])
@login_required
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i,j = 0,0
    while 1:
        _,frame = cap.read()
        faces = extract_faces(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if j%10==0:
                name = newusername+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                i+=1
            j+=1
        if j==500:
            break
        cv2.imshow('Adding new User',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    increase_contrast(userimagefolder, newusername)
    print('Training Model')
    train_model()
    names,rolls,times,l = extract_attendance()
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2())


#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True, port=8081)