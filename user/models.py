from flask import  jsonify, redirect, request, session
from datetime import datetime
from passlib.hash import pbkdf2_sha256
from app import db, datetoday
import uuid

class User:

    def start_session(self, user):
        del user['password']
        session['logged_in'] = True
        session['user'] = user
        return jsonify(user), 200

    def signup(self):

        user = {
            "_id" : uuid.uuid4().hex,
            "name" : request.form.get('name'),
            "email" : request.form.get('email'),
            "password" : request.form.get('password'),
        }

        if db.users.find_one({"email" : user["email"]}):
            return jsonify({"error" : "Email already exists"}), 400

        user["password"] = pbkdf2_sha256.encrypt(user["password"])

        if db.users.insert_one(user):
            return self.start_session(user)

        return jsonify({"error" : "Signup failed"}), 400

    def signout(self):
        session.clear()
        return redirect('/')

    def login(self):
        user = db.users.find_one({
            "email" : request.form.get('email')
        })

        if user:
            if pbkdf2_sha256.verify(request.form.get("password"), user["password"]):
                return self.start_session(user)
            else:
                return jsonify({"error" : "Invalid credentials"}), 401
        else:
            return jsonify({"error" : "Unknown email address"}), 404
        
class Attendance:
    
    def add_attendance(attendance, student):

        current_time = datetime.now().strftime("%H:%M:%S")
        current_date = datetoday()
        
        attendee = {
            "userid" : session['user']['_id'],
            "student_id" : student['student_id'],
            "student_name" : student['student_name'],
            "time" : current_time,
            "date" : current_date
        }

        if db.attendance.find_one({"userid" : attendee['userid'], 
                                   "student_id" : attendee['student_id'], 
                                   "date" : attendee['date']}):
            return False
        
        if db.attendance.insert_one(attendee):
            return True
        else:
            return False
        
    
    def get_attendance(self):
        attendanceList = list()
        for attendance in db.attendance.find({"userid" : session['user']['_id'], "date" : datetoday()}):
            attendanceList.append(attendance)

        return attendanceList