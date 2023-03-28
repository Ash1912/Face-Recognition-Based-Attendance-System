from flask import jsonify, redirect, request, session
from passlib.hash import pbkdf2_sha256
from app import db
import uuid