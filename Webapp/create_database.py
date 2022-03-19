import mysql.connector

mydb = mysql.connector.connect(host="localhost",
                             user="root",
                             password="Password1234")

my_cursor = mydb.cursor()
my_cursor.execute("CREATE DATABASE webapp_database")

from webapp import db
from webapp import models

db.create_all()
db.session.commit()
