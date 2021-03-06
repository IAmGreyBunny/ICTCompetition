from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_restful import Api

# Initializing flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Password1234@localhost/webapp_database'

# Flask extensions
api = Api(app)
db = SQLAlchemy(app)

from webapp import routes
from webapp import api_resources
