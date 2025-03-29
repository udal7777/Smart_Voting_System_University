import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Define declarative base for SQLAlchemy
class Base(DeclarativeBase):
    pass

# Initialize extensions
db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)

# Configure secret key
app.secret_key = os.environ.get("SESSION_SECRET", "dev-key-for-testing")

# Configure session
app.config["PERMANENT_SESSION_LIFETIME"] = 3600  # 1 hour
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False  # Set to True in production with HTTPS

# Configure database - using SQLite for simplicity
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///voting_system.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize extensions with app
db.init_app(app)

# Import models and create tables
with app.app_context():
    import models
    db.create_all()

# Import routes
from routes import *
