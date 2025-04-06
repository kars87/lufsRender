# This file is required for PythonAnywhere
import sys
import os

# Add your project directory to the Python path
path = './app.py'
if path not in sys.path:
    sys.path.append(path)

# Import your Flask app
from app import app as application
