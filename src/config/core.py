import os

from dotenv import load_dotenv

load_dotenv()

class CustomSettings():
    PROJECT_NAME: str = "Eventure AI Module"
    
settings = CustomSettings()
