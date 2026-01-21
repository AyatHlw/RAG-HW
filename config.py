import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

DATA_PATH = "data"
DB_PATH = "vectorstore"
TEMP_UPLOAD_FOLDER = "temp_upload"

HEADER_MARGIN = 50
FOOTER_MARGIN = 50

CHUNK_SIZE = 700
CHUNK_OVERLAP = 250