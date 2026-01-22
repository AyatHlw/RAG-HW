import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

STATIC_DB_PATH = "vectorstore_static"
STATIC_SOURCE_FOLDER = "data/neural_networks"
DATA_PATH = "data"
DB_PATH = "vectorstore"
TEMP_UPLOAD_FOLDER = "temp_upload"

HEADER_MARGIN = 50
FOOTER_MARGIN = 50

CHUNK_SIZE = 700
CHUNK_OVERLAP = 250