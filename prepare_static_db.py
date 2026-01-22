import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import STATIC_DB_PATH, STATIC_SOURCE_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP
from data_processor import extract_text_from_pdf


def build_static_database():
	print(" Starting to build Static Knowledge Base for Neural Networks...")

	if os.path.exists(STATIC_DB_PATH):
		shutil.rmtree(STATIC_DB_PATH)
		print(f"Cleared old database at {STATIC_DB_PATH}")

	if not os.path.exists(STATIC_SOURCE_FOLDER):
		print(f" Error: Folder {STATIC_SOURCE_FOLDER} not found!")
		return

	all_documents = []
	files = [f for f in os.listdir(STATIC_SOURCE_FOLDER) if f.endswith('.pdf')]

	if not files:
		print(" No PDF files found!")
		return

	print(f"Found {len(files)} lectures. Processing...")

	for file_name in files:
		file_path = os.path.join(STATIC_SOURCE_FOLDER, file_name)
		print(f"  - Processing: {file_name}...")
		docs = extract_text_from_pdf(file_path)
		all_documents.extend(docs)

	print("Chunking text...")
	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=CHUNK_SIZE,
		chunk_overlap=CHUNK_OVERLAP,
		separators=["***PAGE_START***", "\n\n", "\n", ".", " "],
		strip_whitespace=True
	)

	raw_chunks = text_splitter.split_documents(all_documents)

	final_chunks = []
	for chunk in raw_chunks:
		content = chunk.page_content.replace("***PAGE_START***", "").strip()
		if len(content) > 50:
			chunk.page_content = content
			final_chunks.append(chunk)

	print(f"Total chunks created: {len(final_chunks)}")

	print("Embedding and saving to ChromaDB (This might take a minute)...")
	embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

	Chroma.from_documents(
		documents=final_chunks,
		embedding=embedding_function,
		persist_directory=STATIC_DB_PATH
	)

	print(" DONE! Static database is ready.")


if __name__ == "__main__":
	build_static_database()