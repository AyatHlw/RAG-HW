import os
import re
import shutil
import fitz
from rapidocr_onnxruntime import RapidOCR
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data"
DB_PATH = "vectorstore"

ocr_engine = RapidOCR()

HEADER_MARGIN = 50
FOOTER_MARGIN = 50


def advanced_clean_text(text):
	text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
	text = re.sub(r'(^|\n)([A-Z][^\n.]{1,60})(\n)', r'\1\2\n\n', text)
	text = re.sub(r'\n(?=\d+\.|•|-|\*)', '\n\n', text)
	text = re.sub(r'(?<![.:])\n(?!\n)', ' ', text)
	text = re.sub(r'[ \t]+', ' ', text)
	text = re.sub(r'\n\s+', '\n', text)
	return text.strip()


def extract_text_from_pdf(pdf_path):
	doc = fitz.open(pdf_path)
	extracted_docs = []

	for page_num, page in enumerate(doc):
		rect = page.rect
		clip_rect = fitz.Rect(0, HEADER_MARGIN, rect.width, rect.height - FOOTER_MARGIN)

		blocks = page.get_text("blocks", sort=True, clip=clip_rect)
		text_blocks = []
		for b in blocks:
			block_text = b[4]
			if block_text.strip():
				text_blocks.append(block_text)

		text = "\n\n".join(text_blocks)

		image_list = page.get_images(full=True)

		if image_list:
			print(f"   [Page {page_num + 1}] Found {len(image_list)} images, scanning for text...")
			for img_index, img in enumerate(image_list):
				xref = img[0]
				try:
					base_image = doc.extract_image(xref)
					image_bytes = base_image["image"]

					if len(image_bytes) < 2000:
						continue

					ocr_result, _ = ocr_engine(image_bytes)
					if ocr_result:
						image_text = " ".join([res[1] for res in ocr_result])

						if len(image_text.strip()) > 5:
							text += f"\n\n[Diagram Text]: {image_text}"

				except Exception as e:
					print(f"   Warning: Failed to OCR image {img_index} on page {page_num + 1}: {e}")

		cleaned_text = advanced_clean_text(text)

		if cleaned_text:
			final_content = f"***PAGE_START***\n\n{cleaned_text}"

			new_doc = Document(
				page_content=final_content,
				metadata={"source": os.path.basename(pdf_path), "page": page_num}
			)
			extracted_docs.append(new_doc)

	return extracted_docs


def create_vector_db():
	if not os.path.exists(DATA_PATH):
		os.makedirs(DATA_PATH)
		print(f"Directory {DATA_PATH} created. Add PDFs there.")
		return

	all_documents = []
	print("Loading and optimizing PDF files...")

	for file in os.listdir(DATA_PATH):
		if file.endswith(".pdf"):
			file_path = os.path.join(DATA_PATH, file)
			try:
				docs = extract_text_from_pdf(file_path)
				all_documents.extend(docs)
				print(f" - Processed {file}: {len(docs)} pages.")
			except Exception as e:
				print(f"Error loading {file}: {e}")

	if not all_documents:
		print("No documents extracted.")
		return

	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=700,
		chunk_overlap=250,
		separators=["***PAGE_START***", "\n\n", "\n", ".", " "],
		strip_whitespace=True
	)

	raw_chunks = text_splitter.split_documents(all_documents)

	if not raw_chunks:
		print("Error: No text chunks created.")
		return

	final_chunks = []
	for chunk in raw_chunks:
		content = chunk.page_content.replace("***PAGE_START***", "").strip()

		if len(content) > 50:
			chunk.page_content = content
			final_chunks.append(chunk)

	print(f"Filtered out {len(raw_chunks) - len(final_chunks)} ghost/empty chunks.")
	print(f"Generating Embeddings for {len(final_chunks)} chunks...")

	embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

	Chroma.from_documents(
		documents=final_chunks,
		embedding=embedding_function,
		persist_directory=DB_PATH
	)
	print("Database Updated Successfully!")


if __name__ == "__main__":
	if os.path.exists(DB_PATH):
		try:
			shutil.rmtree(DB_PATH)
			print("Old database removed.")
		except:
			pass

	print("Starting Ingestion Pipeline...")
	create_vector_db()

	print("\n" + "=" * 40)
	print(" INSPECTING CHUNKS (Quality Check) ")
	print("=" * 40)

	embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
	vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)

	data = vector_db.get(limit=15)

	if data and data['documents']:
		for i, doc in enumerate(data['documents']):
			meta = data['metadatas'][i]
			print(f"\n--- CHUNK {i + 1} ---")
			print(f"Source: {meta.get('source')} | Page: {meta.get('page') + 1}")
			print(f"Length: {len(doc)} chars")
			print("-" * 20)
			print(doc)
			print("-" * 20)
	else:
		print("No chunks found.")
