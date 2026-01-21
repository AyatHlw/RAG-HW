import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from config import DB_PATH, CHUNK_SIZE, CHUNK_OVERLAP, GOOGLE_API_KEY
from data_processor import extract_text_from_pdf


def build_vector_db(file_path):
	if os.path.exists(DB_PATH):
		shutil.rmtree(DB_PATH)

	documents = extract_text_from_pdf(file_path)

	if not documents:
		return None

	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=CHUNK_SIZE,
		chunk_overlap=CHUNK_OVERLAP,
		separators=["***PAGE_START***", "\n\n", "\n", ".", " "],
		strip_whitespace=True
	)

	raw_chunks = text_splitter.split_documents(documents)

	final_chunks = []
	for chunk in raw_chunks:
		content = chunk.page_content.replace("***PAGE_START***", "").strip()
		if len(content) > 50:
			chunk.page_content = content
			final_chunks.append(chunk)

	embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

	vector_db = Chroma.from_documents(
		documents=final_chunks,
		embedding=embedding_function,
		persist_directory=DB_PATH
	)
	return vector_db


def get_answer_from_llm(question):
	embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

	if not os.path.exists(DB_PATH):
		return "Please upload a lecture first.", []

	vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)

	results = vector_db.similarity_search(question, k=5)
	context_text = "\n\n---\n\n".join([doc.page_content for doc in results])

	prompt_template = ChatPromptTemplate.from_template("""
    You are an expert University Tutor.
    Answer the student's question based STRICTLY on the provided lecture context.

    Context:
    {context}

    Instructions:
    1. If context has math formulas, use LaTeX.
    2. If [Diagram Text] is present, explain the diagram.
    3. If the answer is missing, say "Information not found in lecture."

    Question: {question}
    """)

	prompt = prompt_template.format(context=context_text, question=question)

	try:
		model = ChatGoogleGenerativeAI(
			google_api_key=GOOGLE_API_KEY,
			model="models/gemini-2.5-flash",
			temperature=0.1
		)
		response = model.invoke(prompt)

	except Exception:
		model = ChatGoogleGenerativeAI(
			google_api_key=GOOGLE_API_KEY,
			model="models/gemini-flash-latest",
			temperature=0.1
		)
		response = model.invoke(prompt)

	sources = []
	for doc in results:
		src = f"📄 {doc.metadata.get('source')} (Page {doc.metadata.get('page', 0) + 1})"
		sources.append(src)

	return response.content, list(set(sources))