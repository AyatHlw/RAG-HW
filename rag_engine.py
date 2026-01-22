import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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


def rewrite_query(question, chat_history):
	if not chat_history:
		return question

	history_text = "\n".join([f"{role}: {msg}" for role, msg in chat_history])

	prompt_template = PromptTemplate.from_template("""
    Given the following conversation history and a follow-up question, 
    rephrase the follow-up question to be a standalone question. 
    Do NOT answer the question, just rewrite it to include context if needed.

    Chat History:
    {history}

    Follow Up Input: {question}

    Standalone Question:
    """)

	model = ChatGoogleGenerativeAI(
		google_api_key=GOOGLE_API_KEY,
		model="models/gemini-2.5-flash",
		temperature=0.1
	)

	chain = prompt_template | model | StrOutputParser()
	return chain.invoke({"history": history_text, "question": question})


def get_answer_from_llm(question, db_path=DB_PATH, chat_history=[]):
	search_query = rewrite_query(question, chat_history)
	print(f"Original: {question} -> Rewritten: {search_query}")

	embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

	if not os.path.exists(db_path):
		return "Please upload a lecture first.", []

	vector_db = Chroma(persist_directory=db_path, embedding_function=embedding_function)

	results_with_scores = vector_db.similarity_search_with_relevance_scores(search_query, k=5)

	threshold = 0.3
	final_results = [doc for doc, score in results_with_scores if score >= threshold]

	if not final_results and results_with_scores:
		final_results = [doc for doc, score in results_with_scores[:3]]

	if not final_results:
		return "Information not found in lecture.", []

	context_text = "\n\n---\n\n".join([doc.page_content for doc in final_results])

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

	prompt = prompt_template.format(context=context_text, question=search_query)

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
	for doc in final_results:
		src = f"📄 {doc.metadata.get('source')} (Page {doc.metadata.get('page', 0) + 1})"
		sources.append(src)

	return response.content, list(set(sources)), search_query