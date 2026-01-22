import streamlit as st
import os
from rag_engine import build_vector_db, get_answer_from_llm
from config import TEMP_UPLOAD_FOLDER, DB_PATH, STATIC_DB_PATH

st.set_page_config(page_title="Intelligent Course Assistant", layout="wide")

st.title("Intelligent Course Assistant")

with st.sidebar:
	st.header("Knowledge Source")
	source_mode = st.radio("Select Data Source", ["Upload Document", "Neural Networks (Pre-loaded)"])

	active_db_path = None

	if source_mode == "Upload Document":
		uploaded_file = st.file_uploader("Upload PDF File", type="pdf")
		if uploaded_file:
			if not os.path.exists(TEMP_UPLOAD_FOLDER):
				os.makedirs(TEMP_UPLOAD_FOLDER)
			file_path = os.path.join(TEMP_UPLOAD_FOLDER, uploaded_file.name)
			with open(file_path, "wb") as f:
				f.write(uploaded_file.getbuffer())
			if st.button("Analyze Document"):
				with st.spinner("Processing document..."):
					build_vector_db(file_path)
					st.success("Document processed successfully.")
					st.session_state.active_db_path = DB_PATH
					st.session_state.messages = []

		if os.path.exists(DB_PATH):
			active_db_path = DB_PATH

	else:
		st.info("Using Static Knowledge Base: Neural Networks")
		active_db_path = STATIC_DB_PATH

if "messages" not in st.session_state:
	st.session_state.messages = []

for message in st.session_state.messages:
	with st.chat_message(message["role"]):
		if "debug_query" in message and message["debug_query"]:
			with st.expander("Query Reasoning & Rewriting"):
				st.write(f"**Optimized Search Query:** {message['debug_query']}")

		st.markdown(message["content"])

		if "sources" in message and message["sources"]:
			with st.expander("Reference Context"):
				for src in message["sources"]: st.write(src)

if prompt := st.chat_input("Enter your question here..."):
	if not active_db_path or not os.path.exists(active_db_path):
		st.error("Please select a valid source or upload a file first.")
		st.stop()

	chat_history = []
	for msg in st.session_state.messages[-6:]:
		role = "User" if msg["role"] == "user" else "Assistant"
		chat_history.append((role, msg["content"]))

	st.session_state.messages.append({"role": "user", "content": prompt})
	with st.chat_message("user"):
		st.markdown(prompt)

	with st.chat_message("assistant"):
		with st.spinner("Generating response..."):
			answer, sources, debug_query = get_answer_from_llm(
				prompt,
				db_path=active_db_path,
				chat_history=chat_history
			)

			with st.expander("Query Reasoning & Rewriting"):
				st.write(f"**Original Input:** {prompt}")
				st.write(f"**Optimized Search Query:** {debug_query}")

			st.markdown(answer)

			if sources:
				with st.expander("Reference Context"):
					for src in sources: st.write(src)

			st.session_state.messages.append({
				"role": "assistant",
				"content": answer,
				"sources": sources,
				"debug_query": debug_query
			})