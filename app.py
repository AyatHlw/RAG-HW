import streamlit as st
import os
from rag_engine import build_vector_db, get_answer_from_llm
from config import TEMP_UPLOAD_FOLDER

st.set_page_config(page_title="Uni-RAG Assistant", page_icon="🎓")

st.title("🎓 Lecture Genius (Modular RAG)")

# الشريط الجانبي
with st.sidebar:
	st.header("Upload Lecture")
	uploaded_file = st.file_uploader("PDF File", type="pdf")

	if uploaded_file:
		if not os.path.exists(TEMP_UPLOAD_FOLDER):
			os.makedirs(TEMP_UPLOAD_FOLDER)

		file_path = os.path.join(TEMP_UPLOAD_FOLDER, uploaded_file.name)
		with open(file_path, "wb") as f:
			f.write(uploaded_file.getbuffer())

		if st.button("Process PDF 🚀"):
			with st.spinner("Processing..."):
				build_vector_db(file_path)
				st.success("Ready! Ask your questions.")
				st.session_state.db_ready = True

# الشات
if "messages" not in st.session_state:
	st.session_state.messages = []

for message in st.session_state.messages:
	with st.chat_message(message["role"]):
		st.markdown(message["content"])

if prompt := st.chat_input("Ask about the lecture..."):
	if "db_ready" not in st.session_state:
		st.error("Upload a file first!")
	else:
		st.session_state.messages.append({"role": "user", "content": prompt})
		with st.chat_message("user"):
			st.markdown(prompt)

		with st.chat_message("assistant"):
			with st.spinner("Thinking..."):
				answer, sources = get_answer_from_llm(prompt)
				st.markdown(answer)
				with st.expander("Sources"):
					for src in sources: st.write(src)
				st.session_state.messages.append({"role": "assistant", "content": answer})