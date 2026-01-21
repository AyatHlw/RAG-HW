from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DB_PATH = "vectorstore"


def inspect_chunks():
	embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

	vector_db = Chroma(
		persist_directory=DB_PATH,
		embedding_function=embedding_function
	)

	data = vector_db.get(limit=3)

	print(f"Total chunks stored: {len(vector_db.get()['ids'])}")
	print("-" * 30)

	if data['documents']:
		for i, doc in enumerate(data['documents']):
			meta = data['metadatas'][i]
			print(f"CHUNK {i + 1}:")
			print(f"Source: {meta.get('source')} | Page: {meta.get('page')}")
			print(f"Content Preview: {doc[:150]}...")
			print("-" * 30)


if __name__ == "__main__":
	inspect_chunks()