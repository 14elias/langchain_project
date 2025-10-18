from re import search
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma



db = Chroma(
    persist_directory="db/chroma_db",
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)

retriever = db.as_retriever(
    search_type="similarity",          # use similarity instead of threshold filter
    search_kwargs={"k": 2}
)

docs = retriever.invoke("What is constructivism?")
if not docs:
    print("No relevant documents found.")
else:
    print(docs[0].page_content)