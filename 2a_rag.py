import os
from pyexpat import model

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredPowerPointLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma



current_directory = os.path.dirname(os.path.abspath(__file__))
book_dir = os.path.join(current_directory, 'Book')
db_dir = os.path.join(current_directory, 'db')
persistent_file = os.path.join(current_directory, 'db', 'chroma_db_with_metadata')

if not os.path.exists(db_dir):
    print("initializing db")

    if not os.path.exists(book_dir):
        raise FileNotFoundError(
            f"the directory {book_dir} not found"
        )
    
    book_files = [f for f in os.listdir(book_dir) if f.endswith(".txt")]

    documents = []

    for book_file in book_files:
        file_path = os.path.join(book_dir, book_file)
        loader = TextLoader(file_path)
        book_docs = loader.load()

        for doc in book_docs:
            doc.metadata = {"source":file_path}
            documents.append(doc)
    
    texst_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap =100)
    docs = texst_splitter.split_documents(documents)

    print("Document chunks information")
    print(f'number of document chunks: {len(docs)}')

    print("creating embeddings")

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("finished creating embeddings")

    print("creating and processing vector store")

    db = Chroma.from_documents(
        docs, embedding, persist_directory=persistent_file
    )


    print("finished creating and processing vector store")

else:
    print('vectore store already created')
