import os
from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredPowerPointLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
# from langchain.chains import RetrievalQA, ConversationalRetrievalChain


current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, 'Book', 'document.pptx')
persistent_file = os.path.join(current_directory, 'db', 'chroma_db')



if not os.path.exists(persistent_file):
    print("persistent directory does not exist initializing vector store")

    if not os.path.exists(file_path):
       raise FileNotFoundError(f'{file_path} not found, please check the path')
    loader = UnstructuredPowerPointLoader(file_path)
    documents = loader.load()

    text_spliter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_spliter.split_documents(documents)

    print("___Document Chuks Information")
    print(f"number of chunk documents: {len(docs)}")
    print(f"show the first chunk:\n {docs[0].page_content}")

    print("Creating embeddings")



    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("finished embeddings")


    db = Chroma.from_documents(
        docs,
        embedding,
        persist_directory=persistent_file
    )

    print("finished creating vector store")

else:
    print('vector store already exists. no need to initialize.')

    

