import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

def load_documents(folder_path: str) -> List[Document]:
    """
    Loads all PDF and TXT documents from the specified folder path.

    Args:
        folder_path (str): The path to the directory containing the documents.

    Returns:
        List[Document]: A list of loaded LangChain Document objects.
    """
    documents = []
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The directory {folder_path} does not exist.")

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isdir(file_path):
            continue

        try:
            if filename.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif filename.lower().endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading file {filename}: {e}")

    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Splits a list of documents into smaller chunks using RecursiveCharacterTextSplitter.

    Args:
        documents (List[Document]): The list of documents to split.

    Returns:
        List[Document]: A list of document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)

def load_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Loads the Google Generative AI Embeddings model.
    Requires the GOOGLE_API_KEY environment variable to be set.

    Returns:
        GoogleGenerativeAIEmbeddings: The initialized embeddings model.
    """
    # The model "models/embedding-001" is the standard Google GenAI embedding model.
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def create_vector_store(chunks: List[Document], embeddings: GoogleGenerativeAIEmbeddings) -> FAISS:
    """
    Creates a FAISS vector store from document chunks and embeddings.

    Args:
        chunks (List[Document]): The document chunks to index.
        embeddings (GoogleGenerativeAIEmbeddings): The embedding model to use.

    Returns:
        FAISS: The initialized FAISS vector store.
    """
    return FAISS.from_documents(chunks, embeddings)

def create_retriever(vectorstore: FAISS, k: int = 4) -> VectorStoreRetriever:
    """
    Creates a retriever from the given FAISS vector store.

    Args:
        vectorstore (FAISS): The FAISS vector store.
        k (int): The number of documents to retrieve for each query. Defaults to 4.

    Returns:
        VectorStoreRetriever: The configured retriever object.
    """
    return vectorstore.as_retriever(search_kwargs={"k": k})