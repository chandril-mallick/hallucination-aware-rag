import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from embeddings.embedder import Embedder

class VectorStore:
    def __init__(self, embedder: Embedder):
        """
        Initializes the VectorStore with an Embedder instance.
        
        Args:
            embedder (Embedder): The embedding model wrapper.
        """
        self.embedder = embedder.get_embedding_function()
        self.vector_store = None

    def create_index(self, documents: List[Document]):
        """
        Creates a FAISS index from a list of documents.
        
        Args:
            documents (List[Document]): List of chunked documents.
        """
        print(f"Creating FAISS index for {len(documents)} chunks...")
        self.vector_store = FAISS.from_documents(documents, self.embedder)
        print("FAISS index created.")

    def save_index(self, folder_path: str):
        """
        Saves the FAISS index to the specified folder.
        
        Args:
            folder_path (str): The folder path to save the index.
        """
        if self.vector_store:
            self.vector_store.save_local(folder_path)
            print(f"Index saved to {folder_path}")
        else:
            print("No index to save.")

    def load_index(self, folder_path: str):
        """
        Loads a FAISS index from the specified folder.
        
        Args:
            folder_path (str): The folder path containing the index.
        """
        print(f"Loading index from {folder_path}...")
        self.vector_store = FAISS.load_local(folder_path, self.embedder, allow_dangerous_deserialization=True)
        print("Index loaded.")

    def as_retriever(self, search_kwargs: dict = None):
        """
        Returns a retriever object from the vector store.
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Load or create an index first.")
        
        kwargs = search_kwargs or {"k": 3}
        return self.vector_store.as_retriever(search_kwargs=kwargs)
