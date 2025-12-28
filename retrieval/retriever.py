from typing import List
from langchain_core.documents import Document
from retrieval.vector_store import VectorStore

class Retriever:
    def __init__(self, vector_store: VectorStore):
        """
        Initializes the Retriever.
        
        Args:
            vector_store (VectorStore): The initialized VectorStore instance.
        """
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 3) -> List[Document]:
        """
        Retrieves relevant documents for a given query.
        
        Args:
            query (str): The user query.
            top_k (int): Number of documents to retrieve.
            
        Returns:
            List[Document]: List of relevant documents.
        """
        if not self.vector_store.vector_store:
             raise ValueError("Vector store index is not ready.")

        retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(query)
        return docs
