from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the Embedder with a specific SentenceTransformer model.
        
        Args:
            model_name (str): The name of the model to use.
        """
        print(f"Loading embedding model: {model_name}...")
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        print("Embedding model loaded.")

    def get_embedding_function(self):
        """
        Returns the LangChain embedding function/object.
        """
        return self.embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts.
        
        Args:
            texts (List[str]): List of texts to embed.
            
        Returns:
            List[List[float]]: List of embedding vectors.
        """
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query string.
        
        Args:
            text (str): The query text.
            
        Returns:
            List[float]: The embedding vector.
        """
        return self.embeddings.embed_query(text)
