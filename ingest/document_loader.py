import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentLoader:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initializes the DocumentLoader with chunking parameters.
        
        Args:
            chunk_size (int): The maximum size of each text chunk.
            chunk_overlap (int): The overlap between consecutive chunks.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Loads documents from the provided file paths and splits them into chunks.
        
        Args:
            file_paths (List[str]): List of paths to files (PDF or TXT).
            
        Returns:
            List[Document]: A list of chunked Document objects.
        """
        all_docs = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"Warning: File not found {file_path}")
                continue
                
            file_ext = os.path.splitext(file_path)[1].lower()
            
            try:
                if file_ext == ".pdf":
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                elif file_ext == ".txt":
                    loader = TextLoader(file_path)
                    docs = loader.load()
                else:
                    print(f"Warning: Unsupported file type {file_ext} for {file_path}")
                    continue
                
                # Split documents into chunks
                chunks = self.text_splitter.split_documents(docs)
                all_docs.extend(chunks)
                print(f"Loaded {len(chunks)} chunks from {file_path}")
                
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                
        return all_docs

if __name__ == "__main__":
    # Test block
    loader = DocumentLoader()
    # Create a dummy file for testing if needed
    # print(loader.load_documents(["./data/sample.txt"]))
