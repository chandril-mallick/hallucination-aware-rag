import os
from typing import List, Dict
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

class Generator:
    def __init__(self, model_name: str = "mistral"):
        """
        Initializes the LLM Generator.
        
        Args:
            model_name (str): The name of the Ollama model to use.
        """
        self.llm = ChatOllama(model=model_name, temperature=0.1)
        
        # Define the system prompt ensuring strict adherence to context
        self.prompt = ChatPromptTemplate.from_template("""
        You are a trustworthy AI assistant. Answer the user's question based ONLY on the following context.
        I will provide you with a list of context passages, each with an ID. 
        
        Rules:
        1. You must answer the question using ONLY the provided context.
        2. If the answer is not in the context, say "I cannot answer this based on the provided documents."
        3. You must cite the source ID for every sentence you generate. Format: [Source ID]
        4. Do not hallucinate. If you are unsure, state it.
        
        Context:
        {context}
        
        Question: 
        {question}
        
        Answer:
        """)
        
        self.chain = self.prompt | self.llm | StrOutputParser()

    def format_context(self, docs: List[Document]) -> str:
        """
        Formats retrieved documents into a string with IDs.
        """
        formatted_context = ""
        for i, doc in enumerate(docs):
            source = os.path.basename(doc.metadata.get('source', 'unknown'))
            page = doc.metadata.get('page', 0)
            formatted_context += f"[Source {i}] (File: {source}, Page: {page}): {doc.page_content}\n\n"
        return formatted_context

    def generate_answer(self, query: str, context_docs: List[Document]) -> Dict:
        """
        Generates an answer based on the query and context.
        
        Args:
            query (str): The user's question.
            context_docs (List[Document]): The retrieved documents.
            
        Returns:
            Dict: Dictionary containing the answer and input context.
        """
        context_str = self.format_context(context_docs)
        
        response = self.chain.invoke({
            "context": context_str,
            "question": query
        })
        
        return {
            "answer": response,
            "context_str": context_str,
            "context_docs": context_docs
        }
