import re
import numpy as np
from typing import List, Dict
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import nltk
import ssl

# Workaround for NLTK SSL issues on macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Ensure NLTK data (punkt) is available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

class HallucinationDetector:
    def __init__(self, embedder, llm_generator=None):
        """
        Initializes the HallucinationDetector.
        
        Args:
            embedder: An instance of the Embedder class.
            llm_generator: An instance of the Generator class (for verify_claims).
        """
        self.embedder = embedder
        self.llm = llm_generator

    def split_into_sentences(self, text: str) -> List[str]:
        return sent_tokenize(text)

    def check_source_attribution(self, answer: str, context_docs: List[Document]) -> Dict:
        """
        Technique 1: Source Attribution Check
        Verifies if citations in the answer (e.g. [Source 0]) exist in the retrieved docs.
        """
        sentences = self.split_into_sentences(answer)
        results = []
        flagged_sentences = []

        all_citations = []
        
        for sent in sentences:
            # Find all patterns like [Source X]
            citations = re.findall(r'\[Source (\d+)\]', sent)
            all_citations.extend(citations)
            
            supported = False
            if citations:
                # Check if cited indices exist in context docs (naive check)
                # In a real system, we'd check if the content actually matches
                valid_citations = [c for c in citations if int(c) < len(context_docs)]
                if valid_citations:
                    supported = True
            
            # If no citation AND it's a factual claim (heuristic: long enough), flag it
            # Or if citations refer to non-existent docs
            if not citations or not supported:
                 # Allow short conversational fillers without citation
                if len(sent.split()) > 4: 
                    flagged_sentences.append(sent)
                    results.append({"sentence": sent, "status": "unsupported", "detail": "Missing or invalid citation"})
                else:
                    results.append({"sentence": sent, "status": "supported", "detail": "Conversational/Short"})
            else:
                results.append({"sentence": sent, "status": "supported", "citations": citations})

        score = 1.0 - (len(flagged_sentences) / len(sentences)) if sentences else 0.0
        
        return {
            "score_attribution": score,
            "results": results
        }

    def check_semantic_similarity(self, answer: str, context_docs: List[Document], threshold: float = 0.5) -> Dict:
        """
        Technique 2: Semantic Similarity Validation
        Embeds answer sentences and context chunks, checks for high cosine similarity.
        """
        sentences = self.split_into_sentences(answer)
        if not sentences:
            return {"score_similarity": 0.0, "details": []}

        # Embed all context chunks once
        context_texts = [doc.page_content for doc in context_docs]
        context_embeddings = self.embedder.embed_documents(context_texts)
        
        # Embed answer sentences
        sent_embeddings = self.embedder.embed_documents(sentences)
        
        details = []
        supported_count = 0
        
        for i, sent in enumerate(sentences):
            # Calculate sim against all context chunks
            sims = cosine_similarity([sent_embeddings[i]], context_embeddings)[0]
            max_sim = np.max(sims)
            
            status = "supported" if max_sim >= threshold else "hallucinated"
            if status == "supported":
                supported_count += 1
            
            details.append({
                "sentence": sent,
                "max_similarity": float(max_sim),
                "status": status
            })
            
        score = supported_count / len(sentences)
        
        return {
            "score_similarity": score,
            "details": details
        }

    def verify_claims_agent(self, answer: str, context_docs: List[Document]) -> Dict:
        """
        Technique 3: Claim Verification Agent (LLM-based)
        Asks the LLM to verify if the answer is supported by the context.
        This is a 'Faithfulness' check.
        """
        if not self.llm:
            return {"score_faithfulness": 0.0, "reason": "No LLM provided"}

        context_text = "\n\n".join([d.page_content for d in context_docs])
        
        # Simplified prompt for verification
        prompt = f"""
        You are a fact-checking agent. verify if the following Answer is fully supported by the Context.
        
        Context:
        {context_text[:4000]} # Truncate to avoid context limit
        
        Answer:
        {answer}
        
        Task:
        Identify any sentences in the Answer that are NOT supported by the Context.
        Return your response in this format:
        Supported: [Yes/No]
        Unsupported Sentences: [List of sentences or "None"]
        Reasoning: [Brief explanation]
        """
        
        # We invoke the LLM from the generator instance directly (accessing its internal llm/chain logic)
        # Using invoke on the underlying LLM object for raw string output
        response = self.llm.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        is_supported = "Supported: Yes" in content
        
        return {
            "score_faithfulness": 1.0 if is_supported else 0.5, # Simplified scoring
            "agent_response": content
        }
