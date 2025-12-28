from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document

class MetricsCalculator:
    def __init__(self, embedder):
        """
        Initializes the MetricsCalculator.
        
        Args:
            embedder: Instance of Embedder class.
        """
        self.embedder = embedder

    def calculate_metrics(self, query: str, answer: str, context_docs: List[Document], hallucination_results: Dict = None) -> Dict:
        """
        Computes various RAG evaluation metrics.
        """
        metrics = {}
        
        # 1. Faithfulness (based on hallucination check or similarity)
        # Use semantic similarity score as a proxy for faithfulness if explicit check missing
        if hallucination_results and 'score_similarity' in hallucination_results:
            metrics['faithfulness'] = hallucination_results['score_similarity']
        else:
            # Fallback: check if answer is similar to context
            metrics['faithfulness'] = self._calculate_similarity(answer, [d.page_content for d in context_docs])

        # 2. Answer Relevance (Cosine sim between Query and Answer)
        metrics['answer_relevance'] = self._calculate_pair_similarity(query, answer)

        # 3. Context Precision (relevance of retrieved docs to query)
        # Average similarity of top retrieved chunks to the query
        doc_contents = [d.page_content for d in context_docs]
        metrics['context_precision'] = self._calculate_similarity(query, doc_contents)

        # 4. Answer Completeness (Heuristic based on length and relevance)
        # A very short answer might be incomplete.
        length_score = min(len(answer.split()) / 50.0, 1.0) # Cap at 1.0 for ~50 words
        metrics['answer_completeness'] = (length_score + metrics['answer_relevance']) / 2.0

        return metrics

    def _calculate_pair_similarity(self, text1: str, text2: str) -> float:
        embeddings = self.embedder.embed_documents([text1, text2])
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(sim)

    def _calculate_similarity(self, source_text: str, target_texts: List[str]) -> float:
        if not target_texts:
            return 0.0
        
        source_emb = self.embedder.embed_query(source_text)
        target_embs = self.embedder.embed_documents(target_texts)
        
        sims = cosine_similarity([source_emb], target_embs)[0]
        return float(np.mean(sims))
