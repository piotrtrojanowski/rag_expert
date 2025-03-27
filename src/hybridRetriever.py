import numpy as np
from langchain.vectorstores import Chroma
from langchain.schema import BaseRetriever
from ollamaEmbeddings import EmbeddingFunctionBase
from bm25RetrieverWithScores import BM25RetrieverWithScores
from langchain.schema import Document
from typing import List, Optional, Dict, Any

from langchain.schema import Document
from typing import List, Optional, Any, Dict
import numpy as np

class HybridRetriever(BaseRetriever):
    bm25_retriever: BM25RetrieverWithScores = None
    vector_store: BaseRetriever = None
    embedding_function: Optional[Any] = None
    bm25_weight: float = 0.5

    def __init__(self, bm25_retriever: BM25RetrieverWithScores, 
                 vector_retriever: BaseRetriever, 
                 embedding_function: Optional[Any] = None, 
                 bm25_weight: float = 0.5):
        super().__init__()
        self.bm25_retriever = bm25_retriever
        self.vector_store = vector_retriever
        self.embedding_function = embedding_function
        self.bm25_weight = bm25_weight

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        if not scores:
            return []
        min_score, max_score = min(scores), max(scores)
        if max_score == min_score:
            return [0.5] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _get_vector_scores(self, query: str, vector_docs: List[str]) -> List[float]:
        """Compute similarity scores between the query and the vector store documents."""
        # Retrieve query embedding
        query_embedding = self.embedding_function.embed_query(query)
        # Generate embeddings for the vector_docs
        doc_embeddings = [self.embedding_function.embed_query(doc) for doc in vector_docs]  
        # Calculate similarity scores (cosine similarity or dot product)
        vector_scores = [np.dot(query_embedding, doc_embedding) for doc_embedding in doc_embeddings]
        return vector_scores

    def _get_relevant_documents(self, query: str) -> List[str]:
        """Hybrid retrieval with BM25 and Vector scores combined."""
        # Step 1: Get BM25 results with scores
        bm25_results = self.bm25_retriever._get_relevant_documents(query)
        if not bm25_results:
            return []  # No results

        bm25_strings, bm25_scores = zip(*bm25_results)  # Unzip docs and scores

        # Step 2: Retrieve vector-based results for the same query
        vector_results = self.vector_store.get_relevant_documents(query) # chroma does not handle scores
        print("Type of vector_results:", type(vector_results))
        print("First item in vector_results:", vector_results[0])

        vector_strings = [doc.page_content for doc in vector_results]          
        print(vector_strings)
        vector_scores = self._get_vector_scores(query, vector_strings)  # Calculate similarity scores

        # Step 3: Normalize scores
        norm_bm25_scores = self._normalize_scores(list(bm25_scores))
        norm_vector_scores = self._normalize_scores(vector_scores)

        # Step 4: Combine scores using the weight factor
        doc_score_map = {}
        doc_obj_map = {}

        # Add BM25 docs
        for candidateStr, score in zip(bm25_strings, norm_bm25_scores):
            doc_score_map[candidateStr] = score * self.bm25_weight
            doc_obj_map[candidateStr] = candidateStr

        # Add Vector docs
        for candidateStr, score in zip(vector_strings, norm_vector_scores):
            if candidateStr in doc_score_map:
                doc_score_map[candidateStr] += score * (1 - self.bm25_weight)
            else:
                doc_score_map[candidateStr] = score * (1 - self.bm25_weight)
                doc_obj_map[candidateStr] = candidateStr

        # Step 5: Rank documents by combined score
        ranked_docs = sorted(doc_score_map.items(), key=lambda x: x[1], reverse=True)

        # Step 6: Print first 100 characters of each document for debugging
        for i, (doc_content, score) in enumerate(ranked_docs):
            print(f"Doc {i+1}: {doc_content[:100]} (Score: {score:.4f})")

        return [doc_content for doc_content, _ in ranked_docs]

    def invoke(self, query: str, config: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Standard LangChain invoke method."""
        # Step 1: Get the relevant strings (no chunking, pure string list)
        relevant_strings = self._get_relevant_documents(query)
        
        # Step 2: Convert each string to a Document
        documents = [Document(page_content=content) for content in relevant_strings]
        
        return documents
