import pickle
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from langchain.schema import Document

class BM25RetrieverWithScores:
    def __init__(self, documents: List[str], top_k: int = 10):
        """Initialize BM25 retriever with documents."""
        self.k = top_k
        self.docs = documents
        self.tokenized_docs = [doc.split() for doc in self.docs]  # Tokenize
        self.bm25 = BM25Okapi(self.tokenized_docs)  # Initialize BM25

    def _get_relevant_documents(self, query: str) -> List[Tuple[Document, float]]:
        """Retrieve the top-k BM25-ranked documents along with their scores."""
        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)  # Get BM25 similarity scores
        # Create a list of (document, score) tuples
        scored_docs = [(self.docs[i], scores[i]) for i in range(len(scores))]
        # Sort by score, descending
        top_k_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)[:self.k]
        # Return the documents and their scores (not just the documents)
        return top_k_docs

    @classmethod
    def from_texts(cls, texts: List[str], top_k: int = 10):
        """Create BM25RetrieverWithScores from raw text data."""
        return cls(documents=texts, top_k=top_k)

    # Save method with 'k' serialized
    def save(self, file_path: str):
        """Save BM25 retriever to a file."""
        with open(file_path, 'wb') as f:
            pickle.dump({'k': self.k, 'docs': self.docs}, f)

    # Load method with 'k' restored
    @classmethod
    def load(cls, file_path: str):
        """Load BM25 retriever from a file."""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        # Recreate the retriever with the saved 'k' and docs
        return cls(documents=data['docs'], top_k=data['k'])
