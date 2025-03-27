import pickle
from langchain.retrievers.bm25 import BM25Retriever
from langchain.schema import Document

class BM25RetrieverWrapper:
    def create_from_texts(self, texts: list):
        self.bm25_retriever = BM25Retriever.from_texts(texts)

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """Retrieve documents using BM25 retrieval."""
        bm25_docs = self.bm25_retriever.get_relevant_documents(query)
        return bm25_docs  # Return the list of BM25-retrieved documents
    
    def as_retriever(self):
        # Make sure that bm25_retriever is returned as a retriever
        return self.bm25_retriever
    
    def save(self, file_path: str):
        """Save the BM25 retriever to a file."""
        with open(file_path, 'wb') as f:
            pickle.dump(self.bm25_retriever, f)
            
    def load(self, file_path: str):
        """Load a BM25 retriever from a saved file."""
        with open(file_path, 'rb') as f:
            self.bm25_retriever = pickle.load(f)
        
        print("BM25 Retriever af the end of loading:", self.bm25_retriever)


'''
# Example Usage
docs = [
    Document(page_content="Machine learning is amazing."),
    Document(page_content="Deep learning is a subset of machine learning."),
    Document(page_content="Natural language processing enables AI models to understand text."),
]


# Initialize and save
bm25_wrapper = BM25RetrieverWrapper(docs)
bm25_wrapper.save()

# Reload and use
bm25_wrapper.load()
results = bm25_wrapper.retrieve("What is AI?", top_k=2)
for doc in results:
    print(doc.page_content)
'''