import os
import logging

from langchain_ollama import OllamaEmbeddings

from typing import Protocol

class EmbeddingFunctionBase(Protocol):
    def embed_query(self, text: str):
        """Embed a single query."""
        ...

    def embed_documents(self, texts: list):
        """Embed multiple documents."""
        ...

class DebuggableOllamaEmbeddings (EmbeddingFunctionBase):
    def __init__(self, model="default_model"):
        self.embedding_model = OllamaEmbeddings(model=model)
        self.logger = logging.getLogger(__name__)

    def embed_query(self, text):
        """Embed a single query with debugging logs."""
        embedding = self.embedding_model.embed_query(text)
        self.logger.info(f"Embedding for query '{text[:50]}...': {embedding[:5]}...")
        return embedding

    def embed_documents(self, texts):
        """Embed multiple documents one at a time and show progress with detailed debugging logs."""
        self.logger.info(f"Starting to embed {len(texts)} documents one by one.")
        print(f"Starting to embed {len(texts)} documents one by one.")  # Added print

        embeddings = []
        
        # Process each document one at a time
        for i, text in enumerate(texts):
            self.logger.info(f"Processing document {i + 1}/{len(texts)}: '{text[:50]}...'")
            print(f"Processing document {i + 1}/{len(texts)}")

            try:
                # Embed the document
                embedding = self.embedding_model.embed_documents([text])[0]  # Embedding one document at a time
                embeddings.append(embedding)

                # Log the result of embedding for this document
                self.logger.info(f"Embedding {i + 1}/{len(texts)} completed: {embedding[:5]}...")
                # Show progress after every document
                self.logger.info(f"Progress: {i + 1}/{len(texts)} documents processed.")
                print(f"Progress: {i + 1}/{len(texts)} documents processed.")

            except Exception as e:
                self.logger.error(f"Error embedding document {i + 1}: {str(e)}")
                print(f"Error embedding document {i + 1}: {str(e)}")
                continue  # Skip this document and proceed with the next one

        self.logger.info(f"Embedding of all {len(texts)} documents completed.")
        print(f"Embedding of all {len(texts)} documents completed.")

        return embeddings

        '''w/o debugging
           """Embed multiple documents with debugging logs."""
            embeddings = self.embedding_model.embed_documents(texts)
             for i, (text, embedding) in enumerate(zip(texts, embeddings)):
              self.logger.info(f"Embedding {i+1}/{len(texts)} for doc '{text[:50]}...': {embedding[:5]}...")
        return embeddings '''
