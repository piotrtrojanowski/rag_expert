import os
import logging

from PyPDF2 import PdfReader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever
from langchain.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from typing import List, Dict
from bm25RetrieverWithScores import BM25RetrieverWithScores
from ollamaEmbeddings import DebuggableOllamaEmbeddings
from hybridRetriever import HybridRetriever

class PdfRetriever:
    def __init__(self, vector_store_directory: str, pdf_file: str, llm: BaseLanguageModel):
            self.logger = logging.getLogger(self.__class__.__name__)
            self.vector_store_directory = vector_store_directory
            self.llm = llm
            self.embedding_function=DebuggableOllamaEmbeddings(model="mxbai-embed-large")
        
            """Loads or creates a Chroma vector store efficiently."""
            db_file = os.path.join(self.vector_store_directory, "chroma.sqlite3")
            try:
                if os.path.exists(db_file):
                    self.loadPersistedPdfRepresentation()
                else:
                    self.logger.info(f"db file does not exist in: {db_file}. Creating new Chroma vector store in: {self.vector_store_directory}")
                    self.createAndPersistPdfRepresentation(pdf_file)

                self.createRetrievalChain()

            except Exception as e:
                self.logger.error(f"An error occurred: {e}")
                raise

    def invoke(self, query):
        self.logger.debug(f"Processing PDF query: {query}")
        # use the chain of llm + hybrid retriever for retrieval
        result = self.pdf_retrieval_qa.invoke(query)
        print (f"PDF Retriever response: ", result)
        self.logger.debug(f"Retrieved PDF result: {result}")
        return result
    
    def invoke_no_chains(self, query):
        self.logger.debug(f"Processing PDF query: {query}")
        # Retrieve relevant documents using HybridRetriever
        retrieved_docs = self.hybrid_retriever.invoke(query)
        # Extract content from retrieved documents
        retrieved_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        if not retrieved_text.strip():
            self.logger.debug("No relevant documents found.")
            return "I'm sorry, but I couldn't find any relevant information in the document."

        # Format the retrieved content for the LLM
        llm_input = self.QA_PROMPT.format(context=retrieved_text, question=query)
        # Call the LLM to generate an answer strictly based on retrieved content
        llm_response = self.llm.invoke(llm_input)
        self.logger.debug(f"Retrieved PDF result: {llm_response.pretty_repr()}")
        return llm_response.pretty_repr()

    # ==== AUXILIARY PRIVATE FUNCTIONS ====
    def createRetrievalChain(self):
        self.createHybridRetriever()

        self.pdf_qa_template = """You are a strict assistant who must only use the provided context to answer the question.
                                    If the context does not contain the answer, respond with "I don't know."
                                    Here is the Context retrieved from the source pdf document relevant to the question at the end.
                                    Please summarize the text based strictly on the provided.
                                    Do not add new information. Do not search the internet. Do not add anything from yourself. 
                                    Refine and format text. Potentially make it more concise. Do not infer meaning beyond the provided words.
        Context:
                                    
        {context}

        Question: {question}
        Answer:"""

        self.QA_PROMPT = PromptTemplate(template=self.pdf_qa_template, input_variables=["context", "question"])

        self.pdf_retrieval_qa = RetrievalQA.from_chain_type(
                                        llm=self.llm, 
                                        chain_type="stuff", #"map_reduce", 
                                        retriever=self.hybrid_retriever)
                                                
    def createHybridRetriever(self):
        self.chroma_retriever = self.vector_store.as_retriever()
        # Combine both retrievers into a HybridRetriever
        self.hybrid_retriever = HybridRetriever(
            bm25_retriever=self.bm25_retriever, 
            vector_retriever=self.chroma_retriever,
            embedding_function=self.embedding_function,
            bm25_weight=0.5)
    
    def loadPersistedPdfRepresentation(self):
        self.vector_store = Chroma(persist_directory=self.vector_store_directory, 
                                   embedding_function=self.embedding_function)
        self.logger.info("Reassigned embedding function after loading the vector store.")
        self.log_vector_store_summary()
        #load bm25 representation
        self.bm25_retriever = BM25RetrieverWithScores.load(os.path.join(self.vector_store_directory, "bm25_retriever.pkl"))
        print("BM25 instance:", self.bm25_retriever)
    

    def createAndPersistPdfRepresentation(self, pdf_file):
        pdfReader = PdfReader(pdf_file)
        pdf_summary = self.get_pdf_summary(pdfReader)
        self.log_pdf_summary(pdf_summary)
        
        raw_text = ''
        self.logger.info(f"Enumerate over: {pdfReader.pages}...")
        for i, page in enumerate(pdfReader.pages):
            self.logger.info(f"Processing page {i+1} of {len(pdfReader.pages)}")
            text = page.extract_text()
            if text:
                raw_text += text

        self.logger.info(f"Length of raw text: {len(raw_text)}")
        self.logger.info(f"Extracted raw text: {raw_text[:500]}...")  # Log the first 500 characters to check the extraction

        if not raw_text:
            self.logger.info("raw_text is empty. Cannot create a vector store.")

        textSplitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", ". ", "\n"],
            chunk_size=2000,
            chunk_overlap=400,
            length_function=len)
        self.textChunks = textSplitter.split_text(raw_text)
        self.logger.info(f"Length of textChunks: {len(self.textChunks)}")
        
        self.vector_store = Chroma.from_texts(
            texts=self.textChunks,
            embedding=self.embedding_function,
            persist_directory=self.vector_store_directory)
        
        self.log_vector_store_summary()
        self.logger.info(f"Persisting new Chroma vector store to: {self.vector_store_directory}")
        self.vector_store.persist()
        
        # create and save_bm25_retriever        
        self.bm25_retriever = BM25RetrieverWithScores.from_texts(self.textChunks)
        self.bm25_retriever.save(os.path.join(self.vector_store_directory, "bm25_retriever.pkl"))

    def log_vector_store_summary(self):
        """Displays a summary of the Chroma vector store, including embedding function info."""
        try:
            collection = self.vector_store._collection  # Access the underlying collection
            count = collection.count()
            self.logger.info(f"Vector store contains {count} embeddings.")

            # Display embedding function information
            embedding_function = self.vector_store._embedding_function
            self.logger.info(f"Embedding Function Type: {type(embedding_function).__name__}")
            if hasattr(embedding_function, "model_name"):
                self.logger.info(f"Model Name: {embedding_function.model_name}")
            elif hasattr(embedding_function, "model"):
                self.logger.info(f"Model: {embedding_function.model}")

            metadata = collection.metadata
            if metadata:
                self.logger.info("Metadata:")
                for key, value in metadata.items():
                    self.logger.info(f"  {key}: {value}")
            else:
                self.logger.info("No metadata available")

            if count > 0:
                peek = collection.peek()
                self.logger.info("First few documents:")
                for doc in peek['documents'][:min(5, count)]:
                    self.logger.info(f"  {doc[:100]}...")
            else:
                self.logger.info("No documents available")

        except Exception as e:
            self.logger.error(f"Error displaying vector store summary: {e}")

    def get_pdf_summary(self, pdf_reader: PdfReader):
        try:
            info = pdf_reader.metadata
            num_pages = len(pdf_reader.pages)

            summary = {
                "title": info.title,
                "author": info.author,
                "creator": info.creator,
                "producer": info.producer,
                "subject": info.subject,
                "num_pages": num_pages,
            }
            return summary

        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}")
            return None

    def log_pdf_summary(self, summary):
        if summary:
            self.logger.info("PDF Summary:")
            for key, value in summary.items():
                if value:  # Check if value is not None
                    self.logger.info(f"  {key}: {value}")
        else:
            self.logger.info("Could not retrieve PDF summary.")
