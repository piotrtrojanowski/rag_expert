import os
import logging

from PyPDF2 import PdfReader
from langchain.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel

class PdfRetriever:
    def __init__(self, vector_store_directory: str, pdf_file: str, llm: BaseLanguageModel):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.vector_store_directory = vector_store_directory
        self.llm = llm
        """Loads or creates a Chroma vector store efficiently."""
        db_file = os.path.join(vector_store_directory, "chroma.sqlite3")
        try:
            if os.path.exists(db_file):
                self.logger.info(f"Loading existing Chroma vector store from: {vector_store_directory}")
                self.vector_store = Chroma(persist_directory=vector_store_directory)
                #re-assign the embedding function manually since it is not loaded from the file (bug I suppose)
                self.vector_store._embedding_function = self.choose_embedding_function()
                self.logger.info("Reassigned embedding function after loading the vector store.")
        
                '''if (self.vector_store._embedding_function.__class__ != embedding_function.__class__):
                    print("Embedding function has changed. Recreating vectorstore.")
                    shutil.rmtree(vector_store_directory)
                    raise FileNotFoundError #to trigger recreation'''
            else:
                self.logger.info(f"db file does not exist in: {db_file}. Creating new Chroma vector store in: {vector_store_directory}")

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
                    separators=["\n\n", "\n", " "],
                    chunk_size=2000,
                    chunk_overlap=400,
                    length_function=len
                )
                self.textChunks = textSplitter.split_text(raw_text)
                self.logger.info(f"Length of textChunks: {len(self.textChunks)}")
                
                self.vector_store = Chroma.from_texts(
                    texts=self.textChunks,
                    embedding=self.choose_embedding_function(),
                    persist_directory=self.vector_store_directory
                )

                self.logger.info(f"Persisting new Chroma vector store to: {vector_store_directory}")
                self.vector_store.persist()

            self.retriever = self.vector_store.as_retriever()
            self.log_vector_store_summary()

            self.pdf_qa_template = """Here is the Context retrieved from the source document relevant to the question at the end.
                                      Please summarize the text based strictly on the provided sentences from the document.
                                      Do not add new information. Refine and format text, and do not infer meaning beyond the provided words.
            Context:
            {context}

            Question: {question}
            Answer:"""

            self.QA_PROMPT = PromptTemplate(template=self.pdf_qa_template, input_variables=["context", "question"])

            self.pdf_retrieval_qa = RetrievalQA.from_chain_type(
                                            llm=self.llm, 
                                            chain_type="map_reduce", 
                                            retriever=self.retriever) 
                                            #chain_type_kwargs={"llm_chain_kwargs": {"prompt": self.QA_PROMPT}})

        except Exception as e:
            self.logger.error(f"An error occurred with the Chroma vector store: {e}")
            raise

    def invoke(self, query):
        self.logger.debug(f"Processing PDF query: {query}")
        # use the chain of llm + vector_store for retrieval
        result = self.pdf_retrieval_qa.invoke(query)
        self.logger.debug(f"Retrieved PDF result: {result}")
        return result

    # Only necessary before the vector store was created
    def choose_embedding_function(self):
        # Initialize the embedding function
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        return embeddings
    
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
