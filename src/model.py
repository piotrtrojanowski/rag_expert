# Model: Manages data
import logging
from langchain_ollama.chat_models import ChatOllama
from pdfRetriever import PdfRetriever
from ragExpert import RAGExpert

class Model:
    def __init__(self, use_pdf_source, use_llm_source, use_internet_source, final_touch_with_llm):
        # Set up logger for this class
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Model...")

        try:
            self.llm = ChatOllama(model="llama2")
            #self.llm = ChatOllama(model="mistral")
            #self.llm = ChatOllama(model="gemma")
            #self.llm = ChatOllama(model="tinyllama")
            
            self.logger.debug(f"LLM created: {self.llm.invoke("What is the capital of France?")}")
            self.logger.info("LLM proxy initialized")

            vector_store_directory = "/Users/piotrtrojanowski/Documents/book/RAG/vector_store_good_companies"
            pdf_file = "/Users/piotrtrojanowski/Documents/book/RAG/GoodCompanies.pdf"
            #pdf_file = "data/steve-jobs-stanford-university-commencement-speech.pdf"
            #pdf_file = "data/Steve-Jobs-autobiography-book.pdf"
            self.pdfRetriever = PdfRetriever(vector_store_directory, pdf_file, self.llm)
            self.logger.info("PDF Retriever initialized")

            self.ragExpert = RAGExpert(self.llm, self.pdfRetriever, 
                                       use_pdf_source, 
                                       use_llm_source, 
                                       use_internet_source, 
                                       final_touch_with_llm)
            self.logger.info("RAG Expert initialized")

        except Exception as e:
            self.logger.error(f"Error during Model initialization: {e}")
            raise

    def generate_response(self, query):
        try:
            self.logger.debug(f"Processing query: {query}")
            response = self.ragExpert.respond(query)
            self.logger.debug(f"Generated response: {response}")
            return response
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return f"Error processing query: {str(e)}"

