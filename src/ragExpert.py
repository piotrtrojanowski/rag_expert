import logging
import os
from langchain_core.language_models.base import BaseLanguageModel
from pdfRetriever import PdfRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.utilities import SerpAPIWrapper

class RAGExpert:
    def __init__(self, llm: BaseLanguageModel, pdfRetriever: PdfRetriever,
                 use_pdf_source, use_chain, use_llm_source, use_internet_source, final_touch_with_llm):
        self.llm = llm
        self.pdfRetriever = pdfRetriever
        self.use_pdf_source=use_pdf_source
        self.use_chain = use_chain
        self.use_llm_source=use_llm_source
        self.use_internet_source=use_internet_source
        self.final_touch_with_llm=final_touch_with_llm
        self.use_chain=use_chain
        
        # Set up logger for this class
        self.logger = logging.getLogger(__name__)

        #Initialize the internet searches - SerpAPI - a tool for internet search (Crucially with the wrapper)
        self.serpapi_api_key = os.environ.get("SERPAPI_API_KEY") #does not work, so workaround below
        if self.serpapi_api_key:
            self.internet_search = SerpAPIWrapper(serpapi_api_key=self.serpapi_api_key)
        else:
            print("SERPAPI_API_KEY environment variable not set. Internet searches will be unavailable.")
            self.internet_search = [] # search is unavailable if no API key

        self.final_template = """Use the combined context below from PDFs, an LLM, and the internet to answer the question. 
                            Treat the three contexts as three sources of information. If the information is not present in the context say I don't know.

                            Combined Context:
                            {combined_context}

                            Question: {question}
                            Answer:"""

        self.FINAL_PROMPT = PromptTemplate(template=self.final_template, input_variables=["combined_context", "question"])
        self.final_chain = LLMChain(llm=self.llm, prompt=self.FINAL_PROMPT)

        self.logger.info("RAG Expert initialized!")

    def respond(self, query):
        self.logger.debug(f"Processing query: {query}")
        combined_context = self.create_combined_context(query)
        
        response_text = combined_context  # Default response

        if self.final_touch_with_llm:
            raw_response = self.final_chain.invoke({
                "combined_context": combined_context, 
                "question": query
            })     
            # Ensure raw_response is a dictionary before accessing 'text'
            if isinstance(raw_response, dict) and 'text' in raw_response:
                response_text = raw_response['text']
            else:
                response_text = str(raw_response)  # Fallback to string conversion

        # Clean and format the response
        clean_response = self.clean_text_formatting(response_text)

        self.logger.debug(f"Generated response for query: {clean_response}")
        return clean_response
    
    def clean_text_formatting(self, text):
        """Clean up text formatting markers and standardize spacing"""
        if not text:
            return text
            
        # Replace literal \n with actual line breaks
        text = text.replace('\\n', '\n')
        
        # Replace numbered list markers like \n1. with proper formatting
        import re
        text = re.sub(r'\\n(\d+\.)', r'\n\1', text)
        
        # Standardize multiple line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove any leading/trailing whitespace
        text = text.strip()
        
        return text

    def create_combined_context(self, query):
        self.logger.debug("Fetching context from multiple sources")
    
        contexts = []
        
        # PDF Context: Only if USE_PDF is True
        if self.use_pdf_source:
            pdf_context = []
            if self.use_chain:
                pdf_context = self.pdfRetriever.invoke({"query": query})['result']
            else:
                pdf_context = self.pdfRetriever.invoke_no_chains(query)
            
            if pdf_context.strip():
                clean_pdf = self.clean_text_formatting(pdf_context)
                contexts.append(("Context retrieved from the pdf: ", clean_pdf))
        
        # LLM Context: Only if USE_LLM_CONTEXT is True
        if self.use_llm_source:
            try:
                llm_context = self.llm.invoke(query).content
                if llm_context.strip():
                    clean_llm = self.clean_text_formatting(llm_context)
                    contexts.append(("Context retrieved from llm: ", clean_llm))
            except Exception as e:
                self.logger.error(f"Error fetching LLM context: {str(e)}")
        
        # Internet Context: Only if USE_INTERNET is True
        if self.use_internet_source:
            try:
                internet_context = self.internet_search.run(query) if self.internet_search else ""
                if internet_context.strip():
                    clean_internet = self.clean_text_formatting(internet_context)
                    contexts.append(("Context retrieved from the Internet: ", clean_internet))
            except Exception as e:
                self.logger.error(f"Error fetching Internet context: {str(e)}")
        
        # Combine only non-empty contexts
        if contexts:
            combined_context = "\n\n".join(
                f"{source}:\n{content}" 
                for source, content in contexts
            )
        else:
            combined_context = "No relevant context found."
        
        self.logger.debug(f"Combined context:\n{combined_context}")
        return combined_context