import logging
from langchain_core.language_models.base import BaseLanguageModel
from pdfRetriever import PdfRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class RAGExpert:
    def __init__(self, llm: BaseLanguageModel, pdfRetriever: PdfRetriever):
        self.llm = llm
        self.pdfRetriever = pdfRetriever

        # Set up logger for this class
        self.logger = logging.getLogger(__name__)
        self.logger.info("RAG Expert initialized!")

        self.final_template = """Use the combined context below from PDFs, an LLM, and the internet to answer the question. 
                            Treat the three contexts as three sources of information. If the information is not present in the context say I don't know.

                            Combined Context:
                            {combined_context}

                            Question: {question}
                            Answer:"""

        self.FINAL_PROMPT = PromptTemplate(template=self.final_template, input_variables=["combined_context", "question"])
        self.final_chain = LLMChain(llm=self.llm, prompt=self.FINAL_PROMPT)

    def respond(self, query):
        # Replace this with the actual logic for processing queries
        self.logger.debug(f"Processing query: {query}")

        combined_context = self.create_combined_context(query)
        final_answer = self.final_chain.invoke({"combined_context": combined_context, "question": query})
        
        self.logger.debug(f"Generated response for query: {final_answer}")
        return final_answer
    
    def create_combined_context(self, query):
        # Implement the logic to get the context for the query from all three sources
        self.logger.debug("Fetching context from multiple sources")
        
        pdf_context = self.pdfRetriever.invoke({"query": query})['result']
        self.logger.debug(f"PDF context: {pdf_context}")

        llm_context = self.llm.invoke(query).content
        self.logger.debug(f"LLM context: {llm_context}")

        internet_context = "Internet context" # TODO internet_search.run(query) if internet_search else "Internet search is unavailable."
        self.logger.debug(f"Internet context: {internet_context}")

        combined_context = f"PDF Context:\n{pdf_context}\n\nLLM Context:\n{llm_context}\n\nInternet Context:\n{internet_context}"
        self.logger.debug(f"Combined context: {combined_context}")

        return combined_context