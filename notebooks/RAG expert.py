# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
import joblib
import os

from langchain.chat_models import ChatOllama
llm = ChatOllama(model="llama2")
# Example usage
response = llm.invoke("What is the capital of France?")
print(response.content)


# +
def get_pdf_summary(pdf_reader):
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
        print(f"Error processing PDF: {e}")
        return None

def print_pdf_summary(summary):
    if summary:
        print("PDF Summary:")
        for key, value in summary.items():
            if value: #Check if value is not None
                print(f"  {key}: {value}")
    else:
        print("Could not retrieve PDF summary.")



# -

pdfReader=PdfReader("data/SteveJobs autobiography book.pdf")
# steve-jobs-stanford-university-commencement-speech.pdf
summary = get_pdf_summary(pdfReader)
print_pdf_summary(summary)

# +
data_directory = "data"
vector_store_directory = "vector_store"

from langchain_ollama import OllamaEmbeddings


# -

#Only necessary before the vector store was created
def choose_embedding_function ():
    # Initialize the embedding function
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return embeddings


# +
import os
from typing import List
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter    

def load_or_create_chroma(
    pdfReader: PdfReader,
    vector_store_directory: str
) -> Chroma:
    """Loads or creates a Chroma vector store efficiently."""
    db_file = os.path.join(vector_store_directory, "chroma.sqlite3")
    try:
        if os.path.exists(db_file):
            print(f"Loading existing Chroma vector store from: {vector_store_directory}")
            vector_store = Chroma(persist_directory=vector_store_directory)
            
            '''if (vector_store._embedding_function.__class__ != embedding_function.__class__):
                print("Embedding function has changed. Recreating vectorstore.")
                shutil.rmtree(vector_store_directory)
                raise FileNotFoundError #to trigger recreation'''
        else:
            print(f"db file does not exist in: {db_file}")
            raise FileNotFoundError #to trigger creation

    except FileNotFoundError:
        print(f"Received request to recreate vector store. Creating new Chroma vector store in: {vector_store_directory}")

        raw_text=''
        for i, page in enumerate(pdfReader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
    
        if not raw_text:
            print("raw_text is empty. Cannot create a vector store.")
    
        textSplitter = RecursiveCharacterTextSplitter (
            separators=["\n\n", "\n", " "],
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        textChunks = textSplitter.split_text(raw_text)
            
        vector_store = Chroma.from_texts(
            texts=textChunks,
            embedding=choose_embedding_function(),
            persist_directory=vector_store_directory
        )

        print(f"Persisting new Chroma vector store to: {vector_store_directory}")
        try:
            vector_store.persist()
        except Exception as e:
            print(f"An error occurred with the Chroma vector store: {e}")
            raise
        
    return vector_store


# -

def display_vector_store_summary(vector_store: Chroma):
    """Displays a summary of the Chroma vector store, including embedding function info."""
    try:
        collection = vector_store._collection  # Access the underlying collection
        count = collection.count()
        print(f"Vector store contains {count} embeddings.")

        # Display embedding function information
        embedding_function = vector_store._embedding_function
        print("Embedding Function:")
        print(f"  Type: {type(embedding_function).__name__}")
        if hasattr(embedding_function, "model_name"):
            print(f"  Model Name: {embedding_function.model_name}")
        elif hasattr(embedding_function, "model"): #For sentence transformers
            print(f"  Model: {embedding_function.model}")
        # Add other relevant embedding function attributes as needed

        metadata = collection.metadata
        if metadata:
            print("Metadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        else:
          print("No metadata available")

        if count > 0:
            peek = collection.peek()
            print("First few documents:")
            for doc in peek['documents'][:min(5, count)]:
                print(f"  {doc[:100]}...")
        else:
            print("No documents available")

    except Exception as e:
        print(f"Error displaying vector store summary: {e}")



# Example usage (assuming you have defined choose_embedding_function and textChunks):
try:
    vector_store = load_or_create_chroma(pdfReader, vector_store_directory)
    display_vector_store_summary(vector_store)
except ValueError as e:
        print(f"A value error occured: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

retriever = vector_store.as_retriever()

# +
from langchain.chains import RetrievalQA, LLMChain

pdf_qa_template = """Use the following context to answer the question at the end. If you don't know the answer based on the context, just say "I don't know".

Context:
{context}

Question: {question}
Answer:"""

QA_PROMPT = PromptTemplate(template=pdf_qa_template, input_variables=["context", "question"])

pdf_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": QA_PROMPT})

# -

# APPROACH 1: only PDF as a source
# 5. Ask a question
query = "What can you say about Steve Jobs style of management?"
result = pdf_qa.invoke({"query": query})
print(result['result'])

# +
# APPROACH 2: FULL VERSION - Three Sources - the fully blown approach: leverage PDFs + LLM + Internet
from langchain_community.utilities import SerpAPIWrapper
#from langchain.tools import SerpAPIWrapper # Import SerpAPIWrapper explicitly
import os

# 1. Load from PDFs
# This shows how to generalize to a list of pdfs :
''' pdf_paths = ["path/to/pdf1.pdf", "path/to/pdf2.pdf"]  # Replace with your PDF paths
pdf_docs = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    pdf_docs.extend(loader.load())
'''
# 2. Split text from PDFs
# implemented above in the shared section of the notebook so no need to repeat code here

# 3. Create vectorstore from PDFs
# implemented above in the shared section of the notebook so no need to repeat code here

# 4. Initialize LLM (for both internal knowledge and final answer generation)
# implemented above in the shared section of the notebook so no need to repeat code here

# 5. Initialize the internet searches - SerpAPI - a tool for internet search (Crucially with the wrapper)
from langchain_community.utilities import GoogleSearchAPIWrapper

serpapi_api_key = os.environ.get("SERPAPI_API_KEY") #does not work, so workaround below
SERPAPI_API_KEY='49c6ecf880ecddcca49c7795464d7d235a932823cba6f66afe70c01e91383536'
serpapi_api_key=SERPAPI_API_KEY
if serpapi_api_key:
    internet_search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
else:
    print("SERPAPI_API_KEY environment variable not set. Internet searches will be unavailable.")
    internet_search = [] # search is unavailable if no API key

# 6. Define RetrievalQA chain for PDFs
# implemented above in the shared section of the notebook so no need to repeat code her
#QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
#pdf_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=pdf_retriever, chain_type_kwargs={"prompt": QA_PROMPT})


def get_context(query):
    pdf_context = pdf_qa.invoke({"query": query})['result']
#    print(f"PDF context: {pdf_context}")
    llm_context = llm.invoke(query).content
#    print(f"LLM context: {llm_context}")
    internet_context = internet_search.run(query) if internet_search else "Internet search is unavailable." # Direct SerpAPI use
#    print(f"Internet context: {internet_context}")
    combined_context = f"PDF Context:\n{pdf_context}\n\nLLM Context:\n{llm_context}\n\nInternet Context:\n{internet_context}"
#    print(combined_context)
    return combined_context


# +
# 8. Final prompt template and chain
final_template = """Use the combined context below from PDFs, an LLM, and the internet to answer the question. Treat the three contexts as three sources of information. If the information is not present in the context say I don't know.

Combined Context:
{combined_context}

Question: {question}
Answer:"""

FINAL_PROMPT = PromptTemplate(
    template=final_template, input_variables=["combined_context", "question"]
)

final_chain = LLMChain(llm=llm, prompt=FINAL_PROMPT)
# -

# 9. Ask a question
query = "How many children did Steve Jobs have?"
combined_context = get_context(query)
final_answer = final_chain.invoke({"combined_context": combined_context, "question": query})
print(final_answer["text"])

# 9. Ask a question
query = "List names and birthdates of all Steve Jobs's children"
combined_context = get_context(query)
final_answer = final_chain.invoke({"combined_context": combined_context, "question": query})
print(final_answer["text"])

# 9. Ask a question
query = "What can you say about Steve Jobs style of management?"
combined_context = get_context(query)
final_answer = final_chain.invoke({"combined_context": combined_context, "question": query})
print(final_answer["text"])


