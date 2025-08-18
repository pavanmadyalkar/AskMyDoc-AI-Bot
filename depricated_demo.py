
#gradio
#transformers
#sentence-transformers
#langchain
#faiss-cpu

from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
import gradio as gr

# Load the PDF document
def document_loader(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# Split the document into chunks
def text_splitter(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_documents(documents)
    return chunks

# Create embedding model
def get_embedding_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

# Create vector database
def vector_database(chunks, embedding_model):
    vectordb = FAISS.from_documents(chunks, embedding_model)
    return vectordb

# Load the Hugging Face LLM pipeline
def get_llm_pipeline():
    text_gen_pipeline = pipeline(
        "text-generation",
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        max_new_tokens=256,
        temperature=0.5,
        do_sample=True
    )
    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
    return llm

# Create retriever and QA chain
def retriever_qa(file, query):
    documents = document_loader(file.name)
    chunks = text_splitter(documents)
    embedding_model = get_embedding_model()
    vectordb = vector_database(chunks, embedding_model)
    retriever = vectordb.as_retriever()

    llm = get_llm_pipeline()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )

    result = qa_chain.invoke(query)
    return result['result']

# Gradio interface
interface = gr.Interface(
    fn=retriever_qa,
    inputs=[
        gr.File(label="Upload PDF File", file_types=[".pdf"], type="filepath"),
        gr.Textbox(label="Enter your question", lines=2)
    ],
    outputs=gr.Textbox(label="Answer"),
    title="📚 PDF QA Bot",
    description="Upload a PDF and ask questions. The bot will answer using the document content."
)

# Launch the app
interface.launch()
