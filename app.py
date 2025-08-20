import os
import tempfile
import streamlit as st
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline

# Load the PDF document
def document_loader(file_path):
    """
    Takes a list of Streamlit UploadedFile objects,
    loads them into LangChain Documents from different loaders,
    and returns a combined list of documents.
    """
    all_docs = []
    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(tmp_path)
            elif suffix == ".csv":
                loader = CSVLoader(tmp_path)
            elif suffix == ".txt":
                loader = TextLoader(tmp_path)
            elif suffix == ".docx":
                loader = Docx2txtLoader(tmp_path)
            else:
                raise ValueError(f"❌ Unsupported file type: {suffix}")

            docs = loader.load()
            all_docs.extend(docs)
        finally:
            os.remove(tmp_path)  # cleanup

    return all_docs

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
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=256,
        temperature=0.5,
        do_sample=True
    )
    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
    return llm

# Create retriever and QA chain
def retriever_qa(uploaded_files, query):

    documents = document_loader(uploaded_files)
    chunks = text_splitter(documents)
    embedding_model = get_embedding_model()
    vectordb = vector_database(chunks, embedding_model)
    retriever = vectordb.as_retriever()

    llm = get_llm_pipeline()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True   # enable sources if you want to show where it found info
    )

    result = qa_chain.invoke(query)
    return result["result"], result.get("source_documents", [])


# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="📚 Multi-File QA Bot", layout="centered")

st.title("📚 QA Bot")
uploaded_files = st.file_uploader(
    "Upload one or more files",
    type=["pdf", "csv", "txt", "docx"],
    accept_multiple_files=True
)

query = st.text_area("Enter your question")

if st.button("Get Answer"):
    if uploaded_files and query.strip():
        with st.spinner("Processing..."):
            answer, sources = retriever_qa(uploaded_files, query)

        st.success("Answer:")
        st.write(answer)

        if sources:
            st.info("📄 Sources used:")
            for doc in sources:
                st.write(f"- {doc.metadata.get('source', 'Unknown')}")

    else:
        st.warning("Please upload at least one file and enter a question.")