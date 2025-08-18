# 📚 AskMyDoc (Question Answering Bot using LangChain, IBM watsonx.ai & Gradio)

## 🧠 Project Overview

This project demonstrates a **Question Answering (QA) bot** that intelligently responds to user queries based on the content of uploaded PDF documents. It uses **LangChain**, **IBM watsonx.ai**, and **Gradio** to build an interactive assistant capable of parsing and understanding complex documents like legal papers, technical manuals, and research articles.

### ✅ Key Features

- 📄 Load and process PDF documents  
- ✂️ Split text into manageable chunks  
- 🔍 Embed and store document chunks in a vector database  
- 🧭 Retrieve relevant chunks using semantic search  
- 🤖 Generate answers using powerful LLMs via IBM watsonx.ai  
- 🖥️ User-friendly interface built with Gradio  

---

## 🚀 How It Works

1. **Document Loading**: PDFs are loaded and parsed into raw text.  
2. **Text Splitting**: Text is split into chunks using LangChain's `TextSplitter`.  
3. **Embedding**: Chunks are converted into vector embeddings using a pre-trained embedding model.  
4. **Vector Store**: Embeddings are stored in a vector database (e.g., FAISS).  
5. **Retrieval**: Relevant chunks are retrieved based on user queries.  
6. **LLM Response**: Answers are generated using IBM watsonx.ai with the following models:  
   - `ibm/granite-3-3-8b-instruct`  
   - `mistralai/mixtral-8x7b-instruct-v01`  
7. **Frontend**: Gradio provides a simple interface for uploading PDFs and asking questions.

---

## 🖼️ Example Output

![Bot Output]
<img width="1257" height="632" alt="QA_bot" src="https://github.com/user-attachments/assets/93945c82-d644-4a00-844f-5fc79572dfb1" />

---

## 🛠️ Tech Stack

- **LangChain**  
- **IBM watsonx.ai**  
  - `ibm/granite-3-3-8b-instruct`  
  - `mistralai/mixtral-8x7b-instruct-v01`  
- **ChromaDB**  
- **Gradio**  
- **Python**

---

## 📦 Installation

```bash
git clone https://github.com/your-username/AskMyDoc.git
cd AskMyDoc
pip install -r requirements.txt
