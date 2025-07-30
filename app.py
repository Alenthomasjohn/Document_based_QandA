import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import tempfile

# Environment for OpenRouter
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] =os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Set page title
st.set_page_config(page_title="PDF QA with OpenRouter", layout="wide")
st.title("📄 PDF Question Answering with OpenRouter")

# PDF Upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Reading and splitting PDF..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Load and split PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(docs)

        # Embed documents
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()

        # LLM via OpenRouter
        llm = ChatOpenAI(model="mistralai/mistral-7b-instruct", temperature=0.0)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

        st.success("PDF processed and ready!")

        # Query input
        query = st.text_input("Ask a question about the PDF content:")

        if query:
            with st.spinner("Generating answer..."):
                result = qa_chain.invoke({"query": query})
                st.markdown("### 🧠 Answer:")
                st.write(result["result"])

                # Optional: show sources
                st.markdown("### 📚 Source document chunks:")
                for i, doc in enumerate(result["source_documents"]):
                    with st.expander(f"Source Chunk {i+1}"):
                        st.write(doc.page_content)
