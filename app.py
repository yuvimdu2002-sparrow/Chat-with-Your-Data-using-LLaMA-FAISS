import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA

st.title("📄 Chat with your PDF (Local LLM)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split text
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings()

    # Vector DB
    db = FAISS.from_documents(docs, embeddings)

    # LLM
    llm = Ollama(model="llama3")

    # RAG Chain
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

    query = st.text_input("Ask a question:")

    if query:
        result = qa.run(query)
        st.write(result)
