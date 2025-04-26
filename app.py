import os
import streamlit as st
import tempfile
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

st.title("📄 PDF Q&A: Upload Your Book")

# — Session & Memory Setup —
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
if 'history' not in st.session_state:
    st.session_state.history = []

# Frontend PDF upload
uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])
if not uploaded_file:
    st.info("Please upload a PDF to begin your RAG-powered Q&A.")
    st.stop()

# Only when PDF changes, write it once and reload vectordb
if uploaded_file.name != st.session_state.get("pdf_name"):
    st.session_state.pdf_name = uploaded_file.name

    # save in a *stable* path based on filename
    os.makedirs("temp_pdfs", exist_ok=True)
    pdf_path = os.path.join("temp_pdfs", uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())
    st.session_state.pdf_path = pdf_path

    # clear previous vectorstore
    if 'vectordb' in st.session_state:
        del st.session_state.vectordb

# Initialize vectorstore once (cached by Streamlit)
@st.cache_resource
def init_vectorstore(path: str, persist_dir: str = "db"):
    loader = PyPDFLoader(path)
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(raw_docs)
    embedder = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma.from_documents(docs, embedder, persist_directory=persist_dir)

if 'vectordb' not in st.session_state:
    st.session_state.vectordb = init_vectorstore(st.session_state.pdf_path)

# Set Ollama endpoint once
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
llm = Ollama(model="llama3")
retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="refine",
    return_source_documents=True
)

# — Streamlit Q&A UI —
query = st.text_input("Your question:")
status_slot = st.empty()  # placeholder for status updates

if st.button("Submit") and query:
    # Show spinner while the chain is working
    with st.spinner("🤖 Fetching answer…"):
        result = qa_chain({
            "query": query,
            "chat_history": st.session_state.chat_history
        })

    answer = result["result"]
    sources = result["source_documents"]

    # Update status with timestamp
    status_slot.success(f"Answered at {datetime.now().strftime('%H:%M:%S')}")

    # Append to history and display
    st.session_state.history.append((query, answer))
    st.markdown(f"**Answer:** {answer}")
    with st.expander("Sources"):
        for doc in sources:
            page = doc.metadata.get("page", "?")
            snippet = doc.page_content[:200].replace("\n", " ")
            st.write(f"- Page {page}: “{snippet}…”")

# Render full Q&A history
for i, (q, a) in enumerate(st.session_state.history):
    st.markdown(f"**Q{i+1}:** {q}")
    st.markdown(f"**A{i+1}:** {a}")
