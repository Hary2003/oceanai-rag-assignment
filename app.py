
"""
Streamlit UI for OceanAI PDF RAG Assistant (Text + Image) using lightweight Ollama models.

Models
------
* **mistral** – text/table summarisation & QA
* **llava:7b** – image QA (multimodal)
* **nomic-embed-text** – embeddings for vector search

Features
--------
* Upload multiple PDFs
* Parse text, OCR, and extract images via PyMuPDF (fitz) + pytesseract
* Build RAG index with Chroma
* Text QA tab powered by Mistral
* Image QA tab powered by LLaVA (Ollama)
"""

import streamlit as st
import asyncio
import os
import fitz  # PyMuPDF
import io
from PIL import Image
import pytesseract
import tempfile
import base64

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma


# Helpers


def pil_to_b64(img: Image.Image) -> str:
    """Convert PIL image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


# Text + Image PDF Parsing


def extract_documents_from_pdf(uploaded_file):
    """Return a list of LangChain Documents containing text, OCR text, and images."""
    documents = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    pdf = fitz.open(tmp_path)
    for i, page in enumerate(pdf):
        # Extract visible text
        text = page.get_text()
        if text and text.strip():
            documents.append(Document(
                page_content=text,
                metadata={"source": uploaded_file.name, "page": i + 1, "type": "text"}
            ))

        # Extract images
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # OCR fallback for image-only pages
            ocr_text = pytesseract.image_to_string(image)
            if ocr_text.strip():
                documents.append(Document(
                    page_content=ocr_text,
                    metadata={"source": uploaded_file.name, "page": i + 1, "type": "ocr"}
                ))

            # Store image placeholder (actual image kept in metadata)
            documents.append(Document(
                page_content="[IMAGE]",
                metadata={
                    "source": uploaded_file.name,
                    "page": i + 1,
                    "type": "image",
                    "image_data": image,
                    "image_b64": pil_to_b64(image),
                }
            ))

    return documents


# RAG Initialisation


@st.cache_resource(show_spinner=False)
def initialize_rag(uploaded_files):
    try:
        all_documents = []
        for file in uploaded_files:
            all_documents.extend(extract_documents_from_pdf(file))

        # Vectorise only textual chunks (exclude pure image placeholders)
        text_docs = [doc for doc in all_documents if doc.metadata.get("type") != "image"]

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(text_docs)

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_store = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="./chroma_db")

        llm_text = OllamaLLM(model="mistral", temperature=0.3, num_ctx=4096)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_text,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
        )
        return qa_chain, all_documents
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
        print("DEBUG - Init failed:", str(e))
        return None, []


# QA Helpers


def ask_text_question(pipeline, query):
    if pipeline is None:
        return "❌ Error: The RAG system is not initialized. Please upload PDFs and click 'Initialize RAG'."
    try:
        response = pipeline.invoke({"query": query})
        answer = response["result"]
        sources = "\n".join(
            f"• {doc.metadata['source']} (page {doc.metadata.get('page', 'N/A')})"
            for doc in response["source_documents"]
        )
        return f"{answer}\n\nSources:\n{sources}"
    except Exception as e:
        return f"Error processing query: {str(e)}"

def ask_image_question_llava(image: Image.Image, question: str) -> str:
    """Send an image + question to LLaVA via Ollama."""
    try:
        img_b64 = pil_to_b64(image)
        prompt = [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ]
        llava = OllamaLLM(model="llava:7b")
        return llava.invoke(prompt)
    except Exception as e:
        return f"LLaVA error: {str(e)}"


# Streamlit UI

st.set_page_config(page_title="OceanAI PDF RAG Assistant", layout="wide")
st.title("🌊 OceanAI PDF RAG Assistant – Text & Images")

with st.sidebar:
    st.header("1️⃣ Upload PDFs")
    uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        if st.button("🚀 Initialize RAG"):
            with st.spinner("Building RAG pipeline – this may take a moment..."):
                qa_pipeline, all_docs = initialize_rag(uploaded_files)
                st.session_state.qa_pipeline = qa_pipeline
                st.session_state.all_docs = all_docs
            if qa_pipeline:
                st.success("✅ System ready! Switch to the tabs below to ask questions.")
        else:
            st.info("After uploading, click the button to initialize the system.")
    else:
        st.info("Upload one or more PDFs to begin.")


# Tabs for Text & Image QA


if "qa_pipeline" in st.session_state:
    tab_text, tab_img = st.tabs(["📝 Text QA", "🖼 Image QA (LLaVA)"])

    # ---------- Text QA ----------
    with tab_text:
        st.subheader("Ask textual questions about the PDFs")
        text_q = st.text_input("Your question", placeholder="E.g. What is the main conclusion?")
        if st.button("💬 Answer", key="text_btn") and text_q.strip():
            with st.spinner("Searching with Mistral..."):
                result = ask_text_question(st.session_state.qa_pipeline, text_q)
                st.text_area("📘 Answer", value=result, height=300)

    # ---------- Image QA ----------
    with tab_img:
        st.subheader("Ask questions about embedded images")
        img_q = st.text_input("Image question", placeholder="E.g. What does the chart depict?")

        if img_q.strip():
            for doc in st.session_state.all_docs:
                if doc.metadata.get("type") == "image":
                    image = doc.metadata["image_data"]
                    page_no = doc.metadata["page"]
                    st.image(image, caption=f"{doc.metadata['source']} – page {page_no}", use_column_width=True)
                    if st.button(f"Ask about image on page {page_no}", key=f"img_{page_no}"):
                        with st.spinner("Analyzing image with LLaVA..."):
                            answer = ask_image_question_llava(image, img_q)
                            st.success("🧠 LLaVA Answer:")
                            st.write(answer)
else:
    st.warning("Please upload PDFs and initialise the system from the sidebar first.")
