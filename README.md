# 🌊 OceanAI PDF RAG Assistant

A lightweight multimodal **Retrieval‑Augmented Generation (RAG)** application that lets you ask both **textual and visual questions** about the contents of one or more PDF documents.

Built with **Streamlit** for UI and **Ollama** local models for fast, private inference.

---

## ✨ Key Features

| Capability                 | Details                                                                                 |
| -------------------------- | --------------------------------------------------------------------------------------- |
| 📄 **Multi‑PDF ingestion** | Upload any number of PDF files in the sidebar.                                          |
| 🔍 **Hybrid parsing**      | • Extract native text • Run OCR (Tesseract) on scanned pages • Capture embedded images. |
| 🧩 **Vector search**       | Text chunks embedded with **nomic‑embed‑text** → stored in a local **Chroma** DB.       |
| 🤖 **Text QA**             | Questions answered with the 8‑bit **mistral** model via `langchain_ollama`.             |
| 🖼 **Image QA**            | Ask about figures/tables using **llava:7b** (multimodal) — one click per image.         |
| 💾 **Offline‑friendly**    | All models run locally under **Ollama**; no OpenAI key required.                        |

---

## 🚀 Quick Start

> **Prerequisites**
> • Python ≥ 3.9 
> • [Tesseract‑OCR](https://github.com/tesseract-ocr/tesseract) installed and on your `PATH`
> • [Ollama](https://ollama.com) installed and running (`ollama serve`)

```bash
# 1️⃣  Clone the repo
$ git clone https://github.com/YOUR_USERNAME/oceanai‑rag‑assistant.git
$ cd oceanai‑rag‑assistant

# 2️⃣  Install Python deps
$ pip install -r requirements.txt

# 3️⃣  Pull the required models (≈3 GB total)
$ ollama pull mistral
$ ollama pull llava:7b
$ ollama pull nomic-embed-text

# 4️⃣  Launch the app
$ streamlit run streamlit_multimodal_ollama_app.py
```

Then open the local URL shown in your terminal (typically [http://localhost:8501](http://localhost:8501)).

---

## 🖼️ Screenshots

| Upload & Initialise | Text QA         | Image QA        |
| ------------------- | --------------- | --------------- |
| *(placeholder)*     | *(placeholder)* | *(placeholder)* |

Add PNGs (e.g. `assets/screenshot_*.png`) to replace the placeholders.

---

## 🗂️ Folder Structure

```text
oceanai‑rag‑assistant/
├── streamlit_multimodal_ollama_app.py   # 📌 Main Streamlit entry point
├── requirements.txt                    # Python dependencies
├── README.md                           # → you are here
├── assets/                             # Screenshots & logos (optional)
├── chroma_db/                          # Persisted vector store (auto‑created)
└── .gitignore                          # Python, Streamlit & editor artefacts
```

---

## 🔧 Configuration

| Variable      | Purpose                      | Default       |
| ------------- | ---------------------------- | ------------- |
| `CHROMA_PATH` | Persisted vector DB location | `./chroma_db` |
| `NUM_CHUNKS`  | Retriever `k` value          | `4`           |

Edit values at the top of `streamlit_multimodal_ollama_app.py` as needed.

---

## 📖 How It Works

1. **Ingestion** — Each uploaded PDF is temporarily saved, then processed page‑by‑page with *PyMuPDF* (`fitz`).
2. **Extraction** — For every page:

   * Native text → added as a `Document` (type =`text`).
   * Images → OCR via *pytesseract* → (type =`ocr`).
   * Embedded images also saved & referenced (type =`image`).
3. **Chunking & Embedding** — Non‑image docs are chunked (1 000 chars, 20 % overlap) and embedded with **nomic‑embed‑text**.
4. **Vector Store** — Chunks go into a **Chroma** DB; the DB is cached with `st.cache_resource`.
5. **Retrieval + Generation** —

   * **Text tab**: Retriever → `mistral` LLama for answer + sources (via `langchain` `RetrievalQA`).
   * **Image tab**: Selected figure → base64 → prompt sent to **llava:7b** (`OllamaLLM`) together with your question.

---

## 🧩 Tech Stack

| Layer         | Library / Tool               | Why                                  |
| ------------- | ---------------------------- | ------------------------------------ |
| UI            | **Streamlit**                | Rapid data‑app prototyping           |
| PDF handling  | **PyMuPDF (fitz)**           | Fast page parsing & image extraction |
| OCR           | **pytesseract**              | Reliable open‑source OCR             |
| Embeddings    | **nomic‑embed‑text**         | 768‑dim, licence‑friendly            |
| Vector DB     | **Chroma**                   | Lightweight, local vector store      |
| LLMs & Images | **Ollama + LLaVA / Mistral** | Fully local inference                |
| Orchestration | **LangChain**                | Chains, retrievers & agents          |

---

## 📋 Requirements.txt

```
streamlit>=1.35
pymupdf
pillow
pytesseract
langchain
langchain_ollama
chromadb
```

Tesseract‑OCR system package must also be installed (e.g. `sudo apt install tesseract-ocr` on Ubuntu).

---

## 🚧 Roadmap / Future Work

* [ ] **Streaming responses** with Server‑Sent Events.
* [ ] **Multi‑format ingestion** (DOCX, PPTX, webpages).
* [ ] **Better image handling** — highlight bounding boxes in‑place.
* [ ] **Deploy** to Streamlit Cloud using custom Docker w/ Ollama.

Contributions via pull requests or issues are welcome!

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for details.

---

## 🙋‍♂️ Author

|                  |                                                                       |
| ---------------- | --------------------------------------------------------------------- |
| **Harisankar M** | [https://github.com/YOUR\_USERNAME](https://github.com/YOUR_USERNAME) |
