# 🤖 RAG-Based PDF Question Answering System

## 📌 Project Overview
A **Retrieval-Augmented Generation (RAG)** pipeline built from scratch that allows users to ask questions from any PDF document and get accurate, context-aware answers using an LLM — without relying on pre-trained world knowledge.

---

## 🎯 Problem Statement
Large Language Models (LLMs) hallucinate when asked about specific documents they haven't seen. This project solves that by:
- Extracting content from any PDF
- Retrieving only the most relevant chunks for a question
- Forcing the LLM to answer **strictly from the document**

---

## 🏗️ RAG Pipeline Architecture

```
📄 PDF Input
     ↓
📚 Document Loading (PyPDFLoader)
     ↓
✂️  Text Chunking (TokenTextSplitter — 100 tokens, 30 overlap)
     ↓
🔢 Embedding Generation (all-MiniLM-L6-v2)
     ↓
🗂️  Vector Store Indexing (FAISS — IndexFlatL2)
     ↓
❓ User Question → Query Embedding
     ↓
🔍 Similarity Search (Top-K chunks retrieval)
     ↓
📝 Prompt Engineering (Custom Template)
     ↓
🤖 LLM Response (Groq — Qwen3-32B)
     ↓
✅ Grounded Answer
```

---

## 🛠️ Tools & Technologies

| Tool | Purpose |
|---|---|
| **Python** | Core programming language |
| **LangChain** | Document loading & text splitting |
| **PyPDFLoader** | PDF parsing and extraction |
| **TokenTextSplitter** | Token-based chunking |
| **SentenceTransformer** | Generating embeddings (all-MiniLM-L6-v2) |
| **FAISS** | Vector similarity search (IndexFlatL2) |
| **Groq API** | LLM inference (Qwen3-32B) |
| **NumPy** | Embedding array operations |
| **python-dotenv** | Secure API key management |

---

## ⚙️ How It Works — Step by Step

### Step 1 — PDF Loading
```python
pdf_loader = PyPDFLoader(file_path)
docs = pdf_loader.load()
# Loaded 58 pages from the document
```

### Step 2 — Text Chunking
```python
token_text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=30)
# Extracted 431 chunks from the document
```

### Step 3 — Embedding Generation
```python
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# Each chunk converted to 384-dimensional vector
```

### Step 4 — FAISS Vector Index
```python
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
# 431 embeddings stored in FAISS index
```

### Step 5 — Question Answering
```python
question = input("Ask your question: ")
query_embedding = embedding_model.encode([question])
D, I = index.search(query_embedding, k=2)  # Top 2 relevant chunks
response = llm.invoke(filled_template)
```

---

## 📝 Prompt Engineering

Custom prompt template enforces strict grounding:
```
Rules:
- ONLY use retrieved chunks to answer
- DO NOT use any pre-trained world knowledge
- If data is insufficient → say "THE DATA IS INSUFFICIENT TO ANSWER THE QUESTION"
```

---

## 📊 Project Stats

| Metric | Value |
|---|---|
| PDF Pages Processed | 58 |
| Total Chunks Extracted | 431 |
| Embedding Dimensions | 384 |
| Chunk Size | 100 tokens |
| Chunk Overlap | 30 tokens |
| Top-K Retrieval | 2 chunks |
| LLM Used | Qwen3-32B (via Groq) |

---

## 📂 Project Files

| File | Description |
|---|---|
| `Rag_Project.ipynb` | Main notebook with full RAG pipeline |
| `prompt_file.txt` | Custom prompt template for LLM |
| `requirements.txt` | All dependencies |
| `README.md` | Project documentation |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Divya0777/rag-pdf-qa-system
cd rag-pdf-qa-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up API key
Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the notebook
```bash
jupyter notebook Rag_Project.ipynb
```

---

## 📦 Requirements
```
langchain
langchain-community
langchain-text-splitters
langchain-groq
sentence-transformers
faiss-cpu
numpy
pypdf
python-dotenv
```

---

## 💡 What I Learned
- Building a complete RAG pipeline from scratch without frameworks like LlamaIndex
- Token-based chunking strategies and overlap tuning
- Generating and storing vector embeddings using FAISS
- Prompt engineering to prevent LLM hallucination
- Integrating Groq API with LangChain for fast LLM inference
- Secure API key management using environment variables

---

## 🔮 Future Improvements
- [ ] Add support for multiple PDF uploads
- [ ] Implement re-ranking for better chunk retrieval
- [ ] Build a Streamlit UI for easy interaction
- [ ] Add chat history / multi-turn conversation support
- [ ] Experiment with different embedding models

---

## 📬 Connect With Me
[![GitHub](https://img.shields.io/badge/GitHub-Divya0777-black?logo=github)](https://github.com/Divya0777)
