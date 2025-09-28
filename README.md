---
title: AgenticAI
emoji: 📚
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.49.1"
app_file: app.py
pinned: false
---

# RAG (Retrieval-Augmented Generation)

A lightweight hybrid RAG prototype combining retrieval and language model generation to answer queries over documents or knowledge sources.

Accesible in Hugging Face space - [https://huggingface.co/spaces/polojuan/hybridrag](https://huggingface.co/spaces/polojuan/hybridrag)

---

## 🚀 Features

- Hybrid retrieval (semantic + keyword-based) over document corpus  
- Connects retrieval output into an LLM prompt for answer generation  
- Simple, modular architecture for easy experimentation  
- Docker support for reproducible environments  
- Python-based, minimal dependencies  

---

## Architecture & Workflow

1. **Document ingestion & preprocessing**  
   - Load documents (e.g. PDF, text)  
   - Chunk / split into manageable passages  

2. **Embedding & indexing**  
   - Generate embeddings for chunks using e.g. sentence-transformer  
   - Store embeddings in a vector index  

3. **Retrieval (hybrid)**  
   - Semantic (vector) search  
   - Keyword / sparse search (optional)  
   - Fuse or rerank results  

4. **Prompt construction & generation**  
   - Construct the prompt combining query + retrieved context  
   - Send to a language model (e.g. via OpenAI API or local LLM)  
   - Return answer  

5. **(Optional) Postprocessing & filtering**  
   - Clean up output, optionally verify or validate  

---

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.8+  
- Docker (if you choose to use container)  

### Local setup

```bash
git clone https://github.com/palscruz23/rag.git
cd rag

# Install dependencies
pip install -r requirements.txt

# Run streamlit app
streamlit run app.py
```
---

## 📁 Repository Structure
```
.
├── app.py              # Main entry / API or UI driver
├── Dockerfile          # Docker configuration
├── requirements.txt    # Python dependencies
├── utils/              # Utility modules & helpers
└── README.md           # This documentation
```

You may also have submodules under `utils` (e.g. embedding, retrieval, prompt, generation).
---

## 📜 License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.