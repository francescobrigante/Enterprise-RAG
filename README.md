# RAG Pipeline for Enterpise Documents using MilvusDB and LangChain

A specialized Retrieval Augmented Generation (RAG) pipeline designed for corporate documents such as Collective Bargaining Agreements (CCNL), company regulations, and business policies. The system features complete document injection with metadata extraction, intelligent chunking with paragraph titles, and advanced retrieval with document and title routing to provide accurate answers through a Groq-hosted LLM.

> ## 🚧 Work in progress...
>
> This repository is currently **under development**: atm I'm working on paragraph title routing for queries to improve results quality.
> However the main pipeline works just fine so you can test it running `test_rag.py` or `rag.py`
>

## 📂 Folder Structure
```
rag/
├─ datafile/
│  ├─ BIS - Regolamento Aziendale.pdf
│  ├─ ccnl_commercio_terziario_distribuzione_e_servizi.pdf
│  └─ codice etico fittizio_Salute e sicurezza dei lavoratori.pdf
├─ .env
├─ .gitignore
├─ chunk.ipynb
├─ db_injection.py
├─ document_db.db
├─ final_db.db
├─ paragraph_chunker.py
├─ paragraph_injection.py
├─ pdf_chunker.py
├─ pdf_utils.py
├─ rag.py
├─ README.md
├─ retrieve.py
├─ retriever_wrapper.py
├─ test_rag.py
├─ title_extractor.py
└─ test correctness/
   ├─ test_chunking.ipynb
   ├─ test_db.ipynb
   ├─ test_indexing.ipynb
   └─ test_pdf.ipynb
```

## 📄 Files
- `document_db.db`: Milvus file for vector DB.
- `final_db.db`: Additional Milvus database including titles metadata.
- `.env`: Environment variables (API keys, configuration).
- `.gitignore`: Git ignore rules.
- `chunk.ipynb`: Jupyter notebook for chunking experimentation and testing.
- `pdf_utils.py`: PDF parsing helpers (text extraction, header cleanup).
- `pdf_chunker.py`: Token-based page chunking with overlap and metadata.
- `paragraph_chunker.py`: Alternative chunking strategy based on paragraphs.
- `paragraph_injection.py`: Logic for injecting paragraph-based chunks into database.
- `db_injection.py`: Logic for injecting chunks into Milvus.
- `title_extractor.py`: Utility for extracting and processing document titles.
- `retrieve.py`: Retrieval system with semantic search and file routing capabilities.
- `retriever_wrapper.py`: LangChain-compatible wrapper for custom retriever.
- `rag.py`: End-to-end RAG wrapper (Milvus retriever + Groq LLM + prompt).
- `test_rag.py`: Testing RAG pipeline with multiple queries and debugging utilities.
- `datafile/`: Directory containing PDF documents to be indexed.
- `test correctness/`: Files to check correctness of different pipeline stages.
  - `test_chunking.ipynb`: Testing chunking strategies and implementations.
  - `test_db.ipynb`: Database operations and vector storage testing.
  - `test_indexing.ipynb`: Document indexing and embedding testing.
  - `test_pdf.ipynb`: PDF processing and extraction testing.

## 🥳 Features
- **Document Chunking with metadata**: Automatic PDF chunking and vector embedding storage including metadata such as paragraph titles
- **Semantic Search**: Multi-language document retrieval using sentence transformers
- **File Routing**: Intelligent query routing to specific documents using LLM
- **Flexible LLM Support**: Compatible with Qwen, Llama, and other Groq-hosted models
- **Threshold Fallback**: Adaptive similarity thresholds for better retrieval coverage
- **Debug Tools**: Comprehensive testing and debugging utilities
