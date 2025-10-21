# RAG Pipeline for Enterpise Documents using MilvusDB and LangChain

A specialized Retrieval Augmented Generation (RAG) pipeline designed for corporate documents such as Collective Bargaining Agreements (CCNL), company regulations, and business policies. The system features complete document injection with metadata extraction, intelligent chunking with paragraph titles, and advanced retrieval with **intelligent document routing and paragraph-level title understanding**  to provide accurate *Answer*s through a Groq-hosted LLM.

## 🧠 Pipeline Examples

**Input Prompt**
```
Cosa comporta un passaggio di livello in azienda secondo il ccnl?
```

**Output**
```
📄 Document Routing: ccnl_commercio_terziario_distribuzione_e_servizi.pdf
📑 Title Routing: Articolo 3 - Mansioni e passaggi di livello
🧠 *Answer*: Un passaggio di livello, così come disciplinato dal CCNL di riferimento, comporta:

            1. Retribuzione contrattuale  
              - Il dipendente percepisce la retribuzione prevista per il nuovo livello.  
              - Gli scatti di merito non sono assorbibili.

            2. Compensazione temporanea  
              - Se svolte mansioni superiori ≥16 giorni, la differenza retributiva deve essere cor*Answer*.

            3. Durata e definitività  
              - Dopo 6 mesi di mansioni superiori continuative, il passaggio diventa definitivo.

            4. Limiti e condizioni  
              - Eventuali accordi devono rispettare il CCNL e la presenza di rappresentanza sindacale.

            In sintesi: il passaggio di livello adegua la retribuzione, consolida le mansioni dopo sei mesi e non incide sugli scatti di merito.
```

---

**Input Prompt**
```
Se dovessi assentarmi dal lavoro, posso comunicarlo a voce il giorno stesso?
```

**Output**
```
🧠 *Answer*: Il dipendente deve avvertire i propri preposti **prima dell’inizio dell’orario di lavoro**.
              La comunicazione può avvenire per telefono, email o fax. 
              Una comunicazione verbale a giornata iniziata non è conforme, salvo forza maggiore. 
              In tal caso va fornita documentazione entro due giorni.
```

---

**Input Prompt**
```
Si può fumare secondo il regolamento aziendale?
```

**Output**
```
📄 Document Routing: BIS - Regolamento Aziendale.pdf
📑 Title Routing: 16. Divieto di fumo e bevande alcoliche
🧠 *Answer*: No. È vietato fumare in tutti i locali aziendali; è consentito solo all’esterno.
```

---

**Input Prompt**
```
Esistono spazi all'interno per fumare?
```

**Output**
```
🧠 *Answer*: Non vi sono spazi interni per fumare; il divieto è assoluto negli ambienti aziendali.
```

---

**Input Prompt**
```
Chi era Albert Einstein?
```

**Output**
```
🧠 *Answer*: L'argomento della domanda non è pertinente.
```

---

**Input Prompt**
```
Quali sono gli indirizzi IBAN dei conti presenti in tutti i documenti?
```

**Output**
```
🧠 *Answer*: L'argomento della domanda non è pertinente
```





## 📂 Folder Structure
```
rag/
├─ datafile/
│  ├─ BIS - Regolamento Aziendale.pdf (Not added for privacy)
│  ├─ ccnl_commercio_terziario_distribuzione_e_servizi.pdf (Not added for privacy)
│  └─ codice etico fittizio_Salute e sicurezza dei lavoratori.pdf (Not added for privacy)
├─ test correctness/
│  ├─ test_chunking.ipynb
│  ├─ test_db.ipynb
│  ├─ test_indexing.ipynb
│  ├─ test_pdf.ipynb
│  └─ test_titleExtractor.ipynb
├─ db_injection.py
├─ document_db.db (Not added for privacy)
├─ final_db.db (Not added for privacy)
├─ paragraph_chunker.py
├─ paragraph_injection.py
├─ pdf_chunker.py
├─ pdf_utils.py
├─ rag.py
├─ retrieve.py
├─ retriever_wrapper.py
├─ test_rag.ipynb
├─ test_rag.py
└─ title_extractor.py
```

## 📄 Files
- `datafile/`: Directory containing PDF documents to be indexed.
- `test correctness/`: Files to check correctness of different pipeline stages.
  - `test_chunking.ipynb`: Testing chunking implementation.
  - `test_db.ipynb`: Database operations and vector storage testing.
  - `test_indexing.ipynb`: Document indexing and embedding testing.
  - `test_pdf.ipynb`: PDF processing and extraction testing.
  - `test_titleExtractor.ipynb`: Testing title extraction.
- `db_injection.py`: Logic for injecting chunks into Milvus.
- `document_db.db`: Milvus file for vector DB. (Not added for privacy)
- `final_db.db`:  Milvus database including titles metadata. (Not added for privacy)
- `paragraph_chunker.py`: Alternative chunking strategy based on paragraphs.
- `paragraph_injection.py`: Logic for injecting paragraph-based chunks into database.
- `pdf_chunker.py`: Token-based page chunking with overlap and metadata.
- `pdf_utils.py`: PDF parsing helpers (text extraction, header cleanup).
- `rag.py`: End-to-end RAG wrapper (Milvus retriever + Groq LLM + prompt).
- `retrieve.py`: Main class, complete of **Retriever, FileRouter, TitleExtractor**. Features **semantic search to all DB, routed search to specific files and also paragraph title understanding**
- `retriever_wrapper.py`: LangChain-compatible wrapper for custom retriever.
- `test_rag.ipynb`: Jupyter notebook for testing RAG pipeline interactively.
- `test_rag.py`: Testing RAG pipeline with multiple queries and debugging utilities.
- `title_extractor.py`: Title classification using font sizes and extraction.

## 🥳 Features
- **Smart Document Chunking**: Automatic PDF chunking with metadata extraction including paragraph titles using font-size analysis and intelligent page title removal
- **Multi-level Semantic Search**: Three retrieval modes:
  - Standard semantic search across all documents
  - File-routed search with LLM-based document detection
  - Complete search with both file routing and paragraph title matching
- **Intelligent File Routing**: LLM-powered query analysis to automatically identify and route queries to relevant documents
- **Paragraph Title Matching**: Advanced title extraction and matching system that understands query intent and filters results by relevant paragraph sections
- **Flexible LLM Support**: Compatible with Qwen, Llama, GPT-OSS, and other Groq-hosted models
- **Adaptive Thresholds**: Dynamic similarity thresholds with automatic fallback for better retrieval coverage
- **Comprehensive Testing Suite**: Dedicated notebooks for testing chunking, indexing, PDF processing, title extraction, and end-to-end RAG pipeline