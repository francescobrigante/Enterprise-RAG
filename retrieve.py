#==========================================
# This file contains a retrieval system for MilvusDB based on langchain
#==========================================

import os, re, unicodedata, textwrap
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import json

# config constants
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DB_PATH = "./final_db.db"  # changed to use the database with titles
COLLECTION = "pdf_embeddings_with_titles"  # collection with chunk_title metadata

# LLM for NER
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# LLM_NAME = "qwen/qwen3-32b"
LLM_NAME = "openai/gpt-oss-120b"


pdf_files = [
    {
        "path": "datafile/ccnl_commercio_terziario_distribuzione_e_servizi.pdf",
        "START_PAGES_TO_SKIP": 8,
        "END_PAGES_TO_SKIP": 2,
        "num_pages": 66
    },
    {
        "path": "datafile/BIS - Regolamento Aziendale.pdf",
        "START_PAGES_TO_SKIP": 0,
        "END_PAGES_TO_SKIP": 0,
        "num_pages": 4
    },
    {
        "path": "datafile/codice etico fittizio_Salute e sicurezza dei lavoratori.pdf",
        "START_PAGES_TO_SKIP": 0,
        "END_PAGES_TO_SKIP": 0,
        "num_pages": 1
    }
]


# normalize file names
def normalize_name(s):
    s = os.path.splitext(s)[0]
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith("M"))
    s = s.lower()
    s = re.sub(r"[_\-\.\(\)\[\]]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# returns a list of dict with normalized name and original filename, given a directory path
def build_pdf_index(data_dir):
    pdfs = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                norm = normalize_name(f)
                pdfs.append({"norm": norm, "filename": f})
    return pdfs


# printing search results in natural language
def print_results(results: List[Dict[str, Any]], query: str):

    if not results:
        print("No results found.")
        return
    
    print(f"\nFound {len(results)} results for query: '{query}'\n")
    
    for i, result in enumerate(results, 1):
        filename = result.get('filename', 'unknown')
        page = result.get('page_number', 0)
        score = result.get('score', 0)
        text = result.get('text', '')
        
        print(f"Result number {i}:")
        print(f"   From document '{filename}', pages {page}")
        print(f"   Score: {score:.1%}")
        
        if text:
            preview = text.replace('\n', ' ').strip()
            if len(preview) > 200:
                preview = preview[:200] + "..."
            print(f"   Content: {preview}")
        
        print()



# ============================ File Router Class ============================
# uses LLM to extract file references from queries
class FileRouter:
    
    def __init__(self, pdf_index: List[Dict[str, str]]):
        self.pdf_index = pdf_index
        
        # initialize LLM
        if GROQ_API_KEY:
            self.llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name=LLM_NAME,
                temperature=0.0
            )
            
            # NER prompt template
            self.ner_prompt = PromptTemplate(
                input_variables=["query", "available_files"],
                template=textwrap.dedent("""
                    ### System:
                    Analizza la seguente query e identifica se vengono menzionati file PDF specifici presenti in available files.
                    Restituisci SOLO un JSON con questa struttura:
                    {{"mentioned_files": ["nome_file.pdf", "nome_file2.pdf"]}}
                    Se non vengono menzionati file specifici, restituisci:
                    {{"mentioned_files": []}}
                    Rispondi SOLO con il JSON richiesto.
                    NON includere altre informazioni o testo al di fuori del JSON.
                    NON includere i tag <think> </think>.

                    ### Query:
                    "{query}"

                    ### Available files:
                    {available_files}

                    ### Examples:
                    - "cosa dice il ccnl?" → {{"mentioned_files": ["ccnl_commercio.pdf"]}}
                    - "nel regolamento aziendale..." → {{"mentioned_files": ["BIS - Regolamento Aziendale.pdf"]}}
                    - "Secondo il regolamento posso indossare i pantaloncini?" → {{"mentioned_files": ["BIS - Regolamento Aziendale.pdf"]}}
                    - "cosa dice l'articolo 73 comma 2 del regolamento aziendale?" → {{"mentioned_files": ["BIS - Regolamento Aziendale.pdf"]}}
                    - "straordinari retribuiti?" → {{"mentioned_files": []}}
                    - "qual è il codice del tuo pc?" → {{"mentioned_files": []}}

                    JSON:
                """).strip()
            )
        else:
            self.llm = None
    
    # main extraction function
    def extract_mentioned_files(self, query: str) -> Optional[List[str]]:
        
        if not self.llm:
            return None
        
        try:
            # pdf files list
            available_files = "\n".join([f"- {pdf['filename']}" for pdf in self.pdf_index])
            
            # prompt
            prompt = self.ner_prompt.format(
                query=query,
                available_files=available_files
            )
            
            # get LLM response
            print(f"[DEBUG] About to call LLM with query: {query[:50]}...")
            try:
                response = self.llm.invoke(prompt)
                print(f"[DEBUG] LLM call successful, response type: {type(response)}")
                
                if hasattr(response, 'content'):
                    result_text = response.content.strip()
                    print(f"[DEBUG] Got content, length: {len(result_text)}, starts with: {repr(result_text[:50])}")
                else:
                    print(f"[DEBUG] ERROR: No 'content' attribute! Available: {list(response.__dict__.keys())}")
                    return None
                    
            except Exception as llm_error:
                print(f"[DEBUG] LLM call failed: {type(llm_error).__name__}: {llm_error}")
                raise llm_error
            
            # Clean response if it contains thinking tags
            if "</think>" in result_text:
                result_text = result_text.split("</think>", 1)[1].strip()
            
            # parse JSON response
            try:
                result = json.loads(result_text)
                return result.get("mentioned_files", [])
            
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return result.get("mentioned_files", [])
                return []
                
        except Exception as e:
            print(result_text)
            print(f"LLM extraction failed: {e}")
            return None
        
        

class TitleExtractor:
    
    def __init__(self):
        if GROQ_API_KEY:
            self.llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=LLM_NAME, temperature = 0.0)
            
            self.title_prompt = PromptTemplate(
                input_variables=["query", "available_titles"],
                template=textwrap.dedent("""
                    ### System:
                    Analizza la query e identifica se fa riferimento a titoli specifici presenti nella lista.
                    Eventualmente aggiungi quei titoli presenti nella lista in "relevant_titles".
                    Restituisci SOLO un JSON con questa struttura:
                    {{
                        "relevant_titles": ["titolo1", "titolo2"],
                        "has_title_match": true/false,
                        "confidence": "high"/"medium"/"low"
                    }}
                    
                    Criteri:
                    - "high": la query menziona esplicitamente il titolo o numero articolo
                    - "medium": la query fa riferimento al contenuto del titolo  
                    - "low": solo vagamente correlato
                    - Se non trovi match rilevanti: {{"relevant_titles": [], "has_title_match": false, "confidence": "low"}}

                    ### Query:
                    "{query}"

                    ### Available titles:
                    {available_titles}

                    ### Examples:
                    - "Cosa dice l'articolo 23 del CCNL?" → cerca "Articolo 23" o simili nei titoli per la predizione
                    - "Cosa dice il Regolamento Aziendale su salute e sicurezza?" → cerca "salute e sicurezza" o simili nei titoli per la predizione
                    - "Quali sono le procedure e diritti di informazione nel ccnl?" → cerca "procedure e diritti di informazione" o simili nei titoli per la predizione

                    ### Response:
                """).strip()
            )
        else:
            self.llm = None
            
            
    def get_titles_from_query(self, query: str, titles: List[str]) -> Dict[str, Any]:
        if not self.llm or not titles:
            return {"relevant_titles": [], "has_title_match": False, "confidence": "low"}
        
        try:
            available_titles = "\n".join([f"- {title}" for title in titles])
            
            prompt = self.title_prompt.format(query=query, available_titles=available_titles)
            
            response = self.llm.invoke(prompt)
            result_text = response.content.strip()
            
            # clean response if it contains thinking tags
            if "</think>" in result_text:
                result_text = result_text.split("</think>", 1)[1].strip()
            
            # parse JSON
            try:
                result = json.loads(result_text)
                return result
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                return {"relevant_titles": [], "has_title_match": False, "confidence": "low"}
                
        except Exception as e:
            print(f"Title matching failed: {e}")
            return {"relevant_titles": [], "has_title_match": False, "confidence": "low"}

    



# ================================== Main Retriever Class ==================================
class DocumentRetriever:
    
    def __init__(self, model_name = MODEL_NAME, db_path = DB_PATH, collection = COLLECTION):
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name, encode_kwargs={'normalize_embeddings': False}
        )
        
        # milvus db
        absolute_db_path = os.path.abspath(db_path)
        self.vector_store = Milvus(
            embedding_function=self.embeddings,
            collection_name=collection,
            connection_args={"uri": absolute_db_path},
            auto_id=False
        )
        
        
    # regular semantic search 
    def search(self, query: str, k = 5, threshold = 0.6) -> List[Dict[str, Any]]:
        
        docs_scores = self.vector_store.similarity_search_with_score(query, k=k)
        
        # apply chosen threshold
        filtered_results = [
            {
                "text": doc.page_content,
                "filename": doc.metadata.get("filename", "unknown"),
                "page_number": doc.metadata.get("page_number", 0),
                "chunk_id": doc.metadata.get("chunk_id", 0),
                "score": float(score)
            }
            for doc, score in docs_scores if score >= threshold
        ]
        
        # if no results, apply 0.5 threshold
        if not filtered_results:
            filtered_results = [
                {
                    "text": doc.page_content,
                    "filename": doc.metadata.get("filename", "unknown"),
                    "page_number": doc.metadata.get("page_number", 0),
                    "chunk_id": doc.metadata.get("chunk_id", 0),
                    "score": float(score)
                }
                for doc, score in docs_scores if score > 0.5
            ]
        
        return filtered_results
        
    # routed search using file router
    def routed_search(self, query: str, k = 5, threshold = 0.6) -> List[Dict[str, Any]]:
        
        pdf_index = build_pdf_index("./datafile")
        router = FileRouter(pdf_index)
        mentioned_files = router.extract_mentioned_files(query)
        
        # if no files mentioned, do normal search
        if mentioned_files==None or len(mentioned_files)==0:
            return self.search(query, k, threshold)
        
        # else, filter by mentioned files
        filter_expr = f"filename in {mentioned_files}"
        
        print(f"\n==================Routing query to files: {', '.join(mentioned_files)}\n")
        
        docs_scores = self.vector_store.similarity_search_with_score(query, k=k, expr=filter_expr)

        return [
            {
                "text": doc.page_content,
                "filename": doc.metadata.get("filename", "unknown"),
                "page_number": doc.metadata.get("page_number", 0),
                "chunk_id": doc.metadata.get("chunk_id", 0),
                "score": float(score)
            }
            for doc, score in docs_scores #TODO: eventually add threshold?
        ]

    # semantic search with file routing and title matching
    def complete_search(self, query: str, k = 5, threshold = 0.6) -> List[Dict[str, Any]]:
    
        pdf_index = build_pdf_index("./datafile")
        router = FileRouter(pdf_index)
        mentioned_files = router.extract_mentioned_files(query)
        
        # if no files mentioned, do normal search
        if mentioned_files==None or len(mentioned_files)==0:
            return self.search(query, k, threshold)
        
        # we have mentioned files
        else:
            # extract list of titles from Milvus DB for mentioned files
            all_titles = []
            
            try:
                # query Milvus to get all chunk_title metadata for mentioned files
                filter_expr = f"filename in {mentioned_files}"
                
                # get all chunks for these files (high k value to get all)
                docs_scores = self.vector_store.similarity_search_with_score(
                    query="", k=10000, expr=filter_expr
                )
                
                # extract unique titles from metadata, filtering out unwanted ones
                seen_titles = set()
                for doc, score in docs_scores:
                    chunk_title = doc.metadata.get("chunk_title", "")
                    if (chunk_title and 
                        chunk_title not in ["UNKNOWN", "SKIP", "TO_ASSIGN", ""] and 
                        chunk_title not in seen_titles):
                            all_titles.append(chunk_title)
                            seen_titles.add(chunk_title)
                
                print(f"Extracted {len(all_titles)} unique titles from mentioned files")

                # if no titles extracted, perform regular routed search
                if all_titles is None or len(all_titles) == 0:
                    
                    print(f"\n==================Routing query to files: {', '.join(mentioned_files)}\n")
        
                    docs_scores = self.vector_store.similarity_search_with_score(query, k=k, expr=filter_expr)

                    return [
                        {
                            "text": doc.page_content,
                            "filename": doc.metadata.get("filename", "unknown"),
                            "page_number": doc.metadata.get("page_number", 0),
                            "chunk_id": doc.metadata.get("chunk_id", 0),
                            "score": float(score)
                        }
                        for doc, score in docs_scores #TODO: eventually add threshold?
                    ]
                    
                
                # we have titles in our db
                # does the query refer to those titles?
                title_extractor = TitleExtractor()
                result = title_extractor.get_titles_from_query(query, all_titles)
                
                title_routing = result.get('has_title_match', False)
                
                # no titles found, perform regular routed search
                if not title_routing:
                    print(f"\n==================Routing query to files: {', '.join(mentioned_files)}\n")
        
                    docs_scores = self.vector_store.similarity_search_with_score(query, k=k, expr=filter_expr)

                    return [
                        {
                            "text": doc.page_content,
                            "filename": doc.metadata.get("filename", "unknown"),
                            "page_number": doc.metadata.get("page_number", 0),
                            "chunk_id": doc.metadata.get("chunk_id", 0),
                            "score": float(score)
                        }
                        for doc, score in docs_scores #TODO: eventually add threshold?
                    ]
                
                #else titles found
                titles_found = result.get('relevant_titles', [])
                confidence = result.get('confidence', 'N/A')
                
                if titles_found == None or len(titles_found)==0:
                    print(f"\n==================Routing query to files: {', '.join(mentioned_files)}\n")
        
                    docs_scores = self.vector_store.similarity_search_with_score(query, k=k, expr=filter_expr)

                    return [
                        {
                            "text": doc.page_content,
                            "filename": doc.metadata.get("filename", "unknown"),
                            "page_number": doc.metadata.get("page_number", 0),
                            "chunk_id": doc.metadata.get("chunk_id", 0),
                            "score": float(score)
                        }
                        for doc, score in docs_scores #TODO: eventually add threshold?
                    ]
                
                print(f"Found {len(titles_found)} titles with confidence: {confidence}")
                print(f"Relevant titles: {titles_found}")
                
                # filter by titles and document name
                title_filter_parts = [f'chunk_title == "{title}"' for title in titles_found]
                title_filter = " or ".join(title_filter_parts)
                
                combined_filter = f"filename in {mentioned_files} and ({title_filter})"
                
                print(f"\n==================Routing query to files: {', '.join(mentioned_files)}")
                print(f"Filtering by titles: {', '.join(titles_found)}\n")
                
                docs_scores = self.vector_store.similarity_search_with_score(
                    query, k=k, expr=combined_filter
                )
                
                # if not found, apply only document filter
                if not docs_scores:
                    print("No results with specific titles, falling back to file-only search...")
                    file_filter = f"filename in {mentioned_files}"
                    docs_scores = self.vector_store.similarity_search_with_score(
                        query, k=k, expr=file_filter
                    )

                return [
                    {
                        "text": doc.page_content,
                        "filename": doc.metadata.get("filename", "unknown"),
                        "page_number": doc.metadata.get("page_number", 0),
                        "chunk_id": doc.metadata.get("chunk_id", 0),
                        "chunk_title": doc.metadata.get("chunk_title", "unknown"),
                        "score": float(score)
                    }
                    for doc, score in docs_scores
                ]
                
            except Exception as e:
                print(f"Error extracting titles from database: {e}")
                return []
    



# ======================== Main ===================
def main():

    # QUERY = "cosa si dice riguardo la sicurezza dei dipendenti neoassunti?"
    QUERY = "cosa dice l'articolo 23 del ccnl?"         # problema matching (pg19)
    # QUERY = "secondo il codice etico, cosa fa l'azienda?"
    # QUERY = "cosa dice il regolamento aziendale su igiene e sicurezza?"
    
    # QUERY = "cosa succede se non rispetto l'orario di lavoro?"
    # QUERY = "cosa devo fare in caso di assenza?"           # soglia del 40%
    # QUERY = "quali sono gli obblighi del lavoratore?"
    # QUERY = "quali sono i doveri del lavoratore?"
    # QUERY = "quali sono i diritti del lavoratore?"
    # QUERY = "Esistono spazi all'interno per fumare?"
    # QUERY = "Le ferie sono retribuite?"
    # QUERY = "Quante ore settimanali posso lavorare?"
    # QUERY = "Quante ore è previsto che io lavori nella mia azienda?"
    
    if not os.path.exists(DB_PATH):
        print(f"DB not found at: {DB_PATH}")
        return
    
    retriever = DocumentRetriever()
    # results = retriever.search(QUERY, k=5)
    # results = retriever.routed_search(QUERY, k=5, threshold=0.0)
    results = retriever.complete_search(QUERY, k=5, threshold=0.0)  # test the new method

    print_results(results, QUERY)
    
    # print('\n')
    # print("=" * 60)
    # print("Chunks details:")
    
    # for i, r in enumerate(results, 1):
    #     print(f"{i}. Score: {r.get('score', 0):.3f} | "
    #           f"File: {r.get('filename')} | "
    #           f"Page: {r.get('page_number')} | "
    #           f"Chunk: {r.get('chunk_id')}"
    #         )


if __name__ == "__main__":
    main()