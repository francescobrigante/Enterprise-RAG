#==========================================
# Main RAG file including wrapper with LangChain and Groq
#==========================================

import os
import textwrap

# disable warnings on macos
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''

from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from retrieve import DocumentRetriever
from retriever_wrapper import RetrieverWrapper

# load API key from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

# LLM_NAME = "openai/gpt-oss-120b"
# LLM_NAME = "llama-3.3-70b-versatile"
LLM_NAME = "qwen/qwen3-32b"


# ======================== RAG Wrapper ===================
class RAGWrapper:
 
    def __init__(self):
        
        # initialize custom retriever
        self.retriever = DocumentRetriever()
        
        # initialize LLM 
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=LLM_NAME,
            temperature=0.1
        )
        
        # custom prompt
        if LLM_NAME.startswith("qwen"):
                instructions = textwrap.dedent("""
                    ### System:
                    Sei un esperto consulente del lavoro e risorse umane, con un'ampia conoscenza specifica su questo dominio.
                    Usa un tono formale e professionale, ma anche cordiale. Stai parlando con un dipendente dell'azienda.
                    Rispondi in modo pertinente alla domanda basandoti esclusivamente sui documenti forniti.
                    Questi documenti riguardano norme e comportamenti, diritti e doveri nel lavoro, come CCNL, Regolamento Aziendale, ecc.
                    Non dire "secondo i documenti forniti...", "consulta il documento..." se non esplicitamente richiesto, ma rispondi direttamente alla domanda.
                    Se la domanda riguarda l'azienda, le ferie, diritti e doveri del dipendente e dell'azienda, regole, norme, rispondi anche se le informazioni non sono presenti nei documenti.
                    Se invece la domanda non riguarda questi argomenti, quindi non è pertinente, rispondi "L'argomento della domanda non è pertinente."
                    Se non ci sono Reference Documents, se la domanda è pertinente rispondi alla domanda senza aggiungere altro.
                    Se non ci sono Reference Documents e la domanda non è pertinente, rispondi "L'argomento della domanda non è pertinente."
                    Non usare <think> ma rispondi direttamente alla domanda.
                    ### End System.

                    ### Question:
                    {question}
                    
                    ### Reference Documents:
                    {context}

                    ### Response:
                """).strip()
            
        elif LLM_NAME.startswith("llama"):
                instructions = textwrap.dedent("""
                    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                    Sei un esperto consulente del lavoro e risorse umane, con un'ampia conoscenza specifica su questo dominio.
                    Usa un tono formale e professionale, ma anche cordiale. Stai parlando con un dipendente dell'azienda.
                    Rispondi in modo pertinente alla domanda basandoti esclusivamente sui documenti forniti.
                    Questi documenti riguardano norme e comportamenti, diritti e doveri nel lavoro, come CCNL, Regolamento Aziendale, ecc.
                    Non dire "secondo i documenti forniti...", "consulta il documento..." se non esplicitamente richiesto, ma rispondi direttamente alla domanda.
                    Se la domanda riguarda l'azienda, le ferie, diritti e doveri del dipendente e dell'azienda, regole, norme, rispondi anche se le informazioni non sono presenti nei documenti.
                    Se invece la domanda non riguarda questi argomenti, quindi non è pertinente, rispondi "L'argomento della domanda non è pertinente."
                    Se non ci sono Reference Documents, se la domanda è pertinente rispondi alla domanda senza aggiungere altro.
                    Se non ci sono Reference Documents e la domanda non è pertinente, rispondi "L'argomento della domanda non è pertinente."
                    Non usare <think> ma rispondi direttamente alla domanda.
                    <|eot_id|>
                    
                    <|start_header_id|>user<|end_header_id|>
                    Domanda: {question}
                    
                    Documenti di riferimento:
                    {context}
                    <|eot_id|>
                    
                    <|start_header_id|>assistant<|end_header_id|>
                """).strip()
            
        else:
            instructions = textwrap.dedent("""
                System:
                Sei un esperto consulente del lavoro e risorse umane, con un'ampia conoscenza specifica su questo dominio.
                Usa un tono formale e professionale, ma anche cordiale. Stai parlando con un dipendente dell'azienda.
                Rispondi in modo pertinente alla domanda basandoti esclusivamente sui documenti forniti.
                Questi documenti riguardano norme e comportamenti, diritti e doveri nel lavoro, come CCNL, Regolamento Aziendale, ecc.
                Non dire "secondo i documenti forniti...", "consulta il documento..." se non esplicitamente richiesto, ma rispondi direttamente alla domanda.
                Se la domanda riguarda l'azienda, le ferie, diritti e doveri del dipendente e dell'azienda, regole, norme, rispondi anche se le informazioni non sono presenti nei documenti.
                Se invece la domanda non riguarda questi argomenti, quindi non è pertinente, rispondi "L'argomento della domanda non è pertinente."
                Se non ci sono Reference Documents, se la domanda è pertinente rispondi alla domanda senza aggiungere altro.
                Se non ci sono Reference Documents e la domanda non è pertinente, rispondi "L'argomento della domanda non è pertinente."

                Domanda:
                {question}
                
                Documenti di riferimento:
                {context}

                Risposta:
            """).strip()
                        
                        
        self.prompt_template = PromptTemplate(
            input_variables=["question", "context"],
            template=instructions
        )
        
        # QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            # custom retriever using wrapper
            retriever=RetrieverWrapper(self.retriever),
            chain_type_kwargs={"prompt": self.prompt_template},
            return_source_documents=True
        )
    
    # given a query, it retrieves relevant text
    # if text_only == False, then it returns also metadata
    def get_relevant_text(self, query: str, k: int = 5, text_only: bool = True):

        results = self.retriever.search(query, k=k)
        
        if text_only:
            return [result["text"] for result in results]
        else:
            return results
        
    # ask question to get response using RAG
    # if clean_response==True, it removes thinking process of LLM from output
    def ask(self, question: str, clean_response: bool = True, show_prompt: bool = False) -> Dict[str, Any]:

        result = self.qa_chain.invoke({"query": question})
        
        answer = result["result"]
        
        # gets context from source documents
        source_docs = result["source_documents"]
        context = "\n\n".join([doc.page_content for doc in source_docs])
        
        # gets prompt
        full_prompt = self.prompt_template.format(question=question, context=context)
        
        # show prompt
        if show_prompt:
            print("\n" + "="*80)
            print("FULL PROMPT SENT TO LLM:")
            print("="*80)
            print(full_prompt)
            print("="*80 + "\n")
        
        # clean by extracting everything after </think> or remove <think> sections
        if clean_response:
            if "</think>" in answer:
                answer = answer.split("</think>", 1)[1].strip()
            # error case: think opened but not closed
            elif "<think>" in answer:
                answer = "Mi scuso, si è verificato un errore nella generazione della risposta. Potresti ripetere la domanda?"
    
        return {
            "question": question,
            "answer": answer,
            "full_prompt": full_prompt,  # Aggiungi il prompt alla risposta
            "source_documents": [
                {
                    "content": doc.page_content,
                    "filename": doc.metadata.get("filename", "unknown"),
                    "page": doc.metadata.get("page_number", 0)
                }
                for doc in result["source_documents"]
            ]
        }




# ======================== Main ===================

def main():

    try:
        rag = RAGWrapper()
        

        # question = "Quante volte siamo andati nello spazio?"
        # question = "Esistono spazi all'interno per fumare?" # no similarity match
        # question = "Si può fumare secondo il regolamento aziendale?"
        # question = "l'equazione differenziale di primo ordine della tanzania è stata inventata da alfio?"
        # question = "Le ferie sono retribuite?"
        # question = "Si può andare a lavoro in pantaloncini d'estate?"
        #question = "Quante ore settimanali posso lavorare?"
        question = input("Insert the query please ")
        print("\n")
        print("="*60)
        print(f"Question: {question}\n")
        
        print("Retrieved relevant texts:")
        texts = rag.get_relevant_text(question, k=5, text_only=True)
        for i, text in enumerate(texts, 1):
            preview = text[:150] + "..." if len(text) > 150 else text
            print(f"\t{i}. {preview}\n")
        
        # print scores
        print("Metadata:")
        full_results = rag.get_relevant_text(question, k=5, text_only=False)
        for i, result in enumerate(full_results, 1):
            print(f"{i}. Score: {result['score']:.3f} | {result['filename']} | Page: {result['page_number']}")
        
        # RAG response
        print("\n\n\nAnswer:")
        response = rag.ask(question, clean_response=True, show_prompt=False)
        print(response["answer"])
        
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()