import os

# disable warnings on macos
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''

from typing import List
from rag import RAGWrapper

# test queries function ===================
def test_queries(rag: RAGWrapper, queries: List[str], k: int = 5):
    
    for i, question in enumerate(queries, 1):
        print("\n" + "="*80)
        print(f"TEST {i}/{len(queries)}")
        print("="*80)
        print(f"Question: {question}")
        print("-" * 60)
        
        try:
            # Get relevant documents with metadata
            full_results = rag.get_relevant_text(question, k=k, text_only=False)
            
            print("Retrieved Documents:")
            if not full_results:
                print("   No relevant documents found.")
            else:
                for j, result in enumerate(full_results, 1):
                    print(f"\n   {j}. Score: {result['score']:.1%} | {result['filename']} | Page: {result['page_number']}")
                    # Show preview of content
                    preview = result['text'].replace('\n', ' ').strip()
                    if len(preview) > 200:
                        preview = preview[:200] + "..."
                    print(f"      Content: {preview}")
            
            print("\n" + "-" * 60)
            
            # Get RAG response
            response = rag.ask(question, clean_response=True, show_prompt=False)
            print("Answer:")
            print(f"   {response['answer']}")
            
        except Exception as e:
            print(f"   Error processing query: {e}")
        
        print("="*80)


QUERIES = [
    "Cosa dice l'articolo 23 del cnnl?",
    "Chi era Albert Einstein?",
    "Potresti spiegarmi la differenza tra srl ed spa?",
    "Se dovessi assentarmi dal lavoro, posso comunicarlo a voce il giorno stesso?", # regolamento aziendale
    "Cosa comporta un passaggio di livello in azienda secondo il ccnl?",                            # ccnl commercio                               
    "Quali sono i doveri del lavoratore?",                                           # codice etico
    "cosa dice il regolamento aziendale su igiene e sicurezza?",
    "Quante ore è previso che io lavori nella mia azienda?",
    "Cosa dice l'articolo 23 del regolamento aziendale?",
    "Esistono spazi all'interno per fumare?",
    "Si può fumare secondo il regolamento aziendale?",
    "Si può andare a lavoro in pantaloncini d'estate?",
    "cosa devo fare in caso di assenza?"

]


# CHANGE IDX TO CHANGE QUERY <------
idx = 0
print_retieved = True


def main():
    question = QUERIES[idx]
    rag = RAGWrapper()

    print(f"Q[{idx}]: {question}\n")

    if print_retieved:
        texts = rag.get_relevant_text(question, k=5, text_only=True)
        print("Retrieved (before document routing):")
        for i, t in enumerate(texts[:3], 1):
            t = t.replace("\n", " ")
            print(f"- {t[:200]}{'...' if len(t) > 200 else ''}")

    resp = rag.ask(question, clean_response=True)
    print("="*80)
    print("\nAnswer:\n\n" + resp["answer"])


if __name__ == "__main__":
    main()
    
    # rag = RAGWrapper()
    # test_queries(rag, QUERIES, k=5)