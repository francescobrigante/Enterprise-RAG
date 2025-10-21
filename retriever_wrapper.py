from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List, Any

class RetrieverWrapper(BaseRetriever):
    
    retriever: Any
    k: int = 5
    threshold: float = 0.6
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, retriever, k: int = 5, threshold: float = 0.6):
        super().__init__(retriever=retriever, k=k, threshold=threshold)
    
    # dict result from custom retriever to langchain Document
    def _dict_to_document(self, result: dict) -> Document:
        metadata = {
            "filename": result.get("filename", "unknown"),
            "page_number": result.get("page_number", 0),
            "chunk_id": result.get("chunk_id", 0),
            "score": result.get("score", 0)
        }
        return Document(page_content=result["text"], metadata=metadata)
    
    def _get_relevant_documents(self, query: str, *, run_manager) -> List[Document]:
        # CHANGE HERE WITH SPECIFIED SEARCH FUNCTION
        results = self.retriever.complete_search(query, k=self.k, threshold=self.threshold)
        return [self._dict_to_document(result) for result in results]