# ==================================================================
# Injection of chunks with paragraph titles into Milvus vector DB
# ==================================================================

import os
from typing import List, Dict, Any
from pymilvus import DataType
import numpy as np

from pdf_chunker import extract_pages, remove_header
from paragraph_chunker import get_pdf_titles, get_chunks_from_text, get_chunks_from_pdf
from db_injection import MilvusDB

METRIC_TYPE = "COSINE"
DB_PATH = "./final_db.db"


# extended MilvusDB class to support chunk titles metadata
class EnhancedMilvusDB(MilvusDB):
    
    def __init__(self, db_path=DB_PATH, collection_name="pdf_embeddings_with_titles", dimension=768):
        super().__init__(db_path, collection_name, dimension)
    
    # schema setup to include chunk_title field
    def setup_collection(self):
        """
        collection schema: id, vector, text, filename, page_number, chunk_id, chunk_title
        """
        
        if not self.client:
            raise ValueError("Database client not initialized")
        
        collections = self.client.list_collections()
        
        if self.collection_name in collections:
            # collection exists
            self.collection = self.collection_name
        else:
            # create new collection with chunk_title field
            schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
            
            # id and vector field
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dimension)
            
            # text content
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=8192)
            
            # metadata fields
            schema.add_field(field_name="filename", datatype=DataType.VARCHAR, max_length=255)
            schema.add_field(field_name="page_number", datatype=DataType.INT32)
            schema.add_field(field_name="chunk_id", datatype=DataType.INT32)
            schema.add_field(field_name="chunk_title", datatype=DataType.VARCHAR, max_length=512)
            
            self.client.create_collection(collection_name=self.collection_name, schema=schema)
            self.collection = self.collection_name
        
        return self.collection
    
    # insertion method to include chunk titles
    def insert_embeddings_with_titles(self, chunk_data: List[Dict[str, Any]], vectors: List[np.ndarray]):
        
        if not self.collection:
            raise ValueError("Collection missing")
        
        if len(chunk_data) != len(vectors):
            raise ValueError("Number of chunks must match number of vectors")

        data = []
        
        for i, (chunk, vector) in enumerate(zip(chunk_data, vectors)):
            
            if len(vector) != self.dimension:
                raise ValueError(f"Vector dimension {len(vector)} doesn't match expected {self.dimension}")
            
            # truncate text if too long
            text_content = chunk.get('text', '')
            if len(text_content) > 8192:
                text_content = text_content[:8189] + "..."
                print(f"Warning: Text truncated for chunk {chunk.get('chunk_index', i)}")
            
            # truncate title if too long
            chunk_title = chunk.get('chunk_title', 'UNKNOWN')
            if len(chunk_title) > 512:
                chunk_title = chunk_title[:509] + "..."
                print(f"Warning: Title truncated for chunk {chunk.get('chunk_index', i)}")
            
            data.append({
                "vector": vector.tolist(),
                "text": text_content,
                "filename": chunk['source_filename'],
                "page_number": chunk['page_number'],
                "chunk_id": chunk['chunk_index'],
                "chunk_title": chunk_title
            })
        
        try:
            insert_result = self.client.insert(collection_name=self.collection_name, data=data)
            
            # check insertion count
            expected_count = len(data)
            actual_count = insert_result.get("insert_count", 0)
            
            if actual_count != expected_count:
                return {
                    "success": False,
                    "error": f"Insertion mismatch: expected {expected_count}, got {actual_count}",
                    "expected_count": expected_count,
                    "actual_count": actual_count
                }
            
            # force flush
            self.client.flush(collection_name=self.collection_name)
            
            return {
                "success": True,
                "inserted_count": expected_count,
                "verified_count": actual_count,
                "collection_name": self.collection_name
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "attempted_count": len(data)
            }



# ==================================================================
# Main script for injection
# ==================================================================


pdf_files = [
    {
        "path": "datafile/ccnl_commercio_terziario_distribuzione_e_servizi.pdf",
        "START_PAGES_TO_SKIP": 8,
        "END_PAGES_TO_SKIP": 2
    },
    {
        "path": "datafile/BIS - Regolamento Aziendale.pdf",
        "START_PAGES_TO_SKIP": 0,
        "END_PAGES_TO_SKIP": 0
    },
    {
        "path": "datafile/codice etico fittizio_Salute e sicurezza dei lavoratori.pdf",
        "START_PAGES_TO_SKIP": 0,
        "END_PAGES_TO_SKIP": 0
    }
]


if __name__ == "__main__":
    
    print(f"Found {len(pdf_files)} PDF files configured for processing")
    for pdf_config in pdf_files:
        print(f"  - {pdf_config['path']}")
    
    print("\nConnecting to enhanced database")
    db = EnhancedMilvusDB()
    # db.clear_all_embeddings()  # uncomment to clear existing data
    db.create_index()
    print(f"Connected to: {db.db_path}, collection: {db.collection_name}")
    
    all_chunks = []
    
    for pdf_config in pdf_files:
        pdf_file = pdf_config["path"]
        print(f"\nProcessing: {pdf_file}")
        
        try:
            # extract pages and get basic info
            all_pages = extract_pages(pdf_file)
            num_pages = len(all_pages)
            final_pages, common_headers = remove_header(all_pages)
            
            # get titles with custom skip parameters
            titles = get_pdf_titles(pdf_file, common_headers, num_pages, keep_page_title=False, 
                                   start_pages_to_skip=pdf_config["START_PAGES_TO_SKIP"], 
                                   end_pages_to_skip=pdf_config["END_PAGES_TO_SKIP"])
            print(f"  - Extracted {len(titles)} titles from {num_pages} pages")
            
            # get chunks
            chunks = get_chunks_from_pdf(pdf_file, titles)
            
            if not chunks:
                print(f"No chunks generated for {pdf_file}")
                continue
            
            
            all_chunks.extend(chunks)
            
            print(f"  - Total pages: {num_pages}")
            print(f"  - Chunks: {len(chunks)}")
            print(f"  - Tokens: {sum(chunk['token_count'] for chunk in chunks)}")
            
        except Exception as e:
            print(f"Error in {pdf_file}: {e}")
            continue
    
    if not all_chunks:
        print("No chunks generated from PDFs")
        exit(1)
    
    print(f"\nSummary:")
    print(f"   - Total Chunks: {len(all_chunks)}")
    print(f"   - Total tokens: {sum(chunk['token_count'] for chunk in all_chunks)}")
    




    # embedding model
    print("\nInitializing embedding model")
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    
    try:
        embeddings = db.get_embeddings(all_chunks, model_name)
        
        print(f"Generated {len(embeddings)} embeddings")
        print(f"   Shape: {embeddings[0].shape}")
        
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        exit(1)

    print("\nInserting embeddings with titles into MilvusDB")
    result = db.insert_embeddings_with_titles(all_chunks, embeddings)
    
    print("\nInsertion results:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    if result['success']:
        print("\nSuccess: all chunks with titles inserted into database")
    else:
        print("\nError during insertion")
    
    # show final stats
    final_stats = db.get_collection_stats()
    print(f"\nFinal collection stats:")
    print(f"  - Collection: {final_stats.get('collection_name', 'unknown')}")
    print(f"  - Total entities: {final_stats.get('num_entities', 'unknown')}")
    print(f"  - Dimension: {final_stats.get('dimension', 'unknown')}")