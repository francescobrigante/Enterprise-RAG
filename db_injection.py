#==========================================
# This file contains functions to manage MilvusDB for storing PDF embeddings
#==========================================

import os
import glob
from typing import List, Dict, Any
from pymilvus import DataType, MilvusClient
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from pdf_chunker import pdf_to_chunks

METRIC_TYPE = "COSINE"
DB_PATH = "./document_db.db"

class MilvusDB:
    
    def __init__(self, db_path = DB_PATH, collection_name = "pdf_embeddings", dimension = 768):
        """
        Args:
            db_path: path to local database file
            collection_name
            dimension: embedding vector dim
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.dimension = dimension
        self.client = None
        self.collection = None
        
        # initialize database and collection
        self.connect()
        self.setup_collection()
    
    
#=================== Database Functions ===================#    


    # connects to local milvusDB    
    def connect(self):
        try:
            self.client = MilvusClient(uri=self.db_path)
                
        except Exception as e:
            raise Exception(f"Failed to connect to local Milvus database: {e}")
    
    # creates collection if it doesnt exist, otherwise opens it
    def setup_collection(self):
        """
        Collection schema: id, vector, text, filename, page_number, chunk_id
        """
        
        if not self.client:
            raise ValueError("Database client not initialized")
        
        collections = self.client.list_collections()
        
        if self.collection_name in collections:
            # it exists
            self.collection = self.collection_name
        else:
            # create new collection
            schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
            
            # id and vector field
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dimension)
            
            # text
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=8192)
            
            # metadata
            schema.add_field(field_name="filename", datatype=DataType.VARCHAR, max_length=255)
            schema.add_field(field_name="page_number", datatype=DataType.INT32)
            schema.add_field(field_name="chunk_id", datatype=DataType.INT32)
            
            self.client.create_collection(collection_name=self.collection_name, schema=schema)
            
            self.collection = self.collection_name
        
        return self.collection
    
    def disconnect(self):
        if self.client:
            self.client.close()
            
    def create_index(self, index_type = "AUTOINDEX", metric_type = METRIC_TYPE):
        if not self.collection:
            raise ValueError("Collection not initialized.")

        try:
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type=index_type,
                metric_type=metric_type,
            )
            
            self.client.create_index(
                collection_name=self.collection_name,
                index_params=index_params,
            )
            
            return {"success": True, "index_type": index_type, "metric_type": metric_type}
        
        except Exception as e:
            msg = str(e).lower()
            if "exist" in msg or "already" in msg:
                return {"success": True, "exists": True, "index_type": index_type, "metric_type": metric_type}

            raise Exception(f"Failed to create index on '{self.collection_name}'. Details: {e}")



#=================== Embedding Functions ===================#

    # given model_name, it creates embeddings for the provided chunks. Returns list of embeddings
    @staticmethod
    def get_embeddings(chunks: List[Dict[str, Any]], model_name: str):
        
        if not chunks:
            return []
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        model = SentenceTransformer(model_name, device=device)
        
        texts = [chunk['text'] for chunk in chunks]
        
        batch_size = 32 if device == "cuda" else 16
        embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=True)
    
        return embeddings
    
    # inserts embeddings, text content, and metadata into the collection
    def insert_embeddings(self, chunk_data: List[Dict[str, Any]], vectors: List[np.ndarray]):
        
        if not self.collection:
            raise ValueError("Collection missing")
        
        if len(chunk_data) != len(vectors):
            raise ValueError("Number of chunks must match number of vectors")
        

        data = []
        
        for i, (chunk, vector) in enumerate(zip(chunk_data, vectors)):
            
            if len(vector) != self.dimension:
                raise ValueError(f"Vector dimension {len(vector)} doesn't match expected {self.dimension}")
            
            # VARCHAR max length is 8192, truncate if needed
            text_content = chunk.get('text', '')
            if len(text_content) > 8192:
                text_content = text_content[:8189] + "..."
                print(f"Warning: Text truncated for chunk {chunk.get('chunk_index', i)} (original length: {len(chunk.get('text', ''))})")
            
            data.append({
                "vector": vector.tolist(),
                "text": text_content,
                "filename": chunk['source_filename'],
                "page_number": chunk['page_number'],
                "chunk_id": chunk['chunk_index']
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
    
    # gets statistics about the collection
    def get_collection_stats(self) -> Dict[str, Any]:
        if not self.collection:
            return {"error": "Collection not initialized"}
        
        try:

            collection_info = self.client.describe_collection(self.collection_name)
            
            stats = {
                "collection_name": self.collection_name,
                "dimension": self.dimension,
                "db_path": self.db_path,
                "db_file_exists": os.path.exists(self.db_path),
                "collection_info": collection_info
            }
            
            # entities count
            try:
                query_result = self.client.query(
                    collection_name=self.collection_name,
                    filter="",
                    output_fields=["count(*)"]
                )
                stats["num_entities"] = len(query_result) if query_result else 0
            except:
                stats["num_entities"] = "unknown"
            
            return stats
            
        except Exception as e:
            return {
                "error": f"Failed to get collection stats: {e}",
                "collection_name": self.collection_name,
                "db_path": self.db_path
            }


    def clear_all_embeddings(self):

        if not self.collection:
            return {"success": False, "error": "Collection not initialized"}
        
        try:
            
            delete_result = self.client.delete(
                collection_name=self.collection_name,
                filter=""  # Empty filter deletes all
            )

            self.client.flush(collection_name=self.collection_name)
            
            
        except Exception as e:
            raise Exception(f"Failed to clear collection '{self.collection_name}': {e}")


#=================== Main: Actual DB Injection ===================#
    
if __name__ == "__main__":
    
    pdf_files = glob.glob("./datafile/*.pdf")
    
    if not pdf_files:
        print("No PDF files found in ./datafile directory")
        exit(1)
    
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"  - {pdf}")
    
    
    print("Connecting to DB")
    db = MilvusDB()
    # db.clear_all_embeddings()
    db.create_index()
    print(f"Connected to: {db.db_path}, collection: {db.collection_name}")
    print("Note: Database now stores text content along with embeddings for faster retrieval")
    
    
    all_chunks = []
    processing_stats = []
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file}")
        
        try:
            chunks, info = pdf_to_chunks(pdf_file)
            
            if 'error' in info:
                print(f"Error")
                continue
            
            all_chunks.extend(chunks)
            processing_stats.append(info)
            
        except Exception as e:
            print(f"Error in {pdf_file}: {e}")
            continue
    
    if not all_chunks:
        print("No chunks generated for PDF")
        exit(1)
    
    print("\n")
    print(f"   - Processed PDFs: {len(processing_stats)}")
    print(f"   - Total Chunks: {len(all_chunks)}")
    print(f"   - Total tokens: {sum(chunk['token_count'] for chunk in all_chunks)}")
    

    print("\nInitializing embedding model")
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    
    try:
        embeddings = db.get_embeddings(all_chunks, model_name)
        
        print(f"Correctly retrieved {len(embeddings)} embeddings")
        print(f"   Shape: {embeddings[0].shape}")
        
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        exit(1)

    print("\nInserting embeddings in MilvusDB")
    result = db.insert_embeddings(all_chunks, embeddings)
    
    print("\nInserted items:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    if result['success']:
        print("Total insertion completed")
    else:
        print("Error during insertion")