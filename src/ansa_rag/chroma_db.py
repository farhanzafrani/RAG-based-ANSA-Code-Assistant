
import chromadb
from chromadb import PersistentClient
from functools import lru_cache

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import os

#----------------------------------
# Singleton Embedder (loaded only once)
#----------------------------------

@lru_cache(maxsize=1)
def get_embedder():
    # Better embedding models for code and technical documentation
    # Option 1: For mixed code + documentation (recommended)
    model_name = os.getenv("EMBEDDING_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
    
    # Alternative models you can try by setting EMBEDDING_MODEL in .env:
    # "BAAI/bge-large-en-v1.5" - Very good for technical content
    # "intfloat/e5-large-v2" - Strong general embedding
    # "nomic-ai/nomic-embed-text-v1.5" - Excellent for technical docs
    # "mixedbread-ai/mxbai-embed-large-v1" - Good performance
    
    embedder = SentenceTransformerEmbeddingFunction(model_name=model_name)
    return embedder

# --------------------------------------
# Singleton Chroma Client
# --------------------------------------
@lru_cache(maxsize=1)
def create_chroma_client():
    # Create a ChromaDB client with specified settings
    client = PersistentClient(path="./chroma_db")
    return client

# --------------------------------------
# ChromaDB Handler Class
# --------------------------------------
class ChromaDBHandler:
    """
    A high-level wrapper around ChromaDB that manages collections,
    adding/querying data, and performing common operations.
    """

    def __init__(self, collection_name: str):
        self.client = create_chroma_client()
        self.embedder = get_embedder()
        self.collection = self.get_or_create_collection(collection_name)
    
    #-------------------------
    # Utility: Get or create collection
    #-------------------------
    def get_or_create_collection(self, collection_name: str):
        if collection_name in [col.name for col in self.client.list_collections()]:
            return self.client.get_collection(name=collection_name)
        return self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedder
        )

    # -----------------------------
    # UPSERT (insert or replace)
    # -----------------------------
    def upsert_data(self, data, batch_size: int = 500):
        """
        Insert or replace documents by ID with progress reporting.
        """
        ids, docs, metadatas = zip(*data)
        total = len(ids)
        
        print(f"Inserting {total} documents in batches of {batch_size}...")
        
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_num = (start // batch_size) + 1
            total_batches = (total + batch_size - 1) // batch_size
            
            print(f"Processing batch {batch_num}/{total_batches} ({end-start} documents)...")
            
            try:
                self.collection.upsert(
                    ids=list(ids[start:end]),
                    documents=list(docs[start:end]),
                    metadatas=list(metadatas[start:end])
                )
            except Exception as e:
                print(f"Error in batch {batch_num}: {e}")
                raise
    
    def clear_collection(self):
        """
        Safely clear all data from the collection.
        """
        try:
            # Get all document IDs
            all_docs = self.collection.get()
            if all_docs and all_docs['ids']:
                # Delete in batches to avoid memory issues
                batch_size = 1000
                ids = all_docs['ids']
                for i in range(0, len(ids), batch_size):
                    batch_ids = ids[i:i + batch_size]
                    self.collection.delete(ids=batch_ids)
                return len(ids)
            return 0
        except Exception as e:
            # Fallback: delete all with where clause
            self.collection.delete(where={})
            return 0
    
    # -----------------------------
    # Enhanced Query Methods for ANSA API
    # -----------------------------
    def query_data(self, text_query: str, n_results: int = 5, content_type: str = None, 
                   content_category: str = None, module: str = None):
        """
        Query using raw text (auto-embedded) with optional ANSA-specific filtering.
        """
        where_clause = {}
        if content_type:
            where_clause["content_type"] = content_type
        if content_category:
            where_clause["content_category"] = content_category
        if module:
            where_clause["module"] = module
        
        results = self.collection.query(
            query_texts=[text_query],
            n_results=n_results,
            where=where_clause if where_clause else None
        )
        return results
    
    def search_api_functions(self, function_name: str, n_results: int = 5):
        """Search specifically for API functions by name."""
        # Search for chunks that contain API functions
        return self.collection.query(
            query_texts=[f"ansa.base.{function_name} ansa.{function_name}"],
            n_results=n_results,
            where={"has_api_content": True}
        )
    
    def search_by_module(self, module_name: str, query: str, n_results: int = 5):
        """Search within a specific ANSA module (base, mesh, morph, etc.)."""
        return self.query_data(query, n_results, module=module_name)
    
    def search_examples(self, query: str, n_results: int = 5):
        """Search specifically for code examples and usage patterns."""
        return self.query_data(query, n_results, content_category="example")
    
    def search_code(self, query: str, n_results: int = 5):
        """Search specifically for code-related content."""
        return self.query_data(query, n_results, content_type="code")
    
    def search_docs(self, query: str, n_results: int = 5):
        """Search specifically for documentation content."""
        return self.query_data(query, n_results, content_type="text")
    
    def hybrid_search(self, query: str, n_results: int = 8):
        """
        Perform a hybrid search that returns both code and documentation results,
        prioritizing API function matches.
        """
        # First, try to find API function matches
        api_results = self.collection.query(
            query_texts=[query],
            n_results=n_results//3,
            where={"has_api_content": True}
        )
        
        # Then get code examples
        code_results = self.search_code(query, n_results//3)
        
        # Finally get general documentation
        doc_results = self.search_docs(query, n_results//3)
        
        # Combine results
        combined_docs = (api_results.get('documents', [[]])[0] + 
                        code_results.get('documents', [[]])[0] + 
                        doc_results.get('documents', [[]])[0])
        combined_metadatas = (api_results.get('metadatas', [[]])[0] + 
                            code_results.get('metadatas', [[]])[0] + 
                            doc_results.get('metadatas', [[]])[0])
        combined_distances = (api_results.get('distances', [[]])[0] + 
                            code_results.get('distances', [[]])[0] + 
                            doc_results.get('distances', [[]])[0])
        
        return {
            'documents': [combined_docs],
            'metadatas': [combined_metadatas], 
            'distances': [combined_distances]
        }

    
