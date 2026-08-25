"""Document indexing with chunking, embedding, and vector storage."""

import os
import json
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import logging
from tqdm import tqdm
from app.utils import chunk_text_with_overlap, clean_text_for_indexing

logger = logging.getLogger(__name__)


class VectorIndexer:
    """Handles text chunking, embedding generation, and vector indexing."""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_dim: int = 384,  # Dimension for all-MiniLM-L6-v2
        index_path: str = "data/faiss_index",
        metadata_path: str = "data/metadata.json"
    ):
        self.embedding_model_name = embedding_model
        self.vector_dim = vector_dim
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        # Create data directory
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Initialize model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize or load index
        self.index = None
        self.metadata = []
        self.load_index()
    
    def load_index(self):
        """Load existing index or create new one."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded existing index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Could not load index: {e}")
                self.create_new_index()
        else:
            self.create_new_index()
    
    def create_new_index(self):
        """Create a new FAISS index."""
        # Using IndexFlatIP for inner product (works well with normalized vectors)
        self.index = faiss.IndexFlatIP(self.vector_dim)
        self.metadata = []
        logger.info("Created new FAISS index")
    
    def save_index(self):
        """Save index and metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"Saved index with {self.index.ntotal} vectors")
    
    def chunk_pages(
        self, 
        pages: Dict[str, Dict[str, Any]], 
        chunk_size: int = 800,
        chunk_overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """Chunk all pages into smaller segments."""
        all_chunks = []
        
        for url, page_data in pages.items():
            text = page_data.get('text', '')
            if not text:
                continue
            
            # Generate chunks for this page
            chunks = chunk_text_with_overlap(
                text, 
                chunk_size=chunk_size,
                overlap=chunk_overlap,
                respect_sentences=True
            )
            
            # Add metadata to each chunk
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    'chunk_id': f"{url}#chunk_{i}",
                    'url': url,
                    'text': chunk['text'],
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'start_char': chunk['start_char'],
                    'end_char': chunk['end_char'],
                    'timestamp': page_data.get('timestamp', 0)
                }
                all_chunks.append(chunk_data)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
        return all_chunks
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        embeddings = []
        
        # Process in batches for efficiency
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch,
                normalize_embeddings=True,  # Important for cosine similarity
                show_progress_bar=False
            )
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings, dtype=np.float32)
    
    def index_chunks(
        self,
        chunks: List[Dict[str, Any]],
        chunk_size: int = 800,
        chunk_overlap: int = 200
    ) -> Dict[str, Any]:
        """Index chunks into vector store."""
        if not chunks:
            return {
                'vector_count': 0,
                'errors': ['No chunks to index']
            }
        
        try:
            # Extract texts
            texts = [chunk['text'] for chunk in chunks]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} chunks")
            embeddings = self.generate_embeddings(texts)
            
            # Clear existing index
            self.create_new_index()
            
            # Add to index
            self.index.add(embeddings)
            
            # Store metadata
            self.metadata = chunks
            
            # Save to disk
            self.save_index()
            
            # Log configuration
            logger.info(f"Indexing configuration: chunk_size={chunk_size}, overlap={chunk_overlap}, model={self.embedding_model_name}")
            
            return {
                'vector_count': len(chunks),
                'errors': [],
                'config': {
                    'chunk_size': chunk_size,
                    'chunk_overlap': chunk_overlap,
                    'embedding_model': self.embedding_model_name,
                    'vector_dim': self.vector_dim
                }
            }
            
        except Exception as e:
            logger.error(f"Indexing error: {e}")
            return {
                'vector_count': 0,
                'errors': [str(e)]
            }
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks in the index."""
        if not self.index or self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Search index
        distances, indices = self.index.search(
            query_embedding.astype(np.float32),
            min(top_k, self.index.ntotal)
        )
        
        # Collect results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata) and dist > score_threshold:
                chunk = self.metadata[idx].copy()
                chunk['score'] = float(dist)  # Cosine similarity score
                chunk['rank'] = i + 1
                results.append(chunk)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'metadata_count': len(self.metadata),
            'embedding_model': self.embedding_model_name,
            'vector_dimension': self.vector_dim,
            'index_size_mb': os.path.getsize(self.index_path) / 1024 / 1024 if os.path.exists(self.index_path) else 0
        }


async def index_crawled_pages(
    pages: Dict[str, Dict[str, Any]],
    chunk_size: int = 800,
    chunk_overlap: int = 200,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Dict[str, Any]:
    """Convenience function to index crawled pages."""
    indexer = VectorIndexer(embedding_model=embedding_model)
    
    # Chunk the pages
    chunks = indexer.chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Index the chunks
    result = indexer.index_chunks(chunks, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    return result










