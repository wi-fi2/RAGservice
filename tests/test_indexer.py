"""Tests for the indexer module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os
from app.indexer import VectorIndexer
from app.utils import chunk_text_with_overlap


class TestChunking:
    """Test text chunking functionality."""
    
    def test_basic_chunking(self):
        """Test basic text chunking."""
        text = "This is a test sentence. " * 50  # ~1250 characters
        chunks = chunk_text_with_overlap(text, chunk_size=200, overlap=50)
        
        assert len(chunks) > 1
        assert all(len(chunk['text']) <= 200 for chunk in chunks)
        assert chunks[0]['start_char'] == 0
        
        # Check overlap
        if len(chunks) > 1:
            # There should be some overlap between consecutive chunks
            overlap_text = chunks[0]['text'][-50:]
            assert overlap_text in chunks[1]['text']
    
    def test_sentence_aware_chunking(self):
        """Test that chunking respects sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        chunks = chunk_text_with_overlap(
            text, 
            chunk_size=40, 
            overlap=10,
            respect_sentences=True
        )
        
        # Check that chunks end at sentence boundaries
        for chunk in chunks:
            chunk_text = chunk['text'].strip()
            assert chunk_text.endswith('.') or chunk == chunks[-1]
    
    def test_empty_text_chunking(self):
        """Test chunking with empty text."""
        chunks = chunk_text_with_overlap("", chunk_size=100, overlap=20)
        assert chunks == []
    
    def test_single_chunk(self):
        """Test text that fits in a single chunk."""
        text = "Short text."
        chunks = chunk_text_with_overlap(text, chunk_size=100, overlap=20)
        
        assert len(chunks) == 1
        assert chunks[0]['text'] == text
        assert chunks[0]['start_char'] == 0
        assert chunks[0]['end_char'] == len(text)


class TestVectorIndexer:
    """Test vector indexing functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def indexer(self, temp_dir):
        """Create a test indexer instance."""
        return VectorIndexer(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            index_path=os.path.join(temp_dir, "test_index"),
            metadata_path=os.path.join(temp_dir, "test_metadata.json")
        )
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_indexer_initialization(self, mock_st, temp_dir):
        """Test indexer initialization."""
        mock_model = Mock()
        mock_st.return_value = mock_model
        
        indexer = VectorIndexer(
            index_path=os.path.join(temp_dir, "test_index"),
            metadata_path=os.path.join(temp_dir, "test_metadata.json")
        )
        
        assert indexer.index is not None
        assert indexer.metadata == []
        mock_st.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2")
    
    def test_chunk_pages(self, indexer):
        """Test page chunking."""
        pages = {
            "https://example.com": {
                "text": "This is page one content. " * 50,
                "timestamp": 123456789
            },
            "https://example.com/about": {
                "text": "This is page two content. " * 30,
                "timestamp": 123456790
            }
        }
        
        chunks = indexer.chunk_pages(pages, chunk_size=200, chunk_overlap=50)
        
        assert len(chunks) > 2  # Should have multiple chunks
        assert all('chunk_id' in chunk for chunk in chunks)
        assert all('url' in chunk for chunk in chunks)
        assert chunks[0]['url'] == "https://example.com"
    
    @patch('sentence_transformers.SentenceTransformer.encode')
    def test_generate_embeddings(self, mock_encode, indexer):
        """Test embedding generation."""
        # Mock embeddings
        mock_embeddings = np.random.rand(3, 384).astype(np.float32)
        mock_encode.return_value = mock_embeddings
        
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = indexer.generate_embeddings(texts, batch_size=2)
        
        assert embeddings.shape == (3, 384)
        assert mock_encode.call_count == 2  # Called twice due to batch_size=2
    
    @patch('sentence_transformers.SentenceTransformer.encode')
    def test_search(self, mock_encode, indexer):
        """Test vector search functionality."""
        # Setup: index some chunks
        chunks = [
            {"text": "Python programming", "url": "https://example.com/python", "chunk_id": "1"},
            {"text": "Java programming", "url": "https://example.com/java", "chunk_id": "2"},
            {"text": "Machine learning", "url": "https://example.com/ml", "chunk_id": "3"}
        ]
        
        # Mock embeddings for indexing
        index_embeddings = np.random.rand(3, 384).astype(np.float32)
        index_embeddings = index_embeddings / np.linalg.norm(index_embeddings, axis=1, keepdims=True)
        
        # Mock query embedding (similar to first chunk)
        query_embedding = index_embeddings[0:1] + np.random.rand(1, 384) * 0.1
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        mock_encode.side_effect = [index_embeddings, query_embedding]
        
        # Index the chunks
        indexer.index.add(index_embeddings)
        indexer.metadata = chunks
        
        # Search
        results = indexer.search("Python", top_k=2)
        
        assert len(results) <= 2
        assert all('score' in result for result in results)
        assert all('url' in result for result in results)
    
    def test_save_and_load_index(self, indexer):
        """Test saving and loading index."""
        # Create some test data
        chunks = [{"text": "Test chunk", "url": "https://example.com", "chunk_id": "1"}]
        embeddings = np.random.rand(1, 384).astype(np.float32)
        
        # Add to index
        indexer.index.add(embeddings)
        indexer.metadata = chunks
        
        # Save
        indexer.save_index()
        
        # Create new indexer and check it loads
        new_indexer = VectorIndexer(
            index_path=indexer.index_path,
            metadata_path=indexer.metadata_path
        )
        
        assert new_indexer.index.ntotal == 1
        assert len(new_indexer.metadata) == 1
        assert new_indexer.metadata[0]['text'] == "Test chunk"










