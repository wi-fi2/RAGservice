"""Integration tests for the complete RAG pipeline."""

import pytest
import asyncio
from unittest.mock import patch, Mock, AsyncMock
import httpx
from fastapi.testclient import TestClient
from app.server import app
import tempfile
import os


class TestAPIIntegration:
    """Test API endpoints integration."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_crawled_pages(self):
        """Mock crawled pages data."""
        return {
            "https://example.com": {
                "text": "Welcome to Example Company. We provide innovative solutions for data analytics.",
                "url": "https://example.com",
                "timestamp": 1234567890
            },
            "https://example.com/about": {
                "text": "Example Company was founded in 2020. Our mission is to democratize data analytics.",
                "url": "https://example.com/about",
                "timestamp": 1234567891
            },
            "https://example.com/products": {
                "text": "Our main product is CloudAnalytics, a powerful platform for business intelligence.",
                "url": "https://example.com/products",
                "timestamp": 1234567892
            }
        }
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    @patch('app.server.crawl_website', new_callable=AsyncMock)
    def test_crawl_endpoint(self, mock_crawl, client, mock_crawled_pages):
        """Test /crawl endpoint."""
        # Mock crawl result
        mock_crawl.return_value = {
            "page_count": 3,
            "skipped_count": 0,
            "urls": list(mock_crawled_pages.keys()),
            "pages": mock_crawled_pages
        }
        
        response = client.post("/crawl", json={
            "start_url": "https://example.com",
            "max_pages": 10,
            "max_depth": 2,
            "crawl_delay_ms": 500
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["page_count"] == 3
        assert data["skipped_count"] == 0
        assert len(data["urls"]) == 3
    
    @patch('app.server.crawled_pages', new={
        "https://example.com": {
            "text": "Test content for indexing.",
            "url": "https://example.com",
            "timestamp": 1234567890
        }
    })
    def test_index_endpoint(self, client):
        """Test /index endpoint."""
        response = client.post("/index", json={
            "chunk_size": 800,
            "chunk_overlap": 200,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["vector_count"] >= 1
        assert data["errors"] == []
    
    def test_index_without_crawl(self, client):
        """Test indexing without crawling first."""
        # Clear crawled pages
        import app.server
        app.server.crawled_pages = {}
        
        response = client.post("/index", json={
            "chunk_size": 800,
            "chunk_overlap": 200
        })
        
        assert response.status_code == 400
        assert "No pages to index" in response.json()["detail"]


class TestFullPipeline:
    """Test complete crawl -> index -> ask pipeline (network, embeddings model and LLM are faked)."""

    HTML_PAGES = {
        "https://example.com": """
            <html><body>
                <h1>Example Company</h1>
                <p>We are a leading provider of data analytics solutions.</p>
                <a href="/products">Our Products</a>
            </body></html>
        """,
        "https://example.com/products": """
            <html><body>
                <h1>Our Products</h1>
                <p>CloudAnalytics is our flagship product for business intelligence.</p>
            </body></html>
        """,
    }

    def test_complete_pipeline(self):
        """Crawl real parsing code on canned HTML, index it, and ask a question."""
        from app.crawler import PoliteCrawler

        async def fake_fetch(self, session, url, depth):
            html = TestFullPipeline.HTML_PAGES.get(url.rstrip("/"))
            return (html, "text/html", 200) if html else None

        with patch.object(PoliteCrawler, 'initialize_robots', new=AsyncMock()), \
             patch.object(PoliteCrawler, 'fetch_page', new=fake_fetch), \
             patch('app.server.call_ollama_llm', new=AsyncMock(
                 return_value="CloudAnalytics is the flagship product for business intelligence.")):
            with TestClient(app) as client:
                # Step 1: Crawl
                crawl_response = client.post("/crawl", json={
                    "start_url": "https://example.com",
                    "max_pages": 5,
                    "max_depth": 2,
                    "crawl_delay_ms": 100
                })
                assert crawl_response.status_code == 200
                assert crawl_response.json()["page_count"] == 2

                # Step 2: Index
                index_response = client.post("/index", json={
                    "chunk_size": 800,
                    "chunk_overlap": 200
                })
                assert index_response.status_code == 200
                assert index_response.json()["vector_count"] >= 2

                # Step 3: Ask
                ask_response = client.post("/ask", json={
                    "question": "CloudAnalytics flagship product for business intelligence",
                    "top_k": 5
                })
                assert ask_response.status_code == 200
                ask_data = ask_response.json()

                assert "CloudAnalytics" in ask_data["answer"]
                assert any("example.com/products" in s["url"] for s in ask_data["sources"])
                assert ask_data["timings"]["retrieval_ms"] > 0
                assert ask_data["timings"]["total_ms"] > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_ask_with_empty_index(self, client):
        """Test asking questions with empty index."""
        # Reset index
        import app.server
        app.server.indexer.create_new_index()
        app.server.indexer.metadata = []
        
        response = client.post("/ask", json={
            "question": "What is the company mission?",
            "top_k": 5
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "NOT_ENOUGH_INFORMATION" in data["answer"]
        assert data["sources"] == []
    
    def test_invalid_url_format(self, client):
        """Test crawling with invalid URL."""
        response = client.post("/crawl", json={
            "start_url": "not-a-valid-url",
            "max_pages": 10
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_ask_with_long_question(self, client):
        """Test question length limit."""
        long_question = "x" * 501  # Exceeds 500 char limit
        
        response = client.post("/ask", json={
            "question": long_question,
            "top_k": 5
        })
        
        assert response.status_code == 422  # Validation error










