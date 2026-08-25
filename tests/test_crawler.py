"""Tests for the web crawler."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import aiohttp
from app.crawler import PoliteCrawler, crawl_website


class TestPoliteCrawler:
    """Test cases for PoliteCrawler."""
    
    @pytest.fixture
    def crawler(self):
        """Create a test crawler instance."""
        return PoliteCrawler(
            start_url="https://example.com",
            max_pages=10,
            max_depth=2,
            crawl_delay_ms=100
        )
    
    @pytest.mark.asyncio
    async def test_robots_txt_parsing(self, crawler):
        """Test robots.txt parsing."""
        # Mock robots.txt content
        robots_content = """
User-agent: *
Disallow: /admin/
Disallow: /private/
Allow: /public/
        """
        
        # Mock session
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=robots_content)
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        await crawler.initialize_robots(mock_session)
        
        # Test allowed/disallowed URLs
        assert crawler.can_fetch("https://example.com/public/page.html")
        assert not crawler.can_fetch("https://example.com/admin/secret.html")
        assert not crawler.can_fetch("https://example.com/private/data.html")
    
    def test_domain_validation(self, crawler):
        """Test same-domain validation."""
        from app.utils import is_same_domain
        
        # Same domain - should be True
        assert is_same_domain("https://example.com/page1", "https://example.com/page2")
        assert is_same_domain("https://www.example.com/page1", "https://example.com/page2")
        assert is_same_domain("http://example.com/page1", "https://example.com/page2")
        
        # Different domain - should be False
        assert not is_same_domain("https://example.com/page1", "https://different.com/page2")
        assert not is_same_domain("https://example.com/page1", "https://sub.example.different.com/page2")
    
    @pytest.mark.asyncio
    async def test_text_extraction(self, crawler):
        """Test HTML text extraction and cleaning."""
        html_content = """
        <html>
            <head>
                <title>Test Page</title>
                <script>console.log('test');</script>
                <style>body { color: black; }</style>
            </head>
            <body>
                <nav>Navigation menu</nav>
                <main>
                    <h1>Main Content</h1>
                    <p>This is the main content of the page.</p>
                    <a href="/page2">Link to page 2</a>
                    <a href="https://example.com/page3">Link to page 3</a>
                    <a href="https://external.com/page">External link</a>
                </main>
                <footer>Footer content</footer>
            </body>
        </html>
        """
        
        text, links = crawler.extract_text_and_links(html_content, "https://example.com/page1")
        
        # Check text extraction
        assert "Main Content" in text
        assert "This is the main content" in text
        assert "console.log" not in text  # Script removed
        assert "Navigation menu" not in text  # Nav removed
        assert "Footer content" not in text  # Footer removed
        
        # Check links
        assert "https://example.com/page2" in links
        assert "https://example.com/page3" in links
        assert "https://external.com/page" not in links  # External link excluded
    
    @pytest.mark.asyncio
    async def test_max_pages_limit(self):
        """Test that crawler respects max_pages limit."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            # Mock responses
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {'Content-Type': 'text/html'}
            mock_response.text = AsyncMock(return_value="""
                <html><body>
                    <a href="/page2">Page 2</a>
                    <a href="/page3">Page 3</a>
                    <a href="/page4">Page 4</a>
                </body></html>
            """)
            
            mock_session.get = AsyncMock(return_value=mock_response)
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            
            # Create crawler with max_pages=2
            crawler = PoliteCrawler(
                start_url="https://example.com",
                max_pages=2,
                max_depth=3,
                crawl_delay_ms=10
            )
            
            result = await crawler.crawl()
            
            # Should only crawl 2 pages maximum
            assert result['page_count'] <= 2


@pytest.mark.asyncio
async def test_crawl_website_integration():
    """Test the main crawl_website function."""
    with patch('app.crawler.PoliteCrawler.crawl') as mock_crawl:
        mock_crawl.return_value = {
            'page_count': 5,
            'skipped_count': 1,
            'urls': ['https://example.com', 'https://example.com/about'],
            'pages': {
                'https://example.com': {
                    'text': 'Homepage content',
                    'url': 'https://example.com'
                }
            }
        }
        
        result = await crawl_website("https://example.com", max_pages=10)
        
        assert result['page_count'] == 5
        assert result['skipped_count'] == 1
        assert len(result['urls']) == 2










