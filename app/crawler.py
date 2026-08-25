

import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser
import time
from typing import Set, Dict, List, Optional, Tuple
import logging
from readability import Document
from app.utils import normalize_url, is_same_domain, clean_text_for_indexing

logger = logging.getLogger(__name__)


class PoliteCrawler:
    
    
    def __init__(
        self,
        start_url: str,
        max_pages: int = 50,
        max_depth: int = 3,
        crawl_delay_ms: int = 500,
        concurrent_limit: int = 3
    ):
        # Ensure the start_url has a scheme (default to https) so
        # downstream urlparse/normalization works reliably.
        parsed = urlparse(start_url)
        if not parsed.scheme:
            start_url = "https://" + start_url

        self.start_url = normalize_url(start_url)
        self.base_domain = urlparse(self.start_url).netloc
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.crawl_delay_ms = crawl_delay_ms
        self.concurrent_limit = concurrent_limit
        
        self.visited_urls: Set[str] = set()
        self.crawled_pages: Dict[str, Dict[str, any]] = {}
        self.skipped_count = 0
        self.robots_parser = None
        
    async def initialize_robots(self, session: aiohttp.ClientSession):
        """Load and parse robots.txt for the domain."""
        robots_url = urljoin(self.start_url, '/robots.txt')
        self.robots_parser = RobotFileParser()
        self.robots_parser.set_url(robots_url)
        
        try:
            async with session.get(robots_url, timeout=10) as response:
                if response.status == 200:
                    robots_content = await response.text()
                    # parse expects a list of lines
                    self.robots_parser.parse(robots_content.splitlines())
                    logger.info(f"Loaded robots.txt from {robots_url}")
                else:
                    # Treat missing/non-200 robots as permissive (no rules)
                    self.robots_parser.parse([])
        except Exception as e:
            logger.warning(f"Could not fetch robots.txt: {e}")
            # Fallback to permissive rules if robots can't be fetched/parsed
            try:
                self.robots_parser.parse([])
            except Exception:
                # As a last resort, clear the parser so can_fetch returns True
                self.robots_parser = None
    
    def can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt."""
        if not self.robots_parser:
            return True
        try:
            return self.robots_parser.can_fetch("*", url)
        except Exception:
            # If parser malfunctions, default to allowing the fetch
            return True
    
    async def fetch_page(
        self, 
        session: aiohttp.ClientSession, 
        url: str, 
        depth: int
    ) -> Optional[Tuple[str, str, int]]:
        """Fetch a single page and return (content, content_type, status)."""
        if not self.can_fetch(url):
            logger.info(f"Skipping {url} - disallowed by robots.txt")
            self.skipped_count += 1
            return None
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; RAGBot/1.0; +http://example.com/bot)'
            }
            async with session.get(url, headers=headers, timeout=30) as response:
                if response.status != 200:
                    logger.warning(f"Got status {response.status} for {url}")
                    return None
                
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' not in content_type:
                    logger.info(f"Skipping non-HTML content: {content_type}")
                    self.skipped_count += 1
                    return None
                
                content = await response.text()
                return content, content_type, response.status
                
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching {url}")
            self.skipped_count += 1
            return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            self.skipped_count += 1
            return None
    
    def extract_text_and_links(self, html: str, base_url: str) -> Tuple[str, List[str]]:
        """Extract clean text and valid links from HTML."""
        soup = BeautifulSoup(html, 'lxml')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Use readability for main content extraction
        try:
            doc = Document(html)
            summary = doc.summary()
            clean_soup = BeautifulSoup(summary, 'lxml')
            text = clean_soup.get_text(separator=' ', strip=True)
        except Exception:
            # Fallback to simple extraction
            text = soup.get_text(separator=' ', strip=True)
        
        # Clean the text
        text = clean_text_for_indexing(text)
        
        # Extract links
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = normalize_url(href, base_url)
            
            # Only include same-domain links
            if is_same_domain(absolute_url, self.start_url):
                links.append(absolute_url)
        
        return text, links
    
    async def crawl_page(
        self, 
        session: aiohttp.ClientSession,
        url: str, 
        depth: int,
        semaphore: asyncio.Semaphore
    ):
        """Crawl a single page and extract content."""
        async with semaphore:
            if url in self.visited_urls or len(self.crawled_pages) >= self.max_pages:
                return
            
            self.visited_urls.add(url)
            
            # Respect crawl delay
            await asyncio.sleep(self.crawl_delay_ms / 1000.0)
            
            # Fetch page
            result = await self.fetch_page(session, url, depth)
            if not result:
                return
            
            html_content, content_type, status_code = result
            
            # Extract text and links
            text, links = self.extract_text_and_links(html_content, url)
            
            # Store the page
            self.crawled_pages[url] = {
                'url': url,
                'text': text,
                'depth': depth,
                'timestamp': time.time(),
                'status_code': status_code,
                'links_found': len(links)
            }
            
            logger.info(f"Crawled {url} - {len(text)} chars, {len(links)} links")
            
            # Queue child pages if within depth limit
            if depth < self.max_depth:
                tasks = []
                for link in links[:10]:  # Limit breadth
                    if link not in self.visited_urls and len(self.crawled_pages) < self.max_pages:
                        task = self.crawl_page(session, link, depth + 1, semaphore)
                        tasks.append(task)
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
    
    async def crawl(self) -> Dict[str, any]:
        """Main crawl method."""
        logger.info(f"Starting crawl from {self.start_url}")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.concurrent_limit)
        
        # Create a session. Disable SSL verification in this environment
        # to avoid SSL certificate issues on some macOS setups. In
        # production you should enable SSL verification.
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Initialize robots.txt
            await self.initialize_robots(session)
            
            # Start crawling from the root URL
            await self.crawl_page(session, self.start_url, 0, semaphore)
        
        logger.info(f"Crawl complete: {len(self.crawled_pages)} pages, {self.skipped_count} skipped")
        
        return {
            'page_count': len(self.crawled_pages),
            'skipped_count': self.skipped_count,
            'urls': list(self.crawled_pages.keys()),
            'pages': self.crawled_pages
        }


async def crawl_website(
    start_url: str,
    max_pages: int = 50,
    max_depth: int = 3,
    crawl_delay_ms: int = 500
) -> Dict[str, any]:
    """Convenience function to crawl a website."""
    crawler = PoliteCrawler(
        start_url=start_url,
        max_pages=max_pages,
        max_depth=max_depth,
        crawl_delay_ms=crawl_delay_ms
    )
    return await crawler.crawl()


if __name__ == "__main__":
    # Simple CLI entrypoint for manual testing
    import logging, sys
    logging.basicConfig(level=logging.INFO)
    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    c = PoliteCrawler(start_url=test_url, max_pages=5, max_depth=1, crawl_delay_ms=100)
    result = asyncio.run(c.crawl())
    print({
        'page_count': result.get('page_count'),
        'skipped_count': result.get('skipped_count'),
        'urls': result.get('urls')[:10]
    })
