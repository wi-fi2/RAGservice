"""CLI interface for RAG service."""

import asyncio
import json
import click
import httpx
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8000"


@click.group()
def cli():
    """RAG Service CLI - Crawl, index, and query websites."""
    pass


@cli.command()
@click.option('--url', '-u', required=True, help='Starting URL to crawl')
@click.option('--max-pages', '-p', default=50, help='Maximum pages to crawl')
@click.option('--max-depth', '-d', default=1, help='Maximum crawl depth')
@click.option('--delay', default=500, help='Crawl delay in milliseconds')
def crawl(url: str, max_pages: int, max_depth: int, delay: int):
    """Crawl a website."""
    click.echo(f"Crawling {url}...")
    
    payload = {
        "start_url": url,
        "max_pages": max_pages,
        "max_depth": max_depth,
        "crawl_delay_ms": delay
    }
    
    try:
        response = httpx.post(f"{API_BASE_URL}/crawl", json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        
        click.echo(f"✓ Crawled {result['page_count']} pages")
        click.echo(f"  Skipped: {result['skipped_count']} pages")
        click.echo(f"  URLs crawled:")
        for url in result['urls'][:10]:  # Show first 10
            click.echo(f"    - {url}")
        if len(result['urls']) > 10:
            click.echo(f"    ... and {len(result['urls']) - 10} more")
            
    except httpx.HTTPError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--chunk-size', '-c', default=800, help='Chunk size in characters')
@click.option('--overlap', '-o', default=200, help='Chunk overlap in characters')
@click.option('--model', '-m', default='sentence-transformers/all-MiniLM-L6-v2', help='Embedding model')
def index(chunk_size: int, overlap: int, model: str):
    """Index crawled pages."""
    click.echo("Indexing crawled content...")
    
    payload = {
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
        "embedding_model": model
    }
    
    try:
        response = httpx.post(f"{API_BASE_URL}/index", json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        
        click.echo(f"✓ Indexed {result['vector_count']} chunks")
        if result.get('errors'):
            click.echo("  Errors:")
            for error in result['errors']:
                click.echo(f"    - {error}")
                
    except httpx.HTTPError as e:
        click.echo(f"✗ Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument('question')
@click.option('--top-k', '-k', default=5, help='Number of chunks to retrieve')
@click.option('--json-output', is_flag=True, help='Output raw JSON response')
def ask(question: str, top_k: int, json_output: bool):
    """Ask a question about the indexed content."""
    
    payload = {
        "question": question,
        "top_k": top_k
    }
    
    try:
        response = httpx.post(f"{API_BASE_URL}/ask", json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo("\n" + "="*60)
            click.echo(f"Question: {question}")
            click.echo("="*60 + "\n")
            
            click.echo("Answer:")
            click.echo(result['answer'])
            
            if result['sources']:
                click.echo("\nSources:")
                for i, source in enumerate(result['sources'], 1):
                    click.echo(f"\n[{i}] {source['url']}")
                    if source.get('snippet'):
                        click.echo(f"    \"{source['snippet'][:150]}...\"")
            
            click.echo(f"\nTimings:")
            click.echo(f"  Retrieval: {result['timings']['retrieval_ms']:.1f}ms")
            click.echo(f"  Generation: {result['timings']['generation_ms']:.1f}ms")
            click.echo(f"  Total: {result['timings']['total_ms']:.1f}ms")
            
    except httpx.HTTPError as e:
        click.echo(f"✗ Error: {e}", err=True)
        raise click.Abort()


@cli.command()
def stats():
    """Show system statistics."""
    try:
        response = httpx.get(f"{API_BASE_URL}/stats")
        response.raise_for_status()
        stats = response.json()
        
        click.echo("System Statistics:")
        click.echo(f"  Crawled pages: {stats['crawled_pages']}")
        click.echo(f"  Indexed vectors: {stats['index_stats']['total_vectors']}")
        click.echo(f"  Model: {stats['model']}")
        click.echo(f"  Device: {stats['device']}")
        
    except httpx.HTTPError as e:
        click.echo(f"✗ Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument('url')
@click.option('--max-pages', '-p', default=30, help='Maximum pages to crawl')
def demo(url: str, max_pages: int):
    """Run a complete demo: crawl, index, and ask sample questions."""
    click.echo(f"Running demo on {url}\n")
    
    # Crawl
    click.echo("Step 1: Crawling...")
    crawl_response = httpx.post(
        f"{API_BASE_URL}/crawl",
        json={
            "start_url": url,
            "max_pages": max_pages,
            "max_depth": 3,
            "crawl_delay_ms": 500
        },
        timeout=300
    )
    crawl_response.raise_for_status()
    crawl_result = crawl_response.json()
    click.echo(f"✓ Crawled {crawl_result['page_count']} pages\n")
    
    # Index
    click.echo("Step 2: Indexing...")
    index_response = httpx.post(
        f"{API_BASE_URL}/index",
        json={
            "chunk_size": 800,
            "chunk_overlap": 200,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
        },
        timeout=300
    )
    index_response.raise_for_status()
    index_result = index_response.json()
    click.echo(f"✓ Indexed {index_result['vector_count']} chunks\n")
    
    # Ask sample questions
    sample_questions = [
        "What is the main purpose of this website?",
        "Who are the key people or team members mentioned?",
        "What products or services are offered?",
        "What is the company's mission or vision?",
        "How can I contact them?"
    ]
    
    click.echo("Step 3: Asking sample questions...\n")
    for question in sample_questions[:3]:  # Ask first 3 questions
        click.echo(f"Q: {question}")
        ask_response = httpx.post(
            f"{API_BASE_URL}/ask",
            json={"question": question, "top_k": 5},
            timeout=60
        )
        if ask_response.status_code == 200:
            result = ask_response.json()
            answer = result['answer']
            if len(answer) > 200:
                answer = answer[:200] + "..."
            click.echo(f"A: {answer}")
            if result['sources']:
                click.echo(f"   Sources: {len(result['sources'])} pages")
        click.echo()


if __name__ == '__main__':
    cli()








