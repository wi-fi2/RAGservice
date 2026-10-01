"""Shared utilities, constants, and prompt templates."""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, urljoin

# Make heavy ML imports optional so utility functions can be used
# without installing large dependencies (useful for running the crawler).
try:
    import torch
except Exception:
    torch = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    pipeline = None

# Configuration constants
MIN_RELEVANCE_SCORE = 0.5  # Minimum cosine similarity for relevance
MAX_CONTEXT_LENGTH = 2048  # Maximum context tokens for generation (reduced for Phi-2)
DEFAULT_TEMPERATURE = 0.1  # Low temperature for factual generation

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prompt templates
GENERATION_PROMPT_TEMPLATE = """<s>[INST] <<SYS>>
You are a helpful assistant that answers questions ONLY using the provided source documents.

CRITICAL RULES:
1. You MUST only use information from the SOURCES provided below
2. You MUST cite the source URL for every claim you make
3. If the answer cannot be found in the sources, you MUST respond with: "NOT_ENOUGH_INFORMATION"
4. Do NOT add any external knowledge or make assumptions
5. Quote relevant snippets to support your answer
6. Be concise and factual
<</SYS>>

SOURCES:
{sources}

Question: {question}

Provide a clear, grounded answer with citations. If you cannot answer from the sources, say "NOT_ENOUGH_INFORMATION" and explain what information is missing. [/INST]"""

HARDENING_RULES = [
    # Remove potential prompt injections (drop script/style blocks including their contents)
    (r'<script\b[^>]*>.*?</script\s*>', ''),
    (r'<style\b[^>]*>.*?</style\s*>', ''),
    (r'</?script[^>]*>', ''),
    (r'</?style[^>]*>', ''),
    (r'javascript:', ''),
    (r'on\w+\s*=', ''),
    # Remove instruction-like patterns
    (r'ignore previous instructions', '[REMOVED]'),
    (r'disregard all prior', '[REMOVED]'),
    (r'system:', '[REMOVED]'),
    (r'assistant:', '[REMOVED]'),
]


def clean_text_for_indexing(text: str) -> str:
    """Clean and normalize text for indexing."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\'\"]+', ' ', text)
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    return text.strip()


def sanitize_content(text: str) -> str:
    """Apply security hardening to prevent prompt injection."""
    for pattern, replacement in HARDENING_RULES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE | re.DOTALL)
    return text


def is_same_domain(url1: str, url2: str) -> bool:
    """Check if two URLs belong to the same registrable domain."""
    try:
        domain1 = urlparse(url1).netloc.lower()
        domain2 = urlparse(url2).netloc.lower()
        
        # Simple check - for production, use tldextract
        # This handles most cases but not all edge cases
        parts1 = domain1.split('.')
        parts2 = domain2.split('.')
        
        if len(parts1) >= 2 and len(parts2) >= 2:
            # Compare last two parts (domain.tld)
            return parts1[-2:] == parts2[-2:]
        return domain1 == domain2
    except Exception:
        return False


def normalize_url(url: str, base_url: str = None) -> str:
    """Normalize URL for consistency."""
    if base_url:
        url = urljoin(base_url, url)
    
    # Remove fragment
    parsed = urlparse(url)
    normalized = parsed._replace(fragment='').geturl()
    
    # Remove trailing slash unless it's the root
    if normalized.endswith('/') and len(normalized) > len(parsed.scheme + '://' + parsed.netloc + '/'):
        normalized = normalized[:-1]
    
    return normalized


def chunk_text_with_overlap(
    text: str, 
    chunk_size: int = 800, 
    overlap: int = 200,
    respect_sentences: bool = True
) -> List[Dict[str, Any]]:
    """Chunk text with overlap, optionally respecting sentence boundaries."""
    if not text or chunk_size <= 0:
        return []
    
    chunks = []
    sentences = text.split('. ') if respect_sentences else [text]
    
    current_chunk = ""
    current_start = 0
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Add period back if it was removed
        if i < len(sentences) - 1:
            sentence += '.'
        
        # Check if adding this sentence exceeds chunk size
        if current_chunk and len(current_chunk) + len(sentence) + 1 > chunk_size:
            # Save current chunk
            chunks.append({
                'text': current_chunk,
                'start_char': current_start,
                'end_char': current_start + len(current_chunk)
            })
            
            # Start new chunk with overlap
            overlap_start = max(0, len(current_chunk) - overlap)
            current_chunk = current_chunk[overlap_start:] + ' ' + sentence
            current_start = current_start + overlap_start
        else:
            if current_chunk:
                current_chunk += ' ' + sentence
            else:
                current_chunk = sentence
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append({
            'text': current_chunk,
            'start_char': current_start,
            'end_char': current_start + len(current_chunk)
        })
    
    # Fallback to character-based chunking if no sentences
    if not chunks and text:
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            chunks.append({
                'text': chunk_text,
                'start_char': i,
                'end_char': i + len(chunk_text)
            })
    
    return chunks


def format_sources_for_prompt(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks for insertion into prompt."""
    sources_text = ""
    for i, chunk in enumerate(chunks):
        sources_text += f"\nSOURCE_{i+1}: {chunk['url']}\n"
        sources_text += f"SNIPPET_{i+1}: {sanitize_content(chunk['text'])}\n"
        sources_text += f"RELEVANCE_SCORE: {chunk.get('score', 0.0):.3f}\n"
        sources_text += "-" * 40 + "\n"
    return sources_text


class ModelAdapter:
    """Adapter for different LLM backends with fallback options."""
    
    def __init__(self, model_name: str = "microsoft/phi-2"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        logger.info(f"Initializing model {model_name} on {self.device}")
        
    def load_model(self):
        """Load model with fallback to smaller alternatives."""
        try:
            # Try primary model
            self._load_specific_model(self.model_name)
        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}: {e}")
            # Fallback to smaller model
            fallback_models = [
                "microsoft/phi-2",
                "google/flan-t5-large",
                "facebook/opt-1.3b"
            ]
            for fallback in fallback_models:
                try:
                    logger.info(f"Trying fallback model: {fallback}")
                    self._load_specific_model(fallback)
                    self.model_name = fallback
                    break
                except Exception:
                    continue
            
            if not self.model:
                raise RuntimeError("Could not load any model. Check your environment.")
    
    def _load_specific_model(self, model_name: str):
        """Load a specific model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load with appropriate settings for device
        model_kwargs = {
            "dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "low_cpu_mem_usage": True,
        }
        
        if self.device == "cpu":
            # CPU optimizations
            model_kwargs["dtype"] = torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Create pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            max_new_tokens=512,
            temperature=DEFAULT_TEMPERATURE,
            do_sample=False,  # Deterministic for grounding
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        logger.info(f"Successfully loaded {model_name}")
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text from prompt."""
        if not self.pipeline:
            self.load_model()
        
        # Truncate prompt if needed
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        if input_ids.shape[1] > MAX_CONTEXT_LENGTH:
            prompt = self.tokenizer.decode(input_ids[0, -MAX_CONTEXT_LENGTH:])
        
        # Generate
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        # Extract generated text
        generated = outputs[0]['generated_text']
        
        # Remove prompt from output
        if generated.startswith(prompt):
            generated = generated[len(prompt):]
        
        return generated.strip()


def parse_llm_response(response: str) -> Tuple[str, List[Dict[str, str]]]:
    """Parse LLM response to extract answer and sources."""
    # Check for refusal
    if "NOT_ENOUGH_INFORMATION" in response:
        return response.strip(), []
    
    # Extract answer and sources
    answer = response
    sources = []
    
    # Look for citation patterns like [source: URL]
    citation_pattern = r'\[source:\s*([^\]]+)\]'
    citations = re.findall(citation_pattern, response)
    
    for url in citations:
        sources.append({"url": url.strip(), "snippet": ""})
    
    # Remove citations from answer for cleaner text
    answer = re.sub(r'\s+', ' ', re.sub(citation_pattern, '', answer)).strip()
    
    # Try to extract snippets from the response
    snippet_pattern = r'SNIPPET_\d+:\s*([^\n]+)'
    snippets = re.findall(snippet_pattern, response)
    
    for i, snippet in enumerate(snippets[:len(sources)]):
        if i < len(sources):
            sources[i]["snippet"] = snippet.strip()
    
    return answer, sources


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation)."""
    # Rough estimate: 1 token ≈ 4 characters
    return len(text) // 4
