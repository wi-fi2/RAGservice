
# RAG Service with Ollama

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-orange.svg)](https://ollama.ai)

A Retrieval-Augmented Generation (RAG) system that crawls websites, indexes content with vector embeddings, and answers questions using local Ollama LLMs.

## Features

- 🌐 **Web Crawler** - Crawl any website and extract clean content
- 🔍 **Vector Search** - Semantic search using sentence transformers
- 🤖 **Ollama LLM** - Local AI models (Gemma 2, Llama 3, Mistral, etc.)
- 💻 **Web UI** - Beautiful interface for crawling and Q&A
- ⌨️  **CLI** - Command-line interface for automation
- 🆓 **100% Free** - No API keys or subscriptions needed
- 🔒 **Private** - Everything runs locally on your machine

---

## 🚀 Quick Start (3 Steps)

### Step 1: Install Ollama & Pull a Model

```powershell
# Download Ollama from: https://ollama.ai/download
# Then pull a model:
ollama pull gemma2:2b   # Recommended (1.6GB, fast)
```

### Step 2: Install Python Dependencies

```powershell
pip install -r requirements.txt
```

### Step 3: Configure Environment

```powershell
# Copy the example environment file
copy env.example .env

# Edit .env if you want to change the model (optional)
```

### Step 4: Run the Server

```powershell
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```

**That's it!** Open your browser: **http://localhost:8000/static/index.html**

---

## 📝 How to Use

### Web UI (Easiest)

1. **Enter URL** (e.g., `https://en.wikipedia.org/wiki/Python_(programming_language)`)
2. **Set max pages** (e.g., `10`)
3. **Click "🚀 Start Crawling"**
4. **Click "📚 Index Pages"**
5. **Ask your question** and get AI-powered answers with sources!

### CLI (For Automation)

```bash
# Crawl a website
python -m app.cli crawl --url "https://example.com" --max-pages 30

# Index the content
python -m app.cli index

# Ask questions
python -m app.cli ask "Your question here"
```

---

## ⚙️ Alternative: Use Startup Script

```powershell
.\start_rag.ps1
```

This automatically:
- ✅ Checks if Ollama is installed
- ✅ Starts Ollama service
- ✅ Verifies the model exists
- ✅ Starts the RAG server

---

## 🎯 Available Models

| Model | Size | RAM | Quality |
|-------|------|-----|---------|
| `gemma2:2b` | 1.6 GB | 2 GB | Good ⭐ (Recommended) |
| `phi3` | 2.3 GB | 2 GB | Good |
| `mistral` | 4.1 GB | 4 GB | Better |
| `llama3` | 4.7 GB | 8 GB | Best |

```powershell
# Install any model:
ollama pull <model-name>
```

### Switching Models

Edit `.env` file:
```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma2:2b   # Change this to any model
```

---

## 📁 Project Structure

```
 RAG/
├── app/
│   ├── server.py       # Main FastAPI server with Ollama
│   ├── crawler.py      # Web crawler
│   ├── indexer.py      # Vector indexer
│   ├── utils.py        # Utilities
│   ├── metrics.py      # Metrics collection
│   └── cli.py          # CLI interface
│
├── static/
│   └── index.html      # Web UI
│
├── .env                # Configuration
├── README.md           # This file
├── requirements.txt    # Python dependencies
└── start_rag.ps1       # Startup script
```

---

## 🔧 Configuration

The `.env` file controls settings:

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma2:2b
```

---

## 📊 API Endpoints

- `GET /` - Health check
- `GET /stats` - System statistics
- `GET /metrics` - Prometheus metrics
- `POST /crawl` - Crawl a website
- `POST /index` - Index crawled content
- `POST /ask` - Ask a question

### Example API Usage

```bash
# Crawl
curl -X POST http://localhost:8000/crawl \
  -H "Content-Type: application/json" \
  -d '{"start_url": "https://example.com", "max_pages": 30}'

# Index
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"chunk_size": 800, "chunk_overlap": 200}'

# Ask
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Your question?", "top_k": 5}'
```

---

## 🐛 Troubleshooting

### Server not starting

**Check if Ollama is running:**
```powershell
ollama serve
```

**Verify model is installed:**
```powershell
ollama list
ollama pull gemma2:2b
```

### Not getting answers

**Issue**: "NOT_ENOUGH_INFORMATION" message

**Solution**: You need to crawl and index content first!
1. Open http://localhost:8000/static/index.html
2. Enter a URL and click "Start Crawling"
3. Click "Index Pages"
4. Now try asking questions

### Port already in use

```powershell
# Kill process on port 8000
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process
```

### Test if Ollama is working

```powershell
ollama run gemma2:2b "Hello"
```

---

## 💡 How It Works

1. **Crawl**: Fetches web pages, respects robots.txt, extracts clean text
2. **Chunk**: Splits content into 800-character chunks with 200-char overlap
3. **Embed**: Creates vector embeddings using sentence-transformers
4. **Index**: Stores vectors in FAISS for fast similarity search
5. **Search**: Finds relevant chunks using cosine similarity
6. **Generate**: Ollama LLM generates answers based on retrieved context

---

## 📋 Requirements

- Python 3.10+
- Ollama installed and running
- 2-8GB RAM (depending on model)
- Internet connection (for crawling)

---

## 🎓 Example Workflow

```powershell
# 1. Start Ollama (Terminal 1)
ollama serve

# 2. Start RAG server (Terminal 2)
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000

# 3. Open browser
http://localhost:8000/static/index.html

# 4. Try it out:
# - URL: https://en.wikipedia.org/wiki/Mahatma_Gandhi
# - Max Pages: 10
# - Crawl → Index → Ask: "Who was Mahatma Gandhi?"
```

---

## 🛑 Stopping the Server

Press `Ctrl+C` in the terminal window running the server

---

## 💡 Tips

- Start with small websites (5-10 pages) for testing
- Use specific questions for better answers
- The AI cites sources from the crawled pages
- Gemma 2 2B is fastest, Llama 3 is highest quality
- Increase max pages for more comprehensive coverage

---

## 📦 What's Included

This is a clean, production-ready RAG system with:
- ✅ Single FastAPI server with Ollama integration
- ✅ Clean codebase (old demo files removed)
- ✅ Beautiful web UI for easy interaction
- ✅ CLI for automation
- ✅ Vector search with FAISS
- ✅ Local LLM inference (no cloud dependencies)

---

## 🆘 Quick Reference

**Start Server:**
```powershell
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```

**Web UI:**
```
http://localhost:8000/static/index.html
```

**Check Ollama:**
```powershell
ollama list
ollama serve
```

---

## 📄 License

See LICENSE file.

---

## 🎉 You're Ready!

Just run the server and start crawling! Your local RAG system is ready to answer questions about any website you index.

**Happy querying!** 🚀
>>>>>>> 2af9d9a3 (Initial commit)
