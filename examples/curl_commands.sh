#!/bin/bash
# Example cURL commands for RAG Service API

# 1. Crawl a website
echo "=== Crawling website ==="
curl -X POST http://localhost:8000/crawl \
  -H "Content-Type: application/json" \
  -d @examples/crawl_request.json

echo -e "\n\n=== Indexing crawled content ==="
# 2. Index the crawled content
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d @examples/index_request.json

echo -e "\n\n=== Asking an answerable question ==="
# 3. Ask a question that can be answered
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d @examples/ask_request_answerable.json

echo -e "\n\n=== Asking a question that requires refusal ==="
# 4. Ask a question that should be refused
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d @examples/ask_request_refusal.json

echo -e "\n\n=== Getting system statistics ==="
# 5. Get system stats
curl -X GET http://localhost:8000/stats

echo -e "\n\n=== Getting metrics ==="
# 6. Get Prometheus metrics
curl -X GET http://localhost:8000/metrics










