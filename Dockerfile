FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv
# CPU-only torch keeps the image ~1.5 GB smaller. torchvision is pinned because sentence-transformers 2.2.2 requires it
# unpinned, and pip would otherwise pull a newer torchvision that drags in a different torch.
RUN pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN grep -v '^torch==' requirements.txt > /tmp/reqs.txt && pip install -r /tmp/reqs.txt
COPY app ./app
COPY static ./static
ENV PORT=8000 OLLAMA_BASE_URL=http://host.docker.internal:11434
RUN useradd -m appuser && mkdir -p /srv/data && chown -R appuser /srv
USER appuser
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT}"]
