# 🤖 Your Personal Assistant

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14-black?logo=next.js)](https://nextjs.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://www.docker.com/)

Your AI-powered local RAG assistant for chatting with your documents

[Features](#-features) • [Demo](#-demo) • [Architecture](#️-architecture) • [Quick Start](#-quick-start) • [Docker](#-docker)

---

## ✨ Features

- 📄 **Multi-Format Support** — Upload PDFs, text files, and markdown documents
- 🤖 **AI-Powered Chat** — Ask questions and get answers from your documents
- 🔍 **Hybrid Search** — Combines semantic and keyword search (BM25) for better results
- 💾 **Conversation History** — Keep track of your chat sessions
- 🔒 **Privacy-First** — Everything runs locally, no cloud services or API keys needed
- ⚡ **Real-Time Streaming** — Get responses as they're generated
- 🎯 **Background Processing** — Celery workers handle document indexing efficiently

---

## 🎬 Demo

> *Screenshots coming soon*

---

## 🏗️ Architecture

## 🏗️ Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Next.js UI     │────▶│  FastAPI Backend │────▶│  Ollama (Local)  │
│  (port 3000)     │     │   (port 8002)    │     │  Gemma 3 + Embed │
└──────────────────┘     └────────┬─────────┘     └──────────────────┘
                                  │
                     ┌────────────┼────────────┐
                     ▼            ▼            ▼
               ┌──────────┐ ┌──────────┐ ┌──────────┐
               │  Qdrant  │ │PostgreSQL│ │  Redis   │
               │ (Vectors)│ │(Metadata)│ │(History) │
               └──────────┘ └──────────┘ └──────────┘
                     ▲
                     │
               ┌──────────┐
               │  Celery  │
               │ (Workers)│
               └──────────┘
```

### Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Next.js 14, TypeScript, Tailwind CSS |
| Backend | FastAPI, Python 3.11+, LangChain |
| AI/ML | Ollama (Gemma 3 1B), nomic-embed-text |
| Vector DB | Qdrant |
| Databases | PostgreSQL, Redis |
| Task Queue | Celery |
| Infrastructure | Docker, Docker Compose |

---

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- [Ollama](https://ollama.ai/) installed locally
- 8GB RAM minimum
- 10GB disk space

### 1. Clone & Setup

```bash
git clone https://github.com/sanchitshaleen/your-personal-assistant.git
cd your-personal-assistant

---

## ⚙️ Configuration

Key settings in `config/settings.py`:

```python
# Models (default)
LLM_CHAT_MODEL_NAME = "gemma3:1b"      # 815MB, fast
EMB_MODEL_NAME = "nomic-embed-text"     # 274MB

# Retrieval
DOCS_NUM_COUNT = 3                      # Chunks per query
USE_BM25_SEMANTIC_HYBRID = True         # Hybrid search on/off

# Databases
QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333
POSTGRES_HOST = "postgres"
REDIS_HOST = "redis"

# Ollama connection
OLLAMA_BASE_URL = "http://host.docker.internal:11434"
```

Environment variables in `docker-compose.yml`:
```yaml
environment:
  - OLLAMA_BASE_URL=http://host.docker.internal:11434
  - USE_BM25_SEMANTIC_HYBRID=true
  - SEMANTIC_CACHE_ENABLED=true
```

### Switching Models

```bash
# Try different LLM
ollama pull gemma3:2b
# Update config/settings.py: LLM_CHAT_MODEL_NAME = "gemma3:2b"

# Try different embeddings
ollama pull mxbai-embed-large
# Update config/settings.py: EMB_MODEL_NAME = "mxbai-embed-large"

# Restart services
docker-compose restart fastapi celery-worker
```

---

## 📡 API Endpoints

**File Management:**
- `GET /uploads` - List user's files
- `POST /upload` - Upload file
- `DELETE /delete_file` - Delete file and embeddings
- `POST /embed` - Start embedding task
- `GET /embed/status/{task_id}` - Check progress

**Chat:**
- `POST /rag` - Chat with streaming (NDJSON)
- `POST /chat_history` - Get history
- `POST /clear_chat_history` - Clear history

**System:**
- `GET /` - Health check
- `GET /docs` - API documentation

Full docs at http://localhost:8002/docs when running.

---

## 🔧 Troubleshooting

### Services won't start
```bash
# Check status
docker-compose ps

# View logs
docker-compose logs -f fastapi

# Restart specific service
docker-compose restart fastapi
```

### Ollama connection failed
**Error:** `model 'gemma3:1b' not found (status code: 404)`

**Fix:**
```bash
# Verify Ollama is running
ollama list

# Pull missing models
ollama pull gemma3:1b
ollama pull nomic-embed-text

# Restart FastAPI
docker-compose restart fastapi
```

### Files stuck at "pending"
```bash
# Check Celery worker
docker-compose logs celery-worker

# Restart worker
docker-compose restart celery-worker

# Monitor tasks at http://localhost:5555
```

### Port already in use
```bash
# Kill process on port
lsof -ti:3000 | xargs kill -9

# Or change port in docker-compose.yml
ports:
  - "3001:3000"
```

### Out of memory
```bash
# Use smaller model
ollama pull gemma3:1b  # instead of 2b or 4b

# Reduce Docker memory in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 4G
```

### Clear everything and restart
```bash
# Remove all containers and data
docker-compose down -v

# Start fresh
docker-compose up -d
```

### Enable debug logging
Add to `docker-compose.yml` under fastapi service:
```yaml
environment:
  - LOG_LEVEL=DEBUG
```

Then restart and view logs:
```bash
docker-compose restart fastapi
docker-compose logs -f fastapi
```

---

## 🛠️ Tech Stack

**Backend:**
- FastAPI - API server
- LangChain - RAG orchestration
- Celery - Background tasks
- Python 3.11+

**Frontend:**
- Next.js 14 - React framework
- TypeScript
- Tailwind CSS

**Databases:**
- Qdrant - Vector storage
- PostgreSQL - File metadata
- Redis - Chat history & task queue

**AI:**
- Ollama - LLM server
- Gemma 3 1B - Language model
- nomic-embed-text - Embeddings

**Infrastructure:**
- Docker & Docker Compose

---

## 👨‍💻 Development

### Local development setup:

```bash
# Backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8002

# Worker (separate terminal)
celery -A src.worker worker --loglevel=info

# Frontend (separate terminal)
cd front_end
npm install
npm run dev
```

### Running tests:
```bash
# Backend tests
pytest

# Check code formatting
black src/
flake8 src/
```

---

## 🔮 Roadmap

- [ ] **Cloud Drive Integration** — Connect to Google Drive, Dropbox, OneDrive for seamless document sync
- [ ] **Async Folder Monitoring** — Automatic detection of changes in source folders with triggered re-indexing
- [ ] **Multi-Modal Support** — Process images, audio, and video with vision and speech models
- [ ] **Knowledge Graphs support** — Knowledge graph visualization of document relationships


---

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues and pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

GNU General Public License v3.0 - see [LICENSE](LICENSE) file.

- Free to use, modify, and distribute
- Derivative works must be open-source (GPL v3.0)
- Commercial use allowed with attribution

---

Built with ❤️ by [Sanchit Shaleen](https://github.com/sanchitshaleen)

