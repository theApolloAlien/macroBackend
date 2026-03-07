# MacroPollo — Backend

Local AI inference server for the MacroPollo financial analytics terminal.

## Stack
- **FastAPI** — REST API server
- **ChromaDB** — Vector database (institutional memory)
- **Qwen 3.5-4B** — Local LLM via LM Studio
- **SentenceTransformers** — Semantic embeddings
- **Tesseract OCR** — News article image ingestion

## Setup
```bash
pip install -r requirements.txt
python main.py
```
Server runs at `http://localhost:8081`

## Key Endpoints
| Endpoint | Description |
|---|---|
| `POST /analyze` | Core macro risk chain analysis |
| `POST /analyze/portfolio` | Portfolio vulnerability scanner |
| `POST /analyze/red_team` | Devil's Advocate contrarian view |
| `POST /analyze/contagion` | Ripple Effect contagion mapper |
| `POST /analyze/impact_chart` | 6-month asset impact projection |
| `GET /news_feed` | Live RSS macro news feed |
| `POST /remember` | Save to institutional memory |

## Architecture
All inference runs locally — zero data leaves the machine.
