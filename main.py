from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import uvicorn
import feedparser
import requests as req
import uuid
import base64
import io
import re
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from PIL import Image, ImageOps, ImageFilter, ImageEnhance
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ─── Clients ─────────────────────────────────────────────────────────────────
client = OpenAI(base_url="http://192.168.1.17:1234/v1", api_key="not-needed")
chroma_client = chromadb.PersistentClient(path="./institutional_memory_db")
collection = chroma_client.get_or_create_collection(name="institutional_memory")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ─── Topic taxonomy ───────────────────────────────────────────────────────────
VALID_TOPICS = [
    "Monetary Policy", "Geopolitics", "Commodities", "Equities",
    "Fixed Income", "FX & Currency", "Macro Data", "Credit Markets",
    "Energy", "Technology", "General",
]

# ─── News sources ─────────────────────────────────────────────────────────────
NEWS_SOURCES = [
    {"name": "CNBC",        "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html"},
    {"name": "MarketWatch", "url": "https://feeds.marketwatch.com/marketwatch/topstories/"},
    {"name": "Reuters",     "url": "https://feeds.reuters.com/reuters/businessNews"},
]

RSS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def default_meta(source: str = "manual", order: int = 0) -> dict:
    """Minimal metadata for a new memory before AI categorisation runs."""
    return {
        "title":      "",
        "topic":      "General",
        "tags":       "",
        "summary":    "",
        "created_at": now_iso(),
        "order":      order,
        "source":     source,
    }

def ensure_meta(meta: dict | None) -> dict:
    """Fill in any missing fields so old seeded docs never cause KeyErrors."""
    base = default_meta()
    if meta:
        base.update(meta)
    return base

def tags_to_list(tags_str: str) -> list[str]:
    """Parse comma-separated tags string → clean list."""
    return [t.strip() for t in tags_str.split(",") if t.strip()]

def categorize_memory_sync(doc_id: str, text: str):
    """
    Background task: calls LLM to assign title / topic / tags / summary,
    then updates the ChromaDB metadata for that document.
    Runs after the HTTP response has been sent so save is never blocked.
    """
    try:
        response = client.chat.completions.create(
            model="qwen3.5-4b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a macro finance taxonomy engine. "
                        f"Analyse the note and return ONLY valid JSON — no markdown fences, no commentary.\n\n"
                        f"Valid topics: {', '.join(VALID_TOPICS)}\n\n"
                        f"Return this exact schema:\n"
                        f'{{"title":"5-8 word title","topic":"one of the valid topics",'
                        f'"tags":["tag1","tag2","tag3"],"summary":"one sentence under 20 words"}}'
                    ),
                },
                {"role": "user", "content": text[:600]},
            ],
            temperature=0.1,
            max_tokens=150,
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'^```json\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)

        # Fetch existing metadata to preserve fields like created_at and order
        result = collection.get(ids=[doc_id], include=["metadatas"])
        if not result["ids"]:
            return
        existing = ensure_meta(result["metadatas"][0] if result["metadatas"] else None)

        topic = data.get("topic", "General")
        if topic not in VALID_TOPICS:
            topic = "General"

        existing.update({
            "title":   str(data.get("title", text[:50])),
            "topic":   topic,
            "tags":    ",".join(str(t) for t in data.get("tags", [])),
            "summary": str(data.get("summary", "")),
        })

        collection.update(ids=[doc_id], metadatas=[existing])
        print(f"[memory] Categorised {doc_id} → {topic}")

    except Exception as e:
        print(f"[memory] Categorisation failed for {doc_id}: {e}")


def strip_html(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_image_from_html(html: str):
    match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', html or '', re.IGNORECASE)
    return match.group(1) if match else None

def friendly_time(published_str: str) -> str:
    try:
        dt = parsedate_to_datetime(published_str)
        now = datetime.now(dt.tzinfo)
        diff = now - dt
        minutes = int(diff.total_seconds() / 60)
        if minutes < 60:
            return f"{minutes}m ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours}h ago"
        return f"{diff.days}d ago"
    except Exception:
        return ""

def fetch_feed(url: str) -> list:
    try:
        r = req.get(url, headers=RSS_HEADERS, timeout=12)
        r.raise_for_status()
        feed = feedparser.parse(r.content)
        return feed.entries or []
    except Exception as e:
        print(f"[news] fetch_feed failed for {url}: {e}")
        return []

def fetch_og_image(url: str):
    try:
        r = req.get(url, headers=RSS_HEADERS, timeout=4)
        for pattern in [
            r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
        ]:
            m = re.search(pattern, r.text, re.IGNORECASE)
            if m:
                return m.group(1)
        return None
    except Exception:
        return None

def parse_entry_no_image(entry, source_name: str) -> dict:
    image_url = None
    media_content = getattr(entry, "media_content", None)
    if media_content:
        for m in media_content:
            url = m.get("url", "")
            if url and (m.get("medium") == "image" or "image" in m.get("type", "")
                        or re.search(r'\.(jpg|jpeg|png|webp)(\?|$)', url, re.I)):
                image_url = url
                break
        if not image_url:
            image_url = media_content[0].get("url")
    if not image_url:
        thumb = getattr(entry, "media_thumbnail", None)
        if thumb:
            image_url = thumb[0].get("url")
    if not image_url:
        raw_html = entry.get("summary", "") or entry.get("description", "")
        image_url = extract_image_from_html(raw_html)

    summary = strip_html(entry.get("summary", entry.get("description", "")))
    if len(summary) > 180:
        summary = summary[:177] + "…"

    return {
        "title":     entry.get("title", "Untitled"),
        "summary":   summary,
        "url":       entry.get("link", ""),
        "image_url": image_url,
        "published": entry.get("published", ""),
        "time_ago":   friendly_time(entry.get("published", "")),
        "source":     source_name,
        "risk_score":  0,
        "risk_reason": "",
    }

def score_article(article: dict) -> dict:
    """
    Calls the LLM to assign a macro risk score (1–10) and a one-sentence reason.
    Mutates and returns the article dict with 'risk_score' and 'risk_reason' keys.
    Falls back gracefully on any error so the feed always loads.
    """
    text = f"{article.get('title', '')}. {article.get('summary', '')}"
    try:
        response = client.chat.completions.create(
            model="qwen3.5-4b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a macro risk classifier for a professional investment team. "
                        "Rate the macro economic significance of the news headline and summary "
                        "on a scale of 1–10, where:\n"
                        "1–3 = low (company earnings, minor data, soft human-interest)\n"
                        "4–6 = moderate (sector trends, regional policy, mild data surprise)\n"
                        "7–8 = high (central bank signals, geopolitical tension, major data beat/miss)\n"
                        "9–10 = critical (emergency rate decisions, war escalation, systemic crisis)\n\n"
                        "Reply ONLY with valid JSON, no markdown fences:\n"
                        '{"score": <integer 1-10>, "reason": "<one sentence under 15 words>"}'
                    ),
                },
                {"role": "user", "content": text[:300]},
            ],
            temperature=0.0,
            max_tokens=60,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        score = int(data.get("score", 0))
        article["risk_score"]  = max(1, min(10, score))
        article["risk_reason"] = str(data.get("reason", ""))
    except Exception as e:
        print(f"[score] failed for '{article.get('title', '')[:40]}': {e}")
        article.setdefault("risk_score", 0)
        article.setdefault("risk_reason", "")
    return article


def score_articles_batch(articles: list[dict]) -> list[dict]:
    """Score all articles concurrently — max 8 threads to avoid LLM overload."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(score_article, a): i for i, a in enumerate(articles)}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"[score_batch] unexpected error: {e}")
    return articles


def enrich_with_images(articles: list[dict]) -> list[dict]:
    needs = [(i, a) for i, a in enumerate(articles) if not a.get("image_url")]
    if not needs:
        return articles
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(fetch_og_image, a["url"]): i for i, a in needs}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                img = future.result()
                if img:
                    articles[idx]["image_url"] = img
            except Exception:
                pass
    return articles

def is_macro_relevant(text: str) -> tuple[bool, str]:
    text_stripped = text.strip()
    text_lower = text_stripped.lower()
    greetings = {
        "hello", "hi", "hey", "thanks", "thank you", "ok", "okay", "yes",
        "no", "bye", "good morning", "good evening", "sup", "what's up",
        "how are you", "test", "testing", "ping", "yo", "hiya"
    }
    if text_lower in greetings or len(text_stripped) < 8:
        return False, (
            "I only analyse macro economic events. "
            "Try: *'The Fed raises rates by 50bps'* or *'China restricts rare earth exports'*."
        )
    try:
        response = client.chat.completions.create(
            model="qwen3.5-4b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict input classifier for a macro economics risk tool. "
                        "Decide if the input describes a macro economic event suitable for financial risk analysis.\n\n"
                        "RELEVANT: central bank policy, inflation/CPI/PPI, GDP, geopolitical events, "
                        "commodity shocks, currency crises, trade policy, earnings, sanctions, "
                        "supply chain disruptions, sovereign debt, rate changes, energy prices, banking stress.\n\n"
                        "NOT RELEVANT: greetings, general chat, personal questions, unrelated topics.\n\n"
                        "Reply ONLY:\nRELEVANT\nor\nIRRELEVANT: [one short sentence]"
                    ),
                },
                {"role": "user", "content": text_stripped},
            ],
            temperature=0.0,
            max_tokens=60,
        )
        result = response.choices[0].message.content.strip()
        if result.startswith("RELEVANT"):
            return True, ""
        if result.startswith("IRRELEVANT:"):
            reason = result.split("IRRELEVANT:", 1)[-1].strip()
            return False, reason or "Please describe a macro economic event."
        return True, ""
    except Exception:
        return True, ""


# ─── Seed data ────────────────────────────────────────────────────────────────

mock_articles = [

    # ── Storyline 1: The Tech / AI Supply Shock ───────────────────────────────

    (
        "TSMC Warns of Extended Lead Times After Magnitude-7.4 Earthquake Strikes "
        "Hualien, Taiwan — TSMC's advanced N3 and N5 fabs in the Hsinchu Science Park "
        "suspended operations for 72 hours following a major seismic event, triggering "
        "allocation alerts across Apple, NVIDIA, and AMD supply chains. "
        "Spot wafer prices on the secondary market jumped 18% intraday as hyperscalers "
        "scrambled to secure priority capacity, pushing ASML and Tokyo Electron to "
        "multi-month highs on supply-constraint premium repricing.",
        "seed"
    ),
    (
        "U.S. Commerce Department Expands AI Chip Export Controls, Blacklisting "
        "12 Chinese Entities Including SMIC Advanced and Biren Technology — The Bureau "
        "of Industry and Security tightened restrictions on H100-equivalent and above "
        "GPUs, closing the 'distributor loophole' that had allowed re-export via "
        "Singapore and Malaysia. "
        "NVIDIA guided Q3 Data Center revenue $2.1B below consensus on the news, "
        "while domestic Chinese AI semiconductor proxies — Cambricon and Hygon — "
        "surged 30%+ on expectations of accelerated state procurement.",
        "seed"
    ),
    (
        "Global AI Infrastructure Capex Cycle Drives Defensive Tech Re-Rating as "
        "Microsoft, Google, and Meta Collectively Commit $320B in FY2025 Data Center "
        "Spend — Power infrastructure, cooling systems, and fiber backhaul providers "
        "are being reclassified as 'critical utility' assets by institutional allocators, "
        "compressing their equity risk premiums by 80–120bps. "
        "Vertiv, Eaton, and Schneider Electric hit all-time highs as the market priced "
        "a 10-year capex supercycle independent of the broader semiconductor cycle.",
        "seed"
    ),
    (
        "NVIDIA Blackwell Architecture Shipment Delays Push Hyperscaler AI Roadmaps "
        "Back Two Quarters, Sparking Debate Over 'AI Capex Bubble' — CoWoS advanced "
        "packaging constraints at TSMC created a supply bottleneck for GB200 NVL72 "
        "rack systems, with lead times extending to 52 weeks versus an expected 24. "
        "Goldman Sachs downgraded the broader AI infrastructure basket to Neutral, "
        "noting that $150B in committed capex could yield lower-than-modeled GPU "
        "utilisation rates if model efficiency gains (per Deepseek R1 benchmarks) "
        "continue to outpace raw compute demand.",
        "seed"
    ),
    (
        "Taiwan Strait Military Exercises Prompt Lloyd's of London Syndicates to "
        "Re-Price Semiconductor Supply Chain Interruption Coverage — Underwriters "
        "increased annual premiums for fab disruption policies by 40–65% following "
        "a 72-hour PLA naval exclusion zone drill around the Pescadores Islands. "
        "The repricing is accelerating U.S. CHIPS Act disbursements to Intel Foundry "
        "and TSMC Arizona, though analysts note domestic capacity won't be material "
        "until 2027, leaving a multi-year window of concentrated geopolitical risk "
        "for the global semiconductor supply chain.",
        "seed"
    ),

    # ── Storyline 2: Deflationary East vs. Inflationary West ─────────────────

    (
        "China's National Development and Reform Commission Cuts 2025 GDP Growth "
        "Target to 4.5%, Signalling Structural Acceptance of Property Sector "
        "Deflation — The Politburo acknowledged that Evergrande and Country Garden "
        "liquidations have permanently impaired household balance sheets, with "
        "residential property prices in Tier-2 and Tier-3 cities down 22% from "
        "peak on a real basis. "
        "Consumer price index printed at -0.3% YoY for the third consecutive month, "
        "raising the spectre of a Japan-style balance-sheet recession and prompting "
        "the PBOC to cut the 5-year LPR by 25bps while the RRR was lowered to 8.5%.",
        "seed"
    ),
    (
        "U.S. Core PCE Inflation Prints at 3.4% for Third Consecutive Month, "
        "Forcing Federal Reserve to Abandon its March Rate-Cut Guidance — Super-core "
        "services inflation — stripping out housing — re-accelerated to 4.1% as "
        "wage growth in healthcare, hospitality, and financial services remained "
        "sticky above pre-pandemic trends. "
        "Fed funds futures shifted from pricing 75bps of 2025 cuts to just 25bps "
        "by end of week, triggering a 14bps bear-flattening of the 2s10s curve and "
        "a sharp reversal in rate-sensitive equity sectors including REITs and utilities.",
        "seed"
    ),
    (
        "ECB Cuts Rates 25bps While Fed Holds, Creating the Widest EUR/USD Rate "
        "Differential Since 2002 and Pushing the Pair Toward 1.02 Parity — Christine "
        "Lagarde cited 'materially below-target' inflation in Germany and France, "
        "where industrial output contracted 1.8% MoM driven by Chinese EV competition "
        "and elevated energy costs relative to U.S. peers. "
        "The divergence is creating significant carry-trade dynamics: EM central banks "
        "with dollar-denominated debt are facing dual pressure from a strengthening USD "
        "and weakening export demand, with Turkey, Egypt, and Nigeria most exposed.",
        "seed"
    ),
    (
        "Chinese Deflationary Export Pulse Intensifies as PPI Falls to -2.8% YoY, "
        "Flooding Global Markets With Below-Cost Steel, EVs, and Solar Panels — "
        "Chinese steel exports hit a 9-year high of 11.2M tonnes in a single month "
        "as domestic construction demand collapsed, prompting the EU to impose "
        "emergency safeguard tariffs of 25% and the U.S. to triple Section 301 "
        "levies on Chinese EVs to 102.5%. "
        "The 'deflationary export' dynamic is creating a bifurcated global inflation "
        "picture where goods disinflation masks persistent services inflation in "
        "Western economies, complicating central bank reaction functions.",
        "seed"
    ),
    (
        "Japan Intervenes in FX Market for the Third Time in 12 Months, Spending "
        "an Estimated ¥5.5 Trillion to Defend the 158 Level Against the Dollar — "
        "The Ministry of Finance acted after USD/JPY broke through multi-decade "
        "resistance levels, with the carry trade — borrowing in yen to buy "
        "higher-yielding U.S. Treasuries — reaching a notional size of $4.2 trillion "
        "according to BIS estimates. "
        "A disorderly unwind of yen carry positions is considered a systemic tail risk, "
        "as it would simultaneously pressure U.S. Treasury yields higher and trigger "
        "margin calls across leveraged global equity positions.",
        "seed"
    ),

    # ── Storyline 3: Commodity / Geopolitical Squeeze ────────────────────────

    (
        "OPEC+ Announces Surprise 1.65 Million Barrel Per Day Production Cut Effective "
        "Immediately, Blindsiding Markets That Had Priced in a Gradual Tapering of "
        "Existing Restrictions — Saudi Arabia and Russia cited 'global demand "
        "uncertainty' as cover for the decision, though analysts widely interpreted "
        "it as a fiscal defence mechanism with Riyadh requiring $92/bbl to balance "
        "its Vision 2030 budget. "
        "Brent crude spiked 8.4% intraday to $96.70 before settling at $93.10, "
        "reigniting stagflation concerns in G7 economies and forcing a hawkish "
        "reassessment from central banks that had conditionally priced in energy "
        "disinflation as a path to easing.",
        "seed"
    ),
    (
        "Houthi Missile Strikes Force Maersk, Hapag-Lloyd, and MSC to Permanently "
        "Re-Route All Red Sea Traffic Around the Cape of Good Hope, Adding "
        "14 Days and $1.2M Per Voyage in Fuel Costs — Container spot rates on "
        "the Asia-Europe route surged 312% from their 2023 trough to $6,800/FEU "
        "as effective container ship capacity fell 15% due to longer voyage distances. "
        "The disruption is creating a secondary inflationary pulse in European goods "
        "prices estimated at 0.4–0.7% CPI, while Egyptian Suez Canal toll revenues "
        "collapsed 60% YoY, pressuring Egypt's IMF programme covenants.",
        "seed"
    ),
    (
        "Gold Breaks Above $2,800/oz for the First Time on Record as Central Banks "
        "Accelerate De-Dollarisation Purchases and Geopolitical Risk Premia Expand — "
        "The People's Bank of China disclosed its 18th consecutive month of gold "
        "reserve additions, with total PBoC holdings now at 2,264 tonnes, while "
        "Poland, India, and Turkey also reported significant Q3 purchases. "
        "Western institutional flows are simultaneously rotating into gold as a "
        "hedge against both U.S. fiscal sustainability concerns — with the deficit "
        "running at 7.2% of GDP — and the tail risk of a disorderly Treasury "
        "market event given the $9.2 trillion in debt maturing within 12 months.",
        "seed"
    ),
    (
        "U.S. 10-Year Treasury Yield Breaches 5.25%, the Highest Level Since 2007, "
        "as Foreign Demand at Auctions Weakens and Fiscal Deficit Concerns Mount — "
        "The bid-to-cover ratio at a $44B 10-year auction fell to 2.24x, the weakest "
        "since 2011, as Japanese and Chinese buyers — historically the two largest "
        "foreign holders — reduced participation amid their own domestic pressures. "
        "The yield spike triggered an immediate repricing of risk assets: the "
        "equity risk premium inverted for the S&P 500 for the first time since 2002, "
        "theoretically making cash and short-duration Treasuries more attractive "
        "on a risk-adjusted basis than equities at current P/E multiples.",
        "seed"
    ),
    (
        "Iran Seizes Second Commercial Vessel in the Strait of Hormuz in 60 Days, "
        "Pushing Oil Tanker War Risk Insurance Premiums to Levels Last Seen During "
        "the 2019 Abqaiq Attack — The seizure of a Marshall Islands-flagged VLCC "
        "carrying 2M barrels of Iraqi crude prompted the U.S. Fifth Fleet to deploy "
        "two additional destroyers to the region, escalating the potential for "
        "direct confrontation. "
        "Approximately 21% of global oil supply transits the Strait of Hormuz daily; "
        "a full closure scenario is modelled by Goldman Sachs to push Brent crude "
        "to $130–$150/bbl within 30 days, sufficient to trigger a global recession "
        "in most macro stress models.",
        "seed"
    ),
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    if collection.count() == 0:
        print("Seeding database with mock data...")
        for i, (text, source) in enumerate(mock_articles):
            meta = default_meta(source=source, order=i)
            collection.add(
                embeddings=[embedder.encode(text).tolist()],
                documents=[text],
                ids=[f"seed_{i}"],
                metadatas=[meta],
            )
        # Categorise seeds in background
        for i, (text, _) in enumerate(mock_articles):
            try:
                categorize_memory_sync(f"seed_{i}", text)
            except Exception:
                pass
        try:
            for src in NEWS_SOURCES:
                entries = fetch_feed(src["url"])
                if entries:
                    for j, e in enumerate(entries[:5]):
                        text = e.get("title", "") + " — " + strip_html(e.get("summary", ""))
                        meta = default_meta(source="news", order=len(mock_articles) + j)
                        doc_id = f"rss_{j}"
                        collection.add(
                            embeddings=[embedder.encode(text).tolist()],
                            documents=[text],
                            ids=[doc_id],
                            metadatas=[meta],
                        )
                    break
        except Exception as e:
            print(f"RSS seed failed: {e}")
        print(f"Memory ready: {collection.count()} documents.")
    else:
        print(f"Database ready: {collection.count()} documents.")
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Models ──────────────────────────────────────────────────────────────────

class MemoryRequest(BaseModel):
    text: str
    source: str = "manual"

class MacroRequest(BaseModel):
    event: str

class MemoryUpdateRequest(BaseModel):
    text:    Optional[str] = None
    title:   Optional[str] = None
    topic:   Optional[str] = None
    tags:    Optional[str] = None    # comma-separated
    summary: Optional[str] = None

class ReorderRequest(BaseModel):
    ordered_ids: list[str]

class AutoOrganizeApplyRequest(BaseModel):
    assignments: list[dict]   # [{id, title, topic, tags, summary}]

class ImageIngestRequest(BaseModel):
    image_base64: str
    source_label: str = "ocr"

# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/news_feed")
async def get_news_feed():
    for source in NEWS_SOURCES:
        entries = fetch_feed(source["url"])
        if not entries:
            continue
        articles = [parse_entry_no_image(e, source["name"]) for e in entries[:20]]
        articles = enrich_with_images(articles)
        articles = score_articles_batch(articles)
        return {"status": "success", "source": source["name"], "articles": articles, "count": len(articles)}
    return {"status": "error", "message": "All news sources failed.", "articles": [], "count": 0}


@app.post("/remember")
async def add_memory(request: MemoryRequest, background_tasks: BackgroundTasks):
    if not request.text.strip():
        return {"status": "error", "message": "Text cannot be empty"}

    doc_id = f"user_doc_{str(uuid.uuid4())[:8]}"
    order  = collection.count()
    meta   = default_meta(source=request.source, order=order)

    collection.add(
        embeddings=[embedder.encode(request.text).tolist()],
        documents=[request.text],
        ids=[doc_id],
        metadatas=[meta],
    )

    # AI categorisation runs AFTER response is sent — never blocks the save
    background_tasks.add_task(categorize_memory_sync, doc_id, request.text)

    return {"status": "success", "doc_id": doc_id}


@app.get("/memories")
async def list_memories(
    search: str = "",
    topic:  str = "",
    sort:   str = "date",   # date | topic | custom
):
    """
    Returns all memories with full metadata.
    Supports full-text search, topic filter, and three sort modes.
    """
    result = collection.get(include=["documents", "metadatas"])

    memories = []
    for doc_id, doc, raw_meta in zip(
        result["ids"], result["documents"], result["metadatas"]
    ):
        meta = ensure_meta(raw_meta)
        memories.append({
            "id":         doc_id,
            "text":       doc,
            "title":      meta["title"] or doc[:60],
            "topic":      meta["topic"],
            "tags":       tags_to_list(meta["tags"]),
            "summary":    meta["summary"],
            "created_at": meta["created_at"],
            "order":      int(meta.get("order", 0)),
            "source":     meta["source"],
        })

    # Search filter (title + text + tags)
    if search:
        q = search.lower()
        memories = [
            m for m in memories
            if q in m["text"].lower()
            or q in m["title"].lower()
            or any(q in tag for tag in m["tags"])
            or q in m["summary"].lower()
        ]

    # Topic filter
    if topic:
        memories = [m for m in memories if m["topic"] == topic]

    # Sort
    if sort == "date":
        memories.sort(key=lambda m: m["created_at"], reverse=True)
    elif sort == "topic":
        memories.sort(key=lambda m: (m["topic"], m["created_at"]), reverse=False)
    elif sort == "custom":
        memories.sort(key=lambda m: m["order"])

    return {"status": "success", "memories": memories, "count": len(memories)}


@app.get("/topics")
async def list_topics():
    """Returns all unique topics that have at least one memory."""
    result = collection.get(include=["metadatas"])
    topics = set()
    for meta in result["metadatas"]:
        m = ensure_meta(meta)
        if m["topic"]:
            topics.add(m["topic"])
    # Return in taxonomy order, then any custom ones alphabetically
    ordered = [t for t in VALID_TOPICS if t in topics]
    extras  = sorted(t for t in topics if t not in VALID_TOPICS)
    return {"status": "success", "topics": ordered + extras}


@app.put("/memories/{doc_id}")
async def update_memory(doc_id: str, request: MemoryUpdateRequest):
    """
    Update any combination of: text, title, topic, tags, summary.
    If text is changed, re-embeds the document so retrieval stays accurate.
    """
    result = collection.get(ids=[doc_id], include=["documents", "metadatas"])
    if not result["ids"]:
        return {"status": "error", "message": "Memory not found"}

    existing_doc  = result["documents"][0]
    existing_meta = ensure_meta(result["metadatas"][0] if result["metadatas"] else None)

    new_doc  = request.text.strip() if request.text is not None else existing_doc

    if request.title   is not None: existing_meta["title"]   = request.title
    if request.topic   is not None: existing_meta["topic"]   = request.topic
    if request.tags    is not None: existing_meta["tags"]    = request.tags
    if request.summary is not None: existing_meta["summary"] = request.summary

    new_embedding = embedder.encode(new_doc).tolist() if request.text is not None else None

    if new_embedding:
        collection.update(
            ids=[doc_id],
            embeddings=[new_embedding],
            documents=[new_doc],
            metadatas=[existing_meta],
        )
    else:
        collection.update(ids=[doc_id], metadatas=[existing_meta])

    return {"status": "success", "doc_id": doc_id}


@app.delete("/memories/{doc_id}")
async def delete_memory(doc_id: str):
    result = collection.get(ids=[doc_id])
    if not result["ids"]:
        return {"status": "error", "message": "Memory not found"}
    collection.delete(ids=[doc_id])
    return {"status": "success", "deleted": doc_id}


@app.put("/memories/reorder/apply")
async def reorder_memories(request: ReorderRequest):
    """Update the `order` field for each id based on its position in ordered_ids."""
    for pos, doc_id in enumerate(request.ordered_ids):
        result = collection.get(ids=[doc_id], include=["metadatas"])
        if not result["ids"]:
            continue
        meta = ensure_meta(result["metadatas"][0] if result["metadatas"] else None)
        meta["order"] = pos
        collection.update(ids=[doc_id], metadatas=[meta])
    return {"status": "success", "reordered": len(request.ordered_ids)}


@app.post("/memories/auto_organize")
async def auto_organize():
    """
    Scans ALL memories and returns AI-suggested topic/tag/summary assignments.
    Does NOT apply anything — the client must confirm and call /auto_organize/apply.
    Memories are processed in batches of 8 to stay within LLM context limits.
    """
    result = collection.get(include=["documents", "metadatas"])
    if not result["ids"]:
        return {"status": "success", "suggestions": []}

    items = [
        {"id": doc_id, "text": doc[:300]}
        for doc_id, doc in zip(result["ids"], result["documents"])
    ]

    suggestions = []
    batch_size = 8

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_json = json.dumps(batch)

        try:
            response = client.chat.completions.create(
                model="qwen3.5-4b",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are a macro finance memory organiser. "
                            f"For each memory object, assign a title, topic, tags, and summary. "
                            f"Valid topics: {', '.join(VALID_TOPICS)}\n\n"
                            f"Return ONLY a JSON array (no markdown, no commentary):\n"
                            f'[{{"id":"...","title":"...","topic":"...","tags":["..."],"summary":"..."}}]'
                        ),
                    },
                    {"role": "user", "content": batch_json},
                ],
                temperature=0.1,
                max_tokens=600,
            )

            raw = response.choices[0].message.content.strip()
            raw = re.sub(r'^```json\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()
            batch_result = json.loads(raw)
            suggestions.extend(batch_result)

        except Exception as e:
            print(f"[auto-organize] batch {i} failed: {e}")
            # Include passthrough so the client still sees these items
            for item in batch:
                suggestions.append({
                    "id":      item["id"],
                    "title":   item["text"][:50],
                    "topic":   "General",
                    "tags":    [],
                    "summary": "",
                    "error":   str(e),
                })

    return {"status": "success", "suggestions": suggestions, "count": len(suggestions)}


@app.post("/memories/auto_organize/apply")
async def apply_auto_organize(request: AutoOrganizeApplyRequest):
    """Apply the confirmed subset of auto-organize suggestions."""
    applied = 0
    for assignment in request.assignments:
        doc_id = assignment.get("id")
        if not doc_id:
            continue
        result = collection.get(ids=[doc_id], include=["metadatas"])
        if not result["ids"]:
            continue
        meta = ensure_meta(result["metadatas"][0] if result["metadatas"] else None)

        topic = assignment.get("topic", meta["topic"])
        if topic not in VALID_TOPICS:
            topic = "General"

        meta.update({
            "title":   str(assignment.get("title", meta["title"])),
            "topic":   topic,
            "tags":    ",".join(str(t) for t in assignment.get("tags", [])),
            "summary": str(assignment.get("summary", meta["summary"])),
        })
        collection.update(ids=[doc_id], metadatas=[meta])
        applied += 1

    return {"status": "success", "applied": applied}


@app.post("/analyze")
async def analyze_risk(request: MacroRequest):
    if not request.event.strip():
        return {"status": "error", "message": "Event cannot be empty"}

    is_relevant, rejection_reason = is_macro_relevant(request.event)
    if not is_relevant:
        return {"status": "irrelevant", "message": rejection_reason, "content": None, "sources": []}

    query_embedding = embedder.encode(request.event).tolist()

    # ── Topic-aware hybrid retrieval ─────────────────────────────────────────
    # Retrieve top 5 semantically, then format context with topic labels.
    # This lets the LLM understand the provenance of each source and cite more
    # accurately. Topic metadata also improves contextual relevance ranking.
    n = min(5, collection.count())
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas"],
    )

    sources = []
    context_parts = []
    historical_context = "No relevant history found."

    if results["documents"] and results["documents"][0]:
        docs   = results["documents"][0][:3]
        metas  = results["metadatas"][0][:3]

        for i, (doc, raw_meta) in enumerate(zip(docs, metas)):
            meta  = ensure_meta(raw_meta)
            topic = meta.get("topic", "General")
            title = meta.get("title", "")
            label = f"[Source {i+1}] [{topic}]" + (f" — {title}" if title else "")
            context_parts.append(f"{label}:\n{doc}")
            sources.append({"index": i + 1, "text": doc, "topic": topic, "title": title})

        historical_context = "\n\n".join(context_parts)

    response = client.chat.completions.create(
        model="qwen3.5-4b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a quantitative risk analyst. "
                    "Map the risk chain using the provided historical sources. "
                    "CRITICAL INSTRUCTIONS: Be ruthlessly concise. Use extreme brevity. "
                    "Never write a paragraph if a bullet point will do. "
                    "Use directional arrows (➔) to show cause-and-effect. "
                    "Bold key assets and regions. "
                    "Respond strictly in this format:\n\n"
                    "**Event:** [1 sentence max]\n\n"
                    "**Consequence:**\n"
                    "* [Max 2 short bullet points using ➔ to show flow. Cite sources.]\n\n"
                    "**Asset Impact:**\n"
                    "* [Max 2 short bullet points mapping specific assets to outcomes. Cite sources.]"
                ),
            },
            {
                "role": "user",
                "content": f"Historical Sources:\n{historical_context}\n\nNew Macro Event:\n{request.event}",
            },
        ],
        temperature=0.3,
    )

    return {
        "status":  "success",
        "content": response.choices[0].message.content,
        "context": historical_context,
        "sources": sources,
    }


@app.get("/divergences")
async def get_divergences():
    """
    Cross-Asset Divergence Detector.
    Scans the 20 most recent memories for contradictions between asset classes.
    Returns a single JSON object: {headline, description}.
    """
    result = collection.get(include=["documents", "metadatas"])
    if not result["ids"]:
        return {"status": "success", "headline": "None",
                "description": "No memories found."}

    # Sort by created_at descending, take 20 most recent
    items = sorted(
        zip(result["documents"], result["metadatas"]),
        key=lambda x: ensure_meta(x[1]).get("created_at", ""),
        reverse=True,
    )[:20]

    memory_block = ""
    for i, (doc, raw_meta) in enumerate(items):
        meta  = ensure_meta(raw_meta)
        topic = meta.get("topic", "General")
        title = meta.get("title", "")
        label = f"[{i+1}] [{topic}]" + (f" — {title}" if title else "")
        memory_block += f"{label}:\n{doc[:200]}\n\n"

    try:
        response = client.chat.completions.create(
            model="qwen3.5-4b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a quantitative cross-asset analyst. "
                        "Analyze these recent market memories and identify ONE major "
                        "contradiction or divergence between different asset classes "
                        "(e.g., Equities pricing in growth while Fixed Income prices "
                        "in recession). "
                        "Return ONLY a JSON object with no markdown fences:\n"
                        '{"headline": "Brief title of the divergence", '
                        '"description": "Concise explanation of the conflicting signals"}'
                        "\nIf no clear divergence exists, return:\n"
                        '{"headline": "None", "description": "No major divergences detected."}'
                    ),
                },
                {"role": "user", "content": memory_block.strip()},
            ],
            temperature=0.2,
            max_tokens=200,
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)

        return {
            "status":      "success",
            "headline":    str(data.get("headline", "None")),
            "description": str(data.get("description", "")),
        }

    except Exception as e:
        print(f"[divergences] failed: {e}")
        return {"status": "error", "headline": "None",
                "description": f"Analysis failed: {e}"}


@app.post("/analyze/contagion")
async def analyze_contagion(request: MacroRequest):
    """
    Ripple Effect / Contagion Mapper.
    Maps cascading multi-asset effects across three time horizons:
    T+0 (immediate), T+30 days (secondary), T+90 days (systemic).
    """
    if not request.event.strip():
        return {"status": "error", "message": "Event cannot be empty"}

    query_embedding = embedder.encode(request.event).tolist()

    n = min(5, collection.count())
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas"],
    )

    sources = []
    context_parts = []
    historical_context = "No relevant history found."

    if results["documents"] and results["documents"][0]:
        docs  = results["documents"][0][:3]
        metas = results["metadatas"][0][:3]

        for i, (doc, raw_meta) in enumerate(zip(docs, metas)):
            meta  = ensure_meta(raw_meta)
            topic = meta.get("topic", "General")
            title = meta.get("title", "")
            label = f"[Source {i+1}] [{topic}]" + (f" — {title}" if title else "")
            context_parts.append(f"{label}:\n{doc}")
            sources.append({"index": i + 1, "text": doc, "topic": topic, "title": title})

        historical_context = "\n\n".join(context_parts)

    response = client.chat.completions.create(
        model="qwen3.5-4b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a macro contagion strategist. "
                    "Map out the cascading multi-asset effects. "
                    "CRITICAL INSTRUCTION: You MUST output your analysis strictly as a Markdown table. "
                    "Do not include any introductory or concluding text. "
                    "Use this exact format:\n\n"
                    "| Timeframe | Primary Driver | Contagion Impact | Key Assets Affected |\n"
                    "|---|---|---|---|\n"
                    "| T+0 (Immediate) | ... | ... | ... |\n"
                    "| T+30 Days | ... | ... | ... |\n"
                    "| T+90 Days | ... | ... | ... |\n\n"
                    "Keep cell contents under 15 words. Cite sources inline like [1]."
                ),
            },
            {
                "role": "user",
                "content": f"Historical Sources:\n{historical_context}\n\nMacro Event:\n{request.event}",
            },
        ],
        temperature=0.4,
    )

    return {
        "status":  "success",
        "content": response.choices[0].message.content,
        "context": historical_context,
        "sources": sources,
    }


@app.post("/analyze/red_team")
async def analyze_red_team(request: MacroRequest):
    """
    Devil's Advocate / Red Team mode.
    Uses the same semantic retrieval as /analyze but swaps in a
    contrarian hedge-fund system prompt that argues the *opposite*
    of the consensus view.
    """
    if not request.event.strip():
        return {"status": "error", "message": "Event cannot be empty"}

    query_embedding = embedder.encode(request.event).tolist()

    n = min(5, collection.count())
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas"],
    )

    sources = []
    context_parts = []
    historical_context = "No relevant history found."

    if results["documents"] and results["documents"][0]:
        docs  = results["documents"][0][:3]
        metas = results["metadatas"][0][:3]

        for i, (doc, raw_meta) in enumerate(zip(docs, metas)):
            meta  = ensure_meta(raw_meta)
            topic = meta.get("topic", "General")
            title = meta.get("title", "")
            label = f"[Source {i+1}] [{topic}]" + (f" — {title}" if title else "")
            context_parts.append(f"{label}:\n{doc}")
            sources.append({"index": i + 1, "text": doc, "topic": topic, "title": title})

        historical_context = "\n\n".join(context_parts)

    response = client.chat.completions.create(
        model="qwen3.5-4b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a ruthless contrarian hedge fund manager. "
                    "Find the fatal flaw in the consensus view. "
                    "CRITICAL INSTRUCTIONS: Be brutal and concise. "
                    "Limit yourself to exactly two bullet points:\n\n"
                    "* **Consensus Trap:** [1 sentence explaining why the crowd is wrong]\n"
                    "* **Contrarian Reality:** [1 sentence on what will actually happen, "
                    "using ➔ for flow. Cite sources.]"
                ),
            },
            {
                "role": "user",
                "content": f"Historical Sources:\n{historical_context}\n\nMacro Event:\n{request.event}",
            },
        ],
        temperature=0.7,
    )

    return {
        "status":  "success",
        "content": response.choices[0].message.content,
        "context": historical_context,
        "sources": sources,
    }


@app.post("/analyze/impact_chart")
async def analyze_impact_chart(request: MacroRequest):
    """
    Impact Projection Chart.
    Estimates 6-month directional price impact across 5 relevant asset classes.
    Returns structured JSON consumed directly by the Flutter BarChart widget.
    """
    if not request.event.strip():
        return {"status": "error", "message": "Event cannot be empty"}

    query_embedding = embedder.encode(request.event).tolist()
    n = min(5, collection.count())
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas"],
    )

    sources = []
    context_parts = []
    historical_context = "No relevant history found."

    if results["documents"] and results["documents"][0]:
        docs  = results["documents"][0][:3]
        metas = results["metadatas"][0][:3]
        for i, (doc, raw_meta) in enumerate(zip(docs, metas)):
            meta  = ensure_meta(raw_meta)
            topic = meta.get("topic", "General")
            title = meta.get("title", "")
            label = f"[Source {i+1}] [{topic}]" + (f" — {title}" if title else "")
            context_parts.append(f"{label}:\n{doc}")
            sources.append({"index": i + 1, "text": doc, "topic": topic, "title": title})
        historical_context = "\n\n".join(context_parts)

    try:
        response = client.chat.completions.create(
            model="qwen3.5-4b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a quantitative macro analyst. "
                        "Based on the event and historical context, estimate the projected "
                        "6-month price impact (in percentage) on exactly 5 highly relevant "
                        "asset classes. Choose the most impacted assets given the specific event. "
                        "CRITICAL: Output ONLY a raw JSON object. "
                        "No markdown fences, no preamble, no trailing text. "
                        'Schema: {"title": "Projected 6-Month Asset Impact", '
                        '"data": [{"asset": "S&P 500", "impact": -5.5}, '
                        '{"asset": "Gold", "impact": 4.2}]}'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Historical Sources:\n{historical_context}\n\nMacro Event:\n{request.event}",
                },
            ],
            temperature=0.2,
            max_tokens=200,
        )

        raw = response.choices[0].message.content.strip()
        # Strip any accidental markdown fences the model adds despite instructions
        raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)

        # Validate + clamp the payload so Flutter never receives malformed data
        items = data.get("data", [])
        validated = []
        for item in items[:5]:
            asset  = str(item.get("asset", "Unknown"))[:20]
            impact = float(item.get("impact", 0.0))
            impact = max(-50.0, min(50.0, impact))   # clamp to ±50% range
            validated.append({"asset": asset, "impact": round(impact, 1)})

        return {
            "status":  "success",
            "title":   str(data.get("title", "Projected 6-Month Asset Impact")),
            "data":    validated,
            "sources": sources,
        }

    except Exception as e:
        print(f"[impact_chart] failed: {e}")
        return {"status": "error", "message": f"Chart generation failed: {e}"}



class PortfolioRequest(BaseModel):
    event: str
    portfolio: dict   # {"AAPL": "15%", "TLT": "20%", ...}


@app.post("/analyze/portfolio")
async def analyze_portfolio(request: PortfolioRequest):
    """
    Portfolio Vulnerability Scanner.
    Identifies the highest-risk holding, best hedge, and one adjustment
    given a macro shock, informed by institutional memory context.
    """
    if not request.event.strip():
        return {"status": "error", "message": "Event cannot be empty"}
    if not request.portfolio:
        return {"status": "error", "message": "Portfolio cannot be empty"}

    query_embedding = embedder.encode(request.event).tolist()
    n = min(5, collection.count())
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas"],
    )

    sources = []
    context_parts = []
    historical_context = "No relevant history found."

    if results["documents"] and results["documents"][0]:
        docs  = results["documents"][0][:3]
        metas = results["metadatas"][0][:3]
        for i, (doc, raw_meta) in enumerate(zip(docs, metas)):
            meta  = ensure_meta(raw_meta)
            topic = meta.get("topic", "General")
            title = meta.get("title", "")
            label = f"[Source {i+1}] [{topic}]" + (f" — {title}" if title else "")
            context_parts.append(f"{label}:\n{doc}")
            sources.append({"index": i + 1, "text": doc, "topic": topic, "title": title})
        historical_context = "\n\n".join(context_parts)

    # Format portfolio for the prompt
    holdings_str = ", ".join(
        f"{ticker}: {weight}" for ticker, weight in request.portfolio.items()
    )

    try:
        response = client.chat.completions.create(
            model="qwen3.5-4b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior Portfolio Risk Manager at a tier-1 macro hedge fund. "
                        "Your analysis must be financially rigorous and directionally correct. "
                        "Shallow heuristics like 'bonds are safe havens' or 'gold always rises in risk-off' "
                        "are wrong — you must reason from the specific mechanics of each shock.\n\n"

                        "MACRO REGIME REFERENCE — use this to classify the shock before answering:\n"
                        "• TRADE WAR / TARIFFS: Stagflationary. Raises consumer prices (inflation UP) while "
                        "slowing growth (GDP DOWN). Rising inflation expectations push yields HIGHER, which "
                        "means long-duration bonds like TLT FALL in price — TLT is NOT a good hedge here. "
                        "Gold (GLD) IS a good hedge: it benefits from dollar weakness as trading partners "
                        "retaliate, from inflation hedging demand, and from geopolitical uncertainty. "
                        "Cash preserves optionality when direction is unclear.\n"
                        "• RECESSION / DEFLATION: Growth collapses, inflation falls, Fed cuts rates. "
                        "TLT rallies as yields fall. Gold mixed. Equities fall.\n"
                        "• RATE HIKE SHOCK: Yields spike. TLT crashes (long duration = highest sensitivity). "
                        "Gold falls (higher real rates). Cash wins.\n"
                        "• GEOPOLITICAL CRISIS: Flight to safety. Gold and USD rally. Equities fall. "
                        "TLT may rally IF the crisis is deflationary, but not if it disrupts supply chains.\n\n"

                        "HOLDING-SPECIFIC EXPOSURE FACTS — treat these as ground truth:\n"
                        "• NVDA: ~20% of revenue from China; H100/H20 GPU exports restricted by US BIS controls; "
                        "additional tariffs directly reduce China datacenter demand and trigger retaliation risk.\n"
                        "• AAPL: ~90% of iPhone manufacturing in China via Foxconn/Pegatron; "
                        "~19% of revenue is Greater China segment; tariffs raise COGS immediately; "
                        "China has historically singled out Apple for boycott campaigns during US-China trade disputes.\n"
                        "• TLT: Long-duration US Treasuries. Duration ~17 years — the most rate-sensitive asset. "
                        "Hurt badly when inflation expectations rise (yields up = bond prices down).\n"
                        "• GLD: Gold ETF. Benefits from dollar weakness, inflation hedging, and geopolitical risk premium.\n"
                        "• Cash: Preserves optionality. Neither helped nor hurt by macro shocks directly.\n\n"

                        "STRICT RULES:\n"
                        "1. AT-RISK HOLDINGS: Identify ALL holdings that face meaningful downside from this shock, "
                        "ranked PRIMARY (highest exposure) and SECONDARY (significant but less direct). "
                        "For China tariff shocks, BOTH NVDA and AAPL must always appear — one as primary, "
                        "one as secondary — because both have deep, documented China dependency. "
                        "Never pick just one and ignore the other for this shock type.\n\n"

                        "2. BEST HEDGE: First classify the shock using the macro regime reference above. "
                        "Then name the holding whose mechanics best match that regime. "
                        "NEVER recommend TLT as a hedge for a tariff or trade-war shock — tariffs raise "
                        "inflation expectations and hurt long-duration bonds. "
                        "For tariff/trade-war shocks the correct hedge is GLD. Explain why precisely.\n\n"

                        "3. IMMEDIATE ADJUSTMENT: One concrete reallocation between two named holdings, "
                        "expressed as a percentage shift (e.g. 'trim NVDA from 20% to 13%, rotate into GLD'). "
                        "Weights only — never invent dollar figures. "
                        "Logic check: does this reduce the highest-risk holding AND increase the best hedge? "
                        "If not, revise it.\n\n"

                        "4. CITATIONS: Cite historical sources inline by number [1][2][3].\n\n"

                        "Format as exactly four markdown bullet points:\n"
                        "• **Primary Risk: [TICKER]** — [specific channel + magnitude vs other holdings, 2 sentences]\n"
                        "• **Secondary Risk: [TICKER]** — [why this holding is also exposed, 1-2 sentences]\n"
                        "• **Best Hedge: [TICKER]** — [regime type + why this holding benefits, 2 sentences]\n"
                        "• **Immediate Adjustment:** [trim X% → Y%, rotate into Z] — [logic, 1-2 sentences]"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Historical Sources:\n{historical_context}\n\n"
                        f"Macro Event:\n{request.event}\n\n"
                        f"Portfolio Holdings (percentage weights only — never invent dollar figures):\n{holdings_str}\n\n"
                        "Step 1: Classify the shock regime (trade war / recession / rate hike / geopolitical).\n"
                        "Step 2: Score each holding as HURT / HEDGED / NEUTRAL with one-line reasoning.\n"
                        "Step 3: For China tariff shocks — confirm BOTH NVDA and AAPL appear in your risk tier.\n"
                        "Step 4: Write your four bullet-point assessment.\n"
                        "Hard rules: TLT is HURT by tariff shocks (inflation up = yields up = bond prices down). "
                        "GLD is the correct hedge. Never recommend reducing cash in a risk-off shock."
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=650,
        )

        return {
            "status":  "success",
            "content": response.choices[0].message.content,
            "sources": sources,
        }

    except Exception as e:
        print(f"[portfolio] failed: {e}")
        return {"status": "error", "message": f"Analysis failed: {e}"}



@app.post("/ingest_image")
async def ingest_image(request: ImageIngestRequest, background_tasks: BackgroundTasks):
    if not OCR_AVAILABLE:
        return {"status": "error", "message": "OCR unavailable on this server."}
    try:
        raw = request.image_base64
        if "," in raw:
            raw = raw.split(",", 1)[1]
        image = Image.open(io.BytesIO(base64.b64decode(raw)))
        image = ImageOps.exif_transpose(image)
        image = image.convert("L")
        w, h = image.size
        if w < 2000:
            scale = 2000 / w
            image = image.resize((2000, int(h * scale)), Image.LANCZOS)
        image = image.filter(ImageFilter.SHARPEN)
        image = ImageEnhance.Contrast(image).enhance(1.8)

        extracted = pytesseract.image_to_string(image, config=r"--oem 3 --psm 3").strip()
        if not extracted:
            return {"status": "error", "message": "No text extracted. Try a clearer photo."}

        # LLM contextual cleaning
        try:
            clean_response = client.chat.completions.create(
                model="qwen3.5-4b",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a news article extractor. "
                            "You receive raw OCR text from a photo of a news article or webpage. "
                            "Extract ONLY the meaningful journalistic content: headline, subheadline, body. "
                            "Discard navigation, timestamps, share counts, author bios, URLs, ads, and UI chrome. "
                            "Output the cleaned article text only — no labels, no preamble. "
                            "If no meaningful content exists, output: NO_ARTICLE_FOUND"
                        ),
                    },
                    {"role": "user", "content": extracted},
                ],
                temperature=0.1,
                max_tokens=800,
            )
            cleaned = clean_response.choices[0].message.content.strip()
            if not cleaned or cleaned == "NO_ARTICLE_FOUND":
                return {"status": "error", "message": "Could not identify article content. Try capturing just the article text."}
        except Exception as e:
            print(f"[ocr] LLM cleaning failed: {e}")
            lines = [l for l in extracted.splitlines() if len(l.strip()) > 4]
            cleaned = "\n".join(lines)

        return {
            "status":         "success",
            "extracted_text": cleaned,
            "char_count":     len(cleaned),
            "line_count":     len(cleaned.splitlines()),
            "captured_at":    now_iso(),
        }
    except Exception as e:
        return {"status": "error", "message": f"Processing failed: {e}"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
