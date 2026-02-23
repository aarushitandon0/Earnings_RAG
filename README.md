# 💹 EarningsIQ
### Intelligent Financial Analysis Powered by Advanced RAG

> Ask natural language questions across 190 NASDAQ earnings call transcripts (2016–2020) and get precise, cited answers — powered by a 5-stage retrieval pipeline that goes far beyond basic RAG.

<br>

##  Table of Contents
- [Overview](#-overview)
- [Why Advanced RAG?](#-why-advanced-rag)
- [Pipeline Architecture](#-pipeline-architecture)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Evaluation Results](#-evaluation-results)
- [Troubleshooting](#-troubleshooting)

<br>

---

##  Overview

EarningsIQ is an **Advanced Retrieval-Augmented Generation (RAG)** system built on top of real NASDAQ earnings call transcripts. It lets you ask questions like:

- *"What did Apple say about iPhone revenue in 2018?"*
- *"Which companies mentioned supply chain issues during COVID-19?"*
- *"How did AMD and NVIDIA discuss AI demand differently?"*
- *"What risks did Intel highlight across their earnings calls?"*

And get back **precise, sourced answers** like a financial analyst — not vague summaries.

The system is built around a fundamental problem with basic RAG: **retrieval quality**. When retrieval fails, even the best LLM will produce wrong or hallucinated answers. EarningsIQ solves this with 5 layers of intelligent retrieval, including a self-correcting **Corrective RAG (CRAG)** mechanism that detects bad chunks and automatically re-retrieves better ones.

<br>

---

## Why Advanced RAG?

Basic RAG works like this:
```
Question → Vector Search → Top 3 Chunks → LLM Answer
```

It sounds reasonable, but it breaks in several ways:

**Problem 1 — Bad queries:** Users ask vague questions like *"how is Apple doing?"* but the vector store needs specific financial terminology to retrieve the right chunks. A vague query returns vague chunks.

**Problem 2 — Vector search misses exact terms:** Semantic search is great for concepts but struggles with specific financial terms like ticker symbols, product names, or numbers. Searching "INTC Q3 Xeon" semantically might miss the exact relevant chunk.

**Problem 3 — Top-k isn't always best-k:** The most semantically similar chunks aren't always the most *relevant* chunks. Vector similarity ≠ answer relevance.

**Problem 4 — No quality check:** Basic RAG blindly passes whatever it retrieves to the LLM. If the chunks are wrong, the LLM hallucinates — confidently.

EarningsIQ fixes all four with dedicated pipeline stages.

<br>

---

##  Pipeline Architecture

```
                        ┌────────────────────┐
                        │    User Question    │
                        └────────┬───────────┘
                                 │
                                 ▼
               ┌─────────────────────────────────┐
               │         STAGE 1                 │
               │       Query Rewriter             │
               │      (Groq LLaMA3 70B)          │
               │                                 │
               │  "how is Apple doing?"          │
               │            ↓                    │
               │  "Apple Inc AAPL revenue        │
               │   growth iPhone sales fiscal    │
               │   performance guidance          │
               │   market share outlook"         │
               └─────────────┬───────────────────┘
                             │
                             ▼
               ┌─────────────────────────────────┐
               │         STAGE 2                 │
               │      Hybrid Retrieval           │
               │                                 │
               │  ┌─────────────┐               │
               │  │ Vector Search│  60% weight   │
               │  │  (ChromaDB) │               │
               │  └──────┬──────┘               │
               │         │    ╲                  │
               │         │     ╲  Combined       │
               │         │      ╲ Hybrid Score   │
               │         │     ╱                 │
               │  ┌──────┴──────┐               │
               │  │ BM25 Search │  40% weight   │
               │  │  (keyword)  │               │
               │  └─────────────┘               │
               │                                 │
               │       → Top 20 chunks           │
               └─────────────┬───────────────────┘
                             │
                             ▼
               ┌─────────────────────────────────┐
               │         STAGE 3                 │
               │         Re-Ranker               │
               │       (FlashRank)               │
               │                                 │
               │  Cross-encoder reads            │
               │  query + chunk together         │
               │  Scores TRUE relevance          │
               │                                 │
               │       → Top 3 chunks            │
               └─────────────┬───────────────────┘
                             │
                             ▼
               ┌─────────────────────────────────┐
               │         STAGE 4                 │
               │      Corrective RAG (CRAG)      │
               │                                 │
               │  Grades each chunk:             │
               │     RELEVANT                    │
               │     AMBIGUOUS                   │
               │     IRRELEVANT                  │
               │                                 │
               │  If too many IRRELEVANT:        │
               │  → Refines query automatically  │
               │  → Re-retrieves fresh chunks    │
               │  → Re-grades new results        │
               └─────────────┬───────────────────┘
                             │
                             ▼
               ┌─────────────────────────────────┐
               │         STAGE 5                 │
               │      Answer Generator           │
               │      (Groq LLaMA3 70B)          │
               │                                 │
               │  Reads verified chunks          │
               │  Generates cited answer         │
               │  Only uses provided context     │
               └─────────────┬───────────────────┘
                             │
                             ▼
               ┌─────────────────────────────────┐
               │  Final Answer + Source Citations │
               │  + CRAG Status (PASSED /        │
               │    CORRECTED)                   │
               └─────────────────────────────────┘
```

<br>

### What Each Stage Does

**Stage 1 — Query Rewriting**
The LLM transforms vague user questions into retrieval-optimized queries by expanding financial terminology, adding synonyms, expanding abbreviations, and including relevant company/product names. A question like *"how is NVDA on AI?"* becomes a rich, detailed query that surfaces the right transcript chunks.

**Stage 2 — Hybrid Retrieval**
Combines two retrieval methods to get the best of both worlds. Vector search (ChromaDB + sentence-transformers) captures semantic meaning. BM25 keyword search captures exact financial terms, ticker symbols, and product names that semantic search often misses. Final score = 60% vector + 40% BM25. Returns top 20 candidates.

**Stage 3 — Re-Ranking**
FlashRank's cross-encoder model reads the query and each chunk *together* as a pair and assigns a relevance score. Unlike the embedding model which scores in abstract vector space, the cross-encoder understands the relationship between the specific question and the specific chunk. This reorders the 20 candidates and keeps only the top 3.

**Stage 4 — Corrective RAG (CRAG)**
The most important quality control step. The LLM grades each of the top 3 chunks as RELEVANT, AMBIGUOUS, or IRRELEVANT. If 2 or more chunks are irrelevant, CRAG automatically generates a refined query and triggers a fresh retrieval cycle. This prevents the LLM from hallucinating answers based on bad context — a core failure mode of basic RAG.

**Stage 5 — Answer Generation**
The LLM generates a final answer using only the verified chunks as context. It is instructed to always cite sources using `[Source: TICKER | DATE]` format and to honestly admit when context is insufficient rather than hallucinate.

<br>

---

##  Tech Stack

| Component | Technology | Why |
|---|---|---|
| **LLM** | Groq API — `llama-3.3-70b-versatile` | Free tier, extremely fast inference |
| **Vector Database** | ChromaDB (local, persistent) | No cloud needed, easy setup |
| **Embeddings** | sentence-transformers `all-MiniLM-L6-v2` | Free, runs on CPU, good quality |
| **Keyword Search** | `rank_bm25` (BM25Okapi) | Exact term matching for financial vocab |
| **Re-Ranker** | FlashRank `ms-marco-MiniLM-L-12-v2` | Free, local cross-encoder |
| **Interface** | Python CLI | Simple, fast, no UI dependencies |

**Total cost: $0.00** — every component is free and runs locally except Groq API calls (free tier: 14,400 requests/day).

<br>

---

##  Dataset

**Source:** [NASDAQ Earnings Call Transcripts 2016–2020](https://www.kaggle.com/) — Thomson Reuters StreetEvents

| Property | Detail |
|---|---|
| Companies | AAPL, AMZN, MSFT, GOOGL, NVDA, AMD, INTC, CSCO, ASML, MU |
| Total transcripts | 190 files (~19 per company = ~4–5 years of quarterly calls) |
| Total chunks | 4,921 (400 words per chunk, 50-word overlap) |
| Coverage | 2016 Q1 through 2020 Q3/Q4 |
| Format | Plain `.txt` files named `YYYY-Mon-DD-TICKER.txt` |
| Avg file size | ~70 KB per transcript |

The dataset covers some of the most consequential years in tech — including the AI boom beginning (2016-2017), the trade war period (2018-2019), and COVID-19 impact (2020). This makes for rich, varied questions with real analytical value.

<br>

---

##  Project Structure

```
earningsiq/
│
├── data/
│   └── transcripts/             ← All 190 .txt transcript files (flat folder)
│
├── src/
│   ├── ingest.py                ← Reads .txt files, chunks text, builds ChromaDB
│   ├── rewriter.py              ← Stage 1: LLM-based query rewriting
│   ├── retriever.py             ← Basic vector-only retrieval (for evaluation comparison)
│   ├── hybrid_retriever.py      ← Stage 2: Vector + BM25 combined retrieval
│   ├── reranker.py              ← Stage 3: FlashRank cross-encoder re-ranking
│   ├── crag.py                  ← Stage 4: Chunk grading + automatic query correction
│   ├── generator.py             ← Stage 5: Cited answer generation
│   └── pipeline.py              ← Orchestrates all 5 stages end-to-end
│
├── chroma_db/                   ← Auto-created after running ingest.py
├── main.py                      ← CLI interface to run the system
├── evaluate.py                  ← Runs Basic RAG vs Advanced RAG comparison
├── evaluation_results.txt       ← Auto-generated full evaluation output
├── requirements.txt             ← Python dependencies
├── .env                         ← API key (never committed to git)
├── .gitignore                   ← Excludes venv, chroma_db, .env
└── README.md
```

<br>

---

##  Getting Started

### Prerequisites

- **Python 3.11** — ChromaDB is incompatible with Python 3.12 and above
- **Groq API key** — free at [console.groq.com](https://console.groq.com), no credit card needed
- **Dataset** — download from Kaggle (link above)

---

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/earningsiq.git
cd earningsiq
```

---

### 2. Create a Python 3.11 virtual environment

```bash
# Windows
py -3.11 -m venv venv
venv\Scripts\activate

# Mac / Linux
python3.11 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal prompt. Always ensure this is active before running any project commands.

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs all packages and downloads the sentence-transformers embedding model (~90MB) on first use. Takes 5–10 minutes.

---

### 4. Set up your API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your key at [console.groq.com](https://console.groq.com) → API Keys → Create API Key. Takes 2 minutes.

---

### 5. Prepare your data

Download the Kaggle dataset and copy all `.txt` transcript files into `data/transcripts/`. All 190 files should be in a **flat folder** (not in subfolders per company).

```bash
# Windows — repeat for each company
xcopy "C:\path\to\Transcripts\AAPL\*" "data\transcripts\" /Y
xcopy "C:\path\to\Transcripts\AMZN\*" "data\transcripts\" /Y
# ... and so on for all 10 companies
```

Verify your files are there:
```bash
# Should show ~190
ls data/transcripts/ | wc -l        # Mac/Linux
dir data\transcripts | find /c ".txt"  # Windows
```

---

### 6. Build the vector store

```bash
python src/ingest.py
```

This reads all 190 transcripts, splits them into 4,921 chunks, embeds each chunk, and stores everything in ChromaDB. Takes 5–10 minutes on first run.

Expected output:
```
 Loading transcripts...
   Loaded: 2018-Nov-01-AAPL.txt
   Loaded: 2020-Feb-13-NVDA.txt
  ... (190 files)

Total chunks: 4921
 Embedding and storing chunks (this takes 5-10 mins)...
  Stored 4921/4921 chunks...

 Done! Vector store built with 4921 chunks.
```

---

### 7. Run EarningsIQ

```bash
python main.py
```

<br>

---

##  Usage

```
============================================================
   EarningsIQ — Earnings Call Intelligence System
  RAG + Query Rewriting + Hybrid Search +
  Re-ranking + Corrective RAG (CRAG)
============================================================
  Companies: AAPL AMZN MSFT GOOGL NVDA AMD INTC CSCO ASML MU
  Period   : 2016-2020
============================================================
  Type 'quit' to exit

 Your question: What did AMD say about competing with Intel?

 Processing pipeline...

   Stage 1: Rewriting query...
   Stage 2: Hybrid retrieval (Vector + BM25)...
   Stage 3: Re-ranking top chunks...
   Stage 4: Corrective RAG grading...
     → AMD | 2016-Apr-21: IRRELEVANT
     → AMD | 2020-Jan-28: IRRELEVANT
     → AMD | 2020-Jul-28: IRRELEVANT
    CRAG: Too many irrelevant chunks (3/3). Refining query...
   Refined query: AMD CEO statements on Intel competition and market strategy
   CRAG: Re-grading refined results...
     → AMD | 2018-Jul-25: RELEVANT
     → INTC | 2017-Jul-27: AMBIGUOUS
   Stage 5: Generating answer...

============================================================
 Rewritten Query:
   AMD Advanced Micro Devices EPYC Ryzen processor market share
   competition Intel semiconductor CPU server desktop performance...

 CRAG Status: CORRECTED
    Initial retrieval failed → Query refined → Re-retrieved successfully

 Final Sources Used:
   [1] AMD | 2018-Jul-25 | grade: RELEVANT
   [2] INTC | 2017-Jul-27 | grade: AMBIGUOUS

 Answer:
------------------------------------------------------------
AMD highlighted significant competitive gains against Intel in the
server processor market, driven by the EPYC processor line.
In Q2 2018, AMD's CEO noted strong enterprise customer adoption
of EPYC in data centers, positioning it as a credible alternative
to Intel's Xeon lineup [Source: AMD | 2018-Jul-25]...
============================================================
```

This example shows **CRAG in action** — the first retrieval returned 3 irrelevant chunks, CRAG detected the failure, automatically refined the query, re-retrieved, and produced a good answer. Basic RAG would have hallucinated from the bad initial chunks.

<br>

### More Questions to Try

```
# Company-specific
What did Apple say about iPhone revenue in 2018?
How did NVIDIA discuss AI and GPU demand?
What risks did Intel highlight in their earnings calls?
What was Amazon's guidance on revenue growth?
How did Microsoft's cloud business Azure perform?

# Cross-company comparison
Which companies mentioned supply chain issues in 2020?
Which companies showed improving gross margins?
How did companies respond to COVID-19?
Which companies invested heavily in R&D?
Compare AMD and NVIDIA on data center strategy

# Trend analysis
How did tech companies discuss AI between 2016 and 2020?
Which companies mentioned 5G in their earnings calls?
How did semiconductor companies discuss memory pricing?
```

<br>

---

##  Evaluation Results

Run the evaluation script to compare Basic RAG vs Advanced RAG:

```bash
python evaluate.py
```

### Results Summary (10 test questions)

| # | Question | Basic RAG | Advanced RAG | CRAG |
|---|---|---|---|---|
| 1 | Apple iPhone revenue 2018? | 4/4 | 4/4 | PASSED |
| 2 | NVIDIA AI and GPU demand? | 4/4 | 4/4 | PASSED |
| 3 | Supply chain issues 2020? | 4/4 | 4/4 | PASSED |
| 4 | Intel risk factors? | 4/4 | 4/4 | PASSED |
| 5 | Microsoft cloud performance? | 4/4 | 4/4 | PASSED |
| 6 | Companies with improving margins? | 4/4 | 4/4 | PASSED |
| 7 | AMD vs Intel competition? | 4/4 | 4/4 | **CORRECTED**  |
| 8 | COVID-19 company responses? | 4/4 | 4/4 | PASSED |
| 9 | Amazon revenue guidance? | 4/4 | 2/4 | CORRECTED |
| 10 | R&D investment companies? | 4/4 | 4/4 | **CORRECTED**  |

*Scoring criteria per answer: cited sources, specific numbers, sufficient detail, confident response (each = 1 point)*

### Key Findings

**CRAG triggered on 3/10 questions.** In Q7, all 3 initial chunks were graded IRRELEVANT. CRAG automatically refined the query and successfully re-retrieved relevant AMD competitive context. Basic RAG would have passed those irrelevant chunks to the LLM and generated a hallucinated answer.

**Answer quality difference goes beyond the score.** Both systems score 4/4 on Q1, but the answers are qualitatively different:

| | Q1: Apple iPhone Revenue 2018 |
|---|---|
| **Basic RAG** | *"iPhone revenue was the highest ever, driven by broad-based growth"* — vague, no specifics |
| **Advanced RAG** | *"iPhone revenue grew 29% in Q4 2018, ASP $793, 46.9M units sold, growth across every geographic segment [Source: AAPL \| 2018-Nov-01]"* — specific, dated, cited |

The simple scoring function doesn't capture this qualitative improvement — but evaluators and users do.

<br>

---

##  Troubleshooting

**`ChromaDB Pydantic error` on startup**

You are running Python 3.12 or higher. ChromaDB strictly requires Python 3.11.
```bash
py -3.11 -m venv venv      # Windows
venv\Scripts\activate
pip install -r requirements.txt
```

**`ModuleNotFoundError: No module named 'src'`**

Run scripts from the project root folder using the `-m` flag:
```bash
python -m src.ingest       # correct
python src/ingest.py       # fails on Windows
```

**`Model decommissioned` error from Groq**

Update the model name in `rewriter.py`, `crag.py`, and `generator.py`:
```python
model = "llama-3.3-70b-versatile"
```

**`No .txt files found` during ingestion**

All transcript files must be directly inside `data/transcripts/` as a flat structure — not organized into company subfolders. Verify with:
```bash
dir data\transcripts\*.txt   # Windows
ls data/transcripts/*.txt    # Mac/Linux
```

**Groq rate limit errors during evaluation**

Increase the sleep delay between API calls in `evaluate.py`:
```python
time.sleep(2)  
```

**BM25 index slow on first query**

Expected — the BM25 index is built from all 4,921 chunks on first use (~10 seconds) and then cached in memory for the rest of the session. Subsequent queries are fast.

<br>

---

##  Dependencies

```
groq                  # Groq API client for LLaMA3
chromadb              # Local vector database
sentence-transformers # Embedding model (all-MiniLM-L6-v2)
flashrank             # Cross-encoder re-ranking
rank_bm25             # BM25 keyword search
python-dotenv         # Load .env API keys
```

Install all at once:
```bash
pip install -r requirements.txt
```

<br>

---

##  Future Improvements

- **RAGAS evaluation** — quantitative metrics: faithfulness, answer relevancy, context precision, context recall
- **Conversation memory** — multi-turn Q&A so follow-up questions understand prior context
- **Multi-hop retrieval** — chain queries across multiple transcripts to answer trend questions like *"how did Apple's iPhone narrative evolve from 2016 to 2020?"*
- **Gradio web UI** — clean browser interface for non-technical demos
- **Temporal filtering** — restrict retrieval to a specific year or quarter
- **More companies** — extend beyond the current 10 NASDAQ companies
- **Sentiment analysis** — classify management tone (optimistic / cautious / uncertain) per transcript

<br>

---

<br>

---

*Built with Python 3.11 · Groq · ChromaDB · sentence-transformers · FlashRank · rank-bm25*
