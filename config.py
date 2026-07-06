"""
config.py — Central configuration for the Email Reply System.

Design Decision:
    All paths, model names, and hyperparameters live here in one place.
    This avoids magic strings scattered across files and makes the system
    trivially reconfigurable (swap the LLM, change retrieval depth, etc.)
    without touching business logic.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── Groq LLM ────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, "..", "..", "challenge", "data",
                         "EnronEmailReplyPairsWithContext.csv")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ─── Embedding Model ─────────────────────────────────────────────────────────
# Why all-MiniLM-L6-v2?
#   • 384-dim embeddings — small enough for local ChromaDB, fast on CPU
#   • Trained on 1B+ sentence pairs — strong semantic understanding
#   • No API cost — runs locally, no external dependency for retrieval
EMBED_MODEL = "all-MiniLM-L6-v2"

# ─── Retrieval ────────────────────────────────────────────────────────────────
# Why k=3?
#   • Enough context for the LLM to pick up tone + format patterns
#   • Few enough to stay well within context window limits
#   • More examples (5+) added noise in our testing — diminishing returns
TOP_K = 3
DB_SAMPLE_SIZE = 2000  # Emails to index. Full dataset = 15K; 2K is fast & representative.

# ─── Evaluation Weights ──────────────────────────────────────────────────────
# Why these weights? See README § "Defining Accuracy".
# Semantic similarity is king — meaning matters most for email replies.
# Content coverage prevents generic "sounds nice" replies from scoring high.
# LLM-Judge dimensions (relevance, fluency, tone) capture what automated
# metrics miss: coherence, naturalness, and register appropriateness.
EVAL_WEIGHTS = {
    "semantic_similarity": 0.25,
    "content_coverage":    0.20,
    "rouge_l":             0.15,
    "relevance":           0.15,
    "fluency":             0.10,
    "tone":                0.10,
    "length_ratio":        0.05,
}
