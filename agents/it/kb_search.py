"""
IT Agent 2: Knowledge Base Search (RAG)
Loads IT KB files and finds relevant articles.
Uses simple keyword scoring (no external embedding model = free & fast).
"""

import os
import re
from pathlib import Path

KB_DIR = Path(__file__).parent.parent.parent / "data" / "it_kb"


def _load_kb_articles() -> list:
    """Load all .txt files in data/it_kb/ and split on '=== ... ===' headers."""
    articles = []
    if not KB_DIR.exists():
        return articles

    for file in KB_DIR.glob("*.txt"):
        text = file.read_text(encoding="utf-8", errors="ignore")
        # Split on === TITLE ===
        chunks = re.split(r"===\s*(.+?)\s*===", text)
        # chunks = ['', 'TITLE1', 'body1', 'TITLE2', 'body2', ...]
        for i in range(1, len(chunks), 2):
            title = chunks[i].strip()
            body = chunks[i + 1].strip() if i + 1 < len(chunks) else ""
            if title and body:
                articles.append({"title": title, "content": body})
    return articles


def search_kb(query: str, top_k: int = 2) -> list:
    """Simple keyword-based retrieval. Returns top_k most relevant articles."""
    articles = _load_kb_articles()
    if not articles:
        return []

    q = query.lower()
    q_words = set(re.findall(r"\w+", q))

    scored = []
    for art in articles:
        text = (art["title"] + " " + art["content"]).lower()
        text_words = set(re.findall(r"\w+", text))
        overlap = len(q_words & text_words)
        # Bonus if title matches
        title_words = set(re.findall(r"\w+", art["title"].lower()))
        bonus = 3 * len(q_words & title_words)
        scored.append((overlap + bonus, art))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [art for score, art in scored[:top_k] if score > 0]