"""
HR Agent 3: Ranker
Ranks multiple candidates by match score.
"""

def rank_candidates(candidates: list) -> list:
    """
    candidates: list of dicts with keys: name, score, matched, missing, reasoning
    Returns sorted list (descending by score).
    """
    return sorted(candidates, key=lambda c: c.get("score", 0), reverse=True)