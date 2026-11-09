"""Deterministic lexical, phrase overlap, and partial-copy signals."""
import re

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.I)

def tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]

def overlap_signals(left: str, right: str) -> dict[str, float]:
    a, b = tokens(left), tokens(right)
    sa, sb = set(a), set(b)
    lexical = len(sa & sb) / len(sa | sb) if sa | sb else 0.0
    previous = [0] * (len(b) + 1)
    longest = 0
    for x in a:
        current = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                current[j] = previous[j - 1] + 1
                longest = max(longest, current[j])
        previous = current
    shorter = min(len(a), len(b))
    return {
        "lexical_similarity": lexical,
        "phrase_overlap": longest / max(1, max(len(a), len(b))),
        "partial_copy_signal": longest / max(1, shorter) if longest >= 5 else 0.0,
    }

def hybrid_score(semantic: float, signals: dict[str, float], weights: dict[str, float]) -> float:
    if set(weights) != {"semantic", "lexical", "phrase"} or any(v < 0 for v in weights.values()):
        raise ValueError("weights must be nonnegative semantic, lexical, and phrase values")
    total = sum(weights.values())
    if total <= 0: raise ValueError("at least one hybrid weight must be positive")
    return (weights["semantic"] * max(0.0, min(1.0, semantic)) + weights["lexical"] * signals["lexical_similarity"] + weights["phrase"] * signals["phrase_overlap"]) / total

def classify_passages(passages, configured_phrases=()):
    """Mark passages without discarding source text or source offsets."""
    phrases = [" ".join(tokens(p)) for p in configured_phrases if tokens(p)]
    headings = {"references", "bibliography", "works cited", "literature cited"}
    in_references, output = False, []
    for passage in passages:
        normalized = " ".join(tokens(passage.text))
        reason = None
        if normalized in headings:
            in_references, reason = True, "reference_section"
        elif in_references:
            reason = "reference_section"
        elif any(phrase in normalized for phrase in phrases):
            reason = "configured_boilerplate"
        elif len(passage.text.strip()) < 35:
            reason = "insufficient_length"
        output.append(type(passage)(passage.text, passage.start_char, passage.end_char, reason is not None, reason))
    return output
