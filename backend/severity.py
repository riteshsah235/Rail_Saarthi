"""
Severity tagging: combines keyword-based rules with ML for high-priority/critical flagging.
"""
from typing import List, Optional
from .config import SEVERITY_LEVELS, CRITICAL_KEYWORDS, HIGH_KEYWORDS


def get_severity_keyword_score(text: str) -> Optional[str]:
    """
    Rule-based severity hint. Returns 'critical', 'high', or None.
    Used to override or reinforce ML severity for urgent cases.
    """
    if not text:
        return None
    t = text.lower()
    for w in CRITICAL_KEYWORDS:
        if w in t:
            return "critical"
    for w in HIGH_KEYWORDS:
        if w in t:
            return "high"
    return None


def resolve_severity(ml_severity: str, text: str) -> str:
    """
    Final severity: if keyword suggests critical/high, upgrade ML result.
    """
    kw = get_severity_keyword_score(text)
    if kw == "critical":
        return "critical"
    if kw == "high" and ml_severity in ("low", "medium"):
        return "high"
    return ml_severity
