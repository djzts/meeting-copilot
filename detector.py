"""Rule-based question detection for English and Chinese.

The rules are deliberately generous: false positives are cheap because Claude
answers SKIP for anything that does not actually need an answer.
"""
from __future__ import annotations

import re

_EN_QUESTION_STARTS = {
    "what", "why", "how", "when", "where", "who", "whom", "whose", "which",
    "can", "could", "would", "will", "should", "shall", "may", "might", "must",
    "do", "does", "did", "is", "are", "was", "were", "am", "have", "has", "had",
    "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't", "won't",
    "wouldn't", "can't", "couldn't", "shouldn't", "haven't", "hasn't",
    "whats", "hows", "whys", "wheres", "whos",
}
# Leading filler words that are skipped before looking at the first word.
_EN_FILLERS = {
    "so", "and", "but", "ok", "okay", "um", "uh", "er", "hmm", "well", "also", "then",
    "right", "now", "alright", "yeah", "yes", "hey", "oh", "actually", "anyway", "sorry",
    "just", "quick", "question", "one", "more", "last", "follow", "up", "followup",
}
_EN_PHRASES = [
    r"tell me", r"explain", r"walk (?:me|us) through", r"wondering", r"curious",
    r"do you know", r"any (?:idea|thoughts)", r"thoughts on", r"what about", r"how about",
    r"i'd like to (?:know|understand|hear)", r"i would like to (?:know|understand|hear)",
    r"can you", r"could you", r"would you", r"help me understand", r"give (?:me|us) an? (?:overview|update)",
]
_EN_PHRASE_RE = re.compile(r"\b(?:%s)\b" % "|".join(_EN_PHRASES), re.IGNORECASE)

_ZH_MARKERS = [
    "吗", "呢", "什么", "为什么", "怎么", "如何", "哪", "多少", "是否", "是不是", "能不能",
    "有没有", "会不会", "可不可以", "要不要", "对不对", "行不行", "能否", "可否",
    "介绍一下", "讲一下", "讲讲", "说说", "解释一下",
]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？;；])\s*")
_WORD_RE = re.compile(r"[a-z]+(?:['’][a-z]+)?")


def _first_content_word(sentence: str) -> str:
    words = _WORD_RE.findall(sentence.lower().replace("’", "'"))
    for w in words:
        if w not in _EN_FILLERS:
            return w
    return ""


def is_question(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    if not t:
        return False
    if t.endswith(("?", "？")) or "?" in t or "？" in t:
        return True
    if any(m in t for m in _ZH_MARKERS):
        return True
    if _EN_PHRASE_RE.search(t):
        return True
    for sentence in _SENTENCE_SPLIT.split(t):
        w = _first_content_word(sentence)
        if not w:
            continue
        base = w.split("'")[0]
        if w in _EN_QUESTION_STARTS or base in ("what", "how", "why", "where", "who", "when"):
            return True
    return False
