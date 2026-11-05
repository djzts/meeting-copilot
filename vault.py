"""Obsidian vault indexing and retrieval: BM25 (+ optional local embeddings, RRF fused)."""
from __future__ import annotations

import hashlib
import logging
import math
import os
import pickle
import re
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

log = logging.getLogger("vault")

CHUNK_CHARS = 1200
RRF_K = 60
# Weight of the previous utterance relative to the question itself
CONTEXT_WEIGHT_BM25 = 0.15
CONTEXT_WEIGHT_VECTOR = 0.3

# --------------------------------------------------------------------------- #
# Tokenizer (English words + Chinese unigrams/bigrams)
# --------------------------------------------------------------------------- #
EN_STOPWORDS = set(
    """a about above after again against all am an and any are aren't as at be because been before being
    below between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down
    during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here
    here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself
    let's me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves
    out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their
    theirs them themselves then there there's these they they'd they'll they're they've this those through
    to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's
    where where's which while who who's whom why why's will with won't would wouldn't you you'd you'll
    you're you've your yours yourself yourselves also just like get got really maybe yeah okay ok um uh
    tell explain please thanks thank know think going want s t d ll m re ve""".split()
)
ZH_STOP_UNIGRAMS = set("的了是在和与及或也就都而着把被吗呢吧啊呀么这那我你他她它们个有要会能说想去来到上下中")
ZH_STOP_BIGRAMS = {
    "什么", "怎么", "为什么", "如何", "哪些", "哪个", "多少", "是否", "是不是", "能不能", "有没有", "会不会",
    "可以", "一下", "我们", "你们", "他们", "这个", "那个", "一个", "就是", "还是", "然后", "所以", "因为",
    "但是", "如果", "觉得", "知道", "现在", "时候", "已经", "可能", "应该", "需要", "介绍", "讲讲", "说说",
}

_CJK = "㐀-䶿一-鿿豈-﫿"
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?|[%s]+" % _CJK)
_CJK_RE = re.compile(r"[%s]" % _CJK)


def _stem(w: str) -> str:
    """Tiny, consistent plural folding (applied to both query and documents)."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for m in _TOKEN_RE.finditer(text.lower()):
        tok = m.group(0)
        if _CJK_RE.match(tok):
            chars = list(tok)
            for c in chars:
                if c not in ZH_STOP_UNIGRAMS:
                    tokens.append(c)
            for a, b in zip(chars, chars[1:]):
                bg = a + b
                if bg not in ZH_STOP_BIGRAMS:
                    tokens.append(bg)
        else:
            tok = tok.replace("’", "'")
            if tok in EN_STOPWORDS:
                continue
            tok = tok.split("'")[0]
            if not tok or tok in EN_STOPWORDS:
                continue
            if len(tok) == 1 and not tok.isdigit():
                continue
            tokens.append(_stem(tok))
    return tokens


# --------------------------------------------------------------------------- #
# Markdown parsing / chunking
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    id: int
    path: str  # relative to vault root, forward slashes
    note: str  # note name (file stem)
    heading: str
    text: str

    @property
    def title(self) -> str:
        return f"{self.note} › {self.heading}" if self.heading else self.note

    @property
    def sha1(self) -> str:
        return hashlib.sha1((self.title + "\n" + self.text).encode("utf-8")).hexdigest()


_FENCE_RE = re.compile(r"^\s{0,3}(```|~~~)")
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")


def strip_frontmatter(text: str) -> str:
    if text.startswith("﻿"):
        text = text[1:]
    if not text.startswith("---"):
        return text
    lines = text.splitlines(keepends=True)
    if lines[0].strip() != "---":
        return text
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            return "".join(lines[i + 1:])
    return text


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) by headings, ignoring '#' inside code fences."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    in_fence = False
    fence_marker = ""
    for line in text.splitlines():
        m = _FENCE_RE.match(line)
        if m:
            if not in_fence:
                in_fence, fence_marker = True, m.group(1)
            elif m.group(1) == fence_marker:
                in_fence = False
            sections[-1][1].append(line)
            continue
        if not in_fence:
            h = _HEADING_RE.match(line)
            if h:
                sections.append((h.group(2).strip(), []))
                continue
        sections[-1][1].append(line)
    return [(h, "\n".join(body).strip()) for h, body in sections]


def chunk_text(body: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Pack paragraphs into ~max_chars chunks; very long paragraphs are split."""
    if len(body) <= max_chars:
        return [body] if body.strip() else []
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    pieces: list[str] = []
    for p in paras:
        while len(p) > max_chars:
            cut = _find_cut(p, max_chars)
            pieces.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            pieces.append(p)
    chunks: list[str] = []
    cur = ""
    for p in pieces:
        if cur and len(cur) + 2 + len(p) > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def _find_cut(p: str, max_chars: int) -> int:
    window = p[:max_chars]
    for sep in ("\n", "。", ". ", "！", "？", "! ", "? ", "；", "; ", "，", ", ", " "):
        i = window.rfind(sep)
        if i > max_chars * 0.5:
            return i + len(sep)
    return max_chars


def parse_note(text: str, note: str) -> list[tuple[str, str]]:
    """Return [(heading, chunk_text)] for one note."""
    text = strip_frontmatter(text)
    out = []
    for heading, body in split_sections(text):
        for piece in chunk_text(body):
            out.append((heading, piece))
    return out


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #
def glob_to_regex(pattern: str) -> re.Pattern:
    """Glob with ** support, matched against vault-relative posix paths.

    gitignore-like conveniences: "NDA" or "NDA/" means "NDA/**"; a pattern
    without a slash (e.g. "*.excalidraw.md") matches at any depth.
    """
    p = pattern.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    p = p.lstrip("/")
    if p.endswith("/"):
        p += "**"
    elif "*" not in p and "?" not in p and not p.lower().endswith(".md"):
        p += "/**"  # a bare folder name / path excludes that folder
    if "/" not in p:
        p = "**/" + p
    i, out = 0, ""
    while i < len(p):
        c = p[i]
        if p.startswith("**/", i):
            out += r"(?:.*/)?"
            i += 3
        elif p.startswith("**", i):
            out += r".*"
            i += 2
        elif c == "*":
            out += r"[^/]*"
            i += 1
        elif c == "?":
            out += r"[^/]"
            i += 1
        else:
            out += re.escape(c)
            i += 1
    return re.compile(r"^" + out + r"$", re.IGNORECASE)


def iter_markdown_files(root: Path, exclude: Iterable[str] = ()) -> Iterable[tuple[Path, str]]:
    regexes = [glob_to_regex(p) for p in exclude if p and p.strip()]
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for fn in sorted(filenames):
            if not fn.lower().endswith(".md") or fn.startswith("."):
                continue
            full = Path(dirpath) / fn
            rel = full.relative_to(root).as_posix()
            if any(r.match(rel) for r in regexes):
                continue
            yield full, rel


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #
class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.postings[term].append((i, tf))
        self.idf = {
            t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()
        }

    def scores(self, query_tokens, weights: Optional[dict[str, float]] = None) -> dict[int, float]:
        """BM25 scores; `weights` optionally down-weights some query terms."""
        acc: dict[int, float] = defaultdict(float)
        if not self.n:
            return acc
        for term in set(query_tokens):
            plist = self.postings.get(term)
            if not plist:
                continue
            w = weights.get(term, 1.0) if weights else 1.0
            idf = self.idf[term] * w
            for doc, tf in plist:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[doc] / (self.avgdl or 1))
                acc[doc] += idf * tf * (self.k1 + 1) / (tf + norm)
        return acc

    def top(self, query_tokens, n: int, weights: Optional[dict[str, float]] = None) -> list[tuple[int, float]]:
        s = self.scores(query_tokens, weights)
        return sorted(s.items(), key=lambda kv: -kv[1])[:n]


# --------------------------------------------------------------------------- #
# Optional embeddings
# --------------------------------------------------------------------------- #
class Embedder:
    """sentence-transformers wrapper with a sha1(text) -> vector pickle cache."""

    def __init__(self, model_name: str, cache_path: Path):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model_name = model_name
        self.cache_path = cache_path
        self.model = SentenceTransformer(model_name)
        self._lock = threading.Lock()

    def _load_cache(self) -> dict:
        try:
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except Exception as e:  # noqa: BLE001
            log.warning("ignoring unreadable embedding cache: %s", e)
            return {}

    def encode_chunks(self, chunks: list[Chunk], progress: Optional[Callable[[str], None]] = None) -> np.ndarray:
        with self._lock:
            store = self._load_cache()
            cache: dict[str, np.ndarray] = store.get(self.model_name, {})
            keys = [c.sha1 for c in chunks]
            missing = [i for i, k in enumerate(keys) if k not in cache]
            if missing:
                log.info("embedding %d new chunks (%d cached)", len(missing), len(chunks) - len(missing))
                if progress:
                    progress(f"embedding {len(missing)} chunks")
                texts = [chunks[i].title + "\n" + chunks[i].text for i in missing]
                vecs = self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
                for i, v in zip(missing, vecs):
                    cache[keys[i]] = np.asarray(v, dtype=np.float32)
            # keep only vectors that are still in the vault (incremental, no unbounded growth)
            live = set(keys)
            store[self.model_name] = {k: v for k, v in cache.items() if k in live}
            if missing or len(store[self.model_name]) != len(cache):
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.cache_path.with_suffix(".tmp")
                with open(tmp, "wb") as f:
                    pickle.dump(store, f)
                os.replace(tmp, self.cache_path)
            if not keys:
                return np.zeros((0, 1), dtype=np.float32)
            return np.stack([store[self.model_name][k] for k in keys])

    def encode_query(self, text: str, context: str = "", context_weight: float = 0.3) -> np.ndarray:
        texts = [text] + ([context] if context else [])
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        v = np.asarray(vecs[0], dtype=np.float32)
        if context:
            v = v + context_weight * np.asarray(vecs[1], dtype=np.float32)
            v /= np.linalg.norm(v) or 1.0
        return v


# --------------------------------------------------------------------------- #
# Index + search
# --------------------------------------------------------------------------- #
@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25_rank: Optional[int] = None
    vec_rank: Optional[int] = None


@dataclass
class IndexSnapshot:
    chunks: list[Chunk]
    bm25: BM25
    vectors: Optional[np.ndarray] = None
    n_files: int = 0
    built_at: float = field(default_factory=time.time)
    build_sec: float = 0.0


def rrf_fuse(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


class Vault:
    def __init__(
        self,
        path: str,
        exclude: Iterable[str] = (),
        embedding_model: str = "",
        cache_dir: Optional[str] = None,
        top_k: int = 6,
        per_note: int = 2,
    ):
        self.root = Path(os.path.expanduser(path)).resolve() if path else None
        self.exclude = list(exclude or [])
        self.embedding_model = embedding_model or ""
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).resolve().parent / ".cache"
        self.top_k = top_k
        self.per_note = per_note
        self._snap: Optional[IndexSnapshot] = None
        self._embedder: Optional[Embedder] = None
        self._embedder_failed = False
        self._build_lock = threading.Lock()
        self.building = False
        self.last_error: Optional[str] = None

    @property
    def name(self) -> str:
        return self.root.name if self.root else ""

    # -- building ----------------------------------------------------------- #
    def load_chunks(self) -> tuple[list[Chunk], int]:
        if not self.root or not self.root.is_dir():
            raise FileNotFoundError(f"vault path not found: {self.root}")
        chunks: list[Chunk] = []
        n_files = 0
        for full, rel in iter_markdown_files(self.root, self.exclude):
            try:
                text = full.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                log.warning("skip %s: %s", rel, e)
                continue
            n_files += 1
            note = full.stem
            for heading, piece in parse_note(text, note):
                chunks.append(Chunk(id=len(chunks), path=rel, note=note, heading=heading, text=piece))
        return chunks, n_files

    def _get_embedder(self) -> Optional[Embedder]:
        if not self.embedding_model or self._embedder_failed:
            return None
        if self._embedder is None:
            try:
                self._embedder = Embedder(self.embedding_model, self.cache_dir / "embeddings.pkl")
            except Exception as e:  # noqa: BLE001
                self._embedder_failed = True
                log.warning("embeddings disabled (%s); using BM25 only", e)
                return None
        return self._embedder

    def build(self, progress: Optional[Callable[[str], None]] = None) -> IndexSnapshot:
        with self._build_lock:
            self.building = True
            try:
                t0 = time.time()
                chunks, n_files = self.load_chunks()
                # note name and heading count twice: they describe the whole section
                docs = [tokenize(c.title) * 2 + tokenize(c.text) for c in chunks]
                snap = IndexSnapshot(chunks=chunks, bm25=BM25(docs), n_files=n_files)
                self._snap = snap  # BM25 is usable right away
                if progress:
                    progress(f"BM25 ready: {len(chunks)} chunks / {n_files} notes")
                emb = self._get_embedder()
                if emb is not None and chunks:
                    try:
                        vecs = emb.encode_chunks(chunks, progress)
                        snap = IndexSnapshot(chunks=chunks, bm25=snap.bm25, vectors=vecs, n_files=n_files)
                    except Exception as e:  # noqa: BLE001
                        log.warning("embedding failed (%s); BM25 only", e)
                snap.build_sec = time.time() - t0
                self._snap = snap
                self.last_error = None
                log.info("index: %d chunks from %d notes in %.1fs (vectors=%s)", len(chunks), n_files,
                         snap.build_sec, snap.vectors is not None)
                return snap
            except Exception as e:
                self.last_error = str(e)
                raise
            finally:
                self.building = False

    # -- query -------------------------------------------------------------- #
    def stats(self) -> dict:
        s = self._snap
        return {
            "chunks": len(s.chunks) if s else 0,
            "files": s.n_files if s else 0,
            "vectors": bool(s is not None and s.vectors is not None),
            "building": self.building,
            "error": self.last_error,
        }

    def search(self, query: str, top_k: Optional[int] = None, per_note: Optional[int] = None,
               pool: int = 50, context: str = "") -> list[Hit]:
        """Search the vault.

        `query` is the question; `context` (e.g. the previous utterance) also
        contributes at a lower weight, so it helps with follow-ups ("and the
        budget?") without drowning out the question itself.
        """
        snap = self._snap
        top_k = top_k or self.top_k
        per_note = per_note or self.per_note
        if not snap or not snap.chunks or not (query.strip() or context.strip()):
            return []
        q_tokens = tokenize(query)
        c_tokens = [t for t in tokenize(context) if t not in set(q_tokens)] if context else []
        weights = {t: CONTEXT_WEIGHT_BM25 for t in c_tokens}
        bm = snap.bm25.top(q_tokens + c_tokens, pool, weights)
        bm_rank = [d for d, _ in bm]
        rankings = [bm_rank]
        vec_rank: list[int] = []
        if snap.vectors is not None and self._embedder is not None:
            try:
                q = self._embedder.encode_query(query, context, CONTEXT_WEIGHT_VECTOR)
                sims = snap.vectors @ q
                order = np.argsort(-sims)[:pool]
                vec_rank = [int(i) for i in order if sims[i] > 0.2]
                rankings.append(vec_rank)
            except Exception as e:  # noqa: BLE001
                log.warning("vector search failed: %s", e)
        fused = rrf_fuse(rankings)
        bm_pos = {d: i + 1 for i, d in enumerate(bm_rank)}
        vec_pos = {d: i + 1 for i, d in enumerate(vec_rank)}
        per_path: Counter = Counter()
        hits: list[Hit] = []
        for doc, score in fused:
            c = snap.chunks[doc]
            if per_path[c.path] >= per_note:
                continue
            per_path[c.path] += 1
            hits.append(Hit(chunk=c, score=score, bm25_rank=bm_pos.get(doc), vec_rank=vec_pos.get(doc)))
            if len(hits) >= top_k:
                break
        return hits
