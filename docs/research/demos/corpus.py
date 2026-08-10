"""Shared corpus utilities for demos 6-7 (numpy required, `pip install numpy`).

Two sources, one interface:

  build_repo_corpus(root)  — chunk this repository's own markdown into passages
      in reading order, recording (source_file, char_offset, char_length)
      exactly as `passages.jsonl` does, then embed with hashed character
      n-gram TF-IDF + LSA (SVD), L2-normalised. Deterministic, offline,
      dependency-light. NOT the production encoder — a stand-in whose
      geometry is real enough for mechanism-level experiments; every
      retrieval-quality number in demos 6-7 must be re-run against a real
      encoder band before being quoted as an rlat result.

  load_rlat(path)          — read a genuine .rlat archive (format v4:
      bands/base.npz + passages.jsonl, per docs/internal/STORE.md), so the
      same demos run unchanged against a production knowledge model:
      `python3 demo7_latent_graph.py /path/to/model.rlat`.

Corpus = {"band": (N, d) float32 L2-normalised, "coords": [(source_file,
char_offset, char_length)], "texts": [str] | None, "label": str}.
"""

from __future__ import annotations

import io
import json
import zipfile
import zlib
from pathlib import Path

import numpy as np

EXCLUDE_PARTS = {".git", ".claude", "node_modules", "docs/site", "docs/assets"}


def _iter_markdown(root: Path):
    for p in sorted(root.rglob("*.md")):
        rel = p.relative_to(root).as_posix()
        if any(part in rel for part in EXCLUDE_PARTS):
            continue
        yield rel, p.read_text(encoding="utf-8", errors="replace")


def chunk_text(text: str, min_chars: int = 280, max_chars: int = 900):
    """Paragraph-accumulating chunker; yields (char_offset, char_length, chunk).

    Mirrors the spirit of rlat's chunker: paragraph boundaries, bounded size,
    offsets into the ORIGINAL text so receipts stay exact.
    """
    out = []
    pos = 0
    buf_start, buf_parts, buf_len = None, [], 0
    n = len(text)
    while pos < n:
        end = text.find("\n\n", pos)
        if end == -1:
            end = n
        para = text[pos:end]
        if para.strip():
            if buf_start is None:
                buf_start = pos
            buf_parts.append(para)
            buf_len = end - buf_start
            if buf_len >= min_chars:
                out.append((buf_start, buf_len, text[buf_start:buf_start + buf_len]))
                buf_start, buf_parts, buf_len = None, [], 0
        pos = end + 2
    if buf_start is not None and buf_len > 0:
        out.append((buf_start, buf_len, text[buf_start:buf_start + buf_len]))
    # split oversized chunks at max_chars boundaries (rare; keeps receipts exact)
    final = []
    for off, ln, chunk in out:
        while ln > max_chars:
            final.append((off, max_chars, chunk[:max_chars]))
            off, ln, chunk = off + max_chars, ln - max_chars, chunk[max_chars:]
        final.append((off, ln, chunk))
    return final


def _ngrams(text: str, lo: int = 3, hi: int = 5):
    t = " ".join(text.lower().split())
    for k in range(lo, hi + 1):
        for i in range(len(t) - k + 1):
            yield t[i:i + k]


def embed_lsa(texts, dim: int = 256, hash_dim: int = 4096):
    """Hashed char-ngram TF-IDF -> LSA (SVD) -> L2 rows. Deterministic
    (crc32 hashing, no Python hash randomisation)."""
    N = len(texts)
    M = np.zeros((N, hash_dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for g in _ngrams(t):
            M[i, zlib.crc32(g.encode()) % hash_dim] += 1.0
    M = np.log1p(M)
    df = (M > 0).sum(axis=0)
    idf = np.log(1.0 + N / np.maximum(df, 1.0)).astype(np.float32)
    M *= idf
    M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    d = min(dim, len(S))
    X = (U[:, :d] * S[:d]).astype(np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    return X


def build_repo_corpus(root: str | Path, dim: int = 256):
    root = Path(root)
    coords, texts = [], []
    for rel, text in _iter_markdown(root):
        for off, ln, chunk in chunk_text(text):
            coords.append((rel, off, ln))
            texts.append(chunk)
    band = embed_lsa(texts, dim=dim)
    return {"band": band, "coords": coords, "texts": texts,
            "label": f"repo-docs LSA ({len(texts)} passages, {band.shape[1]}d)"}


def load_rlat(path: str | Path):
    """Read a production .rlat (format v4): bands/base.npz + passages.jsonl."""
    with zipfile.ZipFile(path) as z:
        with z.open("bands/base.npz") as f:
            npz = np.load(io.BytesIO(f.read()))
            band = npz[list(npz.keys())[0]].astype(np.float32)
        coords = []
        with z.open("passages.jsonl") as f:
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                if line.strip():
                    row = json.loads(line)
                    coords.append((row["source_file"], row["char_offset"],
                                   row["char_length"]))
    band /= np.maximum(np.linalg.norm(band, axis=1, keepdims=True), 1e-9)
    return {"band": band, "coords": coords, "texts": None,
            "label": f"{Path(path).name} ({band.shape[0]} passages, {band.shape[1]}d)"}


def reading_chains(coords, min_len: int = 3):
    """Group passage indices by source file, ordered by char_offset —
    ground-truth 'reading order' chains for traversal experiments."""
    by_file: dict[str, list[tuple[int, int]]] = {}
    for idx, (src, off, _ln) in enumerate(coords):
        by_file.setdefault(src, []).append((off, idx))
    chains = []
    for src, items in sorted(by_file.items()):
        items.sort()
        if len(items) >= min_len:
            chains.append((src, [idx for _off, idx in items]))
    return chains
