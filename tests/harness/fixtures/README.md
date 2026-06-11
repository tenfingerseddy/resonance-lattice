# Harness fixtures (small, deterministic)

Committed fixtures the per-commit harness suites load by path.

- `encoder_golden.npz` — reference embeddings for 4 fixed strings from the
  pinned encoder revision on the ONNX runtime. `encoder_determinism.py`
  guarantee D3 checks cosine ≥ 0.9999 against it. Regenerate (and re-commit)
  whenever the pinned revision changes; the suite SKIPs on revision mismatch
  rather than failing.
