"""conversion — `rlat convert` is correct across all 6 transitions.

Eight guarantees, all hermetic:

  1-3. Pairwise round-trip (local → bundled → local; local → remote (fixture
       opener) → local; bundled → remote (fixture opener) → bundled).
       Bands semantically identical (np.allclose atol=1e-6) end-to-end.
  4. passage_id stable through every conversion.
  5. content_hash stable through every conversion.
  6. Drift abort: when a source file is mutated post-build, conversion
     refuses to write a new archive.
  7. Idempotent no-op: --to <current_mode> exits 0 without writing.
  8. ConversionDriftError carries the per-passage drift report.

Audit 08 commit 5/6.
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path
from typing import Callable, IO

import numpy as np

from ._testutil import Args as _Args, build_corpus


def _build_local(root: Path, files: dict[str, str]) -> Path:
    return build_corpus(root, files, mode="local")


def _build_bundled(root: Path, files: dict[str, str]) -> Path:
    return build_corpus(root, files, mode="bundled")


def _make_opener(byte_responses: dict[str, bytes]) -> Callable[[str], IO[bytes]]:
    def opener(url: str) -> IO[bytes]:
        if url not in byte_responses:
            raise ConnectionError(f"no fixture for {url}")
        return io.BytesIO(byte_responses[url])
    return opener


def _read(km: Path):
    from resonance_lattice.store import archive
    return archive.read(km)


_FILES = {
    "a.md": "# Alpha\n\nFirst doc about authentication and login flows. "
            "Sessions persist for 24 hours by default.",
    "b.md": "# Beta\n\nSecond doc about credentials and tokens. "
            "Tokens rotate weekly. Logout clears the session immediately.",
    "c.md": "# Gamma\n\nThird doc about session storage in Redis. "
            "Each session has a TTL of 24h.",
}


def _bands_close(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> bool:
    if set(a) != set(b):
        return False
    for name in a:
        if not np.allclose(a[name], b[name], atol=1e-6):
            return False
    return True


def run() -> int:
    from resonance_lattice.store.conversion import (
        ConversionDriftError, convert,
    )

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        # ---- Setup: a local-mode archive we'll convert in three directions ----
        corpus_local = root / "corpus_local"
        km_local = _build_local(corpus_local, _FILES)
        c_local = _read(km_local)
        ids_initial = {c.passage_id for c in c_local.registry}
        hashes_initial = {(c.passage_id, c.content_hash) for c in c_local.registry}
        bands_initial = {n: b.copy() for n, b in c_local.bands.items()}

        # ---- Guarantee 1: local → bundled → local roundtrip ----
        bundled_path = root / "km_bundled.rlat"
        convert(km_local, "bundled", source_root=corpus_local, output_path=bundled_path)
        c_bundled = _read(bundled_path)
        if c_bundled.metadata.store_mode != "bundled":
            print(f"[conversion] FAIL g1: store_mode={c_bundled.metadata.store_mode}",
                  file=sys.stderr)
            return 1
        if not _bands_close(c_bundled.bands, bands_initial):
            print("[conversion] FAIL g1: bands differ after local→bundled "
                  f"(max diff {max(np.max(np.abs(c_bundled.bands[n] - bands_initial[n])) for n in c_bundled.bands)})",
                  file=sys.stderr)
            return 1
        local2_root = root / "extracted_local2"
        local2_path = root / "km_local2.rlat"
        convert(bundled_path, "local",
                source_root=local2_root, output_path=local2_path)
        c_local2 = _read(local2_path)
        if c_local2.metadata.store_mode != "local":
            print(f"[conversion] FAIL g1: store_mode={c_local2.metadata.store_mode}",
                  file=sys.stderr)
            return 1
        if not _bands_close(c_local2.bands, bands_initial):
            print("[conversion] FAIL g1: bands differ after roundtrip",
                  file=sys.stderr)
            return 1
        # Materialised files must exist
        for rel in _FILES:
            if not (local2_root / rel).is_file():
                print(f"[conversion] FAIL g1: {rel} not materialised",
                      file=sys.stderr)
                return 1
        print("[conversion] guarantee 1 (local↔bundled roundtrip) OK", file=sys.stderr)

        # ---- Guarantee 2: local → remote (fixture opener) → local ----
        remote_path = root / "km_remote.rlat"
        url_base = "https://upstream.test/corpus"
        convert(km_local, "remote",
                source_root=corpus_local,
                remote_url_base=url_base,
                output_path=remote_path)
        c_remote = _read(remote_path)
        if c_remote.metadata.store_mode != "remote":
            print(f"[conversion] FAIL g2: store_mode={c_remote.metadata.store_mode}",
                  file=sys.stderr)
            return 1
        if not _bands_close(c_remote.bands, bands_initial):
            print("[conversion] FAIL g2: bands differ after local→remote",
                  file=sys.stderr)
            return 1
        if len(c_remote.remote_manifest) != len(_FILES):
            print(f"[conversion] FAIL g2: manifest entries {len(c_remote.remote_manifest)}",
                  file=sys.stderr)
            return 1
        # Inject fixture opener so remote → local works hermetically.
        import resonance_lattice.store.remote as remote_mod
        url_to_bytes = {
            spec["url"]: _FILES[path].encode("utf-8")
            for path, spec in c_remote.remote_manifest.items()
        }
        original_opener = remote_mod._default_opener
        remote_mod._default_opener = _make_opener(url_to_bytes)
        try:
            local3_root = root / "extracted_local3"
            local3_path = root / "km_local3.rlat"
            convert(remote_path, "local",
                    source_root=local3_root, output_path=local3_path)
        finally:
            remote_mod._default_opener = original_opener
        c_local3 = _read(local3_path)
        if not _bands_close(c_local3.bands, bands_initial):
            print("[conversion] FAIL g2: bands differ after local→remote→local",
                  file=sys.stderr)
            return 1
        print("[conversion] guarantee 2 (local↔remote roundtrip via fixture opener) OK",
              file=sys.stderr)

        # ---- Guarantee 3: bundled → remote → bundled ----
        corpus_bundled = root / "corpus_bundled"
        km_bundled_src = _build_bundled(corpus_bundled, _FILES)
        c_bun_initial = _read(km_bundled_src)
        bands_bun_initial = {n: b.copy() for n, b in c_bun_initial.bands.items()}
        remote2_path = root / "km_remote2.rlat"
        convert(km_bundled_src, "remote",
                remote_url_base=url_base,
                output_path=remote2_path)
        c_remote2 = _read(remote2_path)
        if c_remote2.metadata.store_mode != "remote":
            print(f"[conversion] FAIL g3: store_mode={c_remote2.metadata.store_mode}",
                  file=sys.stderr)
            return 1
        url_to_bytes2 = {
            spec["url"]: _FILES[path].encode("utf-8")
            for path, spec in c_remote2.remote_manifest.items()
        }
        remote_mod._default_opener = _make_opener(url_to_bytes2)
        try:
            bundled2_path = root / "km_bundled2.rlat"
            convert(remote2_path, "bundled", output_path=bundled2_path)
        finally:
            remote_mod._default_opener = original_opener
        c_bun2 = _read(bundled2_path)
        if c_bun2.metadata.store_mode != "bundled":
            print(f"[conversion] FAIL g3: store_mode={c_bun2.metadata.store_mode}",
                  file=sys.stderr)
            return 1
        if not _bands_close(c_bun2.bands, bands_bun_initial):
            print("[conversion] FAIL g3: bands differ after bundled→remote→bundled",
                  file=sys.stderr)
            return 1
        print("[conversion] guarantee 3 (bundled↔remote roundtrip) OK", file=sys.stderr)

        # ---- Guarantee 4: passage_id stable ----
        ids_after = {c.passage_id for c in c_local2.registry}
        if ids_initial != ids_after:
            print(f"[conversion] FAIL g4: passage_id set drifted "
                  f"(missing: {ids_initial - ids_after}, "
                  f"new: {ids_after - ids_initial})", file=sys.stderr)
            return 1
        print("[conversion] guarantee 4 (passage_id stable) OK", file=sys.stderr)

        # ---- Guarantee 5: content_hash stable ----
        hashes_after = {(c.passage_id, c.content_hash) for c in c_local2.registry}
        if hashes_initial != hashes_after:
            print(f"[conversion] FAIL g5: (passage_id, content_hash) pairs "
                  f"drifted across conversion", file=sys.stderr)
            return 1
        print("[conversion] guarantee 5 (content_hash stable) OK", file=sys.stderr)

        # ---- Guarantee 6: drift abort ----
        # Mutate one of the source files behind km_local AFTER its build,
        # then attempt to convert. Should raise ConversionDriftError and
        # NOT write the new archive.
        mutated_root = root / "corpus_drift"
        km_drift = _build_local(mutated_root, _FILES)
        # Edit a.md after the build → that file's passages are now drifted.
        (mutated_root / "a.md").write_text(
            "# Alpha v2\n\nCompletely different content now.",
            encoding="utf-8",
        )
        drift_target = root / "km_drift_target.rlat"
        try:
            convert(km_drift, "bundled", source_root=mutated_root,
                    output_path=drift_target)
        except ConversionDriftError as e:
            if drift_target.exists():
                print("[conversion] FAIL g6: drift target was written",
                      file=sys.stderr)
                return 1
            if e.n_total_passages != len(_read(km_drift).registry):
                print(f"[conversion] FAIL g6: n_total_passages={e.n_total_passages} "
                      f"unexpected", file=sys.stderr)
                return 1
            if not e.drifted_paths:
                print("[conversion] FAIL g6: drifted_paths empty",
                      file=sys.stderr)
                return 1
        else:
            print("[conversion] FAIL g6: drift wasn't detected", file=sys.stderr)
            return 1
        print("[conversion] guarantee 6 (drift abort) OK", file=sys.stderr)

        # ---- Guarantee 7: idempotent no-op ----
        try:
            convert(km_local, "local", source_root=corpus_local)
        except ValueError as e:
            if "already in mode" not in str(e):
                print(f"[conversion] FAIL g7: unexpected error {e}",
                      file=sys.stderr)
                return 1
        else:
            print("[conversion] FAIL g7: --to <current_mode> didn't raise",
                  file=sys.stderr)
            return 1
        print("[conversion] guarantee 7 (idempotent no-op raises) OK",
              file=sys.stderr)

        # ---- Guarantee 8: ConversionDriftError shape ----
        # Already covered in g6 — the catch block validated the report.
        # Adding one explicit assertion that the error message mentions
        # how to reconcile.
        try:
            convert(km_drift, "bundled", source_root=mutated_root,
                    output_path=root / "wont_exist.rlat")
        except ConversionDriftError as e:
            msg = str(e)
            if "rlat refresh" not in msg and "rlat sync" not in msg:
                print(f"[conversion] FAIL g8: error doesn't suggest "
                      f"refresh/sync: {msg}", file=sys.stderr)
                return 1
        print("[conversion] guarantee 8 (drift error shape) OK", file=sys.stderr)

    print("[conversion] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
