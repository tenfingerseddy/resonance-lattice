"""row_mode — the semantic-slicer build surface (semantic-slicer-handoff §11).

Guarantees on the per-row business-key path: a row-mode build turns each
(key, text) row into exactly one passage, keyed and pinned to that key, with
the key threaded all the way to the Fabric slicer output.

  R1. RowSourceWalker yields one (key, text) per valid row; skips empty
      text + blank keys; reports counts.
  R2. build_rlat(row_mode=True, bundled) → one passage per row, key set on
      every coord, passage_id == compute_key_id(key), text round-trips
      through the bundled store EXACTLY (incl. unicode), build_config marks
      row_v1 + row_mode.
  R3. passage_id is pinned to the KEY, not the coordinates: the same key with
      different-length text keeps its id; distinct keys differ; a duplicate
      key raises; row_mode without bundled raises.
  R4. Backward-compat: an ordinary chunked build carries key=None and emits
      no "key" token in passages.jsonl; a coord WITH a key round-trips
      through write/load_jsonl.
  R5. The key is threaded to the verified + Fabric surfaces: verify_hits
      carries it, search() hit dicts include it, slice() returns a deduped
      score-ordered key set.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from ._testutil import ZeroEncoder, check_guarantee


_ROWS = [
    ("L001", "Peaceful and quiet flat away from the traffic, very restful."),
    ("L002", "Bright spacious loft full of natural light near the park."),
    ("L003", "Café façade — naïve £100/night 🌟 tranquil courtyard, no noise."),
]


def _check(ok: bool, label: str) -> bool:
    return check_guarantee(ok, label, "row_mode")


def _build_rows(out: Path, rows, **kw):
    from resonance_lattice.build.pipeline import build_rlat
    from resonance_lattice.build.walker import RowSourceWalker
    from resonance_lattice.config import Kind, StoreMode

    return build_rlat(
        RowSourceWalker(rows, source_name="listings"),
        out,
        store_mode=kw.pop("store_mode", StoreMode.BUNDLED),
        kind=Kind.CORPUS,
        encoder=ZeroEncoder(),
        row_mode=kw.pop("row_mode", True),
        batch_size=4,
        **kw,
    )


def _r1_walker() -> bool:
    from resonance_lattice.build.walker import RowSourceWalker

    w = RowSourceWalker(
        [("L1", "real text one here"), ("L2", "   "), ("", "x"),
         ("L3", "real text three here")],
        source_name="listings",
    )
    yielded = list(w.iter_files())
    keys = [k for k, _ in yielded]
    if keys != ["L1", "L3"]:
        print(f"[row_mode] FAIL R1: yielded keys {keys} != ['L1','L3']", file=sys.stderr)
        return False
    reasons = {k: r for k, r in w.skipped}
    if reasons.get("L2") != "empty" or reasons.get("") != "blank_key":
        print(f"[row_mode] FAIL R1: skipped {w.skipped}", file=sys.stderr)
        return False
    if w.total_files() != 4:
        print(f"[row_mode] FAIL R1: total_files {w.total_files()} != 4", file=sys.stderr)
        return False
    if not w.source_root_for_metadata.startswith("rows://"):
        print(f"[row_mode] FAIL R1: source_root {w.source_root_for_metadata!r}", file=sys.stderr)
        return False
    return True


def _r2_build_and_roundtrip() -> bool:
    from resonance_lattice.store import archive
    from resonance_lattice.store.bundled import BundledStore
    from resonance_lattice.store.registry import compute_key_id

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "km.rlat"
        result = _build_rows(out, _ROWS)
        if result.n_passages != len(_ROWS):
            print(f"[row_mode] FAIL R2: n_passages {result.n_passages} != {len(_ROWS)}",
                  file=sys.stderr)
            return False

        contents = archive.read(out)
        if len(contents.registry) != len(_ROWS):
            print(f"[row_mode] FAIL R2: registry len {len(contents.registry)}", file=sys.stderr)
            return False
        store = BundledStore(out)
        for coord, (key, text) in zip(contents.registry, _ROWS):
            if coord.key != key or coord.source_file != key:
                print(f"[row_mode] FAIL R2: coord.key {coord.key!r} / source_file "
                      f"{coord.source_file!r} != {key!r}", file=sys.stderr)
                return False
            if coord.passage_id != compute_key_id(key):
                print(f"[row_mode] FAIL R2: passage_id not pinned to key for {key!r}",
                      file=sys.stderr)
                return False
            if coord.char_offset != 0 or coord.char_length != len(text):
                print(f"[row_mode] FAIL R2: span ({coord.char_offset},{coord.char_length}) "
                      f"!= (0,{len(text)}) for {key!r}", file=sys.stderr)
                return False
            # The load-bearing unicode check: byte/char fidelity through bundling.
            fetched = store.fetch(coord.source_file, coord.char_offset, coord.char_length)
            if fetched != text:
                print(f"[row_mode] FAIL R2: text round-trip mismatch for {key!r}: "
                      f"{fetched!r} != {text!r}", file=sys.stderr)
                return False
            if store.verify(coord.source_file, coord.char_offset, coord.char_length,
                            coord.content_hash) != "verified":
                print(f"[row_mode] FAIL R2: verify != verified for {key!r}", file=sys.stderr)
                return False

        bc = contents.metadata.build_config
        if bc.get("chunker") != "row_v1" or bc.get("row_mode") is not True:
            print(f"[row_mode] FAIL R2: build_config {bc.get('chunker')!r} "
                  f"row_mode={bc.get('row_mode')!r}", file=sys.stderr)
            return False
        if contents.metadata.store_mode != "bundled":
            print(f"[row_mode] FAIL R2: store_mode {contents.metadata.store_mode!r}", file=sys.stderr)
            return False
    return True


def _r3_id_pinned_and_guards() -> bool:
    from resonance_lattice.build.pipeline import BuildError
    from resonance_lattice.config import StoreMode
    from resonance_lattice.store import archive

    with tempfile.TemporaryDirectory() as d:
        # Same key, different-length text → identical passage_id (pinned to key).
        out_a = Path(d) / "a.rlat"
        out_b = Path(d) / "b.rlat"
        _build_rows(out_a, [("L001", "short text here")])
        _build_rows(out_b, [("L001", "a very much longer description than before, padded out")])
        id_a = archive.read(out_a).registry[0].passage_id
        id_b = archive.read(out_b).registry[0].passage_id
        if id_a != id_b:
            print(f"[row_mode] FAIL R3: id not pinned to key ({id_a} != {id_b})", file=sys.stderr)
            return False

        # Distinct keys → distinct ids.
        out_c = Path(d) / "c.rlat"
        _build_rows(out_c, [("L001", "text one here"), ("L002", "text two here")])
        reg = archive.read(out_c).registry
        if reg[0].passage_id == reg[1].passage_id:
            print("[row_mode] FAIL R3: distinct keys collided on id", file=sys.stderr)
            return False

        # Duplicate key → BuildError.
        try:
            _build_rows(Path(d) / "dup.rlat", [("L001", "one here"), ("L001", "two here")])
            print("[row_mode] FAIL R3: duplicate key did not raise", file=sys.stderr)
            return False
        except BuildError:
            pass

        # row_mode + non-bundled → BuildError.
        try:
            _build_rows(Path(d) / "loc.rlat", [("L001", "x text here")],
                        store_mode=StoreMode.LOCAL)
            print("[row_mode] FAIL R3: row_mode+local did not raise", file=sys.stderr)
            return False
        except BuildError:
            pass
    return True


def _r4_backward_compat() -> bool:
    import zipfile

    from resonance_lattice.store import archive
    from resonance_lattice.store.registry import (
        PassageCoord, load_jsonl, write_jsonl,
    )

    # An ordinary chunked build carries no key and emits no "key" token.
    _FIXTURE = {
        "a.md": "# Auth\n\nLogin via SSO. Sessions persist for twenty-four hours total.",
        "b.md": "# Storage\n\nDocs land in OneLake. The index lives in the .rlat file.",
    }
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        src = root / "src"
        src.mkdir(parents=True)
        for rel, txt in _FIXTURE.items():
            (src / rel).write_text(txt, encoding="utf-8")

        from resonance_lattice.build.pipeline import build_rlat
        from resonance_lattice.build.walker import FilesystemSourceWalker
        from resonance_lattice.config import Kind, StoreMode
        out = root / "chunked.rlat"
        build_rlat(
            FilesystemSourceWalker([src], src), out,
            store_mode=StoreMode.LOCAL, kind=Kind.CORPUS,
            encoder=ZeroEncoder(), min_chars=20, max_chars=400, batch_size=4,
        )
        if any(c.key is not None for c in archive.read(out).registry):
            print("[row_mode] FAIL R4: chunked build set a non-None key", file=sys.stderr)
            return False
        with zipfile.ZipFile(out, "r") as zf:
            passages = zf.read("passages.jsonl").decode("utf-8")
        if '"key"' in passages:
            print("[row_mode] FAIL R4: chunked passages.jsonl leaked a 'key' field",
                  file=sys.stderr)
            return False

    # A coord WITH a key round-trips through write/load_jsonl.
    coords = [PassageCoord(0, "L001", 0, 12, "sha256:aa", "deadbeef", key="L001")]
    reloaded = load_jsonl(write_jsonl(coords).splitlines())
    if reloaded[0].key != "L001" or reloaded[0].passage_id != "deadbeef":
        print(f"[row_mode] FAIL R4: key round-trip lost ({reloaded[0]})", file=sys.stderr)
        return False
    return True


def _r5_key_threaded_to_surfaces() -> bool:
    from resonance_lattice.fabric._runtime import search_with_state, slice_with_state
    from resonance_lattice.store import archive
    from resonance_lattice.store.bundled import BundledStore
    from resonance_lattice.store.verified import verify_hits

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "km.rlat"
        _build_rows(out, _ROWS)
        contents = archive.read(out)
        store = BundledStore(out)

        # verify_hits carries the key from the registry coord.
        hits = verify_hits([(0, 0.9), (1, 0.5)], store, contents.registry)
        if hits[0].key != "L001" or hits[1].key != "L002":
            print(f"[row_mode] FAIL R5: verify_hits keys {[h.key for h in hits]}",
                  file=sys.stderr)
            return False

        state = (contents, store, ZeroEncoder())
        # search() hit dicts include the key.
        sr = search_with_state(state, "quiet place", top_k=3, verified_only=True)
        if not sr["hits"] or "key" not in sr["hits"][0]:
            print(f"[row_mode] FAIL R5: search hit dict missing key: {sr['hits'][:1]}",
                  file=sys.stderr)
            return False

        # slice() returns a deduped, in-corpus key set + per-key receipts.
        sl = slice_with_state(state, "quiet place", top_k=3, verified_only=True)
        valid = {k for k, _ in _ROWS}
        if not sl["keys"]:
            print("[row_mode] FAIL R5: slice returned no keys", file=sys.stderr)
            return False
        if len(sl["keys"]) != len(set(sl["keys"])):
            print(f"[row_mode] FAIL R5: slice keys not deduped: {sl['keys']}", file=sys.stderr)
            return False
        if any(k not in valid for k in sl["keys"]):
            print(f"[row_mode] FAIL R5: slice key not in corpus: {sl['keys']}", file=sys.stderr)
            return False
        if any("key" not in h or "text" not in h for h in sl["hits"]):
            print("[row_mode] FAIL R5: slice hit missing key/text", file=sys.stderr)
            return False
    return True


def _r6_convert_guard() -> bool:
    from resonance_lattice.store.conversion import convert

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "km.rlat"
        _build_rows(out, _ROWS)
        # A row-mode km is bundled-only; convert away from bundled must refuse.
        for target in ("local", "remote"):
            try:
                convert(out, target, source_root=Path(d) / "ext",
                        remote_url_base="https://x.test/v1")
                print(f"[row_mode] FAIL R6: convert --to {target} did not raise",
                      file=sys.stderr)
                return False
            except ValueError as e:
                if "row-mode" not in str(e):
                    print(f"[row_mode] FAIL R6: wrong error for {target}: {e}",
                          file=sys.stderr)
                    return False
    return True


def run() -> int:
    failures = 0
    failures += not _check(_r1_walker(), "R1 (RowSourceWalker)")
    failures += not _check(_r2_build_and_roundtrip(), "R2 (row build + unicode round-trip)")
    failures += not _check(_r3_id_pinned_and_guards(), "R3 (id pinned to key + guards)")
    failures += not _check(_r4_backward_compat(), "R4 (chunked build unaffected)")
    failures += not _check(_r5_key_threaded_to_surfaces(), "R5 (key on verified + fabric)")
    failures += not _check(_r6_convert_guard(), "R6 (convert refuses row-mode away from bundled)")

    if failures:
        print(f"[row_mode] {failures} guarantee(s) failed", file=sys.stderr)
        return 1
    print("[row_mode] all guarantees OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
