# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 2.0.x   | ✅        |
| < 2.0   | ❌ (legacy `v0.11.0` tag preserved for read-only access; no security fixes) |

## Reporting a vulnerability

Please report security issues privately via the project's
[GitHub Security Advisories](https://github.com/tenfingerseddy/resonance-lattice/security/advisories/new)
page, **not** via public issues.

Include:

- A description of the issue and its impact.
- Steps to reproduce, ideally as a minimal working example.
- The version you tested against (`rlat --version`).
- Your suggested mitigation, if any.

A maintainer will acknowledge receipt within a reasonable window and
work with you to confirm the issue and ship a fix. There is no formal
embargo timeline; we'll coordinate disclosure after a patch ships.

## Scope

In scope:

- The `rlat` CLI and Python package.
- The `.rlat` knowledge-model format (parser, archive I/O).
- The optimise pipeline (LLM client, training loop, write-slot path).
- The remote storage mode (HTTP fetch, SHA verification, manifest pin).

Out of scope:

- The encoder model itself (`Alibaba-NLP/gte-modernbert-base`) — please
  report upstream.
- Issues in transitive dependencies (`numpy`, `faiss-cpu`, `torch`,
  `transformers`, `onnxruntime`, `openvino`) — please report upstream;
  we'll bump pins on disclosed CVEs.
- Issues that require a malicious knowledge model the user explicitly
  opens (the format is not a sandbox; treat third-party `.rlat` files
  with the same trust level as the source repository they came from).
