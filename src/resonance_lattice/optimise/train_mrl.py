"""MRL InfoNCE training loop.

Hparams from the original-positive-results spec (mrl_fabric_remote_train.py:407-572):
  steps=250, batch=128, negatives=7, lr=5e-4, tau=0.02
  MRL nested dims = {64, 128, 256, 512}, uniform weighting (0.25 each)
  AdamW(W, lr=5e-4, weight_decay=0.0)   # spec calls out as a critical pitfall
  W init: randn(d_native, backbone_dim) * (1.0 / sqrt(backbone_dim))
  loss precision: fp32 throughout (no autocast in train loop) — temp=0.02
                  saturates fp16 logits because exp(50) overflows max-fp16
  early-kill: dev R@1 at d=512 < 0.2 at step 100 → exit code 4
  no-progress saturation floor: best_loss < 0.01 suppresses kill

Implementation: torch on both CUDA and CPU paths, not numpy on CPU.
`[optimise]` extras pulls torch unconditionally; a parallel numpy MRL
InfoNCE implementation is maintenance burden the audit didn't justify.
Train loop runs fp32 regardless of device — embedding precision is the
encoder's concern, not training's.

Phase 4 deliverable; spec-compliance fixes 2026-04-26 after v3/v4/v5 probes
showed full regression caused by AdamW default weight_decay=0.01 (vs
spec=0.0), xavier_uniform init (vs spec randn*1/sqrt(backbone_dim)),
and fp16 InfoNCE on CUDA (vs spec fp32 throughout).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np

from .device import Device

KillReason = Literal["", "dev_r1_below_threshold", "no_progress_saturation"]

if TYPE_CHECKING:
    import torch as _torch  # noqa: F401 — type-only

# Spec-pinned hyperparameters from mrl_fabric_remote_train.py defaults —
# the values that produced the original Fabric +0.161 R@5 with gte-mb.
STEPS = 250
BATCH = 128
NEGATIVES_PER_QUERY = 7
LR = 5e-4
TEMPERATURE = 0.02
MRL_DIMS = (64, 128, 256, 512)
MRL_WEIGHT = 1.0 / len(MRL_DIMS)
SEED = 0
WEIGHT_DECAY = 0.0  # spec §2: explicit; PyTorch AdamW default is 0.01 — different recipe
EARLY_KILL_R1_THRESHOLD = 0.2
EARLY_KILL_STEP = 100
NO_PROGRESS_WINDOW = 60
SATURATION_FLOOR = 0.01
DEV_FRACTION = 0.1

EXIT_OK = 0
EXIT_EARLY_KILLED = 4


@dataclass
class TrainResult:
    """Trained MRL optimised projection + run diagnostics.

    `kill_reason` is empty string when training completed normally;
    `"dev_r1_below_threshold"` when the step-100 R@1 check fired (truly
    broken run); `"no_progress_saturation"` when loss plateaued for the
    no-progress window without crossing the saturation floor.
    """
    W: np.ndarray              # (d_native=512, backbone_dim=768) float32
    final_loss: float
    final_dev_r1: float
    steps_completed: int
    early_killed: bool
    kill_reason: KillReason = ""


def _l2_normalize_torch(x: "_torch.Tensor", eps: float = 1e-12) -> "_torch.Tensor":
    """L2 normalise along the last dim. Eps-guarded so zero rows don't NaN."""
    import torch
    return x / torch.clamp(x.norm(dim=-1, keepdim=True), min=eps)


def _infonce_loss(
    q: "_torch.Tensor",          # (B, D) — projected query, L2-normalised
    pos: "_torch.Tensor",        # (B, D) — projected positive, L2-normalised
    neg: "_torch.Tensor",        # (B, K, D) — projected negatives, L2-normalised
    temperature: float,
):
    """InfoNCE loss for one (q, pos, K-negatives) batch at a fixed dim.

    Positive at logits column 0; negatives at columns 1..K. cross_entropy
    with target=0 reduces to -log(softmax_0).
    """
    import torch
    pos_sim = (q * pos).sum(dim=-1, keepdim=True) / temperature
    neg_sim = (q.unsqueeze(1) * neg).sum(dim=-1) / temperature
    logits = torch.cat([pos_sim, neg_sim], dim=1)
    target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return torch.nn.functional.cross_entropy(logits, target)


def _mrl_loss(
    q_proj: "_torch.Tensor",     # (B, d_native)
    pos_proj: "_torch.Tensor",   # (B, d_native)
    neg_proj: "_torch.Tensor",   # (B, K, d_native)
    temperature: float,
):
    """Sum the InfoNCE loss across the nested MRL slices with uniform weights.

    For each `d ∈ MRL_DIMS`, slice the first d columns, L2 renorm the
    slice, compute InfoNCE, weight by 1/len(MRL_DIMS) and accumulate.
    """
    import torch
    total = torch.tensor(0.0, device=q_proj.device, dtype=q_proj.dtype)
    for d in MRL_DIMS:
        q_d = _l2_normalize_torch(q_proj[..., :d])
        pos_d = _l2_normalize_torch(pos_proj[..., :d])
        neg_d = _l2_normalize_torch(neg_proj[..., :d])
        total = total + MRL_WEIGHT * _infonce_loss(q_d, pos_d, neg_d, temperature)
    return total


def _dev_r1_at_native(
    W: "_torch.Tensor",
    q_dev: "_torch.Tensor",
    p_emb_full: "_torch.Tensor",
    dev_pos_idx: "_torch.Tensor",
) -> float:
    """Recall@1 of the dev queries at d_native against the full passage band.

    Used at step 100 for the early-kill check; reported in `TrainResult`.
    Hits the full band (not just a sample), but the **dev queries are an
    in-distribution holdout from the training synth-query pool**, not
    real task queries. So a high `final_dev_r1` confirms the projection
    learned the synth-query → target-passage mapping; it does NOT predict
    test-time retrieval lift on real queries (e.g. BEIR test queries).
    The v2-soak BEIR-3 measurement showed dev R@1 0.63-0.86 while real
    nDCG@10 regressed −0.04 — a plain demonstration that the metric is
    informational, not gating.
    """
    import torch
    with torch.no_grad():
        q_proj = _l2_normalize_torch(q_dev @ W.T)
        p_proj = _l2_normalize_torch(p_emb_full @ W.T)
        sims = q_proj @ p_proj.T
        top1 = sims.argmax(dim=1)
        return float((top1 == dev_pos_idx).float().mean())


def train(
    base_band: np.ndarray,             # (N, backbone_dim) — base passages, L2-normed
    query_embeddings: np.ndarray,      # (Q, backbone_dim) — encoded synth queries
    query_passage_idx: np.ndarray,     # (Q,) int — positive passage_idx
    negatives: np.ndarray,             # (Q, NEGATIVES_PER_QUERY) int
    device: Device,
    progress: Callable[[int, float, float], None] | None = None,
) -> TrainResult:
    """Run InfoNCE training and return the trained MRL projection W.

    `progress(step, loss, best_loss)` is called per step if provided —
    used by `cli/optimise.py` to render a progress bar without coupling
    this module to a UI library.
    """
    import torch

    rng = np.random.default_rng(SEED)
    n_queries = query_embeddings.shape[0]
    if n_queries < BATCH:
        raise ValueError(
            f"need ≥{BATCH} synth queries to train; got {n_queries}. "
            f"Re-run synth_queries with a larger corpus or check the LLM "
            f"call returned what was requested."
        )

    perm = rng.permutation(n_queries)
    n_dev = max(1, int(n_queries * DEV_FRACTION))
    dev_idx = perm[:n_dev]
    train_idx = perm[n_dev:]

    backbone_dim = base_band.shape[1]
    d_native = MRL_DIMS[-1]
    torch_device = torch.device(device)
    # fp32 throughout the train loop on every device. Spec §2: "loss
    # precision = fp32 (no autocast inside train loop)". With temp=0.02
    # the logits = sim/0.02 land in [-50,+50]; exp(50) ≈ 5.2e21 vs fp16
    # max ≈ 65504 → silent overflow / saturation that produces stale
    # gradients indistinguishable from convergence.
    dtype = torch.float32

    p_emb_full = torch.tensor(base_band, dtype=dtype).to(torch_device)
    q_emb_t = torch.tensor(query_embeddings, dtype=dtype).to(torch_device)
    pos_idx_t = torch.tensor(np.asarray(query_passage_idx), dtype=torch.long).to(torch_device)
    neg_idx_t = torch.tensor(np.asarray(negatives), dtype=torch.long).to(torch_device)
    dev_idx_t = torch.tensor(dev_idx, dtype=torch.long).to(torch_device)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long).to(torch_device)
    dev_pos_idx_t = pos_idx_t[dev_idx_t]
    dev_q_t = q_emb_t[dev_idx_t]

    # Spec init: randn * 1/sqrt(backbone_dim) — std ≈ 0.0361 for gte-mb.
    # Critical for temp=0.02 stability: any larger spread (xavier_uniform
    # is std≈0.069 for d_native=512, backbone_dim=768) saturates the very
    # first softmax and the InfoNCE recipe collapses on step 1.
    torch.manual_seed(SEED)
    W = torch.empty(d_native, backbone_dim, dtype=torch.float32,
                    device=torch_device, requires_grad=True)
    with torch.no_grad():
        W.normal_(mean=0.0, std=1.0 / (backbone_dim ** 0.5))
    # weight_decay=0.0 per spec §2 — PyTorch AdamW default is 0.01 which
    # continuously decays the small (512,768) projection toward zero,
    # weakening the trained signal across 250 steps.
    optimizer = torch.optim.AdamW([W], lr=LR, weight_decay=WEIGHT_DECAY)

    n_train = train_idx_t.shape[0]
    best_loss = float("inf")
    no_progress_steps = 0
    final_loss = 0.0
    early_killed = False
    kill_reason: KillReason = ""
    steps_completed = 0

    for step in range(1, STEPS + 1):
        replace = n_train < BATCH
        batch_local = torch.tensor(
            rng.choice(n_train, size=BATCH, replace=replace), dtype=torch.long,
        ).to(torch_device)
        batch = train_idx_t[batch_local]

        q = q_emb_t[batch]
        pos = p_emb_full[pos_idx_t[batch]]
        neg = p_emb_full[neg_idx_t[batch]]

        q_proj = q @ W.T
        pos_proj = pos @ W.T
        neg_proj = neg @ W.T

        loss = _mrl_loss(q_proj, pos_proj, neg_proj, TEMPERATURE)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach())
        final_loss = loss_val
        steps_completed = step
        # Epsilon-gated improvement check. The legacy positive-results
        # pipeline used `< best - 1e-4`; the v2-soak shipped with bare
        # `< best` which was reset by sub-noise improvements every step,
        # so the no-progress kill never triggered on plateaus and 250
        # steps ran on a degenerate W. Restoring the legacy contract.
        if loss_val < best_loss - 1e-4:
            best_loss = loss_val
            no_progress_steps = 0
        else:
            no_progress_steps += 1

        if progress is not None:
            progress(step, loss_val, best_loss)

        if step == EARLY_KILL_STEP:
            dev_r1 = _dev_r1_at_native(W, dev_q_t, p_emb_full, dev_pos_idx_t)
            if dev_r1 < EARLY_KILL_R1_THRESHOLD and best_loss >= SATURATION_FLOOR:
                early_killed = True
                kill_reason = "dev_r1_below_threshold"
                final_dev_r1 = dev_r1
                break

        if no_progress_steps >= NO_PROGRESS_WINDOW and best_loss >= SATURATION_FLOOR:
            early_killed = True
            kill_reason = "no_progress_saturation"
            break

    final_dev_r1 = _dev_r1_at_native(W, dev_q_t, p_emb_full, dev_pos_idx_t)
    return TrainResult(
        W=W.detach().to(torch.float32).cpu().numpy(),
        final_loss=final_loss,
        final_dev_r1=final_dev_r1,
        steps_completed=steps_completed,
        early_killed=early_killed,
        kill_reason=kill_reason,
    )
