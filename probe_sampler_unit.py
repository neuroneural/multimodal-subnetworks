#!/usr/bin/env python3
"""
Layer 1 — standalone sampler unit probe (no GPU, no Catalyst, no Mongo).

Instantiates the samplers directly for each rank in a single process and
checks the resulting partition. This isolates the *sampler logic itself*
from anything Catalyst / accelerate does to the DataLoader. It runs anywhere
numpy is available (torch is only needed for the optional --via-dataloader
passthrough check).

Three scenarios are exercised for each (N, batch_size, world_size) combo:

  current        DistributedDBBatchSampler, SAME seed on every rank
                 -> expected: disjoint partition, full coverage (the design
                    only works if every rank shares the same seed).

  old            plain DBBatchSampler with the SAME seed on every rank, each
                 rank iterating ALL batches (this is the commit-5fb4306 DDP
                 path) -> expected: every rank sees the whole dataset
                 (complete duplication, no sharding).

  seed_mismatch  DistributedDBBatchSampler but a DIFFERENT seed per rank
                 -> demonstrates how the design breaks if the module-level
                    SEED is re-drawn per process on spawn (overlaps + gaps).

Exit code is non-zero if the `current` scenario fails its checks, so this can
gate CI.

Examples:
    python probe_sampler_unit.py
    python probe_sampler_unit.py --N 37 200 --batch-size 4 --world-sizes 1 2 4
    python probe_sampler_unit.py --via-dataloader --num-workers 0 2
"""

from __future__ import annotations

import argparse
import sys

import probe_common as pc


def _meta_from_sampler(s):
    """Pull the sampler's own bookkeeping if it exposes it."""
    try:
        return {
            "total_size": int(s.total_size),
            "global_batch_size": int(s.global_batch_size),
            "num_batches": int(s.num_batches),
        }
    except AttributeError:
        return None


def collect_current(Dist, N, B, W, seed):
    """Per-rank batches from DistributedDBBatchSampler, same seed everywhere."""
    rank_to_batches = {}
    meta = None
    for r in range(W):
        s = Dist(list(range(N)), batch_size=B, seed=seed, rank=r, world_size=W)
        if r == 0:
            meta = _meta_from_sampler(s)
        rank_to_batches[r] = list(iter(s))
    return rank_to_batches, meta


def collect_old(Base, N, B, W, seed):
    """Per-rank batches from plain DBBatchSampler (5fb4306 DDP path).

    Every rank uses the same seed and iterates the *entire* set of batches,
    because the old sampler had no notion of rank.
    """
    rank_to_batches = {}
    for r in range(W):
        s = Base(list(range(N)), batch_size=B, seed=seed)
        rank_to_batches[r] = list(iter(s))
    return rank_to_batches


def collect_seed_mismatch(Dist, N, B, W, seed):
    """DistributedDBBatchSampler but each rank gets a different seed."""
    rank_to_batches = {}
    meta = None
    for r in range(W):
        s = Dist(list(range(N)), batch_size=B, seed=seed + r, rank=r, world_size=W)
        if r == 0:
            meta = _meta_from_sampler(s)
        rank_to_batches[r] = list(iter(s))
    return rank_to_batches, meta


def via_dataloader_check(Dist, N, B, W, seed, num_workers_list):
    """Optional: confirm the DataLoader hands the *whole index array* to
    Dataset.__getitem__, and that num_workers does not duplicate/drop indices.

    Needs torch. Skipped with a message if torch is unavailable.
    """
    try:
        import torch  # noqa: F401
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:  # noqa: BLE001
        print(f"[via-dataloader] SKIPPED (torch unavailable: {type(exc).__name__})\n")
        return

    class RecordingDataset(Dataset):
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            # idx is whatever the sampler yielded: here a numpy array of indices
            import numpy as np

            return np.asarray(idx).ravel().tolist()

    for nw in num_workers_list:
        print(f"[via-dataloader] world_size={W}, num_workers={nw}")
        rank_to_batches = {}
        for r in range(W):
            sampler = Dist(list(range(N)), batch_size=B, seed=seed, rank=r, world_size=W)
            loader = DataLoader(
                RecordingDataset(N),
                sampler=sampler,
                collate_fn=lambda x: x,  # x is a list with one element (the array)
                num_workers=nw,
            )
            seen = []
            for item in loader:
                # item == [[i0, i1, ...]] : one chunk per DataLoader step
                seen.append(item[0])
            rank_to_batches[r] = seen
        part = pc.analyze_partition(rank_to_batches, N)
        batches = pc.analyze_batches(rank_to_batches)
        verdict = pc.build_verdict(part, expect_sharded=True)
        pc.print_report(
            f"VIA DATALOADER  N={N} B={B} W={W} num_workers={nw}",
            part, batches, verdict,
        )


def run_combo(Dist, Base, N, B, W, seed):
    overall_ok = True

    # --- current sampler (the one to validate) ---
    rtb, meta = collect_current(Dist, N, B, W, seed)
    part = pc.analyze_partition(rtb, N)
    batches = pc.analyze_batches(rtb)
    verdict = pc.build_verdict(part, meta=meta, expect_sharded=True)
    ok = pc.print_report(
        f"CURRENT  DistributedDBBatchSampler  N={N} B={B} W={W} seed={seed} (same seed/rank)",
        part, batches, verdict,
    )
    overall_ok = overall_ok and ok

    # --- old sampler (demonstration, not gated) ---
    rtb = collect_old(Base, N, B, W, seed)
    part = pc.analyze_partition(rtb, N)
    batches = pc.analyze_batches(rtb)
    verdict = pc.build_verdict(part, expect_sharded=False)
    pc.print_report(
        f"OLD  DBBatchSampler (5fb4306 DDP path)  N={N} B={B} W={W} seed={seed}",
        part, batches, verdict,
    )

    # --- seed mismatch (demonstration of the spawn risk) ---
    if W > 1:
        rtb, meta = collect_seed_mismatch(Dist, N, B, W, seed)
        part = pc.analyze_partition(rtb, N)
        batches = pc.analyze_batches(rtb)
        verdict = pc.build_verdict(part, meta=meta, expect_sharded=True)
        pc.print_report(
            f"SEED-MISMATCH  DistributedDBBatchSampler  N={N} B={B} W={W} "
            f"(seed=SEED+rank; simulates SEED re-drawn per process)",
            part, batches, verdict,
        )

    return overall_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, nargs="+", default=[37, 200],
                    help="dataset size(s) to test (use a prime-ish small N + a realistic one)")
    ap.add_argument("--batch-size", type=int, default=4, help="per-GPU batch size (num_volumes)")
    ap.add_argument("--world-sizes", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-real", action="store_true",
                    help="force the vendored reference samplers even if the real ones import")
    ap.add_argument("--via-dataloader", action="store_true",
                    help="also run the DataLoader passthrough check (needs torch)")
    ap.add_argument("--num-workers", type=int, nargs="+", default=[0, 2],
                    help="num_workers values to test in --via-dataloader")
    args = ap.parse_args()

    Dist, Base, source = pc.load_samplers(prefer_real=not args.no_real)
    print(f"\nSampler source: {source}\n")

    all_ok = True
    for N in args.N:
        for W in args.world_sizes:
            ok = run_combo(Dist, Base, N, args.batch_size, W, args.seed)
            all_ok = all_ok and ok
            if args.via_dataloader and W >= 1:
                via_dataloader_check(Dist, N, args.batch_size, W, args.seed, args.num_workers)

    print("=" * 78)
    print(f"OVERALL (current sampler): {'PASS' if all_ok else 'FAIL'}")
    print("=" * 78)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
