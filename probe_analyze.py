#!/usr/bin/env python3
"""
Analyze the per-rank output of probe_ddp_catalyst.py.

Reads:
  probe_meta_rank{R}.jsonl       (seed, world-size sources, loader type/len)
  probe_idx_{loader}_rank{R}.csv (epoch, step, idx)

and prints, per loader and per epoch:
  * whether every rank saw the SAME tsr.SEED   (the crux of the spawn concern)
  * whether the four world-size sources agree
  * whether engine.prepare re-wrapped the loader (type after prepare)
  * the partition verdict (coverage / balance / duplication-vs-padding)

Usage:
    python probe_analyze.py <logdir> [--data-size N]

If --data-size is omitted it is inferred as the max index seen + 1.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re

import probe_common as pc

IDX_RE = re.compile(r"probe_idx_(?P<loader>.+)_rank(?P<rank>\d+)\.csv$")


def load_meta(logdir):
    """rank -> list of meta dicts (one per loader)."""
    metas = collections.defaultdict(list)
    for path in glob.glob(os.path.join(logdir, "probe_meta_rank*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    metas[d["rank"]].append(d)
    return metas


def load_indices(logdir):
    """loader -> rank -> {epoch -> [ (step, idx), ... ]}."""
    data = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for path in glob.glob(os.path.join(logdir, "probe_idx_*_rank*.csv")):
        m = IDX_RE.search(os.path.basename(path))
        if not m:
            continue
        loader, rank = m.group("loader"), int(m.group("rank"))
        with open(path) as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 3:
                    continue
                epoch, step, idx = int(parts[0]), int(parts[1]), int(parts[2])
                data[loader][rank][epoch].append((step, idx))
    return data


def batches_for_epoch(rank_epoch_rows):
    """Reconstruct per-rank list-of-batches (grouped by step) for one epoch."""
    by_step = collections.defaultdict(list)
    for step, idx in rank_epoch_rows:
        by_step[step].append(idx)
    return [by_step[s] for s in sorted(by_step)]


def report_seed_and_world(metas):
    print("=" * 78)
    print("SEED & WORLD-SIZE CONSISTENCY ACROSS RANKS")
    print("=" * 78)
    # one row per rank (use the first meta entry per rank; they share these)
    rows = {r: ms[0] for r, ms in metas.items() if ms}
    if not rows:
        print("  (no meta files found)")
        return

    # tsr.SEED == the real sampler seed. Must be identical across ranks.
    seeds = {r: rows[r]["tsr_SEED"] for r in sorted(rows)}
    print(f"  tsr.SEED per rank: {seeds}")
    seed_ok = len(set(seeds.values())) == 1
    print(f"  [{'PASS' if seed_ok else 'FAIL'}] all ranks share the same tsr.SEED "
          f"{'' if seed_ok else '<-- DistributedDBBatchSampler partition is INVALID if this fails'}")

    # Controls from the seed experiment (only present if the probe logged them).
    if all("module_seed" in rows[r] for r in rows):
        mod = {r: rows[r]["module_seed"] for r in sorted(rows)}
        if len(set(mod.values())) > 1:
            note = ("DIFFERS (confirms mp.spawn re-imports the module; this is WHY a "
                    "module-level SEED like tsr.SEED is unsafe)")
        else:
            note = "same"
        print(f"  module_seed per rank: {mod} -> {note}")
    if all("guard_seed" in rows[r] for r in rows):
        grd = {r: rows[r]["guard_seed"] for r in sorted(rows)}
        if len(set(grd.values())) == 1:
            note = ("same (a seed drawn under the __main__ guard and passed into the "
                    "runner is broadcast-safe -- the fix pattern)")
        else:
            note = "DIFFERS (unexpected)"
        print(f"  guard_seed  per rank: {grd} -> {note}")

    print("  world-size sources per rank:")
    for r in sorted(rows):
        d = rows[r]
        print(f"    rank {r}: SLURM_GPUS_ON_NODE={d['env_SLURM_GPUS_ON_NODE']} "
              f"cuda_count={d['cuda_device_count']} "
              f"torch_dist={d['torch_dist_world_size']} "
              f"catalyst={d['catalyst_world_size']} "
              f"env_WORLD_SIZE={d['env_WORLD_SIZE']} "
              f"engine_num_processes={d['engine_num_processes']}")
    # agreement among the numeric sources actually defined
    def collect_ws(d):
        vals = []
        for k in ("cuda_device_count", "torch_dist_world_size", "catalyst_world_size",
                  "engine_num_processes"):
            v = d.get(k)
            if isinstance(v, int) and v > 0:
                vals.append(v)
        if d.get("env_SLURM_GPUS_ON_NODE"):
            vals.append(int(d["env_SLURM_GPUS_ON_NODE"]))
        if d.get("env_WORLD_SIZE"):
            vals.append(int(d["env_WORLD_SIZE"]))
        return set(vals)
    all_ws = set()
    for r in rows:
        all_ws |= collect_ws(rows[r])
    ws_ok = len(all_ws) == 1
    print(f"  [{'PASS' if ws_ok else 'FAIL'}] world-size sources agree: {sorted(all_ws)}")

    print("  loader after engine.prepare (double-shard / re-wrap check):")
    for r in sorted(rows):
        for m in metas[r]:
            print(f"    rank {r} [{m['loader']}]: type={m['loader_type_after_prepare']} "
                  f"len={m['loader_len']} sampler={m['sampler_type']}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logdir")
    ap.add_argument("--data-size", type=int, default=None)
    args = ap.parse_args()

    metas = load_meta(args.logdir)
    report_seed_and_world(metas)

    # (rank, loader) -> meta dict, for the OneCycleLR steps_per_epoch check
    meta_by_rank_loader = {
        (d["rank"], d["loader"]): d for r in metas for d in metas[r]
    }

    data = load_indices(args.logdir)
    if not data:
        print("No probe_idx_*.csv files found in", args.logdir)
        return

    # infer data size if not given
    if args.data_size is None:
        mx = -1
        for loader in data:
            for rank in data[loader]:
                for epoch in data[loader][rank]:
                    for _, idx in data[loader][rank][epoch]:
                        mx = max(mx, idx)
        data_size = mx + 1
        print(f"(inferred data_size = {data_size}; pass --data-size to override)\n")
    else:
        data_size = args.data_size

    all_ok = True
    for loader in sorted(data):
        # treat train as sharded (gated); valid mirrors real code's plain
        # sampler -> duplicated across ranks (informational).
        expect_sharded = loader.startswith("train")
        epochs = sorted({e for r in data[loader] for e in data[loader][r]})
        for epoch in epochs:
            rank_to_batches = {
                r: batches_for_epoch(data[loader][r][epoch])
                for r in sorted(data[loader])
                if epoch in data[loader][r]
            }
            part = pc.analyze_partition(rank_to_batches, data_size)
            batches = pc.analyze_batches(rank_to_batches)
            # Layer 2 cannot read the sampler's total_size from CSV; reconstruct
            # the bound from observed slots so padding-aware checks still work.
            meta = None
            if expect_sharded and len(rank_to_batches) > 1:
                meta = {
                    "total_size": part["total_slots"],
                    "global_batch_size": max(
                        (b["first_batch_size"] or 0) for b in batches.values()
                    ) * len(rank_to_batches),
                    "num_batches": max(b["num_batches"] for b in batches.values()),
                }
            verdict = pc.build_verdict(part, meta=meta, expect_sharded=expect_sharded)
            ok = pc.print_report(
                f"LOADER={loader}  EPOCH={epoch}  "
                f"({'sharded/gated' if expect_sharded else 'plain sampler/informational'})",
                part, batches, verdict,
            )

            # --- OneCycleLR steps_per_epoch correctness ---
            # get_scheduler() builds OneCycleLR with steps_per_epoch=len(train_loader)
            # and total_steps = epochs * steps_per_epoch; scheduler.step() is called
            # once per batch per rank. So len(loader) MUST equal the actual number of
            # batches each rank iterates, or the LR cycle is mis-scheduled (and a
            # too-small value makes OneCycleLR raise once .step() overruns total_steps).
            sched_ok = True
            for r in sorted(rank_to_batches):
                declared = meta_by_rank_loader.get((r, loader), {}).get("loader_len")
                observed = batches[r]["num_batches"]
                match = (declared == observed)
                sched_ok = sched_ok and match
                print(f"  [scheduler] rank {r}: len(loader)={declared} (told to OneCycleLR as "
                      f"steps_per_epoch) vs observed batches/epoch={observed} "
                      f"-> {'OK' if match else 'MISMATCH'}")
            print(f"  [{'PASS' if sched_ok else 'FAIL'}] OneCycleLR steps_per_epoch matches actual "
                  f"per-rank batch count\n")

            if expect_sharded:
                all_ok = all_ok and ok and sched_ok

    print("=" * 78)
    print(f"OVERALL (train sharding): {'PASS' if all_ok else 'FAIL'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
