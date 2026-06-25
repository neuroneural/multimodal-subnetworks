#!/usr/bin/env python3
"""
Minimal, dependency-free demonstration of how the "spawn" start method (what
Catalyst's ``mp.spawn`` uses for DDP) treats a MODULE-LEVEL random seed vs a
seed drawn UNDER the ``if __name__ == "__main__"`` guard.

Why this matters: ``train_script_rev.py`` line 40 has
    SEED = random.randint(0, 9999)
at module top level (ABOVE the guard). Catalyst spawns one worker per GPU with
``mp.spawn`` (start method "spawn"), and a spawned worker re-imports the main
module (as ``__mp_main__``), re-running every top-level statement. So that SEED
is re-drawn independently in each worker -> different per rank.

A seed drawn inside ``main()`` (UNDER the guard) runs only in the real parent
``__main__``; spawned workers skip it because of the guard. If you pass that
value into the object that gets pickled to the workers, every rank receives the
SAME value.

This script reproduces exactly that, using only the standard library, so it can
run anywhere (no torch / no GPU). The mechanism is identical to torch's
``mp.spawn``.
"""

import multiprocessing as mp
import os
import random

# --- ABOVE the guard: re-runs in every spawned child ---------------------
# Mirrors train_script_rev.py:40 (SEED = random.randint(0, 9999)).
MODULE_SEED = random.randint(0, 9999)


def worker(rank: int, guard_seed: int, q: "mp.Queue") -> None:
    """Runs in each spawned child."""
    q.put(
        {
            "rank": rank,
            "pid": os.getpid(),
            # read AFTER this module was re-imported in the child:
            "module_seed": MODULE_SEED,
            # received as an argument from the parent:
            "guard_seed": guard_seed,
        }
    )


def main() -> None:
    world_size = 4

    # --- UNDER the guard: runs only in the real parent -------------------
    # This is the "fix" pattern: draw once, then hand it to the workers.
    guard_seed = random.randint(0, 9999)

    ctx = mp.get_context("spawn")  # same start method Catalyst uses
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, guard_seed, q)) for r in range(world_size)]
    for p in procs:
        p.start()
    rows = [q.get() for _ in procs]
    for p in procs:
        p.join()
    rows.sort(key=lambda d: d["rank"])

    print(f"\nparent pid={os.getpid()}  MODULE_SEED={MODULE_SEED}  guard_seed={guard_seed}\n")
    for d in rows:
        print(
            f"  rank {d['rank']} (pid {d['pid']}): "
            f"module_seed={d['module_seed']:>4}   guard_seed={d['guard_seed']:>4}"
        )

    mod_vals = {d["module_seed"] for d in rows}
    grd_vals = {d["guard_seed"] for d in rows}
    print()
    print(
        f"  module-level seed across ranks -> {sorted(mod_vals)}  "
        f"[{'DIFFERS (re-drawn on spawn)' if len(mod_vals) > 1 else 'same'}]"
    )
    print(
        f"  guard-level  seed across ranks -> {sorted(grd_vals)}  "
        f"[{'same (broadcast-safe)' if len(grd_vals) == 1 else 'DIFFERS'}]"
    )
    print()
    print("  => train_script_rev.SEED is module-level, so under Catalyst's mp.spawn it")
    print("     behaves like module_seed (different per rank), which makes")
    print("     DistributedDBBatchSampler's global permutation differ per rank and its")
    print("     per-rank slices overlap / miss data. Drawing the seed under the guard")
    print("     and passing it into the runner (like guard_seed) is broadcast-safe.\n")


if __name__ == "__main__":
    main()
