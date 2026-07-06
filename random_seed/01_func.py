"""
Stage 1 - where does a random seed "live", and what happens under 'spawn'? Standard library only.

LLM talk, may not be true:
This mirrors how Catalyst launches DDP workers: torch.multiprocessing.spawn uses
Python's "spawn" start method, which boots a FRESH interpreter per worker and
RE-IMPORTS this module. So any code written at the TOP of the file runs again in
every worker; code inside main() runs only in the parent.

Run:  python 01_func.py
"""

import multiprocessing as mp
import os
import random

# (A) Seed created at the TOP of the file (module scope).
#     Under "spawn" this exact line re-executes in every worker -> new value each time.
TOP_SEED = random.randint(0, 9999)


def worker(worker_id, passed_seed, q):
    """Runs in a separate spawned process (one per 'rank')."""
    # TOP_SEED  -> read from this module, which was re-imported in this worker
    # passed_seed -> created once in the parent and handed to us as an argument
    q.put((worker_id, os.getpid(), TOP_SEED, passed_seed))


def main():
    n_workers = 4

    # (B) Seed created INSIDE main(). Workers never call main(), so this line
    #     runs exactly once (in the parent). We hand the value to each worker.
    main_seed = random.randint(0, 9999)

    print("=== random seeds under multiprocessing 'spawn' (same mechanism as Catalyst DDP) ===")
    print(f"parent (pid {os.getpid()}): TOP_SEED={TOP_SEED}   main_seed={main_seed}\n")

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(i, main_seed, q)) for i in range(n_workers)]
    for p in procs:
        p.start()
    rows = sorted(q.get() for _ in procs)
    for p in procs:
        p.join()

    for wid, pid, top, passed in rows:
        print(f"  worker {wid} (pid {pid}): TOP_SEED={top:>4}   passed_seed={passed:>4}")

    top_seeds = {top for _, _, top, _ in rows}
    passed = {p for _, _, _, p in rows}
    print()
    print(f"TOP_SEED  (top of file) across workers: {sorted(top_seeds)}  -> "
          f"{'DIFFERENT per worker' if len(top_seeds) > 1 else 'same'}")
    print(f"main_seed (passed in)   across workers: {sorted(passed)}  -> "
          f"{'SAME on every worker' if len(passed) == 1 else 'DIFFERENT'}")
    print("\nTakeaway: 'spawn' re-imports the module, so a top-level random seed is")
    print("re-drawn in each worker. A seed made in main() and passed in is shared.")


if __name__ == "__main__":
    main()
