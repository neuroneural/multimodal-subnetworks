"""
Stage 2 - passing seeds to a CLASS, and how @property behaves under 'spawn' and on re-calls.

Three "seed sources" on one class, each matching a spot in the training code:
  1. passed_seed - a plain attribute set in __init__ from a value passed in.
                   (This is the `self.sampler_seed = sampler_seed` fix pattern.)
  2. @prop_seed  - an @property returning os.urandom on each call.
                   (This is `CustomRunner.seed`, copied verbatim.)
  3. module_seed - a method that reads the module-level TOP_SEED.
                   THIS IS WHAT WAS PASSED TO SAMPLERS IN THE CATALYST EXAMPLE CURRICULUM TRAINING SCRIPT.

Run:  python 02_class.py
"""

import multiprocessing as mp
import os
import random

# Top-of-file seed (Stage 1): re-drawn in every spawned worker.
TOP_SEED = random.randint(0, 9999)


def set_global_seed(s):
    # The real code also seeds numpy and torch; seeding `random` is enough here.
    random.seed(s)


class Experiment:
    def __init__(self, passed_seed):
        # (1) Plain attribute: stored on the instance, so it travels with the pickle.
        self.passed_seed = passed_seed

    @property
    def prop_seed(self) -> int:
        # (2) Verbatim copy of CustomRunner.seed: recomputed on EVERY access.
        random_data = os.urandom(4)
        SEED = int.from_bytes(random_data, byteorder="big")
        set_global_seed(SEED)
        return SEED

    def module_seed(self) -> int:
        # (3) Reads the top-of-file global (old sampler pattern).
        return TOP_SEED


def worker(worker_id, exp, q):
    """Runs in a spawned process; `exp` arrived pickled from the parent."""
    q.put((
        worker_id, os.getpid(),
        exp.passed_seed,    # frozen at creation -> same as parent
        exp.prop_seed,      # fresh each access
        exp.prop_seed,      # fresh again -> differs from the read above
        exp.module_seed(),  # TOP_SEED, re-drawn in this worker
    ))


def main():
    n = 4
    main_seed = random.randint(0, 9999)  # safe across workers, passed to the class
    exp = Experiment(main_seed)   # created ONCE, in the parent

    print("=== Stage 2: seeds on a CLASS under 'spawn' ===")
    print(f"parent: created exp with passed_seed={exp.passed_seed}")
    print(f"parent: @prop_seed read#1={exp.prop_seed} read#2={exp.prop_seed}")
    print("        ^ two reads differ -> the @property is recomputed each time, NOT fixed at creation")
    print(f"parent: module_seed()={exp.module_seed()} (== TOP_SEED)\n")

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(i, exp, q)) for i in range(n)]
    for p in procs:
        p.start()
    rows = sorted(q.get() for _ in procs)
    for p in procs:
        p.join()

    for wid, pid, passed, p1, p2, mod in rows:
        print(f"worker {wid} (pid {pid}): passed_seed={passed:<6} "
              f"@prop_seed#1={p1:<11} @prop_seed#2={p2:<11} module_seed={mod}")

    passed_vals = {r[2] for r in rows}
    prop_vals = {r[3] for r in rows} | {r[4] for r in rows}
    mod_vals = {r[5] for r in rows}
    print()
    print(f"passed_seed (attribute, passed in) : {sorted(passed_vals)}  "
          f"-> SAME everywhere (pickled with the object)")
    print(f"@prop_seed  (os.urandom)           : {len(prop_vals)} distinct values over {2*n} reads  "
          f"-> DIFFERENT on every access & every worker")
    print(f"module_seed (reads TOP_SEED)       : {sorted(mod_vals)}  "
          f"-> DIFFERENT per worker (module re-imported)")
    print("\nTakeaway: only the attribute set once and passed in is consistent across")
    print("ranks. A @property is a function re-run on every read, so there is no stored")
    print("value to inherit; and a top-of-file global is re-drawn in each worker.")


if __name__ == "__main__":
    main()
