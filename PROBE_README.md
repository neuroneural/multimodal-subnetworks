# DDP sampler probe

Tools to verify how data is split across GPUs under DDP in `train_script_rev.py`:
batch sizes, batch counts, sample repetition, coverage, the random seed each rank
gets, and the world size reported by each source. Tests both the current
`DistributedDBBatchSampler` and the old plain `DBBatchSampler` (commit `5fb4306`)
so you can tell whether the distributed sampler is actually required and correct.

Uses random in-memory data and a 1-parameter dummy model — no Mongo, no ResNet.
Only the samplers and batch generation are exercised.

## Files

| File | What it is |
|------|-----------|
| `probe_spawn_seed_demo.py` | Standalone, stdlib-only proof of the spawn seed behavior (module-level seed differs per worker; guard-level seed passed in stays identical). Run: `python probe_spawn_seed_demo.py`. |
| `probe_common.py` | Shared: reference samplers (with real-import fallback) + partition/coverage/duplication checkers. No torch import. |
| `probe_sampler_unit.py` | **Layer 1** — standalone, no GPU/Catalyst/Mongo. Tests the sampler logic in isolation. Runs anywhere numpy is present. |
| `probe_ddp_catalyst.py` | **Layer 2** — real Catalyst `Runner` + `DistributedDataParallelEngine`. Run under SLURM. Records per-rank seed/world-size/indices. |
| `probe_analyze.py` | Aggregates Layer 2's per-rank CSVs into a verdict. |
| `probe_ddp.sh` | SLURM launcher for Layer 2 (1/2/4 GPUs, current vs old). |

## Layer 1 — sampler unit probe (fast, run anywhere)

```bash
python probe_sampler_unit.py                       # defaults: N in {37,200}, B=4, W in {1,2,4}
python probe_sampler_unit.py --N 37 200 --batch-size 4 --world-sizes 1 2 4
python probe_sampler_unit.py --via-dataloader --num-workers 0 2   # also test DataLoader passthrough (needs torch)
python probe_sampler_unit.py --no-real             # force the vendored reference copies
```

For each `(N, batch_size, world_size)` it prints three scenarios:

- **CURRENT** — `DistributedDBBatchSampler`, same seed on every rank. Expected: full
  coverage, balanced ranks, duplication only from tail padding (`< global_batch_size`).
  Exit code is non-zero if this fails.
- **OLD** — plain `DBBatchSampler`, the `5fb4306` DDP path (every rank iterates *all*
  batches). Shows full cross-rank duplication = no sharding. Informational.
- **SEED-MISMATCH** — `DistributedDBBatchSampler` with a different seed per rank.
  Demonstrates how the design breaks (missing coverage + excess duplication) if the
  module-level `SEED` is re-drawn per process.

By default it imports the **real** classes from `train_script_rev` /
`mindfultensors`; if those aren't importable it falls back to vendored reference
copies and says so in the header (`Sampler source: ...`).

## Layer 2 — Catalyst integration probe (SLURM)

This is the one that answers the questions Layer 1 can't, because it runs through the
real spawn + `engine.prepare` path:

```bash
sbatch --gres=gpu:A100:1 probe_ddp.sh current
sbatch --gres=gpu:A100:2 probe_ddp.sh current
sbatch --gres=gpu:A100:4 probe_ddp.sh current
sbatch --gres=gpu:A100:2 probe_ddp.sh old        # compare the old sampler under the real stack
```

Tunables: `PROBE_N` (default 200), `PROBE_BS` (default 4), `PROBE_EPOCHS` (default 2).
The job runs the probe and then `probe_analyze.py` automatically; results are in the
`#SBATCH --output` file and under `./_probe_out/job_<id>_gpus<N>_<sampler>/`.

To re-analyze later:

```bash
python probe_analyze.py _probe_out/job_<id>_gpus2_current/current_n200_bs4 --data-size 200
```

### What Layer 2 reports

- **SEED per rank** — `tsr.SEED` on every rank, with a PASS/FAIL on whether they're
  identical. *This is the crux*: `DistributedDBBatchSampler` slices a per-rank portion
  out of a global permutation, which is only a valid partition if every rank built the
  same permutation, i.e. got the same seed. Two controls are logged alongside it:
  `module_seed` (drawn above the `__main__` guard, like `tsr.SEED` — expected to DIFFER
  per rank, confirming the spawn re-import) and `guard_seed` (drawn under the guard and
  passed into the runner — expected IDENTICAL, the broadcast-safe fix pattern).
- **OneCycleLR steps_per_epoch** — `get_scheduler()` builds OneCycleLR with
  `steps_per_epoch=len(self.loaders["train"])` and `total_steps = epochs × steps_per_epoch`,
  and `scheduler.step()` is called once per batch per rank. The analyzer compares the
  `len(loader)` each rank reported (after `engine.prepare`) against the actual number of
  batches it iterated. They must match, or the LR cycle is mis-scheduled.
- **World-size sources** — `SLURM_GPUS_ON_NODE`, `cuda.device_count()`,
  `torch.distributed.get_world_size()`, Catalyst's `get_world_size()`, env `WORLD_SIZE`,
  and the engine's `num_processes`, with a PASS/FAIL on agreement.
- **Loader after `engine.prepare`** — the loader's type/len and sampler type *after*
  Catalyst (HuggingFace accelerate) prepares it, to detect re-wrapping / double sharding.
- **Per loader, per epoch** — coverage, per-rank batch counts and sizes, and the
  padding-aware duplication verdict. `train` is gated; `valid` is informational (the real
  code uses a plain `DBBatchSampler` for valid, so it is duplicated across ranks by design).

## Pass criteria (for the current sampler, train loader)

- All ranks report the **same `SEED`**.
- The world-size sources **agree**.
- **Full coverage**: every dataset index appears at least once across ranks.
- **Balanced**: equal per-rank batch counts.
- **Duplication only from padding**: total duplicated index-slots `== total_size - data_size`,
  and that padding is `< global_batch_size`.
- `engine.prepare` did **not** silently re-shard on top of the custom sampler.
- **OneCycleLR**: `len(train_loader)` (what is passed as `steps_per_epoch`) equals the actual
  per-rank batch count, on every rank.

## Notes / things this surfaced while building

- The samplers are used as `sampler=` (not `batch_sampler=`): each yielded item is a whole
  numpy array of indices, so the **effective per-GPU batch size = `num_volumes`** and
  `len(sampler) = ceil(data_size / global_batch_size)`.
- Tail-padding duplication across ranks is **normal and bounded** (PyTorch's own
  `DistributedSampler` does the same); the checker only flags duplication beyond the padding.
- Only `train` uses the sharded sampler; `valid`/`infer` use plain `DBBatchSampler`, so every
  rank validates on the full set and the metric mean-reduce averages duplicates.
- The seed concern is real because Catalyst spawns workers via `mp.spawn` (start method
  "spawn"), which re-imports the module and re-runs `SEED = random.randint(...)`. Layer 2
  measures the actual per-rank value rather than assuming.
