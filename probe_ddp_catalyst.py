#!/usr/bin/env python3
"""
Layer 2 — Catalyst integration probe (real DDP, run under SLURM).

This drives the *real* sampler through the *real* Catalyst stack so we capture
everything the Layer-1 unit probe cannot:

  * the per-rank value of the module-level ``SEED`` that actually feeds the
    sampler (this is the crux: Catalyst spawns workers with ``mp.spawn`` /
    start-method "spawn", which re-imports the module and may RE-DRAW that
    random SEED per rank -> the partition logic silently breaks);
  * agreement between the four world-size sources
    (SLURM_GPUS_ON_NODE / cuda.device_count / torch.distributed / catalyst);
  * what ``engine.prepare(loader)`` does to the loader (does HuggingFace
    accelerate re-shard / re-wrap it on top of our sampler -> double sharding?);
  * the actual indices each rank consumes, recorded to per-rank CSVs.

It mirrors ``train_script_rev.CustomRunner``'s DDP-relevant structure exactly
(same ``get_engine`` decision, same ``BatchPrefetchLoaderWrapper(DataLoader(
sampler=...))`` construction, train uses the sharded sampler while valid uses a
plain ``DBBatchSampler`` — just like the real code) but swaps in a dummy
in-memory dataset and a 1-parameter model so only the sampler/loader plumbing
is exercised. No Mongo, no ResNet.

Run it via probe_ddp.sh on 1, 2 and 4 GPUs, then run probe_analyze.py on the
output dir to get a verdict per allocation.

Outputs (in --logdir):
    probe_meta_rank{R}.jsonl       one line per loader: seed, world sizes, loader type/len
    probe_idx_{loader}_rank{R}.csv columns: epoch, step, idx
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")  # quiet pynvml/pydantic import warnings (before torch import)

import argparse
import csv
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from catalyst import dl
from catalyst.data import BatchPrefetchLoaderWrapper
from catalyst.utils import distributed as cat_dist

# Import the REAL training module. This gives us the real samplers AND the real
# module-level SEED, and — because mp.spawn re-imports this probe module (which
# imports train_script_rev) in every worker — it reproduces the exact per-rank
# SEED behaviour of the real job.
import train_script_rev as tsr  # noqa: E402
from mindfultensors.utils import DBBatchSampler  # noqa: E402


# --- Seed experiment -------------------------------------------------------
# MODULE_SEED is drawn ABOVE the __main__ guard, exactly like
# train_script_rev.SEED (line 40). Under Catalyst's mp.spawn ("spawn" start
# method) every worker re-imports this module and re-runs this line, so this
# value is expected to DIFFER per rank -- the same failure mode that affects
# the real SEED. (See probe_spawn_seed_demo.py for the isolated proof.)
# The "guard" seed is drawn in main() (under the guard) and passed into the
# runner, so it is expected to be IDENTICAL on every rank: the fix pattern.
MODULE_SEED = random.randint(0, 9999)


def _safe_rank() -> int:
    r = cat_dist.get_rank()
    return 0 if r is None or r < 0 else r


def probe_collate(results):
    """The DataLoader yields a list with ONE element (the sampler's chunk).

    Each element is ``(indices_list, features)`` from the dataset. Mirrors the
    real collate's ``results[0]`` access pattern.
    """
    indices, feats = results[0]
    return torch.tensor(indices, dtype=torch.long), feats


class IndexDataset(Dataset):
    """Returns the indices it is asked for, so we can see what each rank gets.

    ``DBBatchSampler``/``DistributedDBBatchSampler`` are used as ``sampler=`` and
    yield whole numpy arrays of indices; PyTorch passes such an array straight to
    ``__getitem__`` as ``idx``.
    """

    def __init__(self, n: int):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        arr = np.asarray(idx).ravel()
        feats = torch.zeros((len(arr), 1), dtype=torch.float32)  # dummy 1-feature input
        return arr.tolist(), feats


class ProbeRunner(dl.Runner):
    def __init__(self, logdir, n, batch_size, epochs, sampler_kind, num_workers, guard_seed):
        super().__init__()
        self._logdir = logdir
        self._n = n
        self._batch_size = batch_size
        self._epochs = epochs
        self._sampler_kind = sampler_kind  # "current" or "old"
        self._num_workers = num_workers
        # drawn under the __main__ guard in main(), shipped here by pickling ->
        # expected identical on every rank (contrast with MODULE_SEED / tsr.SEED)
        self._guard_seed = guard_seed
        os.makedirs(logdir, exist_ok=True)

    # ---- mirror CustomRunner's engine decision exactly ----
    def get_engine(self):
        if os.environ.get("PROBE_FORCE_CPU") == "1":
            return dl.CPUEngine()
        n_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", torch.cuda.device_count()))
        if n_gpus > 1:
            return dl.DistributedDataParallelEngine(
                process_group_kwargs={"backend": os.environ.get("PROBE_BACKEND", "nccl")},
            )
        return dl.GPUEngine() if torch.cuda.is_available() else dl.CPUEngine()

    @property
    def num_epochs(self) -> int:
        return self._epochs

    @property
    def seed(self) -> int:
        return 42  # fixed; the sampler seed is tsr.SEED, tracked separately

    def get_loggers(self):
        # Override to {} like CustomRunner does. The base Runner.get_loggers()
        # reads self._loggers, which the base __init__ never sets -> AttributeError.
        # The probe writes its own per-rank CSVs, so it needs no Catalyst loggers
        # (also avoids the tensorboard/csv loggers the base would auto-add).
        return {}

    def _build_loader(self, sampler):
        kwargs = dict(
            sampler=sampler,
            collate_fn=probe_collate,
            num_workers=self._num_workers,
        )
        if self._num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 2
        loader = DataLoader(IndexDataset(self._n), **kwargs)
        # Faithful wrapper, but it needs CUDA streams; skip on CPU debug runs.
        if torch.cuda.is_available():
            loader = BatchPrefetchLoaderWrapper(loader, num_prefetches=2)
        return loader

    def get_loaders(self):
        rank, world_size = tsr.get_rank_world()

        # TRAIN: sharded sampler under DDP (the thing we are validating),
        # exactly as train_script_rev does.
        if self._sampler_kind == "current" and self.engine.is_ddp:
            train_sampler = tsr.DistributedDBBatchSampler(
                IndexDataset(self._n), batch_size=self._batch_size,
                seed=tsr.SEED, rank=rank, world_size=world_size,
            )
        else:
            # "old" path (commit 5fb4306 DDP) or single-GPU: plain sampler.
            train_sampler = DBBatchSampler(
                IndexDataset(self._n), batch_size=self._batch_size, seed=tsr.SEED,
            )

        # VALID: plain DBBatchSampler even under DDP — mirrors the real code,
        # so the probe also reveals that validation is duplicated across ranks.
        valid_sampler = DBBatchSampler(
            IndexDataset(self._n), batch_size=self._batch_size, seed=tsr.SEED,
        )

        return {
            "train": self._build_loader(train_sampler),
            "valid": self._build_loader(valid_sampler),
        }

    def get_model(self):
        return torch.nn.Linear(1, 1)  # 1 param group, all params used -> DDP-happy

    def get_criterion(self):
        return None

    def get_optimizer(self, model):
        return torch.optim.SGD(model.parameters(), lr=0.0)  # lr=0: don't actually move

    def get_scheduler(self, optimizer):
        return None

    def get_callbacks(self):
        return {}  # no checkpoint/logging callbacks needed

    # ---- recording ----
    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        rank = _safe_rank()
        loader = self.loaders[self.loader_key]

        # one-time-per-loader metadata row
        meta = {
            "loader": self.loader_key,
            "sampler_kind": self._sampler_kind,
            "rank": rank,
            "tsr_SEED": int(tsr.SEED),          # the REAL sampler seed (module-level)
            "module_seed": int(MODULE_SEED),    # control: module-level -> expect DIFFERENT per rank
            "guard_seed": int(self._guard_seed),  # control: guard-level -> expect SAME per rank
            "is_ddp": bool(self.engine.is_ddp),
            "engine_process_index": int(getattr(self.engine, "process_index", -1)),
            "engine_num_processes": int(getattr(self.engine, "num_processes", -1)),
            "env_SLURM_GPUS_ON_NODE": os.environ.get("SLURM_GPUS_ON_NODE"),
            "env_RANK": os.environ.get("RANK"),
            "env_WORLD_SIZE": os.environ.get("WORLD_SIZE"),
            "env_LOCAL_RANK": os.environ.get("LOCAL_RANK"),
            "cuda_device_count": torch.cuda.device_count(),
            "torch_dist_world_size": (
                torch.distributed.get_world_size()
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else None
            ),
            "torch_dist_rank": (
                torch.distributed.get_rank()
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else None
            ),
            "catalyst_world_size": cat_dist.get_world_size(),
            "catalyst_rank": cat_dist.get_rank(),
            "loader_type_after_prepare": type(loader).__name__,
            "loader_len": len(loader),
            "sampler_type": type(getattr(loader, "sampler", None)).__name__,
        }
        with open(os.path.join(self._logdir, f"probe_meta_rank{rank}.jsonl"), "a") as f:
            f.write(json.dumps(meta) + "\n")

        # open per-rank index CSV for this loader
        self._csv_path = os.path.join(
            self._logdir, f"probe_idx_{self.loader_key}_rank{rank}.csv"
        )
        new = not (os.path.isfile(self._csv_path) and os.path.getsize(self._csv_path) > 0)
        self._csv_f = open(self._csv_path, "a", newline="")
        self._csv_w = csv.writer(self._csv_f)
        if new:
            self._csv_w.writerow(["epoch", "step", "idx"])

    def handle_batch(self, batch):
        indices, feats = batch
        # trivial forward/backward so DDP has gradients to sync (lr=0 -> no drift)
        out = self.model(feats)
        loss = out.float().pow(2).mean()
        if self.is_train_loader:
            self.engine.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()

        idx_list = indices.detach().cpu().numpy().ravel().tolist()
        for idx in idx_list:
            self._csv_w.writerow([self.epoch_step, self.batch_step, int(idx)])

        self.batch_metrics.update({"loss": loss})

    def on_loader_end(self, runner):
        if getattr(self, "_csv_f", None):
            self._csv_f.flush()
            self._csv_f.close()
            self._csv_f = None
        super().on_loader_end(runner)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--logdir", default="./_probe_out", help="where to write per-rank CSVs")
    ap.add_argument("--n", type=int, default=200, help="dummy dataset size")
    ap.add_argument("--batch-size", type=int, default=4, help="per-GPU batch size (num_volumes)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--sampler", choices=["current", "old"], default="current",
                    help="'current' = DistributedDBBatchSampler; 'old' = plain DBBatchSampler (5fb4306 DDP path)")
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    logdir = os.path.join(args.logdir, f"{args.sampler}_n{args.n}_bs{args.batch_size}")
    os.makedirs(logdir, exist_ok=True)

    # Drawn UNDER the guard, in the real __main__ parent only. Spawned workers
    # never re-run this; they receive it via the pickled runner -> same on all.
    guard_seed = random.randint(0, 9999)

    print(f"[probe] logdir={logdir} sampler={args.sampler} n={args.n} "
          f"batch_size={args.batch_size} epochs={args.epochs} "
          f"SLURM_GPUS_ON_NODE={os.environ.get('SLURM_GPUS_ON_NODE')} "
          f"tsr.SEED(parent)={tsr.SEED} MODULE_SEED(parent)={MODULE_SEED} guard_seed={guard_seed}")

    runner = ProbeRunner(
        logdir=logdir, n=args.n, batch_size=args.batch_size,
        epochs=args.epochs, sampler_kind=args.sampler, num_workers=args.num_workers,
        guard_seed=guard_seed,
    )
    runner.run()
    print(f"[probe] done. Analyze with:  python probe_analyze.py {logdir}")


if __name__ == "__main__":
    main()
