"""
Stage 4 - feed each candidate seed to its own DataLoader and see what data each
rank gets, under REAL Catalyst DDP - using the SAME sampler as the training script.

This uses mindfultensors.utils.DBBatchSampler (exactly what train_script_rev.py
builds in get_loaders), a plain sampler that seeds itself and yields index chunks;
Catalyst/accelerate then shards those chunks across ranks. So it is directly
comparable to the real training data path.

For each candidate seed source we build one loader and report, per source:
  - the seed value each rank used
  - the data (range(N) indices) each rank received
  - the UNION across ranks
  - which original indices are MISSING (never seen by any rank)

Seed sources (each maps to a spot in the training code):
  - passed  : the attribute passed into __init__   -> SAME on every rank  -> clean partition, nothing missing
  - @prop   : the os.urandom @property             -> DIFFERENT per rank  -> gaps + overlaps, data missing
  - module  : the top-of-file TOP_SEED             -> DIFFERENT per rank  -> gaps + overlaps, data missing

Ranks follow SLURM_GPUS_ON_NODE.
"""

import warnings
warnings.filterwarnings("ignore")  # quiet pynvml FutureWarning / pydantic warnings (before torch import)

import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from catalyst import dl, utils
from mindfultensors.utils import DBBatchSampler   # the SAME sampler train_script_rev.py uses

# Top of the file: re-executed in every spawned worker -> re-drawn per rank.
TOP_SEED = random.randint(0, 9999)
N = 10          # tiny dataset: range(10)
BATCH = 2       # per-chunk size (analog of num_volumes in training)
SEED_SOURCES = ["passed", "@prop", "module"]


def n_gpus_allocated() -> int:
    return int(os.environ.get("SLURM_GPUS_ON_NODE", torch.cuda.device_count()))


class IndexDataset(Dataset):
    """DBBatchSampler yields an ARRAY of indices per item (like MongoDataset).
    __getitem__ receives that whole array and returns (indices, dummy_features)."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        arr = np.asarray(idx).ravel()
        x = torch.tensor(arr, dtype=torch.float32).view(-1, 1)  # dummy features [chunk, 1]
        return arr.tolist(), x


def chunk_collate(results):
    # DataLoader hands us a length-1 list holding one chunk's result (mirrors the
    # training collate's `results = results[0]`).
    indices, x = results[0]
    return torch.tensor(indices, dtype=torch.long), x


class SeedRunner(dl.Runner):
    def __init__(self, passed_seed, epochs=1):
        super().__init__()
        self.passed_seed = passed_seed
        self._epochs = epochs

    @property
    def prop_seed(self) -> int:
        SEED = int.from_bytes(os.urandom(4), "big")
        utils.set_global_seed(SEED)
        return SEED

    def module_seed(self) -> int:
        return TOP_SEED

    def _seed_for(self, source) -> int:
        return {"passed": self.passed_seed, "@prop": self.prop_seed, "module": TOP_SEED}[source]

    @property
    def num_epochs(self) -> int:
        return self._epochs

    def get_engine(self):
        n = n_gpus_allocated()
        if n > 1:
            return dl.DistributedDataParallelEngine(
                process_group_kwargs={"backend": "nccl"},
                num_node_workers=n,
                world_size=n,
            )
        return dl.GPUEngine() if torch.cuda.is_available() else dl.CPUEngine()

    # --- all base get_* return self._xxx (unset here), so override every one ---
    def get_loggers(self):
        return {}

    def _make_loader(self, seed):
        ds = IndexDataset(N)
        return DataLoader(
            ds,
            sampler=DBBatchSampler(ds, batch_size=BATCH, seed=int(seed)),
            collate_fn=chunk_collate,
        )

    def get_loaders(self):
        # Trivial loader to satisfy Catalyst's setup/loop; the real inspection
        # uses its own per-seed loaders in on_experiment_start (below).
        return {"train": self._make_loader(self.passed_seed)}

    def get_model(self):
        return torch.nn.Linear(1, 1)

    def get_criterion(self):
        return None

    def get_optimizer(self, model):
        return torch.optim.SGD(model.parameters(), lr=0.0)

    def get_scheduler(self, optimizer):
        return None

    def get_callbacks(self):
        return {}

    def handle_batch(self, batch):
        _, x = batch
        loss = self.model(x.view(-1, 1).float()).sum()
        if self.is_train_loader:
            self.engine.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.batch_metrics.update({"loss": loss})

    def _collect_rank_data(self):
        """For each seed source: build a DBBatchSampler loader, prepare (shard) it,
        and return the seed used + the indices THIS rank received."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        record = {"rank": rank, "seed": {}, "data": {}}
        for src in SEED_SOURCES:
            seed = self._seed_for(src)   # NB: "@prop" is fresh os.urandom on each read
            record["seed"][src] = int(seed)
            loader = self.engine.prepare(self._make_loader(seed))
            seen = []
            for idx, _ in loader:
                seen.extend(int(i) for i in idx.view(-1).tolist())
            record["data"][src] = sorted(seen)
        return record

    def on_experiment_start(self, runner):
        super().on_experiment_start(runner)  # DDP is initialized after this
        world = dist.get_world_size() if dist.is_initialized() else 1
        record = self._collect_rank_data()

        if dist.is_initialized() and world > 1:
            gathered = [None] * world
            dist.all_gather_object(gathered, record)
        else:
            gathered = [record]

        if (dist.get_rank() if dist.is_initialized() else 0) == 0:
            self._report(gathered, world)

    def _report(self, gathered, world):
        gathered = sorted(gathered, key=lambda g: g["rank"])
        full = set(range(N))
        print(f"\n=== Stage 4: per-seed DDP sampling via DBBatchSampler "
              f"(N={N}, batch={BATCH}, world_size={world}) ===")
        for src in SEED_SOURCES:
            print(f"\nseed source = {src}")
            print(f"  seed per rank : { {g['rank']: g['seed'][src] for g in gathered} }")
            union = set()
            for g in gathered:
                print(f"  rank {g['rank']} data : {g['data'][src]}")
                union |= set(g["data"][src])
            missing = sorted(full - union)
            tag = "OK - full coverage" if not missing else f"{len(missing)} MISSING"
            print(f"  UNION         : {sorted(union)}  ({len(union)}/{N})")
            print(f"  MISSING       : {missing}   [{tag}]")


def main():
    passed_seed = random.randint(0, 9999)  # drawn once here; same on every rank once passed in
    print(
        f"[parent] n_gpus(SLURM_GPUS_ON_NODE)={n_gpus_allocated()}  "
        f"cuda.device_count()={torch.cuda.device_count()}  "
        f"passed_seed={passed_seed}  TOP_SEED={TOP_SEED}  N={N}  BATCH={BATCH}",
        flush=True,
    )
    SeedRunner(passed_seed=passed_seed).run()


if __name__ == "__main__":
    main()
