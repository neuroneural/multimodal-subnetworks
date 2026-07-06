"""
Stage 5 - what data each rank actually receives, per candidate seed, under real
Catalyst DDP - using the SAME sampler and the SAME construction point as training.

The three loaders are built IN get_loaders() (exactly where train_script_rev.py
defines its sampler), each with mindfultensors.utils.DBBatchSampler seeded by a
different candidate seed. Catalyst prepares (accelerate-shards) them and runs each
as a train loader. handle_batch records which range(N) indices this rank saw, and
at the end rank 0 reports, per seed source:
  - the seed value each rank used
  - the data each rank received
  - the UNION across ranks
  - which original indices are MISSING (never seen by any rank)

Seed sources:
  passed  : attribute passed into __init__   -> SAME on every rank -> full coverage
  @prop   : os.urandom @property             -> DIFFERENT per rank -> gaps + overlaps
  module  : top-of-file TOP_SEED             -> DIFFERENT per rank -> gaps + overlaps

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
from mindfultensors.utils import DBBatchSampler   # SAME sampler train_script_rev.py uses

# Top of the file: re-executed in every spawned worker -> re-drawn per rank.
TOP_SEED = random.randint(0, 9999)
N = 10        # tiny dataset: range(10)
BATCH = 2     # per-chunk size (analog of num_volumes)

# display label -> Catalyst loader key (all start with "train" so they're treated
# as train loaders and iterated the same way)
SOURCES = {"passed": "train_passed", "@prop": "train_prop", "module": "train_module"}
KEY2DISP = {v: k for k, v in SOURCES.items()}


def n_gpus_allocated() -> int:
    return int(os.environ.get("SLURM_GPUS_ON_NODE", torch.cuda.device_count()))


class IndexDataset(Dataset):
    """DBBatchSampler yields an ARRAY of indices per item (like MongoDataset)."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        arr = np.asarray(idx).ravel()
        x = torch.tensor(arr, dtype=torch.float32).view(-1, 1)  # dummy features [chunk, 1]
        return arr.tolist(), x


def chunk_collate(results):
    # DataLoader hands a length-1 list holding one chunk's result (mirrors the
    # training collate's `results = results[0]`).
    indices, x = results[0]
    return torch.tensor(indices, dtype=torch.long), x


class SeedRunner(dl.Runner):
    def __init__(self, passed_seed, epochs=1):
        super().__init__()
        self.passed_seed = passed_seed
        self._epochs = epochs
        self._seen = {}     # display -> indices this rank saw
        self._seeds = {}    # display -> seed used

    @property
    def prop_seed(self) -> int:
        SEED = int.from_bytes(os.urandom(4), "big")
        utils.set_global_seed(SEED)
        return SEED

    def module_seed(self) -> int:
        return TOP_SEED

    def _seed_for(self, disp) -> int:
        return {"passed": self.passed_seed, "@prop": self.prop_seed, "module": TOP_SEED}[disp]

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
        # Build one loader per candidate seed, HERE (the sampler-definition point),
        # exactly like train_script_rev.py builds its train sampler.
        loaders = {}
        for disp, key in SOURCES.items():
            seed = self._seed_for(disp)   # NB: "@prop" is fresh os.urandom on each read
            self._seeds[disp] = int(seed)
            loaders[key] = self._make_loader(seed)
        return loaders

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
        idx, x = batch
        disp = KEY2DISP.get(self.loader_key, self.loader_key)
        self._seen.setdefault(disp, []).extend(int(i) for i in idx.view(-1).tolist())
        loss = self.model(x.view(-1, 1).float()).sum()
        if self.is_train_loader:
            self.engine.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.batch_metrics.update({"loss": loss})

    def on_experiment_end(self, runner):
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        record = {"rank": rank, "seed": self._seeds, "data": {k: sorted(v) for k, v in self._seen.items()}}
        if dist.is_initialized() and world > 1:
            gathered = [None] * world
            dist.all_gather_object(gathered, record)
        else:
            gathered = [record]
        if rank == 0:
            self._report(gathered, world)
        super().on_experiment_end(runner)  # engine.cleanup() -> destroy_process_group

    def _report(self, gathered, world):
        gathered = sorted(gathered, key=lambda g: g["rank"])
        full = set(range(N))
        print(f"\n=== Stage 5: per-seed DDP data via DBBatchSampler "
              f"(N={N}, batch={BATCH}, world_size={world}) ===")
        for disp in SOURCES:
            print(f"\nseed source = {disp}")
            print(f"  seed per rank : { {g['rank']: g['seed'].get(disp) for g in gathered} }")
            union = set()
            for g in gathered:
                d = g["data"].get(disp, [])
                print(f"  rank {g['rank']} data : {d}")
                union |= set(d)
            missing = sorted(full - union)
            tag = "OK - full coverage" if not missing else f"{len(missing)} MISSING"
            print(f"  UNION         : {sorted(union)}  ({len(union)}/{N})")
            print(f"  MISSING       : {missing}   [{tag}]")


def main():
    passed_seed = random.randint(0, 9999)  # drawn once; same on every rank once passed in
    print(
        f"[parent] n_gpus(SLURM_GPUS_ON_NODE)={n_gpus_allocated()}  "
        f"cuda.device_count()={torch.cuda.device_count()}  "
        f"passed_seed={passed_seed}  TOP_SEED={TOP_SEED}  N={N}  BATCH={BATCH}",
        flush=True,
    )
    SeedRunner(passed_seed=passed_seed).run()


if __name__ == "__main__":
    main()
