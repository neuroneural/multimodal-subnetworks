"""
Stage 4 - which seeds actually arrive where the sampler is built, and what data
each rank ends up with, under REAL Catalyst DDP.

In train_script_rev.py the sampler is created inside get_loaders() with some
seed=... . This runner snapshots the three candidate seeds AT THAT POINT, per
rank, so you can see which one is safe to feed the sampler:
  - passed_seed : the attribute passed into __init__  -> SAME on every rank
  - @prop_seed  : the os.urandom @property            -> DIFFERENT each access & rank
  - module_seed : the top-of-file TOP_SEED            -> DIFFERENT per rank (re-import)

It then builds a tiny loader over range(10) (an IndexDataset that returns its own
index, like the old probe) seeded with passed_seed, lets Catalyst/accelerate shard
it, and each rank prints the data items it received. With a cross-rank-consistent
seed the shards are disjoint and together cover range(10).

Ranks follow the GPUs allocated to the job (SLURM_GPUS_ON_NODE).
Run e.g.:  sbatch --gres=gpu:A100:2 03_catalyst.sh   (point it at 04_sampler.py)
"""

import warnings
warnings.filterwarnings("ignore")  # quiet pynvml FutureWarning / pydantic warnings (before torch import)

import os
import random

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from catalyst import dl, utils

# Top of the file: re-executed in every spawned worker -> re-drawn per rank.
TOP_SEED = random.randint(0, 9999)
N = 10  # tiny dataset: range(10)


def n_gpus_allocated() -> int:
    return int(os.environ.get("SLURM_GPUS_ON_NODE", torch.cuda.device_count()))


class IndexDataset(Dataset):
    """Returns its own index, so we can see exactly which items a rank received."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


class SeedRunner(dl.Runner):
    def __init__(self, passed_seed, epochs=1):
        super().__init__()
        self.passed_seed = passed_seed
        self._epochs = epochs
        self._seen = []                 # indices this rank actually processed
        self._seeds_at_get_loaders = None

    @property
    def prop_seed(self) -> int:
        # Same body as CustomRunner.seed: recomputed on EVERY access.
        SEED = int.from_bytes(os.urandom(4), "big")
        utils.set_global_seed(SEED)
        return SEED

    def module_seed(self) -> int:
        return TOP_SEED

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

    def get_loaders(self):
        # This is exactly where train_script_rev.py defines its sampler.
        # Snapshot the three candidate seeds AS SEEN HERE, on this rank:
        self._seeds_at_get_loaders = {
            "passed": self.passed_seed,   # attribute -> same on every rank
            "prop": self.prop_seed,       # @property -> fresh os.urandom
            "module": TOP_SEED,           # top-of-file global -> per-rank
        }
        # Feed the sampler the cross-rank-consistent seed. accelerate shards the
        # (identically-shuffled) order across ranks -> disjoint per-rank slices.
        # (Swap manual_seed(self.passed_seed) for self.prop_seed / TOP_SEED to
        #  watch the partition break: gaps + overlaps.)
        g = torch.Generator().manual_seed(self.passed_seed)
        loader = DataLoader(IndexDataset(N), batch_size=1, shuffle=True, generator=g)
        return {"train": loader}

    def handle_batch(self, batch):
        idxs = batch.view(-1).tolist()
        self._seen.extend(int(i) for i in idxs)
        # trivial forward/backward so DDP is happy (all params used, lr=0)
        loss = self.model(batch.view(-1, 1).float()).sum()
        if self.is_train_loader:
            self.engine.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.batch_metrics.update({"loss": loss})

    def on_experiment_end(self, runner):
        if dist.is_available() and dist.is_initialized():
            rank, world = dist.get_rank(), dist.get_world_size()
        else:
            rank, world = 0, 1
        s = self._seeds_at_get_loaders
        print(
            f"[rank {rank}/{world}] seeds@get_loaders: "
            f"passed={s['passed']:<6} @prop={s['prop']:<11} module={s['module']:<6} "
            f"| sampler seed=passed | data this rank={sorted(self._seen)}",
            flush=True,
        )
        super().on_experiment_end(runner)  # engine.cleanup() -> destroy_process_group


def main():
    passed_seed = random.randint(0, 9999)  # drawn once here, same on every rank once passed in
    print(
        f"[parent] n_gpus(SLURM_GPUS_ON_NODE)={n_gpus_allocated()}  "
        f"cuda.device_count()={torch.cuda.device_count()}  "
        f"passed_seed={passed_seed}  TOP_SEED={TOP_SEED}  N={N}",
        flush=True,
    )
    SeedRunner(passed_seed=passed_seed).run()


if __name__ == "__main__":
    main()
