"""
Stage 4 - which seeds are available at get_loaders, under real Catalyst DDP.

get_loaders() is exactly where train_script_rev.py builds its sampler. This runner
does nothing but report, at that point, the three candidate seeds on each rank:
  passed  : attribute passed into __init__   -> SAME on every rank
  @prop   : os.urandom @property             -> DIFFERENT per rank
  module  : top-of-file TOP_SEED             -> DIFFERENT per rank

No data / no batch logging - just the seeds. (Stage 5 shows what data each rank
gets from these seeds.) Ranks follow SLURM_GPUS_ON_NODE.
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


def n_gpus_allocated() -> int:
    return int(os.environ.get("SLURM_GPUS_ON_NODE", torch.cuda.device_count()))


class TinyDataset(Dataset):
    """Minimal loader payload so Catalyst has something to iterate."""

    def __init__(self, n=4):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return float(i), i


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

    # base get_* all return self._xxx (unset here) -> override each
    def get_loggers(self):
        return {}

    def get_loaders(self):
        # This is the sampler-definition point in train_script_rev.py.
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        print(
            f"[rank {rank}/{world}] seeds@get_loaders: "
            f"passed={self.passed_seed:<6} @prop={self.prop_seed:<11} module={TOP_SEED}",
            flush=True,
        )
        return {"train": DataLoader(TinyDataset(), batch_size=1)}

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
        x, _ = batch
        loss = self.model(x.view(-1, 1).float()).sum()
        if self.is_train_loader:
            self.engine.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.batch_metrics.update({"loss": loss})


def main():
    passed_seed = random.randint(0, 9999)  # drawn once; same on every rank once passed in
    print(
        f"[parent] n_gpus(SLURM_GPUS_ON_NODE)={n_gpus_allocated()}  "
        f"cuda.device_count()={torch.cuda.device_count()}  "
        f"passed_seed={passed_seed}  TOP_SEED={TOP_SEED}",
        flush=True,
    )
    SeedRunner(passed_seed=passed_seed).run()


if __name__ == "__main__":
    main()
