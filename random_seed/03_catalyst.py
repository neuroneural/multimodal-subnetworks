"""
Stage 3 - the same three seed behaviors as 02_class.py, but under REAL Catalyst DDP.

LLM talk, may not be true:
Catalyst spawns one worker process per GPU with torch.multiprocessing.spawn
("spawn" start method), then ships the runner object (pickled) to each worker.
This is the real version of what 01/02 showed with plain multiprocessing.

Three seed sources on the runner (each maps to a spot in train_script_rev.py):
  1. passed_seed - passed into __init__, stored as an attribute
                   (the `self.sampler_seed = sampler_seed` fix). -> SAME on every rank.
  2. @prop_seed  - an @property returning os.urandom on each access
                   (mirrors CustomRunner.seed's body).           -> DIFFERENT each access & rank.
  3. module_seed - a method reading the module-level TOP_SEED
                   (old sampler reading the top-of-file `SEED`).  -> DIFFERENT per rank (re-import).
                   THIS IS WHAT WAS PASSED TO SAMPLERS IN THE CATALYST EXAMPLE CURRICULUM TRAINING SCRIPT.

The number of ranks follows the GPUs allocated to the job (SLURM_GPUS_ON_NODE).

Each rank prints one line; compare the columns across ranks.
"""

import os
import random

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset

from catalyst import dl, utils

# Top of the file: re-executed in every spawned worker -> re-drawn per rank.
TOP_SEED = random.randint(0, 9999)


def n_gpus_allocated() -> int:
    return int(os.environ.get("SLURM_GPUS_ON_NODE", torch.cuda.device_count()))


class SeedRunner(dl.Runner):
    def __init__(self, passed_seed, logdir="./_seed_out", epochs=1):
        super().__init__()
        self.passed_seed = passed_seed  # (1) passed in -> travels with the pickle -> same on all ranks
        self._logdir = logdir
        self._epochs = epochs

    @property
    def prop_seed(self) -> int:
        # (2) Same body as CustomRunner.seed. Recomputed on EVERY access.
        # NB: named prop_seed here, not `seed`, so we don't hijack Catalyst's own
        # internal seeding (in the real code it IS named `seed`, so Catalyst reads it).
        SEED = int.from_bytes(os.urandom(4), "big")
        utils.set_global_seed(SEED)
        return SEED

    def module_seed(self) -> int:
        # (3) Reads the top-of-file global.
        return TOP_SEED

    @property
    def num_epochs(self) -> int:
        return self._epochs

    def get_engine(self):
        n = n_gpus_allocated()
        if n > 1:
            # Pin spawn/world size to the allocation (see world-size discussion).
            return dl.DistributedDataParallelEngine(
                process_group_kwargs={"backend": "nccl"},
                num_node_workers=n,
                world_size=n,
            )
        return dl.GPUEngine() if torch.cuda.is_available() else dl.CPUEngine()

    def get_loggers(self):
        # Base Runner.get_loggers() reads self._loggers, which is never set -> override.
        return {}

    def get_loaders(self):
        x = torch.arange(8, dtype=torch.float32).view(8, 1)
        return {"train": DataLoader(TensorDataset(x, x), batch_size=2)}

    def get_model(self):
        return torch.nn.Linear(1, 1)

    def get_criterion(self):
        return None

    def get_optimizer(self, model):
        return torch.optim.SGD(model.parameters(), lr=0.0)  # lr=0: nothing actually trains

    def on_experiment_start(self, runner):
        # super() runs engine.setup() -> dist.init_process_group(), so DDP is live after it.
        super().on_experiment_start(runner)
        if dist.is_available() and dist.is_initialized():
            rank, world = dist.get_rank(), dist.get_world_size()
        else:
            rank, world = 0, 1
        print(
            f"[rank {rank}/{world}] "
            f"passed_seed={self.passed_seed:<6} "
            f"@prop_seed#1={self.prop_seed:<11} @prop_seed#2={self.prop_seed:<11} "
            f"module_seed(TOP_SEED)={self.module_seed()}",
            flush=True,
        )

    def handle_batch(self, batch):
        x, y = batch
        loss = ((self.model(x) - y) ** 2).mean()
        if self.is_train_loader:
            self.engine.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.batch_metrics.update({"loss": loss})


def main():
    # This is a good seed that once passed to the runner, will be the same on every rank.
    passed_seed = random.randint(0, 9999)
    print(
        f"[parent] n_gpus(SLURM_GPUS_ON_NODE)={n_gpus_allocated()}  "
        f"cuda.device_count()={torch.cuda.device_count()}  "
        f"passed_seed={passed_seed}  TOP_SEED={TOP_SEED}",
        flush=True,
    )
    SeedRunner(passed_seed=passed_seed).run()


if __name__ == "__main__":
    main()
