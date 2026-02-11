import os
import torch
import random

def setup_distributed_port(seed=42, low=2000, high=2200):
    """
    Sets MASTER_PORT in os.environ based on SLURM_JOB_ID or random seed.
    Range default: 20000-29999 (Safe user range).
    """
    if "MASTER_PORT" in os.environ:
        print(f"[Init] MASTER_PORT used by catalyst DDP already set to {os.environ['MASTER_PORT']}")
        return

    if "SLURM_JOB_ID" in os.environ:
        job_id = int(os.environ["SLURM_JOB_ID"])
        # Map the huge Job ID to our specific range
        # Formula: Low_Bound + (Job_ID % Range_Size)
        range_size = high - low
        port = low + (job_id % range_size)
        
        print(f"[Init] MASTER_PORT used by catalyst DDP set to {port} (Derived from SLURM_JOB_ID {job_id})")
    else:
        # Fallback: Deterministic random based on global SEED
        # We use a local Random instance to avoid messing up the global random state
        rng = random.Random(seed)
        port = rng.randint(low, high)
        print(f"[Init] MASTER_PORT used by catalyst DDP set to {port} (Random selection based on SEED {seed})")
    
    os.environ["MASTER_PORT"] = str(port)
    
def qnormalize(img, qmin=0.02, qmax=0.98):
    """Unit interval preprocessing with clipping"""
    qlow = torch.quantile(img, qmin)
    qhigh = torch.quantile(img, qmax)
    img = (img - qlow) / (qhigh - qlow)
    img = torch.clamp(img, 0, 1)  # Clip the values to be between 0 and 1
    return img

class ProductScheduler:
    def __init__(self, scheduler1, scheduler2):
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.initial_lr = scheduler1.optimizer.param_groups[0]["lr"]

    def step(self):
        lr1 = self.scheduler1.get_last_lr()[0]
        lr2 = self.scheduler2.get_last_lr()[0]
        combined_lr = lr1 * lr2
        self.scheduler1.step()
        self.scheduler2.step()
        self.scheduler1.optimizer.param_groups[0]["lr"] = combined_lr
        return combined_lr


def assert_equal_length(*args):
    """Enhanced version that shows which parameters have mismatched lengths"""
    if not all(len(arg) == len(args[0]) for arg in args):
        print("\nParameter length mismatch detected:")
        print("{:<15} {:<10}".format("Parameter", "Length"))
        print("-" * 25)
        param_names = [
            "cubesizes", "numcubes", "numvolumes", "weights",
            "databases", "collections", "epochs", 
            "prefetches", "attenuates"
        ]
        for name, arg in zip(param_names, args):
            print("{:<15} {:<10}".format(name, len(arg)))
        print()
        
        # Show first few elements of each list for comparison
        print("First few elements of each list:")
        for name, arg in zip(param_names, args):
            print(f"{name}: {arg[:3]}...")
        print()
        
        raise AssertionError("Not all parameter lists have the same length!")