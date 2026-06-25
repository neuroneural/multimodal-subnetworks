"""
probe_common.py — shared building blocks for the DDP sampler probes.

Holds three things:

1. Reference ("vendored") copies of ``DBBatchSampler`` and
   ``DistributedDBBatchSampler`` that mirror the real code, used so the
   Layer-1 unit probe can run anywhere (e.g. with only numpy installed).
2. ``load_samplers()`` which prefers the *real* classes from
   ``train_script_rev`` / ``mindfultensors`` and only falls back to the
   vendored copies when those imports are unavailable. On the cluster you
   therefore always test the real code.
3. Pure-python/numpy checkers (no torch) that judge a per-rank partition:
   disjointness, coverage, duplication, batch sizes, and a pass/fail verdict.

Nothing here imports torch at module load, so it is safe to import in a
minimal environment.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Vendored reference samplers (mirror of the real implementations)
# ---------------------------------------------------------------------------
# These are byte-for-byte equivalent in logic to:
#   * mindfultensors/utils.py :: DBBatchSampler
#   * train_script_rev.py     :: DistributedDBBatchSampler
# Kept here only so the unit probe is runnable without the full project deps.
# ``load_samplers()`` prefers the real classes when they import cleanly.


class _RefDBBatchSampler:
    """Reference copy of mindfultensors.utils.DBBatchSampler.

    A batch sampler over a random permutation. Used as ``sampler=`` on a
    DataLoader: every yielded item is a *whole array of indices* (one batch).
    """

    def __init__(self, data_source, batch_size=1, seed=None):
        self.batch_size = batch_size
        self.data_source = data_source
        self.data_size = len(data_source)
        self.seed = seed

    @staticmethod
    def _chunks(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i : i + n]

    def __iter__(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        return self._chunks(np.random.permutation(self.data_size), self.batch_size)

    def __len__(self):
        return (self.data_size + self.batch_size - 1) // self.batch_size


def _ref_get_rank_world():
    import os

    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


class _RefDistributedDBBatchSampler(_RefDBBatchSampler):
    """Reference copy of train_script_rev.DistributedDBBatchSampler."""

    def __init__(self, data_source, batch_size=1, seed=None, rank=None, world_size=None):
        super().__init__(data_source, batch_size=batch_size, seed=seed)
        detected_rank, detected_world_size = _ref_get_rank_world()
        self.rank = detected_rank if rank is None else rank
        self.world_size = detected_world_size if world_size is None else world_size
        self.global_batch_size = self.batch_size * self.world_size
        self.num_batches = int(math.ceil(self.data_size / self.global_batch_size))
        self.total_size = self.num_batches * self.global_batch_size

    def __iter__(self):
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
            indices = rng.permutation(self.data_size)
        else:
            indices = np.random.permutation(self.data_size)

        padding_size = self.total_size - len(indices)
        if padding_size > 0 and len(indices) > 0:
            repeats = int(math.ceil(padding_size / len(indices)))
            padding = np.tile(indices, repeats)[:padding_size]
            indices = np.concatenate([indices, padding])

        rank_batches = []
        for start in range(0, self.total_size, self.global_batch_size):
            global_batch = indices[start : start + self.global_batch_size]
            rank_start = self.rank * self.batch_size
            rank_end = rank_start + self.batch_size
            rank_batches.append(global_batch[rank_start:rank_end])
        return iter(rank_batches)

    def __len__(self):
        return self.num_batches


def load_samplers(prefer_real: bool = True):
    """Return ``(DistributedDBBatchSampler, DBBatchSampler, source)``.

    Tries the real project classes first; falls back to the vendored
    reference copies. ``source`` is a human-readable string saying which set
    was used, so probes can print it and you always know what was tested.
    """
    if prefer_real:
        try:
            from train_script_rev import DistributedDBBatchSampler  # type: ignore
            from mindfultensors.utils import DBBatchSampler  # type: ignore

            return DistributedDBBatchSampler, DBBatchSampler, "real:train_script_rev+mindfultensors"
        except Exception as exc:  # noqa: BLE001 - any import issue -> fallback
            fallback_reason = f"{type(exc).__name__}: {exc}"
    else:
        fallback_reason = "prefer_real=False"

    return (
        _RefDistributedDBBatchSampler,
        _RefDBBatchSampler,
        f"vendored (real import skipped: {fallback_reason})",
    )


# ---------------------------------------------------------------------------
# Checkers — all pure python / numpy, judge a per-rank partition
# ---------------------------------------------------------------------------


def _as_int_list(batches: Sequence) -> List[int]:
    """Flatten a list of index-arrays/lists into a flat python int list."""
    flat: List[int] = []
    for b in batches:
        flat.extend(int(x) for x in np.asarray(b).ravel().tolist())
    return flat


def analyze_partition(
    rank_to_batches: Dict[int, Sequence],
    data_size: int,
) -> Dict:
    """Compute coverage / disjointness / duplication stats for one epoch.

    ``rank_to_batches`` maps rank -> list of per-step index arrays that the
    rank actually consumed. Returns a dict of metrics (JSON-friendly).
    """
    ranks = sorted(rank_to_batches)
    rank_flat = {r: _as_int_list(rank_to_batches[r]) for r in ranks}
    rank_sets = {r: set(rank_flat[r]) for r in ranks}

    # per-rank counts
    per_rank_total = {r: len(rank_flat[r]) for r in ranks}
    per_rank_unique = {r: len(rank_sets[r]) for r in ranks}
    per_rank_internal_dups = {r: per_rank_total[r] - per_rank_unique[r] for r in ranks}

    # coverage
    union = set().union(*rank_sets.values()) if rank_sets else set()
    full = set(range(data_size))
    missing = sorted(full - union)
    extra = sorted(union - full)  # indices outside [0, data_size) (shouldn't happen)

    # cross-rank overlap
    pairwise_overlap = {}
    overlapping_indices = set()
    for i, ri in enumerate(ranks):
        for rj in ranks[i + 1 :]:
            inter = rank_sets[ri] & rank_sets[rj]
            pairwise_overlap[f"{ri}&{rj}"] = len(inter)
            overlapping_indices |= inter

    # how many distinct indices appear on >1 rank
    appearance = {}
    for r in ranks:
        for idx in rank_sets[r]:
            appearance[idx] = appearance.get(idx, 0) + 1
    indices_on_multiple_ranks = sum(1 for c in appearance.values() if c > 1)

    counts = list(per_rank_total.values())
    balanced = len(set(counts)) <= 1

    total_slots = sum(counts)
    coverage_count = len(union)
    # duplicated *slots* overall: how many index occurrences are repeats of an
    # already-seen value (internal repeats + cross-rank repeats combined).
    total_duplication = total_slots - coverage_count

    return {
        "data_size": data_size,
        "ranks": ranks,
        "per_rank_total": per_rank_total,
        "per_rank_unique": per_rank_unique,
        "per_rank_internal_dups": per_rank_internal_dups,
        "balanced_counts": balanced,
        "total_slots": total_slots,
        "coverage_count": coverage_count,
        "coverage_fraction": (coverage_count / data_size) if data_size else 0.0,
        "total_duplication": total_duplication,
        "missing_count": len(missing),
        "missing_sample": missing[:20],
        "extra_count": len(extra),
        "extra_sample": extra[:20],
        "pairwise_overlap": pairwise_overlap,
        "total_cross_rank_overlap_pairs": sum(pairwise_overlap.values()),
        "indices_on_multiple_ranks": indices_on_multiple_ranks,
        "is_disjoint": all(v == 0 for v in pairwise_overlap.values()),
    }


def analyze_batches(rank_to_batches: Dict[int, Sequence]) -> Dict:
    """Per-rank batch-size and batch-count summary."""
    out = {}
    for r in sorted(rank_to_batches):
        sizes = [int(np.asarray(b).size) for b in rank_to_batches[r]]
        out[r] = {
            "num_batches": len(sizes),
            "batch_sizes": sizes,
            "distinct_batch_sizes": sorted(set(sizes)),
            "first_batch_size": sizes[0] if sizes else None,
            "last_batch_size": sizes[-1] if sizes else None,
        }
    return out


def build_verdict(
    partition: Dict,
    meta: Dict = None,
    expect_sharded: bool = True,
) -> "OrderedDict[str, Dict]":
    """Turn raw metrics into named pass/fail checks.

    ``meta`` (optional) carries the sampler's own declared bookkeeping:
    ``{"total_size", "global_batch_size", "num_batches"}``. When present, the
    duplication checks become *padding-aware*: a correct DDP sampler is allowed
    to repeat up to ``total_size - data_size`` index slots (the tail padding
    that keeps ranks balanced, exactly as torch's DistributedSampler does), and
    that padding must be smaller than one global batch. Excess duplication or
    any missing coverage is a real failure.

    ``expect_sharded=False`` switches to informational reporting (used for the
    old non-sharding sampler, where duplication is expected and not gated).
    """
    checks: "OrderedDict[str, Dict]" = OrderedDict()

    checks["full_coverage"] = {
        "pass": partition["missing_count"] == 0 and partition["extra_count"] == 0,
        "detail": (
            f"covered {partition['coverage_count']}/{partition['data_size']} "
            f"({partition['coverage_fraction']*100:.1f}%), "
            f"missing={partition['missing_count']}, out_of_range={partition['extra_count']}"
        ),
    }
    checks["balanced_counts"] = {
        "pass": partition["balanced_counts"],
        "detail": f"per-rank totals={partition['per_rank_total']}",
    }

    if not expect_sharded:
        checks["duplication_observed"] = {
            "pass": True,  # informational only for the old sampler
            "detail": (
                f"indices_on_multiple_ranks={partition['indices_on_multiple_ranks']}, "
                f"total_duplication={partition['total_duplication']}, "
                f"pairwise_overlap={partition['pairwise_overlap']}"
            ),
        }
        return checks

    if meta:
        padding = meta["total_size"] - partition["data_size"]
        gbs = meta["global_batch_size"]
        checks["matches_declared_total"] = {
            "pass": partition["total_slots"] == meta["total_size"],
            "detail": (
                f"consumed slots={partition['total_slots']} vs sampler.total_size={meta['total_size']} "
                f"(num_batches={meta['num_batches']}, global_batch_size={gbs})"
            ),
        }
        checks["padding_bounded"] = {
            "pass": 0 <= padding < gbs,
            "detail": f"tail padding={padding} (must be in [0, global_batch_size={gbs}))",
        }
        # With full coverage + matching total, duplication is exactly padding.
        checks["duplication_only_from_padding"] = {
            "pass": (partition["missing_count"] == 0)
            and (partition["total_duplication"] == padding),
            "detail": (
                f"total_duplication={partition['total_duplication']} "
                f"(expected == padding={padding}); "
                f"cross-rank overlap (info)={partition['pairwise_overlap']}"
            ),
        }
    else:
        # No sampler metadata available (e.g. probing arbitrary batches):
        # fall back to strict disjointness.
        checks["disjoint_across_ranks"] = {
            "pass": partition["is_disjoint"],
            "detail": (
                f"pairwise_overlap={partition['pairwise_overlap']}, "
                f"indices_on_multiple_ranks={partition['indices_on_multiple_ranks']}"
            ),
        }
    return checks


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def print_report(title: str, partition: Dict, batches: Dict, verdict: Dict) -> bool:
    """Print a human-readable block. Returns True if every check passed."""
    line = "=" * 78
    print(line)
    print(title)
    print(line)
    print(
        f"data_size={partition['data_size']}  ranks={partition['ranks']}  "
        f"coverage={partition['coverage_count']}/{partition['data_size']} "
        f"({partition['coverage_fraction']*100:.1f}%)"
    )
    for r in partition["ranks"]:
        b = batches[r]
        print(
            f"  rank {r}: {b['num_batches']} batches, "
            f"batch_sizes(distinct)={b['distinct_batch_sizes']}, "
            f"total={partition['per_rank_total'][r]} "
            f"unique={partition['per_rank_unique'][r]} "
            f"internal_dups={partition['per_rank_internal_dups'][r]}"
        )
    print(
        f"  total slots={partition['total_slots']}  "
        f"total duplication={partition['total_duplication']}  "
        f"cross-rank pairwise overlap={partition['pairwise_overlap']}"
    )
    if partition["missing_count"]:
        print(f"  MISSING {partition['missing_count']} indices, e.g. {partition['missing_sample']}")
    print("  --- checks ---")
    all_pass = True
    for name, res in verdict.items():
        flag = "PASS" if res["pass"] else "FAIL"
        if not res["pass"]:
            all_pass = False
        print(f"  [{flag}] {name}: {res['detail']}")
    print(f"  => {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print()
    return all_pass
