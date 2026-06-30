import warnings
warnings.filterwarnings("ignore")

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
import csv
import random
import shutil
from packaging import version
import yaml

from catalyst import dl, metrics, utils
from catalyst.data import BatchPrefetchLoaderWrapper
from catalyst.utils import distributed

import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from resnet import ResNet3D

from mindfultensors.mongoloader import MongoClient
from mindfultensors.utils import DBBatchSampler

def safe_normalize(img):
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        return torch.zeros_like(img)
    return (img - mn) / (mx - mn)


from src.db_client import ClientCreator
from src.customMongoDataset import CustomMongoDataset, MultimodalMongoDataset, multimodal_collate, make_serial
from src.masked_model import MultiMaskSNIPWrapper
from src.utils import setup_distributed_port

SEED = random.randint(0, 9999)
utils.set_global_seed(SEED)
setup_distributed_port(seed=SEED)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
# os.environ["NCCL_SOCKET_IFNAME"] = "ib0"
# os.environ["NCCL_P2P_LEVEL"] = "NVL"

torch_version = torch.__version__
if version.parse(torch_version) >= version.parse("2.3"):
    scaler = torch.amp.GradScaler()
else:
    scaler = torch.cuda.amp.GradScaler()


# CustomRunner – PyTorch for-loop decomposition
# https://github.com/catalyst-team/catalyst#minimal-examples
class CustomRunner(dl.Runner):
    def __init__(
        self,
        logdir: str,
        wandb_project: str,
        wandb_experiment: str,
        model_path: str,
        n_channels: int,
        n_classes: int,
        n_epochs: int,
        optimize_inline: bool,
        validation_percent: float,
        onecycle_lr: float,
        rmsprop_lr: float,
        num_subcubes: int,
        num_volumes: int,
        client_creator,
        off_brain_weight: float,
        indexid: str,
        # modelconfig: str,
        db_host: str,
        db_name: str,
        db_collection: str,
        wandb_team: str,
        db_fields: tuple,
        meta_fields: tuple,
        groupnorm=False,
        prefetches=8,
        num_workers=6,
        prefetch_factor=2,
        train_prefetches=None,
        train_num_workers=None,
        train_prefetch_factor=None,
        train_persistent_workers=True,
        eval_prefetches=None,
        eval_num_workers=None,
        eval_prefetch_factor=None,
        eval_persistent_workers=True,
        volume_shape=[256] * 3,
        subvolume_shape=[256] * 3,
        lowprecision=False,
        lossweight=[1, 0],
        maxshape=300,
        hparams=None,
        sampler_seed=None,
    ):
        super().__init__()
        self._logdir = logdir
        self.wandb_project = wandb_project
        self.wandb_experiment = wandb_experiment
        self.model_path = model_path
        self.n_channels = n_channels
        self.n_classes = n_classes
        # self.config_file = modelconfig
        self.optimize_inline = optimize_inline
        self.onecycle_lr = onecycle_lr
        self.validation_percent = validation_percent
        self.rmsprop_lr = rmsprop_lr
        self.prefetches = prefetches
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.train_prefetches = train_prefetches if train_prefetches is not None else prefetches
        self.train_num_workers = train_num_workers if train_num_workers is not None else num_workers
        self.train_prefetch_factor = train_prefetch_factor if train_prefetch_factor is not None else prefetch_factor
        self.train_persistent_workers = train_persistent_workers
        self.eval_prefetches = eval_prefetches if eval_prefetches is not None else prefetches
        self.eval_num_workers = eval_num_workers if eval_num_workers is not None else num_workers
        self.eval_prefetch_factor = eval_prefetch_factor if eval_prefetch_factor is not None else prefetch_factor
        self.eval_persistent_workers = eval_persistent_workers

        self.db_host = db_host
        self.db_name = db_name
        self.db_collection = db_collection
        self.db_fields = db_fields
        self.meta_fields = meta_fields

        self.shape = subvolume_shape[0]
        self.num_subcubes = num_subcubes
        self.num_volumes = num_volumes
        self.n_epochs = n_epochs
        self.off_brain_weight = off_brain_weight
        self.client_creator = client_creator
        self.funcs = None
        self.collate = None
        self.bit16 = lowprecision
        self.index_id = indexid
        self.groupnorm = groupnorm
        self.loss_weight = lossweight
        self.wandb_team = wandb_team
        self.maxshape = maxshape
        self._hparams = hparams

        # Seed for the data samplers. Must be IDENTICAL across DDP ranks: the
        # module-level SEED is re-drawn per worker under Catalyst's mp.spawn, so
        # using it directly makes each rank shuffle differently and breaks the
        # shard partition. This value is drawn once in main() (the parent
        # __main__ process) and passed in, so it ships to every rank via pickling.
        self.sampler_seed = sampler_seed if sampler_seed is not None else SEED

        self.masked = self._hparams["model"].get("masked", False)

    @property
    def _metric_keys(self):
        return ["loss", "accuracy", "learning rate"]

    def get_engine(self):
        # Use SLURM-allocated GPU count, not total visible GPUs on the node.
        # torch.cuda.device_count() is unreliable on this cluster: CUDA_VISIBLE_DEVICES
        # is unset, so it reports ALL physical GPUs on the node (e.g. 5) regardless of
        # the allocation. SLURM_GPUS_ON_NODE tracks the real allocation. Catalyst's
        # engine otherwise spawns one process per *physical* GPU (via device_count),
        # so we pin the spawn/world size with num_node_workers/world_size.
        n_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", torch.cuda.device_count()))
        if n_gpus > 1:
            return dl.DistributedDataParallelEngine(
                # mixed_precision="fp16",
                # ddp_kwargs={"backend": "nccl"},
                process_group_kwargs={"backend": "nccl"},
                num_node_workers=n_gpus,
                world_size=n_gpus,
            )
        else:
            return dl.GPUEngine()

    def get_loggers(self):
        return {
            "console": dl.ConsoleLogger(),
            "csv": dl.CSVLogger(logdir=self._logdir),
            "wandb": dl.WandbLogger(
                project=self.wandb_project,
                name=self.wandb_experiment,
                entity=self.wandb_team,
            ),
        }

    @property
    def stages(self):
        return ["train"]

    @property
    def num_epochs(self) -> int:
        return self.n_epochs

    @property
    def seed(self) -> int:
        """Experiment's seed for reproducibility."""
        random_data = os.urandom(4)
        SEED = int.from_bytes(random_data, byteorder="big")
        utils.set_global_seed(SEED)
        return SEED

    def get_stage_len(self) -> int:
        return self.n_epochs

    def get_loaders(self):
        #MM
        self.multimodal = True if (len(self.db_fields) > 1 or self.masked) else False

        self.funcs = {
            "createclient": self.client_creator.create_client,
            "createVclient": self.client_creator.create_client,
            "mycollate": self.client_creator.mycollate,
            "mycollate_full": self.client_creator.mycollate_full,
            "mytransform": self.client_creator.mytransform,
        }
        
        self.collate = (
            multimodal_collate if self.multimodal else #MM
            self.funcs["mycollate_full"]
            if self.shape == 256
            else self.funcs["mycollate"]
        )

        print(
            "[LoaderConfig] "
            f"train_workers={self.train_num_workers}, train_prefetch_factor={self.train_prefetch_factor}, "
            f"train_prefetches={self.train_prefetches}, train_persistent={self.train_persistent_workers}; "
            f"eval_workers={self.eval_num_workers}, eval_prefetch_factor={self.eval_prefetch_factor}, "
            f"eval_prefetches={self.eval_prefetches}, eval_persistent={self.eval_persistent_workers}; "
            f"cudnn_benchmark={torch.backends.cudnn.benchmark}"
        )

        # get all IDs with the required modalities, pull their labels for cross-validation splits

        client = MongoClient("mongodb://" + self.db_host + ":27017")
        db = client[self.db_name]
        posts_bin = db[self.db_collection + ".bin"]
        posts_meta = db[self.db_collection + ".meta"]

        # get ids, pull labels
        all_ids = posts_meta.distinct( # pull all unique IDs (subjects) with at least one modality in db_fields
            "id",
            {'modalities': {"$in": self.db_fields}}
        )
        all_ids = sorted(all_ids)
        # print(all_ids)

        # Fetch all split labels in one query, preserving all_ids order below.
        label_field = self.meta_fields[0]
        meta_docs = {
            doc["id"]: doc
            for doc in posts_meta.find(
                {"id": {"$in": all_ids}},
                {"id": 1, label_field: 1, "_id": 0},
            )
        }
        missing_label_ids = [id for id in all_ids if id not in meta_docs]
        if missing_label_ids:
            raise ValueError(f"Missing labels for ids: {missing_label_ids[:10]}")
        labels = np.array([meta_docs[id][label_field] for id in all_ids])
    
        # Create CV split
        cv_folds = StratifiedKFold(n_splits=self._hparams["experiment"]["cv_folds"], shuffle=True, random_state=self._hparams["experiment"].get("cv_seed", 42))
        train_idx, test_idx = list(cv_folds.split(all_ids, labels))[self._hparams["fold_idx"]]
        # split train into train and validation
        train_idx, valid_idx = train_test_split(train_idx, test_size=self.validation_percent, stratify=labels[train_idx], random_state=self._hparams["experiment"].get("cv_seed", 42))

        all_ids = np.array(all_ids)
        train_ids = all_ids[train_idx].tolist() # mongo expects default python list, not numpy array
        valid_ids = all_ids[valid_idx].tolist()
        test_ids = all_ids[test_idx].tolist()
        # get data for masks calculation
        if self.masked:
            print("Preparing SNIP mask data...")
            snip_batch_size = self._hparams["model"].get("snip_batch_size", 20)
            # Use the cross-rank-consistent sampler seed (NOT the per-process module
            # SEED): runs in every DDP worker, so all ranks must select the SAME snip
            # batch and build the SAME mask instead of relying on DDP's rank-0 broadcast.
            rng = random.Random(self.sampler_seed)
            snip_batch_ids = rng.sample(train_ids, k=min(snip_batch_size, len(train_ids)))

            snip_data, snip_modalities, snip_labels = self.get_snip_data(posts_bin, posts_meta, snip_batch_ids)
            self.snip_data = (snip_data, snip_modalities, snip_labels)
            print(f"SNIP mask data prepared. Data shape: {snip_data.shape}, Modalities: {snip_modalities.shape}, Labels shape: {snip_labels.shape}")


        # save splits into logdir
        with open(os.path.join(self._logdir, 'train_ids.txt'), 'w') as f:
            for id in train_ids:
                f.write(f"{id}\n")
        with open(os.path.join(self._logdir, 'valid_ids.txt'), 'w') as f:
            for id in valid_ids:
                f.write(f"{id}\n")
        with open(os.path.join(self._logdir, 'test_ids.txt'), 'w') as f:
            for id in test_ids:
                f.write(f"{id}\n")


        usedDataset = MultimodalMongoDataset if self.multimodal else CustomMongoDataset #MM
        # Create dataloaders
        train_dataset = usedDataset(
            train_ids, 
            self.funcs["mytransform"],
            None,
            self.db_fields,
            self.meta_fields,
            normalize=safe_normalize,
            id=self.index_id,
        )
        
        # Train uses a plain DBBatchSampler with a cross-rank-consistent seed
        # (self.sampler_seed, identical on every rank). Catalyst/accelerate's
        # engine.prepare() ALREADY shards the DataLoader across ranks, so a plain
        # sampler with the same seed yields a clean disjoint partition with full
        # coverage (verified by the DDP probe: 100% coverage, disjoint, balanced).
        # DistributedDBBatchSampler is intentionally NOT used here: it pre-shards,
        # and accelerate then shards again -> double-sharding that drops
        # ~(1 - 1/world_size) of the data each epoch.
        # cv_seed (config) is used only for the CV splits above.
        train_sampler = DBBatchSampler(train_dataset, batch_size=self.num_volumes, seed=self.sampler_seed)

        train_loader_kwargs = {
            "sampler": train_sampler,
            "collate_fn": self.collate,
            "pin_memory": True,
            "worker_init_fn": self.funcs["createclient"],
            "num_workers": self.train_num_workers,
        }
        if self.train_num_workers > 0:
            train_loader_kwargs["persistent_workers"] = self.train_persistent_workers
            train_loader_kwargs["prefetch_factor"] = self.train_prefetch_factor
        train_dataloader = BatchPrefetchLoaderWrapper(
            DataLoader(train_dataset, **train_loader_kwargs),
            num_prefetches=self.train_prefetches,
        )

        valid_dataset = usedDataset(
            valid_ids,#take first validation_percent percent from list
            self.funcs["mytransform"],
            None,
            self.db_fields,
            self.meta_fields,
            normalize=safe_normalize,
            id=self.index_id,
        )

        valid_sampler = DBBatchSampler(valid_dataset, batch_size=self.num_volumes, seed=self.sampler_seed)
        valid_loader_kwargs = {
            "sampler": valid_sampler,
            "collate_fn": self.collate,
            "pin_memory": True,
            "worker_init_fn": self.funcs["createVclient"],
            "num_workers": self.eval_num_workers,
        }
        if self.eval_num_workers > 0:
            valid_loader_kwargs["persistent_workers"] = self.eval_persistent_workers
            valid_loader_kwargs["prefetch_factor"] = self.eval_prefetch_factor
        valid_dataloader = BatchPrefetchLoaderWrapper(
            DataLoader(valid_dataset, **valid_loader_kwargs),
            num_prefetches=self.eval_prefetches,
        )

        test_dataset = usedDataset(
            test_ids,#take first validation_percent percent from list
            self.funcs["mytransform"],
            None,
            self.db_fields,
            self.meta_fields,
            normalize=safe_normalize,
            id=self.index_id,
        )
        test_sampler = DBBatchSampler(test_dataset, batch_size=self.num_volumes, seed=self.sampler_seed)
        test_loader_kwargs = {
            "sampler": test_sampler,
            "collate_fn": self.collate,
            "pin_memory": True,
            "worker_init_fn": self.funcs["createVclient"],
            "num_workers": self.eval_num_workers,
        }
        if self.eval_num_workers > 0:
            test_loader_kwargs["persistent_workers"] = self.eval_persistent_workers
            test_loader_kwargs["prefetch_factor"] = self.eval_prefetch_factor
        test_dataloader = BatchPrefetchLoaderWrapper(
            DataLoader(test_dataset, **test_loader_kwargs),
            num_prefetches=self.eval_prefetches,
        )

        return {"train": train_dataloader, "valid": valid_dataloader, "infer": test_dataloader}

    def get_snip_data(self, posts_bin, posts_meta, snip_ids):
        snip_dict = {}

        # 1. Fetch all binary data for SNIP in one batch
        snip_samples = list(
            posts_bin.find(
                {
                    "id": {"$in": snip_ids},
                    "kind": {"$in": self.db_fields}, 
                },
                {"id": 1, "chunk": 1, "kind": 1, "chunk_id": 1},
            )
        )

        # Pre-group chunks by (id, kind) for O(N) access
        chunks_by_id_kind = {}
        for s in snip_samples:
            key = (s["id"], s["kind"])
            if key not in chunks_by_id_kind:
                chunks_by_id_kind[key] = []
            chunks_by_id_kind[key].append(s)

        # 2. Fetch all metadata for SNIP in one batch
        all_meta = list(
            posts_meta.find(
                {"id": {"$in": snip_ids}},
                list(self.meta_fields) + ["modalities", "id"],
            )
        )
        meta_lookup = {meta["id"]: meta for meta in all_meta}

        for id in snip_ids:
            # get ID's label and modalities
            meta_for_id = meta_lookup.get(id)
            if meta_for_id is None:
                print(f"[WARN] No metadata for SNIP subject {id}, skipping.")
                continue

            label = meta_for_id[self.meta_fields[0]]
            modalities = meta_for_id["modalities"]
            id_modalities = set(modalities).intersection(set(self.db_fields))

            for mod in id_modalities:
                # Optimized: get pre-grouped chunks and sort them
                samples_for_id_kind = chunks_by_id_kind.get((id, mod), [])
                if not samples_for_id_kind:
                    continue
                
                samples_for_id_kind.sort(key=lambda x: x["chunk_id"])
                data = b"".join([s["chunk"] for s in samples_for_id_kind])

                result = {
                    "input": safe_normalize(self.funcs["mytransform"](data).float()),
                    "modality": mod,
                    "label": torch.tensor(label).unsqueeze(0),
                }

                snip_dict[str(id)+'_'+mod] = result

        return multimodal_collate({0:snip_dict}) # dict is expected in collate

    def get_model(self):
        model_init_seed = self._hparams["model"].get("model_init_seed", None)
        if model_init_seed is not None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(model_init_seed)

        model = ResNet3D(
            in_channels=1,
            n_classes=self.n_classes,
            channels=self.n_channels
        )

        if model_init_seed is not None:
            torch.set_rng_state(rng_state)
            print(f"Model initialized with fixed seed {model_init_seed}")

        init_weights_path = self._hparams["model"].get("init_weights_path", None)
        if init_weights_path and os.path.exists(init_weights_path):
            model.load_state_dict(torch.load(init_weights_path, map_location="cpu"))
            print(f"Loaded init weights from {init_weights_path}")
        elif init_weights_path:
            raise FileNotFoundError(f"init_weights_path not found: {init_weights_path}")

        if self.masked:
            print("Using MultiMaskSNIPWrapper for masked training")
            model = MultiMaskSNIPWrapper(
                model,
                sparsity=self._hparams["model"].get("sparsity", 0.9),
            )

            use_smart_init = self._hparams["model"].get("smart_init", False)
            use_warmup_masks = self._hparams["model"].get("warmup_mask_init", False)
            use_disjoint_masks = self._hparams["model"].get("disjoint_mask_init", False)
            unimodal_paths = self._hparams["model"].get("unimodal_model_paths", None)

            MOD_NAME_TO_INT = {"smri": 0, "falff": 1, "dwi": 2}

            if use_warmup_masks and unimodal_paths:
                print("Using warmup mask initialization from dense unimodal checkpoints...")
                print(f"Unimodal paths config: {unimodal_paths}")

                unimodal_checkpoints = {}
                for mod_name, path in unimodal_paths.items():
                    if path is None:
                        continue
                    int_mod_id = MOD_NAME_TO_INT.get(mod_name)
                    if int_mod_id is None:
                        raise ValueError(f"Unknown modality name '{mod_name}', expected one of {list(MOD_NAME_TO_INT)}")
                    if os.path.exists(path):
                        print(f"Loading dense unimodal checkpoint for modality {mod_name} ({int_mod_id}) from {path}")
                        unimodal_checkpoints[int_mod_id] = torch.load(path, map_location='cpu')
                    else:
                        raise FileNotFoundError(f"Unimodal model path not found: {path}")

                snip_data, snip_modalities, snip_labels = self.snip_data
                model.register_masks_from_dense_checkpoints(
                    unimodal_checkpoints,
                    snip_data=(snip_data, snip_modalities, snip_labels),
                )
                print("Warmup mask initialization complete!")

            elif use_disjoint_masks and unimodal_paths:
                print("Using disjoint mask initialization from dense unimodal checkpoints...")
                print(f"Unimodal paths config: {unimodal_paths}")

                unimodal_checkpoints = {}
                for mod_name, path in unimodal_paths.items():
                    if path is None:
                        continue
                    int_mod_id = MOD_NAME_TO_INT.get(mod_name)
                    if int_mod_id is None:
                        raise ValueError(f"Unknown modality name '{mod_name}', expected one of {list(MOD_NAME_TO_INT)}")
                    if os.path.exists(path):
                        print(f"Loading dense unimodal checkpoint for modality {mod_name} ({int_mod_id}) from {path}")
                        unimodal_checkpoints[int_mod_id] = torch.load(path, map_location='cpu')
                    else:
                        raise FileNotFoundError(f"Unimodal model path not found: {path}")

                snip_data, snip_modalities, snip_labels = self.snip_data
                model.register_disjoint_masks_from_dense_checkpoints(
                    unimodal_checkpoints,
                    snip_data=(snip_data, snip_modalities, snip_labels),
                )
                print("Disjoint mask initialization complete!")

            elif use_smart_init and unimodal_paths:
                print("Using smart initialization from unimodal models...")
                print(f"Unimodal paths config: {unimodal_paths}")

                unimodal_checkpoints = {}
                for mod_id, path in unimodal_paths.items():
                    print(f"Processing modality: {mod_id}, path: {path}")
                    if os.path.exists(path):
                        print(f"Loading unimodal model for modality {mod_id} from {path}")
                        checkpoint = torch.load(path, map_location='cpu')
                        unimodal_checkpoints[mod_id] = checkpoint
                    else:
                        raise FileNotFoundError(f"Unimodal model path not found: {path}")

                snip_data, snip_modalities, snip_labels = self.snip_data
                model.initialize_from_unimodal_models(
                    unimodal_checkpoints,
                    snip_data=(snip_data, snip_modalities, snip_labels)
                )
                print("Smart initialization complete!")

            else:
                print("Initializing masks from scratch using SNIP...")
                snip_data, snip_modalities, snip_labels = self.snip_data
                model.register_multimodal_masks(snip_modalities, snip_data, snip_labels)
                print("Masks initialized.")

        return model

    def get_criterion(self):
        return torch.nn.BCEWithLogitsLoss()

    def get_optimizer(self, model):
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=self.rmsprop_lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.onecycle_lr)
        return optimizer

    def get_scheduler(self, optimizer):
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.onecycle_lr,
            div_factor=100,
            pct_start=0.1,
            epochs=self.num_epochs,
            steps_per_epoch=len(self.loaders["train"]),
        )
        return scheduler

    def get_callbacks(self):
        checkpoint_params = {
            # "sync": False,
            "save_best": True,
            "metric_key": "loss",
            "loader_key": "valid",
            "minimize": True,
        }
        # checkpoint_params = {
        #     # "sync": False,
        #     "save_best": True,
        #     "metric_key": "accuracy",
        #     "loader_key": "valid",
        #     "minimize": False,
        # }
        if self.model_path:
            checkpoint_params.update({"resume_model": self.model_path})
        return {
            "checkpoint": dl.CheckpointCallback(
                self._logdir, **checkpoint_params
            ),
            "tqdm": dl.TqdmCallback(),
        }

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveValueMetric(compute_on_call=False)
            for key in self._metric_keys
        }
        self.meters["auc"] = metrics.AUCMetric(compute_on_call=False)

        rank = distributed.get_rank()
        loader_key = self.loader_key
        self.csv_filename = os.path.join(self._logdir, f"raw_preds_{loader_key}_rank_{rank}.csv")
        file_exists = os.path.isfile(self.csv_filename) and os.path.getsize(self.csv_filename) > 0
        self.csv_file = open(self.csv_filename, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        if not file_exists:
            self.csv_writer.writerow(["epoch", "probability", "target"])

    def on_loader_end(self, runner):
        for key in self._metric_keys:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        self.loader_metrics["auc"] = self.meters["auc"].compute()[2]

        if self.engine.is_ddp:
            world_size = distributed.get_world_size()
            for key in ["loss", "accuracy", "auc"]:
                local_val = self.loader_metrics[key]
                val_tensor = torch.tensor([local_val], device=self.engine.device)
                avg_tensor = distributed.mean_reduce(val_tensor, world_size)
                self.loader_metrics[key] = avg_tensor.item()

        if hasattr(self, 'csv_file') and self.csv_file:
            self.csv_file.close()

        super().on_loader_end(runner)

    # model train/valid step
    def handle_batch(self, batch):
        if self.multimodal:
            sample, modality, label = batch
        else:
            sample, label = batch
            modality = None

        if self.model.training:
            if self.bit16:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    y_hat = self.model.forward(sample) if not self.masked else self.model.forward(sample, modality)
                    loss = self.criterion(y_hat, label.float())
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                self.scheduler.step()
                scaler.update()
                self.optimizer.zero_grad()
            else:
                y_hat = self.model.forward(sample) if not self.masked else self.model.forward(sample, modality)
                loss = self.criterion(y_hat, label.float())
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
        else:
            with torch.no_grad():
                y_hat = self.model.forward(sample) if not self.masked else self.model.forward(sample, modality)
                loss = self.criterion(y_hat, label.float())

        with torch.no_grad():
            proba_preds = torch.sigmoid(y_hat)
            preds = proba_preds > 0.5
            accuracy = (preds == label).float().mean()
            probs_np = proba_preds.detach().cpu().numpy().flatten()
            targets_np = label.detach().cpu().numpy().flatten()
            self.csv_writer.writerows(zip([self.epoch_step] * len(probs_np), probs_np, targets_np))

        self.batch_metrics.update({
            "loss": loss,
            "accuracy": accuracy,
            "learning rate": torch.tensor(self.optimizer.param_groups[0]["lr"]),
        })
        for key in self.batch_metrics:
            self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)
        self.meters["auc"].update(proba_preds, label)

        del sample, label, y_hat, loss

@hydra.main(config_path="conf", config_name="new_conf", version_base=None)
def main(cfg: DictConfig):
    # Loading common parameters
    # Model parameters
    n_classes = cfg.model.n_classes
    # config_file = cfg.model.config_file
    optimize_inline = cfg.model.optimize_inline
    model_channels = cfg.model.model_channels
    use_groupnorm = cfg.model.use_groupnorm
    model_path = cfg.paths.model if cfg.paths.loadcheckpoint else ""
    db_host = cfg.mongo.host_slurm if os.environ.get("SLURM_JOB_ID") else cfg.mongo.host

    validation_percent = cfg.mongo.validation_percent
    wandb_project = cfg.wandb.project
    bit16 = cfg.bit16

    client_creator = ClientCreator(
        db_host, crop_tensor=cfg.client_creator.crop_tensor
    )

    # Evaluate the Python code from the YAML config
    experiment_name = cfg.experiment.experiment_name
    cubesizes = cfg.experiment.cubesizes
    numcubes = cfg.experiment.numcubes
    numvolumes = cfg.experiment.numvolumes
    weights = cfg.experiment.weights
    databases = cfg.experiment.databases
    collections = cfg.experiment.collections
    # dbfields = [tuple(fields) for fields in cfg.experiment.dbfields]  # Convert to tuples
    dbfields = tuple(cfg.experiment.dbfields)
    metafields = tuple(cfg.experiment.metafields)
    epochs = cfg.experiment.epochs
    prefetches = cfg.experiment.prefetches
    num_workers = cfg.experiment.num_workers
    prefetch_factor = cfg.experiment.prefetch_factor
    train_prefetches = cfg.experiment.get("train_prefetches", None)
    train_num_workers = cfg.experiment.get("train_num_workers", None)
    train_prefetch_factor = cfg.experiment.get("train_prefetch_factor", None)
    train_persistent_workers = cfg.experiment.get("train_persistent_workers", True)
    eval_prefetches = cfg.experiment.get("eval_prefetches", None)
    eval_num_workers = cfg.experiment.get("eval_num_workers", None)
    eval_prefetch_factor = cfg.experiment.get("eval_prefetch_factor", None)
    eval_persistent_workers = cfg.experiment.get("eval_persistent_workers", True)
    cudnn_benchmark = cfg.experiment.get("cudnn_benchmark", False)
    max_folds = cfg.experiment.get("max_folds", None)
    attenuates = cfg.experiment.attenuates
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

    # we need oneCycleLR, but not the rest of the curiculum
    subvolume_shape = [cubesizes] * 3
    # Use SLURM_GPUS_ON_NODE — same source as get_engine() — since DDP is not
    # initialized yet when main() runs (Catalyst forks processes inside runner.run()).
    world_size = int(os.environ.get("SLURM_GPUS_ON_NODE", torch.cuda.device_count()))
    onecycle_lr = rmsprop_lr = (
        attenuates # this comes from 0.8/0.2 training? what is this input for oneCycleLR? TODO: trace it further
        * 1
        * cfg.experiment.lr_scale
        * numcubes
        * numvolumes
        * world_size
        / 256
    )
    wandb_experiment = (
        f"{experiment_name}: {collections}, {dbfields}-{metafields}, masked={cfg.model.get('masked', False)}, sps={cfg.model.get('sparsity', None)}"
    )

    # Set database parameters
    client_creator.set_database(databases)
    client_creator.set_collection(collections)
    client_creator.set_num_subcubes(numcubes)
    client_creator.set_shape(subvolume_shape)

    # paths:
    #     loadcheckpoint: False
    #     model: "../logs/tmp/new_test_fbirn_falff/model.last.pth"
    #     logdir: "./logs/tmp/new_test_fbirn_falff/"
    logdir = f"{cfg.paths.logdir}/{experiment_name}_{collections}_{dbfields}_{metafields}_masked_{cfg.model.get('masked', False)}_sps_{cfg.model.get('sparsity', None)}"
    os.makedirs(logdir, exist_ok=True)

    # Set hparams
    hparams = OmegaConf.to_container(cfg, resolve=True)

    folds_to_run = cfg.experiment.cv_folds if max_folds is None else min(cfg.experiment.cv_folds, int(max_folds))
    if folds_to_run < 1:
        raise ValueError("experiment.max_folds must be at least 1 when set")

    # Sampler seed: drawn ONCE here in the (parent) __main__ process and passed
    # into the runner so every DDP rank receives the same value via pickling.
    # The module-level SEED is re-drawn per worker by mp.spawn and is NOT safe
    # for the distributed sampler (it would differ per rank).
    sampler_seed = random.randint(0, 9999)

    # run cross-validation
    for fold_idx in range(folds_to_run):

        print(f"Starting fold {fold_idx+1}/{cfg.experiment.cv_folds}")
        hparams["fold_idx"] = fold_idx

        rundir = f"{logdir}/fold_{fold_idx}"
        os.makedirs(rundir, exist_ok=True)

        runner = CustomRunner(
            logdir=rundir, # this is self._logdir
            wandb_project=wandb_project,
            wandb_experiment=wandb_experiment,
            model_path=model_path,
            n_channels=model_channels,
            n_classes=n_classes,
            # modelconfig=config_file,
            n_epochs=epochs,
            optimize_inline=optimize_inline,
            validation_percent=validation_percent,
            onecycle_lr=onecycle_lr,
            rmsprop_lr=rmsprop_lr,
            num_subcubes=numcubes,
            num_volumes=numvolumes,
            groupnorm=use_groupnorm,
            client_creator=client_creator,
            off_brain_weight=weights,
            prefetches=prefetches,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            train_prefetches=train_prefetches,
            train_num_workers=train_num_workers,
            train_prefetch_factor=train_prefetch_factor,
            train_persistent_workers=train_persistent_workers,
            eval_prefetches=eval_prefetches,
            eval_num_workers=eval_num_workers,
            eval_prefetch_factor=eval_prefetch_factor,
            eval_persistent_workers=eval_persistent_workers,
            indexid=cfg.mongo.index_id,
            db_collection=collections,
            db_name=databases,
            db_fields=dbfields,
            meta_fields=metafields,
            subvolume_shape=subvolume_shape,
            lowprecision=bit16,
            lossweight = [w / sum(cfg.model.loss_weight) for w in cfg.model.loss_weight] if sum(cfg.model.loss_weight) != 0 else ValueError("The sum of loss weights cannot be zero."),
            db_host=db_host,
            wandb_team=cfg.wandb.team,
            maxshape=cfg.model.maxshape,
            hparams=hparams,
            sampler_seed=sampler_seed,
        )
        runner.run()
        del runner
        torch.cuda.empty_cache()
        import gc; gc.collect()


if __name__ == "__main__":
    main()
