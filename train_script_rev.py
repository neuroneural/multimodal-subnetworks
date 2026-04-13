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
from mindfultensors.utils import unit_interval_normalize, DBBatchSampler

def safe_normalize(img):
    """Unit interval normalization with epsilon protection against zero-variance volumes."""
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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
#os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
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
        prefetches=4,
        num_workers=8,
        volume_shape=[256] * 3,
        subvolume_shape=[256] * 3,
        lowprecision=False,
        lossweight=[1, 0],
        maxshape=300,
        hparams=None,
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

        self.masked = self._hparams["model"].get("masked", False)

    def get_engine(self):
        if torch.cuda.device_count() > 1:
            return dl.DistributedDataParallelEngine(
                # mixed_precision="fp16",
                # ddp_kwargs={"backend": "nccl"},
                process_group_kwargs={"backend": "nccl"},
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
                log_batch_metrics=True,
                # log_epoch_metrics=True,
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

        # get all IDs with the required modalities, pull their labels for cross-validation splits

        client = MongoClient("mongodb://" + self.db_host + ":27017")
        db = client[self.db_name]
        posts_bin = db[self.db_collection + ".bin"]
        posts_meta = db[self.db_collection + ".meta"]

        # get ids, pull labels
        all_ids = posts_meta.distinct( # pull all unique IDs (subjects) with at least one modality in db_fields
            "id",
            {'modalities': {"$in": self.db_fields}}
            #{'modalities': {"$all": self.db_fields}}
        )
        all_ids = sorted(all_ids)
        # print(all_ids)

        # Batch label fetch — single query instead of one find_one per subject
        meta_docs = {
            doc["id"]: doc[self.meta_fields[0]]
            for doc in posts_meta.find(
                {"id": {"$in": all_ids}},
                {"id": 1, self.meta_fields[0]: 1, "_id": 0}
            )
        }
        labels = np.array([meta_docs[i] for i in all_ids])
    
        # Create CV split
        cv_folds = StratifiedKFold(n_splits=self._hparams["experiment"]["cv_folds"], shuffle=True, random_state=self._hparams["experiment"].get("cv_seed", 42))
        train_idx, test_idx = list(cv_folds.split(all_ids, labels))[self._hparams["fold_idx"]]
        # split train into train and validation
        train_idx, valid_idx = train_test_split(train_idx, test_size=self.validation_percent, stratify=labels[train_idx], random_state=self._hparams["experiment"].get("cv_seed", 42))

        all_ids = np.array(all_ids)
        train_ids = all_ids[train_idx].tolist() # mongo expects default python list, not numpy array
        valid_ids = all_ids[valid_idx].tolist()
        test_ids = all_ids[test_idx].tolist()

        if self.masked:
            print("Preparing SNIP mask data...")
            snip_batch_size = self._hparams["model"].get("snip_batch_size", 20)
            rng = random.Random(SEED) 
            snip_batch_ids = rng.sample(train_ids, len(train_ids))[:snip_batch_size]

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
        train_sampler = (
            DBBatchSampler(train_dataset, batch_size=self.num_volumes, seed=SEED)
            if self.engine.is_ddp
            else DBBatchSampler(train_dataset, batch_size=self.num_volumes)
        )
        train_dataloader = BatchPrefetchLoaderWrapper(
            DataLoader(
                train_dataset,
                sampler=train_sampler,
                collate_fn=self.collate,
                pin_memory=not self.multimodal,
                worker_init_fn=self.funcs["createclient"],
                persistent_workers=False,
                prefetch_factor=2,
                num_workers=self.num_workers,
            ),
            num_prefetches=self.prefetches,
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
        valid_sampler = (
            DBBatchSampler(valid_dataset, batch_size=self.num_volumes, seed=SEED)
            if self.engine.is_ddp
            else DBBatchSampler(
                valid_dataset, batch_size=self.num_volumes, seed=SEED
            )
        )
        valid_dataloader = BatchPrefetchLoaderWrapper(
            DataLoader(
                valid_dataset,
                sampler=valid_sampler,
                collate_fn=self.collate,
                pin_memory=not self.multimodal,
                worker_init_fn=self.funcs["createVclient"],
                persistent_workers=False,
                prefetch_factor=2,
                num_workers=self.num_workers,
            ),
            num_prefetches=self.prefetches,
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
        test_sampler = (
            DBBatchSampler(test_dataset, batch_size=self.num_volumes, seed=SEED)
            if self.engine.is_ddp
            else DBBatchSampler(
                test_dataset, batch_size=self.num_volumes, seed=SEED
            )
        )
        test_dataloader = BatchPrefetchLoaderWrapper(
            DataLoader(
                test_dataset,
                sampler=test_sampler,
                collate_fn=self.collate,
                pin_memory=not self.multimodal,
                worker_init_fn=self.funcs["createVclient"],
                persistent_workers=False,
                prefetch_factor=2,
                num_workers=self.num_workers,
            ),
            num_prefetches=self.prefetches,
        )

        return {"train": train_dataloader, "valid": valid_dataloader, "infer": test_dataloader}

    def get_snip_data(self, posts_bin, posts_meta, snip_ids):
        snip_dict = {}

        snip_samples = list(
            posts_bin.find(
                {
                    "id": {"$in": snip_ids},
                    "kind": {"$in": self.db_fields}, # .bin contains 3D kinds like 'smri', 'falff', 'dwi'. Scalar labels are stored in .meta
                },
                {"id": 1, "chunk": 1, "kind": 1, "chunk_id": 1},
            )
        )

        for id in snip_ids:
            # get ID's label and modalities
            meta_for_id = list(
                posts_meta.find(
                    {
                        "id": id,
                    },
                    list(self.meta_fields) + ["modalities"],
                )
            )

            assert len(meta_for_id) != 0, f"No meta entries found for id {id}"
            assert len(meta_for_id) < 2, f"More than one meta entry found for id {id}"

            label = meta_for_id[0][self.meta_fields[0]]
            modalities = meta_for_id[0]["modalities"]
            id_modalities = set(modalities).intersection(set(self.db_fields))

            # Get samples for this ID
            samples_for_id = [
                sample
                for sample in snip_samples
                if sample["id"] == id
            ]

            for mod in id_modalities:
                data = make_serial(samples_for_id, mod)

                for mod in id_modalities:
                    data = make_serial(samples_for_id, mod)

                    result = {
                        "input": safe_normalize(self.funcs["mytransform"](data).float()),
                        "modality": mod,
                        "label": torch.tensor(label).unsqueeze(0),
                    }

                    snip_dict[str(id)+'_'+mod] = result

        return multimodal_collate({0:snip_dict}) # dict is expected in collate

    def get_model(self):
        model = ResNet3D(
            in_channels=1, 
            n_classes=self.n_classes, 
            channels=self.n_channels
        )

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

            # Check if we should use smart initialization from unimodal models
            use_smart_init = self._hparams["model"].get("smart_init", False)
            unimodal_paths = self._hparams["model"].get("unimodal_model_paths", None)
            
            if use_smart_init and unimodal_paths:
                print("Using smart initialization from unimodal models...")
                print(f"Unimodal paths config: {unimodal_paths}")
                
                # Load unimodal model state_dicts
                unimodal_checkpoints = {}
                for mod_id, path in unimodal_paths.items():
                    print(f"Processing modality: {mod_id}, path: {path}")
                    if os.path.exists(path):
                        print(f"Loading unimodal model for modality {mod_id} from {path}")
                        checkpoint = torch.load(path, map_location='cpu')
                        unimodal_checkpoints[mod_id] = checkpoint
                    else:
                        raise FileNotFoundError(f"Unimodal model path not found: {path}")
                
                # Initialize from unimodal models: load pretrained masks, apply SNIP
                # on fixed-init weights, intersect masks, then average weights
                snip_data, snip_modalities, snip_labels = self.snip_data
                model.initialize_from_unimodal_models(
                    unimodal_checkpoints,
                    snip_data=(snip_data, snip_modalities, snip_labels)
                )
                print("Smart initialization complete!")
                
            else:
                # Standard SNIP initialization from scratch
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
            for key in ["loss", "accuracy", "learning rate"]
        }
        self.meters["auc"] = metrics.AUCMetric(
            compute_on_call=False
        )

        # --- CSV LOGGING SETUP ---
        rank = distributed.get_rank()
        loader_key = self.loader_key # e.g., "train", "valid"
        self.csv_filename = os.path.join(
            self._logdir, 
            f"raw_preds_{loader_key}_rank_{rank}.csv"
        )
        file_exists = os.path.isfile(self.csv_filename) and os.path.getsize(self.csv_filename) > 0

        self.csv_file = open(self.csv_filename, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write header only if file is new
        if not file_exists:
            self.csv_writer.writerow(["epoch", "probability", "target"])


    def on_loader_end(self, runner):
        for key in ["loss", "accuracy", "learning rate"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        self.loader_metrics["auc"] = self.meters["auc"].compute()[2]

        if self.engine.is_ddp:
            # Get world_size explicitly
            world_size = distributed.get_world_size()
            
            for key in ["loss", "accuracy"]:
                local_val = self.loader_metrics[key]
                
                # Create a tensor on the correct device
                # self.engine.device is reliable for the current worker's device
                val_tensor = torch.tensor([local_val], device=self.engine.device)
                
                # FIX: Pass world_size to mean_reduce
                avg_tensor = distributed.mean_reduce(val_tensor, world_size)
                self.loader_metrics[key] = avg_tensor.item()

        # CSV Safety Close
        if hasattr(self, 'csv_file') and self.csv_file:
            self.csv_file.close()

        super().on_loader_end(runner)

    # model train/valid step
    def handle_batch(self, batch):

        #print(f'[DEBUG] batch_step={self.batch_step}', flush=True)
        # # Add synchronization before processing
        # if self.engine.is_ddp:
        #     torch.cuda.synchronize()
        
        if self.multimodal: #MM
            sample, modality, label = batch
        else:
            sample, label = batch

        # run model forward/backward pass
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
                if torch.isnan(sample).any() or torch.isinf(sample).any():
                    print(f'[WARN] Bad input at step {self.batch_step}: nan={torch.isnan(sample).sum()} inf={torch.isinf(sample).sum()}')
                    print(f"[WARN] Input shape: {sample.shape}, dtype: {sample.dtype}, min: {sample.min()}, max: {sample.max()}")
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

        # Metrics calculation and CSV logging
        with torch.no_grad():
            proba_preds = torch.sigmoid(y_hat)
            preds = proba_preds > 0.5
            accuracy = (preds == label).float().mean()
            
            # CSV logging: Move to CPU / Numpy
            probs_np = proba_preds.detach().cpu().numpy().flatten()
            targets_np = label.detach().cpu().numpy().flatten()
            epochs_np = [self.epoch_step] * len(probs_np)
            rows = zip(epochs_np, probs_np, targets_np)
            self.csv_writer.writerows(rows)


        self.batch_metrics.update({
            "loss": loss,
            "accuracy": accuracy, 
            "learning rate": torch.tensor(
                    self.optimizer.param_groups[0]["lr"]
            )
        })
        for key in self.batch_metrics:
            self.meters[key].update(
                self.batch_metrics[key].item(), self.batch_size
            )
        self.meters["auc"].update(proba_preds, label)

        del sample
        del label
        del y_hat
        del loss
        del proba_preds
        del preds
        del accuracy
        if self.multimodal:
            del modality

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
    attenuates = cfg.experiment.attenuates



    # paths:
    #     loadcheckpoint: False
    #     model: "../logs/tmp/new_test_fbirn_falff/model.last.pth"
    #     logdir: "./logs/tmp/new_test_fbirn_falff/"
    logdir = f"{cfg.paths.logdir}/{experiment_name}_{collections}_{dbfields}_{metafields}_masked_{cfg.model.get('masked', False)}_sps_{cfg.model.get('sparsity', None)}"
    os.makedirs(logdir, exist_ok=True)

    # Set hparams
    hparams = OmegaConf.to_container(cfg, resolve=True)

    # run cross-validation
    for fold_idx in range(cfg.experiment.cv_folds):
        subvolume_shape = [cubesizes] * 3
        onecycle_lr = rmsprop_lr = (
            attenuates 
            * 1
            * cfg.experiment.lr_scale
            * numcubes
            * numvolumes
            / 256
        )
        wandb_experiment = (
            f"{experiment_name}: {collections}, {dbfields}-{metafields}, masked={cfg.model.get('masked', False)}, sps={cfg.model.get('sparsity', None)}"
        )

        print(f"Starting fold {fold_idx+1}/{cfg.experiment.cv_folds}")
        hparams["fold_idx"] = fold_idx

        # Set database parameters
        client_creator.set_database(databases)
        client_creator.set_collection(collections)
        client_creator.set_num_subcubes(numcubes)
        client_creator.set_shape(subvolume_shape)

        rundir = f"{logdir}/fold_{fold_idx}"
        os.makedirs(logdir, exist_ok=True)

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
        )
        runner.run()
        del runner
        torch.cuda.empty_cache()
        import gc; gc.collect()


if __name__ == "__main__":
    main()
