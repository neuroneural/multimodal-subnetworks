import warnings
warnings.filterwarnings("ignore")

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
import random
import shutil
from packaging import version
import yaml

from catalyst import dl, metrics, utils
from catalyst.data import BatchPrefetchLoaderWrapper

import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from resnet import ResNet3D

from mindfultensors.mongoloader import MongoClient
from mindfultensors.utils import unit_interval_normalize, DBBatchSampler

from src.db_client import ClientCreator
from src.customMongoDataset import CustomMongoDataset

SEED = random.randint(0, 9999)
utils.set_global_seed(SEED)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
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
        self.funcs = {
            "createclient": self.client_creator.create_client,
            "createVclient": self.client_creator.create_client,
            "mycollate": self.client_creator.mycollate,
            "mycollate_full": self.client_creator.mycollate_full,
            "mytransform": self.client_creator.mytransform,
        }

        self.collate = (
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
        )
        all_ids = sorted(all_ids)
        # print(all_ids)

        labels = []
        for id in all_ids:
            label = posts_meta.find_one({"id": id}, self.meta_fields)[self.meta_fields[0]] # get label for the id
            labels.append(label)
        labels = np.array(labels)
    
        # Create CV split
        cv_folds = StratifiedKFold(n_splits=self._hparams["experiment"]["cv_folds"], shuffle=True, random_state=SEED)
        train_idx, test_idx = list(cv_folds.split(all_ids, labels))[self._hparams["fold_idx"]]
        # split train into train and validation
        train_idx, valid_idx = train_test_split(train_idx, test_size=self.validation_percent, stratify=labels[train_idx], random_state=SEED)

        all_ids = np.array(all_ids)
        train_ids = all_ids[train_idx].tolist() # mongo expects default python list, not numpy array
        valid_ids = all_ids[valid_idx].tolist()
        test_ids = all_ids[test_idx].tolist()

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

        # Create dataloaders
        train_dataset = CustomMongoDataset(
            train_ids, 
            self.funcs["mytransform"],
            None,
            self.db_fields,
            self.meta_fields,
            normalize=unit_interval_normalize,
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
                pin_memory=True,
                worker_init_fn=self.funcs["createclient"],
                persistent_workers=True,
                prefetch_factor=2,
                num_workers=4,  # self.prefetches,
                # prefetch_factor=None,
                # num_workers=1,  # self.prefetches,
            ),
            num_prefetches=self.prefetches,
        )

        valid_dataset = CustomMongoDataset(
            valid_ids,#take first validation_percent percent from list
            self.funcs["mytransform"],
            None,
            self.db_fields,
            self.meta_fields,
            normalize=unit_interval_normalize,
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
                pin_memory=True,
                worker_init_fn=self.funcs["createVclient"],
                persistent_workers=True,
                # prefetch_factor=4,
                # num_workers=4,  # self.prefetches,
                prefetch_factor=2,
                num_workers=4,  # self.prefetches,
            ),
            num_prefetches=self.prefetches,
        )

        test_dataset = CustomMongoDataset(
            test_ids,#take first validation_percent percent from list
            self.funcs["mytransform"],
            None,
            self.db_fields,
            self.meta_fields,
            normalize=unit_interval_normalize,
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
                pin_memory=True,
                worker_init_fn=self.funcs["createVclient"],
                persistent_workers=True,
                # prefetch_factor=4,
                # num_workers=4,  # self.prefetches,
                prefetch_factor=2,
                num_workers=4,  # self.prefetches,
            ),
            num_prefetches=self.prefetches,
        )

        return {"train": train_dataloader, "valid": valid_dataloader, "infer": test_dataloader}

    def get_model(self):
        model = ResNet3D(
            in_channels=1, 
            n_classes=self.n_classes, 
            channels=self.n_channels
        )
        # if self.model_path and os.path.exists(self.model_path):
        #     model.load_state_dict(torch.load(self.model_path))
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

    def on_loader_end(self, runner):
        for key in ["loss", "accuracy", "learning rate"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        self.loader_metrics["auc"] = self.meters["auc"].compute()[2]

        super().on_loader_end(runner)

    # model train/valid step
    def handle_batch(self, batch):

        # Add synchronization before processing
        if self.engine.is_ddp:
            torch.cuda.synchronize()
        
        sample, label = batch

        # run model forward/backward pass
        if self.model.training:
            if self.bit16:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    y_hat = self.model.forward(sample)
                    loss = self.criterion(y_hat, label.float())
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                self.scheduler.step()
                scaler.update()
                self.optimizer.zero_grad()
            else:
                y_hat = self.model.forward(sample)
                loss = self.criterion(y_hat, label.float())
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
        else:
            with torch.no_grad():
                y_hat = self.model.forward(sample)
                loss = self.criterion(y_hat, label.float())

        # Metrics calculation
        with torch.no_grad():
            proba_preds = torch.sigmoid(y_hat)
            preds = proba_preds > 0.5
            accuracy = (preds == label).float().mean()


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
    attenuates = cfg.experiment.attenuates

    # we need oneCycleLR, but not the rest of the curiculum
    subvolume_shape = [cubesizes] * 3
    onecycle_lr = rmsprop_lr = (
        attenuates # this comes from 0.8/0.2 training? what is this input for oneCycleLR? TODO: trace it further
        * 1
        * cfg.experiment.lr_scale
        * numcubes
        * numvolumes
        / 256
    )
    wandb_experiment = (
        f"{experiment_name}: {collections}, {dbfields}-{metafields}"
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
    logdir = f"{cfg.paths.logdir}/{experiment_name}_{collections}_{dbfields}_{metafields}"
    os.makedirs(logdir, exist_ok=True)

    # Set hparams
    hparams = OmegaConf.to_container(cfg, resolve=True)

    # run cross-validation
    for fold_idx in range(cfg.experiment.cv_folds):
        print(f"Starting fold {fold_idx+1}/{cfg.experiment.cv_folds}")
        hparams["fold_idx"] = fold_idx

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


if __name__ == "__main__":
    main()
