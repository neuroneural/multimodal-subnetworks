import hydra
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

from resnet import ResNet3D, enMesh_checkpoint, enMesh  # Import from resnet.py
from mindfultensors.gencoords import CoordsGenerator
from mindfultensors.utils import unit_interval_normalize, DBBatchSampler

from mindfultensors.mongoloader import (
    create_client,
    collate_subcubes,
    mcollate,
    MongoDataset,
    MongoClient,
    MongoheadDataset,
    mtransform,
)

SEED = random.randint(0, 9999)
utils.set_global_seed(SEED)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

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
        validation_percent: float,
        onecycle_lr: float,
        num_subcubes: int,
        num_volumes: int,
        client_creator,
        indexid: str,
        db_host: str,
        db_name: str,
        db_collection: str,
        wandb_team: str,
        db_fields: tuple,
        prefetches=8,
        volume_shape=[256] * 3,
        subvolume_shape=[256] * 3,
        hparams=None,
    ):
        super().__init__()
        self._logdir = logdir
        self.wandb_project = wandb_project
        self.wandb_experiment = wandb_experiment
        self.model_path = model_path
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.prefetches = prefetches
        self.db_host = db_host
        self.db_name = db_name
        self.db_collection = db_collection
        self.db_fields = db_fields
        self.shape = subvolume_shape[0]
        self.num_subcubes = num_subcubes
        self.num_volumes = num_volumes
        self.n_epochs = n_epochs
        self.client_creator = client_creator
        self.index_id = indexid
        self.wandb_team = wandb_team
        self._hparams = hparams
        self.onecycle_lr = onecycle_lr

    def get_engine(self):
        return dl.GPUEngine()  # Simplified for ResNet3D

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

    def get_loaders(self):
        client = MongoClient("mongodb://" + self.db_host + ":27017")
        db = client[self.db_name]
        posts = db[self.db_collection + ".bin"]
        num_examples = posts.count_documents({})

        tdataset = MongoheadDataset(
            range(num_examples),
            self.client_creator.mytransform,
            None,
            self.db_fields,
            normalize=unit_interval_normalize,
            id=self.index_id,
        )

        tdataloader = BatchPrefetchLoaderWrapper(
            DataLoader(
                tdataset,
                batch_size=self.num_volumes,
                collate_fn=self.client_creator.mycollate_full,
                pin_memory=True,
                num_workers=4,
            ),
            num_prefetches=self.prefetches,
        )

        vdataset = MongoDataset(
            range(32),  # Fixed validation set size
            self.client_creator.mytransform,
            None,
            self.db_fields,
            normalize=unit_interval_normalize,
            id=self.index_id,
        )

        vdataloader = BatchPrefetchLoaderWrapper(
            DataLoader(
                vdataset,
                batch_size=self.num_volumes,
                collate_fn=self.client_creator.mycollate_full,
                pin_memory=True,
                num_workers=4,
            ),
            num_prefetches=self.prefetches,
        )

        return {"train": tdataloader, "valid": vdataloader}

    def get_model(self):
        model = ResNet3D(
            in_channels=1,  # Your MRI channels
            n_classes=self.n_classes,
            channels=self.n_channels
        )
        if self.model_path and os.path.exists(self.model_path):
            model.load_state_dict(torch.load(self.model_path))
        return model

    def get_criterion(self):
        return torch.nn.BCELoss()  # For binary classification

    def get_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.onecycle_lr)

    def get_scheduler(self, optimizer):
        return OneCycleLR(
            optimizer,
            max_lr=self.onecycle_lr,
            total_steps=self.n_epochs * len(self.loaders["train"]),
        )

    def handle_batch(self, batch):
        sample, label = batch
        
        # Forward pass
        y_hat = self.model(sample)
        loss = self.criterion(y_hat, label.float())
        
        # Backward pass
        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

        # Metrics
        accuracy = ((y_hat > 0.5).float() == label.float()).float().mean()
        self.batch_metrics.update({
            "loss": loss,
            "accuracy": accuracy
        })

@hydra.main(config_path="conf", config_name="resnet3d_gender_bn_64base_2.2.2.2_exp01", version_base=None)
def main(cfg: DictConfig):
    # Initialize client creator
    client_creator = ClientCreator(
        cfg.mongo.host_slurm if os.environ.get("SLURM_JOB_ID") else cfg.mongo.host,
        crop_tensor=cfg.client_creator.crop_tensor
    )

    # Create runner with ResNet3D-specific parameters
    runner = CustomRunner(
        logdir=cfg.paths.logdir,
        wandb_project=cfg.wandb.project,
        wandb_experiment=f"resnet3d_{cfg.model.base_channels}base",
        model_path=cfg.paths.model if cfg.paths.loadcheckpoint else "",
        n_channels=cfg.model.base_channels,
        n_classes=cfg.model.n_classes,
        n_epochs=cfg.experiment.epochs_code[0],  # First epoch value
        validation_percent=cfg.mongo.validation_percent,
        onecycle_lr=cfg.experiment.attenuates_code[0] * cfg.experiment.lr_scale,
        num_subcubes=cfg.experiment.numcubes_code[0],
        num_volumes=cfg.experiment.numvolumes_code[0],
        client_creator=client_creator,
        indexid=cfg.mongo.index_id,
        db_host=cfg.mongo.host,
        db_name=cfg.mongo.dbname,
        db_collection=cfg.mongo.collection,
        wandb_team=cfg.wandb.team,
        db_fields=(cfg.mongo.datafield, cfg.mongo.labelfield),
        prefetches=cfg.experiment.prefetches_code[0],
        volume_shape=cfg.model.volume_shape,
        hparams=OmegaConf.to_container(cfg)
    )
    
    runner.run()

if __name__ == "__main__":
    main()
