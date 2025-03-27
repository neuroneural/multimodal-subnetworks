import hydra
from omegaconf import DictConfig, OmegaConf
import os
import random
import shutil
import yaml
from catalyst import dl, metrics, utils
from catalyst.data import BatchPrefetchLoaderWrapper
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from resnet import ResNet3D  # Import ResNet3D from your resnet.py
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

class ClientCreator:
    def __init__(self, mongohost, volume_shape=[256] * 3, crop_tensor=False):
        self.mongohost = mongohost
        self.volume_shape = volume_shape
        self.subvolume_shape = None
        self.dbname = None
        self.collection = None
        self.num_subcubes = None
        self.crop_tensor = crop_tensor

    def set_shape(self, shape):
        self.subvolume_shape = shape
        self.coord_generator = CoordsGenerator(
            self.volume_shape, self.subvolume_shape
        )

    def set_collection(self, collection):
        self.collection = collection

    def set_database(self, database):
        self.dbname = database

    def set_num_subcubes(self, num_subcubes):
        self.num_subcubes = num_subcubes

    def create_client(self, x):
        return create_client(
            x,
            dbname=self.dbname,
            colname=self.collection,
            mongohost=self.mongohost,
        )

    def create_v_client(self, x):
        return create_client(
            x,
            dbname=self.dbname,
            colname=self.collection,
            mongohost=self.mongohost,
        )

    def mycollate(self, x):
        return collate_subcubes(
            x,
            self.coord_generator,
            samples=self.num_subcubes,
        )

    def mycollate_full(self, x):
        return mcollate(x)  # Removed crop_tensor for simplicity

    def mytransform(self, x):
        return mtransform(x)

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
        client_creator,
        indexid: str,
        db_collection: str,
        db_name: str,
        db_fields: tuple,
        subvolume_shape: list,
        db_host: str,
        wandb_team: str,
        hparams: dict,
        prefetches=8,
        num_volumes=4,
        **kwargs
    ):
        super().__init__()
        self._logdir = logdir
        self.wandb_project = wandb_project
        self.wandb_experiment = wandb_experiment
        self.model_path = model_path
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_epochs = n_epochs
        self.client_creator = client_creator
        self.index_id = indexid
        self.db_collection = db_collection
        self.db_name = db_name
        self.db_fields = db_fields
        self.subvolume_shape = subvolume_shape
        self.db_host = db_host
        self.wandb_team = wandb_team
        self.prefetches = prefetches
        self.num_volumes = num_volumes
        self._hparams = hparams

    def get_engine(self):
        if torch.cuda.device_count() > 1:
            return dl.DistributedDataParallelEngine(
                process_group_kwargs={"backend": "nccl"}
            )
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
            ),
        }

    def get_loaders(self):
        client = MongoClient(f"mongodb://{self.db_host}:27017")
        db = client[self.db_name]
        posts = db[f"{self.db_collection}.bin"]
        num_examples = posts.count_documents({})

        # Training dataset
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
                persistent_workers=True,
            ),
            num_prefetches=self.prefetches,
        )

        # Validation dataset
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
            in_channels=1,  # MRI data is single channel
            n_classes=self.n_classes,
            channels=self.n_channels
        )
        if self.model_path and os.path.exists(self.model_path):
            model.load_state_dict(torch.load(self.model_path))
        return model

    def get_criterion(self):
        return torch.nn.BCEWithLogitsLoss()  # Better for numerical stability

    def get_optimizer(self, model):
        return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def get_scheduler(self, optimizer):
        # Get the actual number of epochs (first element if it's a list)
        if isinstance(self.n_epochs, str):
            # Evaluate the epochs code if it's a string
            context = {"maxreps": 10}  # Default value if needed
            epochs_list = eval(self.n_epochs, globals(), context)
            n_epochs = int(epochs_list[0])  # Take first element
        elif isinstance(self.n_epochs, list):
            n_epochs = int(self.n_epochs[0])
        else:
            n_epochs = int(self.n_epochs)
        
        train_loader_len = len(self.loaders["train"])
        total_steps = n_epochs * train_loader_len
        
        return OneCycleLR(
            optimizer,
            max_lr=1e-3,
            total_steps=total_steps,
            pct_start=0.3
        )

    def get_callbacks(self):
        return {
            "checkpoint": dl.CheckpointCallback(
                self._logdir,
                save_best=True,
                metric_key="accuracy",
                loader_key="valid",
                minimize=False
            ),
            "tqdm": dl.TqdmCallback(),
        }

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveValueMetric(compute_on_call=False)
            for key in ["loss", "accuracy"]
        }

    def on_loader_end(self, runner):
        for key in ["loss", "accuracy"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)

    def handle_batch(self, batch):
        sample, label = batch
        
        # Forward pass
        y_hat = self.model(sample)
        loss = self.criterion(y_hat, label.float())
        
        # Backward pass and optimization
        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

        # Metrics calculation
        with torch.no_grad():
            preds = torch.sigmoid(y_hat) > 0.5
            accuracy = (preds == label).float().mean()

        self.batch_metrics.update({
            "loss": loss,
            "accuracy": accuracy
        })

        for key in ["loss", "accuracy"]:
            self.meters[key].update(
                self.batch_metrics[key].item(), self.num_volumes
            )


@hydra.main(config_path="conf", config_name="resnet3d_gender_bn_64base_2.2.2.2_exp01", version_base=None)
def main(cfg: DictConfig):
    # Load parameters from config
    volume_shape = cfg.model.volume_shape
    n_classes = cfg.model.n_classes
    model_channels = cfg.model.base_channels
    model_path = cfg.paths.model if cfg.paths.loadcheckpoint else ""
    logdir = cfg.paths.logdir
    db_host = cfg.mongo.host_slurm if os.environ.get("SLURM_JOB_ID") else cfg.mongo.host

    # Initialize client creator
    client_creator = ClientCreator(
        db_host,
        volume_shape=volume_shape,
        crop_tensor=cfg.client_creator.crop_tensor
    )

    # Set database parameters from config
    client_creator.set_database(cfg.mongo.dbname)
    client_creator.set_collection(cfg.mongo.collection)
    client_creator.set_shape(volume_shape)

    # Prepare hyperparameters
    hparams = OmegaConf.to_container(cfg, resolve=True)

    # Initialize and run training
    runner = CustomRunner(
        logdir=logdir,
        wandb_project=cfg.wandb.project,
        wandb_experiment=f"resnet3d_{cfg.model.base_channels}base",
        model_path=model_path,
        n_channels=model_channels,
        n_classes=n_classes,
        n_epochs=cfg.experiment.epochs_code[0],  # Use first epoch value
        client_creator=client_creator,
        indexid=cfg.mongo.index_id,
        db_collection=cfg.mongo.collection,
        db_name=cfg.mongo.dbname,
        db_fields=(cfg.mongo.datafield, cfg.mongo.labelfield),
        subvolume_shape=volume_shape,
        db_host=db_host,
        wandb_team=cfg.wandb.team,
        hparams=hparams
    )
    
    runner.run()

if __name__ == "__main__":
    main()
