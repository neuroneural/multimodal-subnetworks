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

from resnet import ResNet3D

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
# os.environ["NCCL_SOCKET_IFNAME"] = "ib0"
# os.environ["NCCL_P2P_LEVEL"] = "NVL"

torch_version = torch.__version__
if version.parse(torch_version) >= version.parse("2.3"):
    scaler = torch.amp.GradScaler()
else:
    scaler = torch.cuda.amp.GradScaler()


def qnormalize(img, qmin=0.02, qmax=0.98):
    """Unit interval preprocessing with clipping"""
    qlow = torch.quantile(img, qmin)
    qhigh = torch.quantile(img, qmax)
    img = (img - qlow) / (qhigh - qlow)
    img = torch.clamp(img, 0, 1)  # Clip the values to be between 0 and 1
    return img


def crop_tensor(tensor, label, percentile=10):

    # Use torch.quantile instead of kthvalue for potentially faster operation
    threshold = torch.quantile(tensor.flatten(), percentile / 100)

    # Create a mask on the original device
    mask = tensor > threshold

    # If the mask is all False, return the original tensors
    if not torch.any(mask):
        return tensor, label

    # Find the bounding box (this part is already efficient)
    nonzero = torch.nonzero(mask)
    min_coords, _ = torch.min(nonzero, dim=0)
    max_coords, _ = torch.max(nonzero, dim=0)

    # Crop the original tensor and label using the bounding box
    slices = tuple(
        slice(min_coord.item(), max_coord.item() + 1)
        for min_coord, max_coord in zip(min_coords[2:], max_coords[2:])
    )
    cropped_tensor = tensor[(slice(None), slice(None)) + slices]
    cropped_label = label[(slice(None),) + slices]

    return cropped_tensor, cropped_label


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
        modelconfig: str,
        db_host: str,
        db_name: str,
        db_collection: str,
        wandb_team: str,
        db_fields: tuple,
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
        self.config_file = modelconfig
        self.optimize_inline = optimize_inline
        self.onecycle_lr = onecycle_lr
        self.rmsprop_lr = rmsprop_lr
        self.prefetches = prefetches
        self.db_host = db_host
        self.db_name = db_name
        self.db_collection = db_collection
        self.db_fields = db_fields
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
            # "tensorboard": dl.TensorboardLogger(logdir=self._logdir,
            #                                     log_batch_metrics=True),
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
            "createVclient": self.client_creator.create_v_client,
            "mycollate": self.client_creator.mycollate,
            "mycollate_full": self.client_creator.mycollate_full,
            "mytransform": self.client_creator.mytransform,
        }

        self.collate = (
            self.funcs["mycollate_full"]
            if self.shape == 256
            else self.funcs["mycollate"]
        )

        client = MongoClient("mongodb://" + self.db_host + ":27017")
        db = client[self.db_name]
        posts = db[self.db_collection + ".bin"]
        num_examples = int(
            posts.find_one(sort=[(self.index_id, -1)])[self.index_id] + 1
        )

        tdataset = MongoheadDataset(
            range(num_examples),
            # [
            #     int(x)
            #     for x in np.random.permutation(
            #         list(np.random.randint(0, num_examples, 8)) * 100
            #     )
            # ],
            self.funcs["mytransform"],
            None,
            self.db_fields,
            normalize=unit_interval_normalize,
            id=self.index_id,
        )

        tsampler = (
            DBBatchSampler(tdataset, batch_size=self.num_volumes, seed=SEED)
            if self.engine.is_ddp
            else DBBatchSampler(tdataset, batch_size=self.num_volumes)
        )

        tdataloader = BatchPrefetchLoaderWrapper(
            DataLoader(
                tdataset,
                sampler=tsampler,
                collate_fn=self.collate,
                pin_memory=True,
                worker_init_fn=self.funcs["createclient"],
                persistent_workers=True,
                prefetch_factor=4,
                num_workers=4,  # self.prefetches,
            ),
            num_prefetches=self.prefetches,
        )

        vdataset = MongoDataset(
            range(32),
            self.funcs["mytransform"],
            None,
            self.db_fields,
            normalize=unit_interval_normalize,
            id=self.index_id,
        )

        vsampler = (
            DBBatchSampler(vdataset, batch_size=self.num_volumes, seed=SEED)
            if self.engine.is_ddp
            else DBBatchSampler(
                vdataset, batch_size=self.num_volumes, seed=SEED
            )
        )

        vdataloader = BatchPrefetchLoaderWrapper(
            DataLoader(
                vdataset,
                sampler=vsampler,
                collate_fn=self.collate,
                pin_memory=True,
                worker_init_fn=self.funcs["createVclient"],
                persistent_workers=True,
                prefetch_factor=2,
                num_workers=4,  # self.prefetches,
            ),
            num_prefetches=self.prefetches,
        )

        return {"train": tdataloader, "valid": vdataloader}

    def get_model(self):
        model = ResNet3D(
            in_channels=1, 
            n_classes=self.n_classes, 
            channels=self.n_channels
        )
        if self.model_path and os.path.exists(self.model_path):
            model.load_state_dict(torch.load(self.model_path))
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
            "metric_key": "accuracy",
            "loader_key": "valid",
            "minimize": False,
        }
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
            for key in ["loss", "accuracy"]
        }

    def on_loader_end(self, runner):
        for key in ["loss", "accuracy"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)



    # model train/valid step
    def handle_batch(self, batch):
        # Add synchronization before processing
        if self.engine.is_ddp:
            torch.cuda.synchronize()
        
        sample, label = batch
        # np.save("labels.npy", label.cpu().numpy())
        # np.save("input.npy", sample.cpu().numpy())
        # stop
        # run model forward/backward pass
        if self.model.training:
            if self.shape > self.maxshape:
                if self.engine.is_ddp:
                    with self.model.no_sync():
                        loss, y_hat = self.model.forward(
                            x=sample,
                            y=label,
                            loss=self.criterion,
                            verbose=False,
                        )
                    torch.distributed.barrier()
                else:
                    loss, y_hat = self.model.forward(
                        x=sample, y=label, loss=self.criterion, verbose=False
                    )
            else:
                if self.bit16:
                    with torch.amp.autocast(
                        device_type="cuda", dtype=torch.float16
                    ):
                        y_hat = self.model.forward(sample)
                        # print("y_hat.shape: ", y_hat.shape)
                        # print("label.shape: ", label.shape)
                        # stop

                        loss = self.criterion(y_hat, label)
                    scaler.scale(loss).backward()
                else:
                    y_hat = self.model.forward(sample)
                    loss = self.criterion(y_hat, label)
                    loss.backward()
            if not self.optimize_inline:
                if self.bit16:
                    scaler.step(self.optimizer)
                    self.scheduler.step()
                    scaler.update()
                    self.optimizer.zero_grad()
                else:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
        else:
            with torch.no_grad():
                y_hat = self.model.forward(sample)
                loss = self.criterion(y_hat, label)
        with torch.inference_mode():
            result = torch.squeeze(torch.argmax(y_hat, 1)).long()
            labels = torch.squeeze(label)

        # Metrics calculation
        with torch.no_grad():
            preds = torch.sigmoid(y_hat) > 0.5
            accuracy = (preds == label).float().mean()

        self.batch_metrics.update({
            "loss": loss,
            "accuracy": accuracy
        })

        del sample
        del label
        del y_hat
        del result
        del labels
        del loss


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
            dbname="multimodalSubnetworks",
            colname="fbirn_falff",
            mongohost=self.mongohost,
        )

    def mycollate(self, x):
        return collate_subcubes(
            x,
            self.coord_generator,
            samples=self.num_subcubes,
        )

    def mycollate_full(self, x):
        return crop_tensor(*mcollate(x)) if self.crop_tensor else mcollate(x)

    def mytransform(self, x):
        return mtransform(x)


def assert_equal_length(*args):
    assert all(
        len(arg) == len(args[0]) for arg in args
    ), "Not all parameter lists have the same length!"



@hydra.main(config_path="conf", config_name="resnet3d_gender_bn_64base_2.2.2.2_exp01", version_base=None)
def main(cfg: DictConfig):
    # Loading common parameters
    # Model parameters
    volume_shape = cfg.model.volume_shape
    n_classes = cfg.model.n_classes
    config_file = cfg.model.config_file
    optimize_inline = cfg.model.optimize_inline
    model_channels = cfg.model.model_channels
    model_label = cfg.model.model_label
    use_groupnorm = cfg.model.use_groupnorm
    model_path = cfg.paths.model if cfg.paths.loadcheckpoint else ""
    logdir = cfg.paths.logdir
    db_host = cfg.mongo.host_slurm if os.environ.get("SLURM_JOB_ID") else cfg.mongo.host

    # MongoDB parameters
    validation_percent = cfg.mongo.validation_percent

    wandb_project = cfg.wandb.project

    bit16 = cfg.bit16

    client_creator = ClientCreator(
        db_host, crop_tensor=cfg.client_creator.crop_tensor
    )

    # Specify curriculum parameters
    # Set up the environment for eval
    context = {"maxreps": cfg.experiment.maxreps}

    # Evaluate the Python code from the YAML config
    cubesizes = eval(cfg.experiment.cubesizes_code, globals(), context)
    numcubes = eval(cfg.experiment.numcubes_code, globals(), context)
    numvolumes = eval(cfg.experiment.numvolumes_code, globals(), context)
    weights = eval(cfg.experiment.weights_code, globals(), context)
    databases = eval(cfg.experiment.databases_code, globals(), context)
    collections = eval(cfg.experiment.collections_code, globals(), context)
    dbfields = eval(cfg.experiment.dbfields_code, globals(), context)
    epochs = eval(cfg.experiment.epochs_code, globals(), context)
    prefetches = eval(cfg.experiment.prefetches_code, globals(), context)
    attenuates = eval(cfg.experiment.attenuates_code, globals(), context)

    assert_equal_length(
        cubesizes,
        numcubes,
        numvolumes,
        weights,
        databases,
        collections,
        epochs,
        prefetches,
        attenuates,
    )

    start_experiment = 0
    for experiment in range(len(cubesizes)):
        subvolume_shape = [cubesizes[experiment]] * 3
        onecycle_lr = rmsprop_lr = (
            attenuates[experiment] ** experiment
            * 8
            * cfg.experiment.lr_scale
            * numcubes[experiment]
            * numvolumes[experiment]
            / 256
        )
        wandb_experiment = (
            f"{start_experiment + experiment:02} cube "
            + str(subvolume_shape[0])
            + " "
            + collections[experiment]
            + model_label
        )

        # Set database parameters
        client_creator.set_database(databases[experiment])
        client_creator.set_collection(collections[experiment])
        client_creator.set_num_subcubes(numcubes[experiment])
        client_creator.set_shape(subvolume_shape)

        with open(cfg.model.config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
            hparams = {"model_arch": config_dict, **OmegaConf.to_container(cfg)}

        runner = CustomRunner(
            logdir=logdir,
            wandb_project=wandb_project,
            wandb_experiment=wandb_experiment,
            model_path=model_path,
            n_channels=model_channels,
            n_classes=n_classes,
            modelconfig=config_file,
            n_epochs=epochs[experiment],
            optimize_inline=optimize_inline,
            validation_percent=validation_percent,
            onecycle_lr=onecycle_lr,
            rmsprop_lr=rmsprop_lr,
            num_subcubes=numcubes[experiment],
            num_volumes=numvolumes[experiment],
            groupnorm=use_groupnorm,
            client_creator=client_creator,
            off_brain_weight=weights[experiment],
            prefetches=prefetches[experiment],
            indexid=cfg.mongo.index_id,
            db_collection=collections[experiment],
            db_name=databases[experiment],
            db_fields=dbfields[experiment],
            subvolume_shape=subvolume_shape,
            lowprecision=bit16,
            lossweight = [w / sum(cfg.model.loss_weight) for w in cfg.model.loss_weight] if sum(cfg.model.loss_weight) != 0 else ValueError("The sum of loss weights cannot be zero."),
            db_host=db_host,
            wandb_team=cfg.wandb.team,
            maxshape=cfg.model.maxshape,
            hparams=hparams,
        )
        runner.run()

        shutil.copy(
            logdir + "/model.last.pth",
            logdir
            + "/model.last."
            + str(subvolume_shape[0])
            + f".run{experiment:02}.curriculum.pth",
        )

        model_path = logdir + "model.last.pth"


if __name__ == "__main__":
    main()