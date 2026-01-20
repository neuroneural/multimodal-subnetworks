from mindfultensors.gencoords import CoordsGenerator

from mindfultensors.mongoloader import (
    create_client,
    collate_subcubes,
    mcollate,
    MongoDataset,
    MongoClient,
    MongoheadDataset,
    mtransform,
)
import torch

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
            colname="fbirn_falff.bin",
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