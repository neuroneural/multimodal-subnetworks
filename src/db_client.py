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

from src.utils import crop_tensor

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