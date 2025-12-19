from mindfultensors.mongoloader import MongoDataset
import torch
from mindfultensors.utils import unit_interval_normalize

class CustomMongoDataset(MongoDataset):
    def __init__(
        self,
        indices,
        transform,
        collection,
        sample,
        meta_sample,
        normalize=unit_interval_normalize,
        id="id",
    ):
        super(CustomMongoDataset, self).__init__(indices, transform, collection, sample, normalize, id)
        self.meta_sample = meta_sample

    def __getitem__(self, batch):
        # Fetch all samples for ids in the batch and where 'kind' is either
        # data or label as specified by the sample parameter
        
        samples = list(
            self.collection["bin"].find(
                {
                    self.id: {"$in": [self.indices[_] for _ in batch]},
                    "kind": {"$in": self.sample}, # .bin contains 3D kinds like 'smri', 'falff', 'dwi'. Scalar labels are stored in .meta
                },
                self.fields,
            )
        )

        results = {}
        for id in batch:
            # Separate samples for this id
            samples_for_id = [
                sample
                for sample in samples
                if sample[self.id] == self.indices[id]
            ]

            # Separate processing for each 'kind' # TODO: for multimodal, pull all kinds here and then just match them with labels properly
            data = self.make_serial(samples_for_id, self.sample[0])

            meta_for_id = list(
                self.collection["meta"].find(
                    {
                        self.id: self.indices[id],
                    },
                    self.meta_sample,
                )
            )

            assert len(meta_for_id) != 0, f"No meta entries found for id {id}"
            assert len(meta_for_id) < 2, f"More than one meta entry found for id {id}"
            
            label = meta_for_id[0][self.meta_sample[0]]

            # Add to results
            results[id] = {
                "input": self.normalize(self.transform(data).float()),
                "label": torch.tensor(label).unsqueeze(0),
            }

        return results