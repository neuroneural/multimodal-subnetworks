from mindfultensors.mongoloader import MongoDataset
import torch
from mindfultensors.utils import unit_interval_normalize

class CustomMongoDataset(MongoDataset):
    """
    CustomMongoDataset is designed to work with the new "MindfulTensors" databasde organization.
    Gender labels there are stored in the .meta collection, while the 3D image data is stored in the .bin collection.
    This class is designed to fetch SINGLE MODALITY and SINGLE LABEL from both collections and returns them appropriately.
    USE MultimodalMongoDataset TO pull MULTIPLE modalities per subject.
    """
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

        # Batched .meta query for all subjects in batch — one query instead of N
        meta_results = {
            doc[self.id]: doc[self.meta_sample[0]]
            for doc in self.collection["meta"].find(
                {self.id: {"$in": [self.indices[_] for _ in batch]}},
                {self.id: 1, self.meta_sample[0]: 1, "_id": 0}
            )
        }

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

            #assert len(meta_for_id) != 0, f"No meta entries found for id {id}"
            #assert len(meta_for_id) < 2, f"More than one meta entry found for id {id}"
            assert self.indices[id] in meta_results, f"No meta entries found for id {id}"

            
            label = meta_results[self.indices[id]] #meta_for_id[0][self.meta_sample[0]]

            # Add to results
            results[id] = {
                "input": self.normalize(self.transform(data).float()),
                "label": torch.tensor(label).unsqueeze(0),
            }

        return results
    
class MultimodalMongoDataset(MongoDataset):
    """
    MultimodalMongoDataset is designed to work with the new "MindfulTensors" databasde organization.
    Gender labels there are stored in the .meta collection, while the 3D image data is stored in the .bin collection.
    This class is designed to fetch SINGLE OR MULTIPLE MODALITIES, MODALITY CODE, and SINGLE LABEL.
    Modality codes are assigned using the following dictionary:
    modality_mapping = {
        "smri": 0,
        "falff": 1,
        "dwi": 2,
    }
    """
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
        super(MultimodalMongoDataset, self).__init__(indices, transform, collection, sample, normalize, id)
        self.meta_sample = meta_sample

    def __getitem__(self, batch):
        # Fetch all samples for ids in the batch and where 'kind' is either
        # data or label as specified by the sample parameter
        # TODO: make it respect the batch size; right now it returns a bigger batch with multiple modalities per id
        
        samples = list(
            self.collection["bin"].find(
                {
                    self.id: {"$in": [self.indices[_] for _ in batch]},
                    "kind": {"$in": self.sample}, # .bin contains 3D kinds like 'smri', 'falff', 'dwi'. Scalar labels are stored in .meta
                },
                self.fields,
            )
        )

        # Batched .meta query for all subjects in batch — one query instead of N
        meta_results = {
            doc[self.id]: {
                self.meta_sample[0]: doc[self.meta_sample[0]],
                "modalities": doc["modalities"],
            }
            for doc in self.collection["meta"].find(
                {self.id: {"$in": [self.indices[_] for _ in batch]}},
                {self.id: 1, self.meta_sample[0]: 1, "modalities": 1, "_id": 0}
            )
        }


        results = {}
        for id in batch:
            # get ID's label and modalities
            

            assert self.indices[id] in meta_results, f"No meta entries found for id {id}"
            meta = meta_results[self.indices[id]]
            label = meta[self.meta_sample[0]]

            modalities =  meta["modalities"] #meta_for_id[0]["modalities"]
            id_modalities = set(modalities).intersection(set(self.sample))
                

            # Get samples for this ID
            samples_for_id = [
                sample
                for sample in samples
                if sample[self.id] == self.indices[id]
            ]

            for mod in id_modalities:
                data = self.make_serial(samples_for_id, mod)

                result = {
                    "input": self.normalize(self.transform(data).float()),
                    "modality": mod,
                    "label": torch.tensor(label).unsqueeze(0),
                }


                # Add to results
                results[str(id)+'_'+mod] = result

        return results
    

def multimodal_collate(results, field=("input", "modality", "label")):
    """
    Use this collate function with BatchPrefetchLoaderWrapper when using MultimodalMongoDataset.
    It will stack the inputs, modality codes, and labels into tensors properly.
    """
    results = results[0]
    # Assuming 'results' is your dictionary containing all the data
    input_tensors = [results[id_][field[0]] for id_ in results.keys()]
    modalities = [results[id_][field[1]] for id_ in results.keys()]
    label_tensors = [results[id_][field[2]] for id_ in results.keys()]
    # Stack all input tensors into a single tensor
    stacked_inputs = torch.stack(input_tensors)
    # Stack all label tensors into a single tensor
    stacked_modalities = torch.stack([torch.tensor(map_modality_codes(mod)) for mod in modalities])
    stacked_labels = torch.stack(label_tensors)
    return stacked_inputs.unsqueeze(1), stacked_modalities.long(), stacked_labels.long()

def map_modality_codes(mod):
    """
    Maps modality strings to integer codes.
    """
    modality_mapping = {
        "smri": 0,
        "falff": 1,
        "dwi": 2,
    }
    return modality_mapping[mod]

def make_serial(samples_for_id, kind):
    """
    Serializes chunks into a single binary blob. From MongoDataset self methods.
    """
    return b"".join(
        [
            sample["chunk"]
            for sample in sorted(
                (
                    sample
                    for sample in samples_for_id
                    if sample["kind"] == kind
                ),
                key=lambda x: x["chunk_id"],
            )
        ]
    )