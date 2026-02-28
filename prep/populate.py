import os
import io
import glob
import torch
import nibabel as nib
import numpy as np
import bson
import pandas as pd
from pymongo import MongoClient

# config
MONGO_HOST = "10.245.12.58"
DB_NAME = "multimodalSubnetworks"
BIN_COLLECTION = "ukb.bin"
META_COLLECTION = "ukb.meta"
FALFF_DIR = "/data/users2/jwardell1/multimodal-subnetworks/groupedData/ukb_falff/images/conformed"
SMRI_DIR = "/data/users2/jwardell1/multimodal-subnetworks/groupedData/ukb_smri/images/conformed"
DWI_DIR = "/data/users2/jwardell1/multimodal-subnetworks/groupedData/ukb_dwi/images/conformed"
DEMOGRAPHICS_CSV = "/data/users2/jwardell1/multimodal-subnetworks/prep/my_ukb_data.csv"
CHUNK_SIZE_MB = 10

def qnormalize(img, qmin=0.01, qmax=0.99):
    img = (img - img.quantile(qmin)) / (img.quantile(qmax) - img.quantile(qmin))
    return img

def tensor2bin(tensor):
    tensor_1d = (tensor * 255).clamp(0, 255).to(torch.uint8)
    buffer = io.BytesIO()
    torch.save(tensor_1d, buffer)
    return buffer.getvalue()

def chunk_binobj(tensor_binary, idx, subject_id, kind, chunksize):
    chunksize_bytes = chunksize * 1024 * 1024
    num_chunks = len(tensor_binary) // chunksize_bytes
    if len(tensor_binary) % chunksize_bytes != 0:
        num_chunks += 1
    chunks = []
    for i in range(num_chunks):
        start = i * chunksize_bytes
        end = min((i + 1) * chunksize_bytes, len(tensor_binary))
        chunk = tensor_binary[start:end]
        chunks.append({
            "id": idx,
            "subject_id": subject_id,
            "chunk_id": i,
            "kind": kind,
            "chunk": bson.Binary(chunk),
        })
    return chunks

def load_nifti(filepath):
    img = nib.load(filepath)
    data = img.get_fdata(dtype=np.float32)
    tensor = torch.from_numpy(data)
    tensor = qnormalize(tensor)
    return tensor

def load_demographics_for_subjects(csv_path, subject_ids):
    """Load demographics only for specific subjects"""
    print(f"loading demographics for {len(subject_ids)} subjects...")
    
    # Read only needed columns
    df = pd.read_csv(csv_path, usecols=['eid', 'sex_f31_0_0'])
    
    # Convert to set for faster lookup
    subject_set = set(subject_ids)
    
    demo_dict = {}
    for _, row in df.iterrows():
        sub_id = str(int(row['eid']))
        if sub_id not in subject_set:
            continue
            
        gender_code = row['sex_f31_0_0']
        if pd.isna(gender_code):
            continue
        gender_code = int(gender_code)
        
        if gender_code == 0:
            gender = 'M'
            gender_encoded = 0
        elif gender_code == 1:
            gender = 'F'
            gender_encoded = 1
        else:
            continue
        demo_dict[sub_id] = {'gender': gender, 'gender_encoded': gender_encoded}
    
    print(f"found demographics for {len(demo_dict)} subjects")
    return demo_dict

def main():
    client = MongoClient(MONGO_HOST)
    db = client[DB_NAME]
    bin_coll = db[BIN_COLLECTION]
    meta_coll = db[META_COLLECTION]

    # Drop and recreate collections
    print("Dropping existing collections...")
    bin_coll.drop()
    meta_coll.drop()

    # Get all subjects with fALFF
    print("\nFinding fALFF subjects...")
    falff_files = glob.glob(os.path.join(FALFF_DIR, "*_fALFF.nii"))
    falff_subjects = {os.path.basename(f).replace("_fALFF.nii", ""): f for f in falff_files}
    print(f"found {len(falff_subjects)} fALFF subjects")

    # Get all subjects with sMRI
    print("Finding sMRI subjects...")
    smri_files = glob.glob(os.path.join(SMRI_DIR, "*_smri.nii.gz"))
    smri_subjects = {os.path.basename(f).replace("_smri.nii.gz", ""): f for f in smri_files}
    print(f"found {len(smri_subjects)} sMRI subjects")

    # Get all subjects with DWI
    print("Finding DWI subjects...")
    dwi_files = glob.glob(os.path.join(DWI_DIR, "*_dwi.nii.gz"))
    dwi_subjects = {os.path.basename(f).replace("_dwi.nii.gz", ""): f for f in dwi_files}
    print(f"found {len(dwi_subjects)} DWI subjects")

    # Get union of all subjects
    all_subjects = sorted(set(falff_subjects.keys()) | set(smri_subjects.keys()) | set(dwi_subjects.keys()))
    print(f"total unique subjects: {len(all_subjects)}")

    # Load demographics only for these subjects
    demographics = load_demographics_for_subjects(DEMOGRAPHICS_CSV, all_subjects)

    # Filter to subjects with demographics
    subjects_with_demo = [s for s in all_subjects if s in demographics]
    print(f"subjects with demographics: {len(subjects_with_demo)}")

    # Process each subject
    inserted = 0
    skipped = 0

    for idx, subject_id in enumerate(subjects_with_demo):
        try:
            modalities = []
            bin_docs = []

            # Process fALFF if available
            if subject_id in falff_subjects:
                tensor = load_nifti(falff_subjects[subject_id])
                tensor_binary = tensor2bin(tensor)
                chunks = chunk_binobj(tensor_binary, idx, subject_id, "falff", CHUNK_SIZE_MB)
                bin_docs.extend(chunks)
                modalities.append("falff")

            # Process sMRI if available
            if subject_id in smri_subjects:
                tensor = load_nifti(smri_subjects[subject_id])
                tensor_binary = tensor2bin(tensor)
                chunks = chunk_binobj(tensor_binary, idx, subject_id, "smri", CHUNK_SIZE_MB)
                bin_docs.extend(chunks)
                modalities.append("smri")

            # Process DWI if available
            if subject_id in dwi_subjects:
                tensor = load_nifti(dwi_subjects[subject_id])
                tensor_binary = tensor2bin(tensor)
                chunks = chunk_binobj(tensor_binary, idx, subject_id, "dwi", CHUNK_SIZE_MB)
                bin_docs.extend(chunks)
                modalities.append("dwi")

            # Insert binary data
            if bin_docs:
                bin_coll.insert_many(bin_docs)

            # Create meta entry
            demo = demographics[subject_id]
            meta_doc = {
                "id": idx,
                "subject_id": subject_id,
                "gender": demo["gender"],
                "modalities": sorted(modalities),
                "data_types": {
                    "image": "Normalized (0-255 uint8)",
                    "label": "Binary (0: Male, 1: Female, uint8)"
                },
                "normalization": "Quantile (0.01-0.99)",
                "chunk_size_mb": CHUNK_SIZE_MB,
                "lz4_compressed": False,
                "gender_encoded": demo["gender_encoded"]
            }
            meta_coll.insert_one(meta_doc)

            inserted += 1
            if inserted % 100 == 0:
                print(f"[{idx}] inserted {inserted} subjects...")

        except Exception as e:
            print(f"ERROR on {subject_id}: {e}")
            skipped += 1

    print(f"\ndone. inserted={inserted}, skipped={skipped}")
    print(f"total bin documents: {bin_coll.count_documents({})}")
    print(f"total meta documents: {meta_coll.count_documents({})}")

if __name__ == "__main__":
    main()
