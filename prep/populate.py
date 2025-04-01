import os
import nibabel as nib
import numpy as np
from glob import glob
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor
from dask import delayed, compute
import bson
import io
import torch
import pandas as pd

print("=== Starting FBIRN Data Pipeline ===")

# Configuration
MONGOHOST = "10.245.12.58"
MONGODB = "multimodalSubnetworks"
BASE_PATH = "/data/users2/jwardell1/multimodal-subnetworks/groupedData"
CHUNK_SIZE = 10  # MB
DEMOGRAPHICS_FILE = 'FBIRN_PANSS_Details.xlsx'

# Initialize MongoDB Connection
print(f"\nConnecting to MongoDB at {MONGOHOST}...")
try:
    client = MongoClient(f"{MONGOHOST}:27017", serverSelectionTimeoutMS=5000)
    client.server_info()  # Test connection
    db = client[MONGODB]
    print("✓ MongoDB connection established")
except Exception as e:
    print(f"✗ MongoDB connection failed: {e}")
    exit(1)

# Load and Process Demographic Data
print(f"\nLoading demographics from {DEMOGRAPHICS_FILE}...")
try:
    demographics_df = pd.read_excel(DEMOGRAPHICS_FILE)
    print(f"✓ Loaded {len(demographics_df)} records")
    
    # Standardize gender values
    demographics_df['gender_clean'] = (
        demographics_df['Demographics_sDEMOG_GENDER']
        .str.upper()
        .str.strip()
        .replace({'MALE': 'M', 'FEMALE': 'F'})
    )
    
    # Encode gender numerically
    gender_encoded = np.where(
        demographics_df['gender_clean'] == 'M', 0,
        np.where(
            demographics_df['gender_clean'] == 'F', 1,
            -1  # Invalid
        )
    ).astype(np.uint8)
    
    print("Gender Distribution:")
    print(pd.Series(gender_encoded).value_counts()
          .rename(index={0: 'Male', 1: 'Female', -1: 'Invalid'}))
    
    # Create ID mapping
    gender_mapping = {}
    for idx, row in demographics_df.iterrows():
        original_id = str(row['SubjectID']).strip()
        clean_id = original_id.lstrip('0')  # Remove leading zeros
        
        # Store gender data
        gender_data = {
            'original': row['Demographics_sDEMOG_GENDER'],
            'encoded': int(gender_encoded[idx])  # Convert numpy to native int
        }
        
        # Map both ID formats
        gender_mapping[original_id] = gender_data
        if clean_id != original_id:
            gender_mapping[clean_id] = gender_data
    
    print(f"\n✓ Created gender mapping for {len(demographics_df)} subjects")
    print("Sample mappings:", {k: gender_mapping[k] for k in list(gender_mapping.keys())[:3]})

except Exception as e:
    print(f"✗ Failed to process demographics: {e}")
    exit(1)

# Core Functions
def qnormalize(img, qmin=0.02, qmax=0.98):
    """Quantile normalization with clipping"""
    qlow = np.quantile(img, qmin)
    qhigh = np.quantile(img, qmax)
    img = (img - qlow) / (qhigh - qlow)
    return np.clip(img, 0, 1)

def tensor2bin(tensor):
    """Serialize tensor to binary"""
    buffer = io.BytesIO()
    torch.save(tensor.to(torch.uint8), buffer)
    return buffer.getvalue()

def chunk_binobj(tensor_compressed, idx, kind, chunksize):
    """Split binary data into chunks"""
    chunksize_bytes = chunksize * 1024 * 1024
    num_chunks = len(tensor_compressed) // chunksize_bytes + 1
    for i in range(num_chunks):
        start = i * chunksize_bytes
        end = min((i + 1) * chunksize_bytes, len(tensor_compressed))
        yield {
            "id": idx,
            "chunk_id": i,
            "kind": kind,
            "chunk": bson.Binary(tensor_compressed[start:end]),
        }

def process_subject(image_path, collection_name, id):
    try:
        print(f"\n--- Processing Subject {id} ---")
        print(f"Image: {os.path.basename(image_path)}")
        
        # Load and process image
        img = np.asarray(nib.load(image_path).dataobj)
        img_norm = (qnormalize(img) * 255).astype(np.uint8)
        img_bin = tensor2bin(torch.from_numpy(img_norm))
        print(f"✓ Processed image ({len(img_bin)/1024:.1f} KB binary)")
        
        # Extract and match ID
        filename = os.path.basename(image_path)
        file_id = filename.split('_')[0]
        clean_id = file_id.lstrip('0')
        
        print(f"Extracted IDs: File={file_id}, Cleaned={clean_id}")
        
        # Find matching gender data
        gender_data = None
        matching_id = None
        for id_format in [file_id, clean_id]:
            if id_format in gender_mapping:
                gender_data = gender_mapping[id_format]
                matching_id = id_format
                print(f"✓ Matched using ID format: {id_format}")
                break
        
        if not gender_data:
            available_ids = list(gender_mapping.keys())[:5]
            raise ValueError(f"No gender match. Tried: {file_id}, {clean_id}\nFirst 5 available: {available_ids}")
        
        if gender_data['encoded'] not in {0, 1}:
            raise ValueError(f"Invalid gender code: {gender_data}")
        
        print(f"Gender: {gender_data}")
        
        # Create metadata document
        meta = {
            "id": id,
            "collection": collection_name,
            "filename": filename,
            "id_formats": {
                "from_filename": file_id,
                "cleaned": clean_id,
                "matched": matching_id
            },
            "gender": gender_data,
            "data_types": {
                "image": "Normalized (0-255 uint8)",
                "label": "Binary (0: Male, 1: Female, uint8)"
            },
            "processing_log": {
                "normalization": "Quantile (0.02-0.98)",
                "chunk_size_mb": CHUNK_SIZE,
                "status": "processed"
            }
        }

        # MongoDB operations
        col_bin = db[f"{collection_name}.bin"]
        col_meta = db[f"{collection_name}.meta"]
        
        print("Inserting metadata...")
        meta_result = col_meta.insert_one(meta)
        print(f"✓ Metadata inserted: {meta_result.inserted_id}")
        
        # Insert binary data
        gender_value = gender_data['encoded']  # From your existing code
        gender_tensor = torch.tensor([gender_value], dtype=torch.long)  # Wrap in list to make 1D
        gender_binary = gender_tensor.numpy().tobytes()

        gender_chunks = list(chunk_binobj(gender_binary, id, "gender", CHUNK_SIZE))
        img_chunks = list(chunk_binobj(img_bin, id, "image", CHUNK_SIZE))   
        print(f"Inserting {len(img_chunks)} binary chunks...")
        bin_result = col_bin.insert_many(img_chunks + gender_chunks)
        print(f"✓ Inserted {len(bin_result.inserted_ids)} chunks")
        
        return True
        
    except Exception as e:
        print(f"✗ Processing failed: {str(e)}")
        return False



# Main Execution
if __name__ == "__main__":
    print("\n=== Starting Main Processing ===")
    
    # Process each collection
    collections = [d for d in os.listdir(BASE_PATH) 
                  if os.path.isdir(os.path.join(BASE_PATH, d))]
    
    for collection in collections:
        print(f"\nProcessing Collection: {collection}")
        image_dir = os.path.join(BASE_PATH, collection, "images")
        images = glob(os.path.join(image_dir, "*.nii"))
        if not images:
            images = glob(os.path.join(image_dir, "*.nii.gz"))
        
        print(f"Found {len(images)} images")
        if not images:
            print("✗ No images found! Check file pattern.")
            continue
            
        # Process in parallel
        with ThreadPoolExecutor() as executor:
            futures = []
            for idx, img_path in enumerate(images):
                futures.append(executor.submit(
                    process_subject, 
                    img_path, 
                    collection, 
                    idx
                ))
            
            # Track progress
            results = [f.result() for f in futures]
            success_rate = sum(results)/len(results)
            print(f"\nCollection {collection} Results:")
            print(f"Success: {sum(results)}/{len(results)} ({success_rate:.1%})")
    
    print("\n=== Processing Complete ===")
    print("Final checks:")
    print(f"Metadata documents: {db['fbirn_falff.meta'].count_documents({})}")
    print(f"Binary chunks: {db['fbirn_falff.bin'].count_documents({})}")
