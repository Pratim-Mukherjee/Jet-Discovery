#!/usr/bin/env python
"""
Download the Top Quark Tagging dataset from Zenodo.
"""
import os
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile

ZENODO_RECORD = "2603256"
BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

# Files to download
FILES = {
    "train-raw.parquet": "train-raw.parquet",
    "train-labels.parquet": "train-labels.parquet",
    "validation-raw.parquet": "validation-raw.parquet",
    "validation-labels.parquet": "validation-labels.parquet",
    "test-raw.parquet": "test-raw.parquet",
    "test-labels.parquet": "test-labels.parquet",
}

def download_file(url, dest_path):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print("Downloading Top Quark Tagging dataset from Zenodo...")
    print(f"Record: {ZENODO_RECORD}")
    
    for filename, dest in FILES.items():
        dest_path = data_dir / dest
        if dest_path.exists():
            print(f"  {filename} already exists, skipping...")
            continue
        
        url = f"{BASE_URL}/{filename}"
        print(f"  Downloading {filename}...")
        try:
            download_file(url, dest_path)
        except Exception as e:
            print(f"  Error downloading {filename}: {e}")
            print("  Please download manually from https://zenodo.org/records/2603256")
    
    print("\nDownload complete!")

if __name__ == "__main__":
    main()