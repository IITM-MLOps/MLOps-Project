# download_mnist.py
import os
import urllib.request
import gzip
import shutil

def download_file(url, dest_path):
    """Download a file from a URL to a destination path."""
    if not os.path.exists(dest_path):
        print(f"Downloading {url} to {dest_path}...")
        urllib.request.urlretrieve(url, dest_path)
    else:
        print(f"File {dest_path} already exists, skipping download.")

def extract_gz(file_path, dest_path):
    """Extract a .gz file to a destination path."""
    if not os.path.exists(dest_path):
        print(f"Extracting {file_path} to {dest_path}...")
        with gzip.open(file_path, 'rb') as f_in:
            with open(dest_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        print(f"File {dest_path} already extracted, skipping.")

def download_mnist():
    """Download and extract MNIST dataset files."""
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = [
        ("train-images-idx3-ubyte.gz", "mnist/train-images.idx3-ubyte"),
        ("train-labels-idx1-ubyte.gz", "mnist/train-labels.idx1-ubyte"),
        ("t10k-images-idx3-ubyte.gz", "mnist/t10k-images.idx3-ubyte"),
        ("t10k-labels-idx1-ubyte.gz", "mnist/t10k-labels.idx1-ubyte")
    ]
    
    os.makedirs("mnist", exist_ok=True)
    
    for gz_file, dest_file in files:
        gz_path = f"mnist/{gz_file}"
        download_file(base_url + gz_file, gz_path)
        extract_gz(gz_path, dest_file)

if __name__ == "__main__":
    download_mnist()
