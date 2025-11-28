import os
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    dataset_name = "paultimothymooney/chest-xray-pneumonia"
    download_path = "data/raw"
    
    print(f"Checking Kaggle configuration...")
    
    # Check for credentials
    if not (os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY')) and not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
        print("\nERROR: Kaggle credentials not found!")
        print("To download the dataset, you need to set up the Kaggle API.")
        print("\nOption 1: Place 'kaggle.json' in ~/.kaggle/")
        print("Option 2: Set KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
        print("\nYou can get your API token from: https://www.kaggle.com/settings -> API -> Create New Token")
        sys.exit(1)

    print(f"Authenticating with Kaggle...")
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"Error authenticating: {e}")
        print("Please check your credentials.")
        sys.exit(1)

    print(f"Downloading {dataset_name} to {download_path}...")
    if not os.path.exists(download_path):
        os.makedirs(download_path)
        
    try:
        api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        print("Download complete.")
        print(f"Data is ready in {download_path}")
        
    except Exception as e:
        print(f"Error downloading: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_dataset()
