import os
import requests
from pathlib import Path

def download_sam_model():
    model_path = Path("models/sam_vit_h_4b8939.pth")
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        print("Downloading SAM model...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete!")

if __name__ == "__main__":
    download_sam_model()