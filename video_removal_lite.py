import streamlit as st
import cv2
import numpy as np
import tempfile
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import os
from pathlib import Path
import requests
import hashlib
import shutil

# Constants
MODEL_PATH = Path("models/sam_vit_h_4b8939.pth")
TEMP_DIR = Path("temp")
MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
CHUNK_SIZE = 1024 * 1024  # 1MB chunks
EXPECTED_SIZE = 2564500325  # Expected file size in bytes

def calculate_sha256(file_path):
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def download_with_progress(url, destination, chunk_size=CHUNK_SIZE):
    """Download file with progress bar and size verification"""
    try:
        # Create temporary download path
        temp_path = destination.with_suffix('.temp')
        
        # Check if partial download exists
        initial_size = temp_path.stat().st_size if temp_path.exists() else 0
        
        # Setup download headers for resume
        headers = {'Range': f'bytes={initial_size}-'} if initial_size > 0 else {}
        
        # Make request
        response = requests.get(url, stream=True, headers=headers)
        total_size = int(response.headers.get('content-length', 0)) + initial_size
        
        # Create progress bar
        progress_text = st.empty()
        progress_bar = st.progress(0)
        progress_text.text("Downloading SAM model... This may take a few minutes.")
        
        # Open file in append mode if resuming, write mode if new
        mode = 'ab' if initial_size > 0 else 'wb'
        with open(temp_path, mode) as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    current_size = f.tell()
                    progress = min(current_size / EXPECTED_SIZE, 1.0)
                    progress_bar.progress(progress)
                    progress_text.text(f"Downloading... {current_size/1024/1024:.1f}MB / {EXPECTED_SIZE/1024/1024:.1f}MB")
        
        # Verify file size
        if temp_path.stat().st_size != EXPECTED_SIZE:
            raise ValueError("Downloaded file size does not match expected size")
        
        # Move temp file to final destination
        shutil.move(str(temp_path), str(destination))
        progress_text.text("Download completed successfully!")
        return True
        
    except Exception as e:
        st.error(f"Download error: {str(e)}")
        # Clean up partial download
        if temp_path.exists():
            temp_path.unlink()
        return False

def ensure_model_downloaded():
    """Ensure model is downloaded and valid"""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if model exists and has correct size
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size == EXPECTED_SIZE:
        return True
    
    # Remove potentially corrupted file
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
    
    # Download model
    return download_with_progress(MODEL_URL, MODEL_PATH)

@st.cache_resource
def load_models():
    """Load YOLO and SAM models"""
    try:
        # Ensure model is downloaded
        if not ensure_model_downloaded():
            raise RuntimeError("Failed to download SAM model")
        
        # Load YOLO model
        yolo_model = YOLO('yolov8n.pt')
        
        # Load SAM model
        model_type = "vit_h"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        sam = sam_model_registry[model_type](checkpoint=str(MODEL_PATH))
        sam.to(device=device)
        predictor = SamPredictor(sam)
        
        return yolo_model, predictor
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        raise

def main():
    st.title("Video Object Removal App")
    st.write("Upload a video and select objects to remove")
    
    try:
        # Load models
        yolo_model, predictor = load_models()
        
        # Create temp directory if it doesn't exist
        TEMP_DIR.mkdir(exist_ok=True)
        
        # Rest of your application code here
        uploaded_file = st.file_uploader("Upload video file", type=["mp4", "mov", "avi", "mkv"])
        
        if uploaded_file is not None:
            # Process video code here
            st.write("Video uploaded successfully! Processing...")
            
    except Exception as e:
        st.error("An error occurred. Please try refreshing the page.")
        st.error(f"Error details: {str(e)}")

if __name__ == "__main__":
    main()