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
MODEL_PATH = Path("models/sam_vit_b_01ec64.pth")  # Changed to ViT-B model
TEMP_DIR = Path("temp")
MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"  # Changed URL
CHUNK_SIZE = 1024 * 1024  # 1MB chunks
EXPECTED_SIZE = 375001389  # Updated size for ViT-B model

def download_with_progress(url, destination, chunk_size=CHUNK_SIZE):
    """Download file with progress bar"""
    try:
        # Create parent directory if it doesn't exist
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup temporary file
        temp_path = destination.with_suffix('.temp')
        
        # Download with progress bar
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        with open(temp_path, 'wb') as f:
            downloaded_size = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    progress = min(downloaded_size / total_size, 1.0)
                    progress_bar.progress(progress)
                    progress_text.text(f"Downloading SAM model... {downloaded_size/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB")
        
        # Move temp file to final destination
        shutil.move(str(temp_path), str(destination))
        progress_text.text("Download completed successfully!")
        return True
        
    except Exception as e:
        st.error(f"Download error: {str(e)}")
        if temp_path.exists():
            temp_path.unlink()
        return False

@st.cache_resource
def load_models():
    """Load YOLO and SAM models"""
    try:
        # Download model if needed
        if not MODEL_PATH.exists():
            if not download_with_progress(MODEL_URL, MODEL_PATH):
                raise RuntimeError("Failed to download SAM model")
        
        # Load YOLO model
        yolo_model = YOLO('yolov8n.pt')
        
        # Load SAM model
        model_type = "vit_b"  # Changed to vit_b
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        sam = sam_model_registry[model_type](checkpoint=str(MODEL_PATH))
        sam.to(device=device)
        predictor = SamPredictor(sam)
        
        return yolo_model, predictor
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        # Clean up potentially corrupted file
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        raise

def process_frame(frame, predictor, selected_objects, yolo_model):
    """Process a single frame"""
    results = yolo_model(frame)
    
    # Create mask for selected objects
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    predictor.set_image(frame)
    
    for r in results:
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            if yolo_model.names[int(cls)] in selected_objects:
                box = box.cpu().numpy().astype(int)
                input_box = np.array([box[0], box[1], box[2], box[3]])
                
                masks, _, _ = predictor.predict(
                    box=input_box,
                    multimask_output=False
                )
                mask = np.logical_or(mask, masks[0]).astype(np.uint8) * 255
    
    # Remove objects using inpainting
    if np.any(mask):
        return cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
    return frame

def process_video(video_path, selected_objects, yolo_model, predictor):
    """Process video and remove selected objects"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer
    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4').name
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(temp_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    progress_bar = st.progress(0)
    frame_text = st.empty()
    
    try:
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame = process_frame(frame, predictor, selected_objects, yolo_model)
            out.write(processed_frame)
            
            # Update progress
            progress = (frame_idx + 1) / total_frames
            progress_bar.progress(progress)
            frame_text.text(f"Processing frame {frame_idx + 1} of {total_frames}")
            
    finally:
        cap.release()
        out.release()
    
    return temp_output

def main():
    st.title("Video Object Removal App")
    st.write("Upload a video and select objects to remove")
    
    try:
        # Load models
        yolo_model, predictor = load_models()
        
        # File uploader
        video_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if video_file:
            # Save uploaded video
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_video.write(video_file.read())
            
            # Show original video
            st.video(temp_video.name)
            
            # Detect objects in first frame
            cap = cv2.VideoCapture(temp_video.name)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Get available objects
                results = yolo_model(frame)
                detected_objects = []
                for r in results:
                    for cls in r.boxes.cls:
                        obj_name = yolo_model.names[int(cls)]
                        if obj_name not in detected_objects:
                            detected_objects.append(obj_name)
                
                # Object selection
                selected_objects = st.multiselect(
                    "Select objects to remove",
                    options=detected_objects
                )
                
                if selected_objects and st.button("Process Video"):
                    with st.spinner("Processing video..."):
                        output_path = process_video(temp_video.name, selected_objects, yolo_model, predictor)
                        st.success("Processing complete!")
                        st.video(output_path)
                        
                        # Cleanup
                        os.unlink(temp_video.name)
                        os.unlink(output_path)
            
    except Exception as e:
        st.error("An error occurred. Please try again.")
        st.error(f"Error details: {str(e)}")

if __name__ == "__main__":
    main()