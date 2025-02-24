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
import asyncio
import platform

# Constants
MODEL_PATH = Path("models/sam_vit_b_01ec64.pth")
TEMP_DIR = Path("temp")
MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
CHUNK_SIZE = 1024 * 1024
EXPECTED_SIZE = 375001389

# Video constraints
MAX_VIDEO_DURATION = 3
TARGET_FPS = 5
MAX_FILE_SIZE = 50 * 1024 * 1024

def download_with_progress(url, destination, chunk_size=CHUNK_SIZE):
    """Download file with progress bar"""
    try:
        # Create parent directory if it doesn't exist
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination.with_suffix('.temp')
        
        # Make HTTP request
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise exception for bad status codes
        total_size = int(response.headers.get('content-length', 0))
        
        # Set up progress tracking
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        # Download file in chunks
        with open(temp_path, 'wb') as f:
            downloaded_size = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    progress = min(downloaded_size / total_size, 1.0)
                    progress_bar.progress(progress)
                    progress_text.text(f"تحميل نموذج SAM... {downloaded_size/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB")
        
        # Move temp file to final destination
        shutil.move(str(temp_path), str(destination))
        progress_text.text("اكتمل التحميل بنجاح!")
        return True
        
    except Exception as e:
        st.error(f"خطأ في التحميل: {str(e)}")
        if temp_path.exists():
            temp_path.unlink()
        return False

def init_torch():
    """Initialize PyTorch settings"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def check_video_constraints(video_path):
    """Check if video meets the required constraints"""
    cap = cv2.VideoCapture(video_path)
    
    # Check duration
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    cap.release()
    
    if duration > MAX_VIDEO_DURATION:
        raise ValueError(f"Video duration ({duration:.1f}s) exceeds maximum allowed duration ({MAX_VIDEO_DURATION}s)")
    
    # Check file size
    file_size = os.path.getsize(video_path)
    if file_size > MAX_FILE_SIZE:
        raise ValueError(f"File size ({file_size/1024/1024:.1f}MB) exceeds maximum allowed size (50MB)")
    
    return fps, total_frames

@st.cache_resource
def load_models():
    """Load YOLO and SAM models with error handling"""
    try:
        device = init_torch()
        
        # Ensure MODEL_PATH parent directory exists
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        if not MODEL_PATH.exists():
            if not download_with_progress(MODEL_URL, MODEL_PATH):
                raise RuntimeError("فشل تحميل نموذج SAM")
        
        # Load YOLO with error handling
        try:
            yolo_model = YOLO('yolov8n.pt')
        except Exception as e:
            raise RuntimeError(f"فشل تحميل نموذج YOLO: {str(e)}")
        
        # Load SAM with error handling
        try:
            model_type = "vit_b"
            sam = sam_model_registry[model_type](checkpoint=str(MODEL_PATH))
            sam.to(device=device)
            predictor = SamPredictor(sam)
        except Exception as e:
            raise RuntimeError(f"فشل تحميل نموذج SAM: {str(e)}")
        
        return yolo_model, predictor
    
    except Exception as e:
        st.error(f"خطأ في تحميل النماذج: {str(e)}")
        raise

def process_frame(frame, predictor, selected_objects, yolo_model):
    """Process a single frame with error handling"""
    try:
        results = yolo_model(frame)
        
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
        
        if np.any(mask):
            return cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
        return frame
    
    except Exception as e:
        st.error(f"خطأ في معالجة الإطار: {str(e)}")
        return frame

def process_video(video_path, selected_objects, yolo_model, predictor):
    """Process video with reduced frame rate"""
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame sampling
    frame_interval = max(1, int(original_fps / TARGET_FPS))
    processed_fps = original_fps / frame_interval
    
    # Create temporary output file
    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(temp_output, cv2.VideoWriter_fourcc(*'mp4v'), processed_fps, (width, height))
    
    progress_bar = st.progress(0)
    frame_text = st.empty()
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process only selected frames
            if frame_count % frame_interval == 0:
                processed_frame = process_frame(frame, predictor, selected_objects, yolo_model)
                out.write(processed_frame)
                
                # Update progress
                progress = (frame_count + 1) / total_frames
                progress_bar.progress(progress)
                frame_text.text(f"معالجة الإطار {frame_count + 1} من {total_frames}")
            
            frame_count += 1
            
    finally:
        cap.release()
        out.release()
    
    return temp_output

def main():
    st.title("تطبيق إزالة الكائنات من الفيديو")
    st.write("قم بتحميل فيديو واختر الكائنات المراد إزالتها")
    
    try:
        # Initialize models
        with st.spinner("جاري تحميل النماذج..."):
            yolo_model, predictor = load_models()
        
        video_file = st.file_uploader("اختر ملف فيديو", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if video_file:
            # Create temporary directory if it doesn't exist
            TEMP_DIR.mkdir(parents=True, exist_ok=True)
            
            # Save uploaded video
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=str(TEMP_DIR))
            temp_video.write(video_file.read())
            
            try:
                # Check video constraints
                fps, total_frames = check_video_constraints(temp_video.name)
                
                # Show original video
                st.video(temp_video.name)
                
                # Detect objects in first frame
                cap = cv2.VideoCapture(temp_video.name)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    results = yolo_model(frame)
                    detected_objects = []
                    for r in results:
                        for cls in r.boxes.cls:
                            obj_name = yolo_model.names[int(cls)]
                            if obj_name not in detected_objects:
                                detected_objects.append(obj_name)
                    
                    selected_objects = st.multiselect(
                        "اختر الكائنات المراد إزالتها",
                        options=detected_objects
                    )
                    
                    if selected_objects and st.button("معالجة الفيديو"):
                        with st.spinner("جاري معالجة الفيديو..."):
                            output_path = process_video(temp_video.name, selected_objects, yolo_model, predictor)
                            st.success("اكتملت المعالجة!")
                            st.video(output_path)
                            
                            # Cleanup
                            os.unlink(output_path)
            
            except ValueError as e:
                st.error(str(e))
            
            finally:
                # Cleanup temporary video file
                if os.path.exists(temp_video.name):
                    os.unlink(temp_video.name)
                
    except Exception as e:
        st.error(f"حدث خطأ: {str(e)}")

if __name__ == "__main__":
    # Set up event loop
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    
    main()