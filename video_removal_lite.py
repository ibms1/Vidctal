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
MODEL_PATH = Path("models/sam_vit_b_01ec64.pth")
TEMP_DIR = Path("temp")
MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
CHUNK_SIZE = 1024 * 1024
EXPECTED_SIZE = 375001389

# New constants for video constraints
MAX_VIDEO_DURATION = 3  # Maximum duration in seconds
TARGET_FPS = 5  # Reduced FPS for processing
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB maximum file size

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

def download_with_progress(url, destination, chunk_size=CHUNK_SIZE):
    """Download file with progress bar"""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination.with_suffix('.temp')
        
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
                    progress_text.text(f"تحميل نموذج SAM... {downloaded_size/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB")
        
        shutil.move(str(temp_path), str(destination))
        progress_text.text("اكتمل التحميل بنجاح!")
        return True
        
    except Exception as e:
        st.error(f"خطأ في التحميل: {str(e)}")
        if temp_path.exists():
            temp_path.unlink()
        return False

@st.cache_resource
def load_models():
    """Load YOLO and SAM models"""
    try:
        if not MODEL_PATH.exists():
            if not download_with_progress(MODEL_URL, MODEL_PATH):
                raise RuntimeError("فشل تحميل نموذج SAM")
        
        yolo_model = YOLO('yolov8n.pt')
        
        model_type = "vit_b"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        sam = sam_model_registry[model_type](checkpoint=str(MODEL_PATH))
        sam.to(device=device)
        predictor = SamPredictor(sam)
        
        return yolo_model, predictor
    
    except Exception as e:
        st.error(f"خطأ في تحميل النماذج: {str(e)}")
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        raise

def process_frame(frame, predictor, selected_objects, yolo_model):
    """Process a single frame"""
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

def process_video(video_path, selected_objects, yolo_model, predictor):
    """Process video with reduced frame rate"""
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame sampling
    frame_interval = max(1, int(original_fps / TARGET_FPS))
    processed_fps = original_fps / frame_interval
    
    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4').name
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
        yolo_model, predictor = load_models()
        
        video_file = st.file_uploader("اختر ملف فيديو", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if video_file:
            # Save uploaded video
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
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
                            os.unlink(temp_video.name)
                            os.unlink(output_path)
                            
            except ValueError as e:
                st.error(str(e))
                os.unlink(temp_video.name)
            
    except Exception as e:
        st.error("حدث خطأ. الرجاء المحاولة مرة أخرى.")
        st.error(f"تفاصيل الخطأ: {str(e)}")

if __name__ == "__main__":
    main()