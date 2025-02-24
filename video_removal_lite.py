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

# Initialize asyncio event loop for Windows
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def init_torch():
    """Initialize PyTorch settings"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

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
                # Process video
                process_video(temp_video.name, predictor, yolo_model)
            finally:
                # Cleanup
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