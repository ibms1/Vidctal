import streamlit as st
import cv2
import numpy as np
import tempfile
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import os

@st.cache_resource
def load_sam_model():
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return sam

def process_video(video_path, selected_objects):
    # Load models
    yolo_model = YOLO('yolov8n.pt')
    sam = load_sam_model()
    predictor = SamPredictor(sam)
    
    # Read video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create temporary file for output
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)
        
        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update progress
            progress_bar.progress((frame_idx + 1) / frame_count)
            
            # YOLO detection
            results = yolo_model(frame)
            
            # Prepare frame for SAM
            predictor.set_image(frame)
            
            # Create mask for selected objects
            final_mask = np.zeros((height, width), dtype=np.uint8)
            
            for r in results:
                for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                    if yolo_model.names[int(cls)] in selected_objects:
                        # Convert box to input for SAM
                        box = box.cpu().numpy().astype(int)
                        input_box = np.array([box[0], box[1], box[2], box[3]])
                        
                        # Get SAM mask
                        masks, _, _ = predictor.predict(
                            box=input_box,
                            multimask_output=False
                        )
                        
                        # Combine masks
                        final_mask = np.logical_or(final_mask, masks[0]).astype(np.uint8) * 255
            
            # Inpainting with combined mask
            if np.any(final_mask):
                processed_frame = cv2.inpaint(frame, final_mask, 3, cv2.INPAINT_TELEA)
            else:
                processed_frame = frame
                
            out.write(processed_frame)
            
    finally:
        cap.release()
        out.release()
        
    return temp_output

def main():
    st.title("Advanced Video Object Removal App")
    
    # Download SAM model if not exists
    if not os.path.exists("sam_vit_h_4b8939.pth"):
        st.warning("Please download the SAM model first from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        return
    
    # ...existing code...

if __name__ == "__main__":
    main()
