import streamlit as st
import cv2
import numpy as np
import tempfile
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import os
from pathlib import Path

# Constants
MODEL_PATH = Path("models/sam_vit_h_4b8939.pth")
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

@st.cache_resource
def load_sam_model():
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not MODEL_PATH.exists():
        st.error("SAM model not found. Please download it first.")
        st.stop()
        
    sam = sam_model_registry[model_type](checkpoint=str(MODEL_PATH))
    sam.to(device=device)
    return sam

@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n.pt')

def get_frame_from_video(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def process_frame(frame, predictor, point_coords=None, point_labels=None):
    if point_coords is None or point_labels is None:
        return None
    
    # Get mask from SAM
    masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False
    )
    
    # Convert mask to uint8
    mask = masks[0].astype(np.uint8) * 255
    
    # Inpainting
    if np.any(mask):
        return cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
    return frame

def process_video(video_path, selected_frame, point_coords, point_labels, progress_text):
    # Load models
    sam = load_sam_model()
    predictor = SamPredictor(sam)
    
    # Read video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    temp_output = str(TEMP_DIR / "output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Changed from 'avc1' to 'mp4v'
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)
        
        # Process reference frame first
        ref_frame = get_frame_from_video(video_path, selected_frame)
        predictor.set_image(ref_frame)
        ref_mask = process_frame(ref_frame, predictor, point_coords, point_labels)
        
        # Process all frames
        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update progress
            progress = (frame_idx + 1) / frame_count
            progress_bar.progress(progress)
            progress_text.text(f"Processing frame {frame_idx + 1} of {frame_count}")
            
            # Process frame
            predictor.set_image(frame)
            processed_frame = process_frame(frame, predictor, point_coords, point_labels)
            if processed_frame is None:
                processed_frame = frame
                
            out.write(processed_frame)
            
    finally:
        cap.release()
        out.release()
        
    # Convert output to web-compatible format
    final_output = str(TEMP_DIR / "final_output.mp4")
    os.system(f"ffmpeg -i {temp_output} -vcodec libx264 {final_output}")
    
    return final_output

def main():
    st.title("Video Object Removal App")
    st.write("Upload a video and click on objects you want to remove")
    
    # Session state for coordinates
    if 'points' not in st.session_state:
        st.session_state.points = []
        st.session_state.labels = []
    
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])
    
    if uploaded_file is not None:
        # Save uploaded video
        video_path = str(TEMP_DIR / "input.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
            
        # Get total frames
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Frame selection slider
        frame_number = st.slider("Select frame for object selection", 0, total_frames-1, 0)
        
        # Display selected frame
        frame = get_frame_from_video(video_path, frame_number)
        if frame is not None:
            # Convert frame to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create clickable image
            clicked = st.image(frame_rgb, use_column_width=True)
            
            # Handle click events
            if st.button("Add Click Point"):
                # In a real implementation, you would get coordinates from mouse click
                # For now, we'll use center point as example
                h, w = frame.shape[:2]
                st.session_state.points.append([w//2, h//2])
                st.session_state.labels.append(1)  # 1 for foreground
                
            # Show selected points
            if st.session_state.points:
                st.write("Selected points:", st.session_state.points)
                
                if st.button("Clear Points"):
                    st.session_state.points = []
                    st.session_state.labels = []
                
                if st.button("Process Video"):
                    progress_text = st.empty()
                    with st.spinner("Processing..."):
                        point_coords = np.array(st.session_state.points)
                        point_labels = np.array(st.session_state.labels)
                        
                        output_path = process_video(
                            video_path,
                            frame_number,
                            point_coords,
                            point_labels,
                            progress_text
                        )
                        
                        st.success("Processing complete!")
                        st.video(output_path)

if __name__ == "__main__":
    main()