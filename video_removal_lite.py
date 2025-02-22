import streamlit as st
import cv2
import numpy as np
import tempfile
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import os
from pathlib import Path
from PIL import Image
import time

# تعريف المسارات والثوابت
MODEL_PATH = Path("models/sam_vit_h_4b8939.pth")
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

@st.cache_resource
def load_models():
    # تحميل نموذج YOLO
    yolo_model = YOLO('yolov8n.pt')
    
    # تحميل نموذج SAM
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not MODEL_PATH.exists():
        st.error("SAM model not found. Please download it first.")
        st.stop()
        
    sam = sam_model_registry[model_type](checkpoint=str(MODEL_PATH))
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    return yolo_model, predictor

def extract_frames(video_path, frame_dir):
    """استخراج إطارات الفيديو وحفظها كصور"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    progress_bar = st.progress(0)
    frame_index = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_path = frame_dir / f"frame_{frame_index:04d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frames.append(frame_path)
        
        # تحديث شريط التقدم
        progress = (frame_index + 1) / frame_count
        progress_bar.progress(progress)
        frame_index += 1
    
    cap.release()
    return frames, fps

def detect_objects(frame_path, yolo_model):
    """اكتشاف الكائنات في الإطار باستخدام YOLO"""
    results = yolo_model(str(frame_path))
    objects = []
    
    for r in results:
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            if conf > 0.5:  # حد الثقة
                objects.append({
                    'box': box.cpu().numpy(),
                    'class': yolo_model.names[int(cls)],
                    'confidence': float(conf)
                })
    
    return objects

def process_frame(frame_path, selected_objects, predictor, yolo_model):
    """معالجة إطار واحد وإزالة الكائنات المحددة"""
    frame = cv2.imread(str(frame_path))
    predictor.set_image(frame)
    
    # اكتشاف الكائنات
    objects = detect_objects(frame_path, yolo_model)
    final_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # إنشاء قناع للكائنات المحددة
    for obj in objects:
        if obj['class'] in selected_objects:
            box = obj['box'].astype(int)
            masks, _, _ = predictor.predict(
                box=box,
                multimask_output=False
            )
            final_mask = np.logical_or(final_mask, masks[0]).astype(np.uint8) * 255
    
    # إزالة الكائنات المحددة
    if np.any(final_mask):
        processed_frame = cv2.inpaint(frame, final_mask, 3, cv2.INPAINT_TELEA)
        cv2.imwrite(str(frame_path), processed_frame)

def create_video(frame_paths, output_path, fps):
    """إنشاء فيديو من الإطارات المعالجة"""
    frame = cv2.imread(str(frame_paths[0]))
    height, width = frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        out.write(frame)
    
    out.release()
    
    # تحويل الفيديو إلى تنسيق متوافق مع الويب
    final_output = output_path.parent / "final_output.mp4"
    os.system(f"ffmpeg -i {output_path} -vcodec libx264 {final_output}")
    return final_output

def main():
    st.title("Video Object Removal App")
    st.write("Upload a video and select objects to remove")
    
    # تحميل النماذج
    yolo_model, predictor = load_models()
    
    uploaded_file = st.file_uploader("Upload video file", type=["mp4", "mov", "avi", "mkv"])
    
    if uploaded_file is not None:
        # إنشاء مجلد مؤقت للإطارات
        frames_dir = TEMP_DIR / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        # حفظ الفيديو المحمل
        video_path = TEMP_DIR / "input.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # عرض الفيديو الأصلي
        st.video(video_path)
        
        # استخراج الإطارات
        with st.spinner("Extracting frames..."):
            frames, fps = extract_frames(str(video_path), frames_dir)
        
        if frames:
            # اكتشاف الكائنات في الإطار الأول
            first_frame = frames[0]
            objects = detect_objects(first_frame, yolo_model)
            
            # عرض الإطار الأول مع الكائنات المكتشفة
            frame = cv2.imread(str(first_frame))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # إنشاء صورة مع إطارات حول الكائنات
            annotated_frame = frame_rgb.copy()
            for obj in objects:
                box = obj['box'].astype(int)
                cv2.rectangle(annotated_frame, 
                            (box[0], box[1]), 
                            (box[2], box[3]), 
                            (0, 255, 0), 2)
                cv2.putText(annotated_frame, 
                          f"{obj['class']} ({obj['confidence']:.2f})",
                          (box[0], box[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # عرض الإطار المشروح
            st.image(annotated_frame, caption="Detected Objects", use_column_width=True)
            
            # إنشاء قائمة بالكائنات المكتشفة
            detected_classes = list(set(obj['class'] for obj in objects))
            selected_objects = st.multiselect(
                "Select objects to remove",
                detected_classes
            )
            
            if selected_objects and st.button("Process Video"):
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                # معالجة كل إطار
                for i, frame_path in enumerate(frames):
                    process_frame(frame_path, selected_objects, predictor, yolo_model)
                    progress = (i + 1) / len(frames)
                    progress_bar.progress(progress)
                    progress_text.text(f"Processing frame {i + 1} of {len(frames)}")
                
                # إنشاء الفيديو النهائي
                with st.spinner("Creating final video..."):
                    output_path = TEMP_DIR / "output.mp4"
                    final_output = create_video(frames, output_path, fps)
                
                st.success("Processing complete!")
                st.video(str(final_output))
                
                # تنظيف الملفات المؤقتة
                for frame_path in frames:
                    os.remove(frame_path)
                frames_dir.rmdir()

if __name__ == "__main__":
    main()