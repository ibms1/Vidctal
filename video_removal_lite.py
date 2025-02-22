import streamlit as st
import cv2
import numpy as np
import tempfile
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import os
from pathlib import Path

# تعريف مسار النموذج
MODEL_PATH = Path("models/sam_vit_h_4b8939.pth")

@st.cache_resource
def load_sam_model():
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not MODEL_PATH.exists():
        st.error("نموذج SAM غير موجود. يرجى تحميله أولاً.")
        st.stop()
        
    sam = sam_model_registry[model_type](checkpoint=str(MODEL_PATH))
    sam.to(device=device)
    return sam

@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n.pt')

def process_video(video_path, selected_objects, progress_text):
    # تحميل النماذج
    yolo_model = load_yolo_model()
    sam = load_sam_model()
    predictor = SamPredictor(sam)
    
    # قراءة الفيديو
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # إنشاء ملف مؤقت للمخرجات
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
            
            # تحديث شريط التقدم
            progress = (frame_idx + 1) / frame_count
            progress_bar.progress(progress)
            progress_text.text(f"معالجة الإطار {frame_idx + 1} من {frame_count}")
            
            # YOLO كشف الكائنات باستخدام
            results = yolo_model(frame)
            
            # تحضير الإطار لـ SAM
            predictor.set_image(frame)
            
            # إنشاء قناع للكائنات المحددة
            final_mask = np.zeros((height, width), dtype=np.uint8)
            
            for r in results:
                for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                    if yolo_model.names[int(cls)] in selected_objects:
                        box = box.cpu().numpy().astype(int)
                        input_box = np.array([box[0], box[1], box[2], box[3]])
                        
                        masks, _, _ = predictor.predict(
                            box=input_box,
                            multimask_output=False
                        )
                        
                        final_mask = np.logical_or(final_mask, masks[0]).astype(np.uint8) * 255
            
            # معالجة الإطار
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
    st.title("تطبيق إزالة الكائنات من الفيديو")
    st.write("قم بتحميل فيديو وحدد الكائنات التي تريد إزالتها")
    
    uploaded_file = st.file_uploader("قم بتحميل ملف فيديو", type=["mp4", "mov", "avi", "mkv"])
    
    if uploaded_file is not None:
        # حفظ الفيديو المحمل
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        # عرض الفيديو الأصلي
        st.video(video_path)
        
        # تحميل نموذج YOLO وعرض الكائنات المتاحة
        yolo_model = load_yolo_model()
        object_names = yolo_model.names
        selected_objects = st.multiselect("حدد الكائنات المراد إزالتها", object_names)
        
        if st.button("معالجة الفيديو"):
            progress_text = st.empty()
            with st.spinner("جاري المعالجة..."):
                output_path = process_video(video_path, selected_objects, progress_text)
                st.success("اكتملت المعالجة!")
                st.video(output_path)
                
                # تنظيف الملفات المؤقتة
                os.unlink(tfile.name)
                os.unlink(output_path)

if __name__ == "__main__":
    main()