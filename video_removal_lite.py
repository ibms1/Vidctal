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
import shutil

# إضافة تشخيص النماذج
def verify_yolo_model(model):
    """التحقق من تحميل نموذج YOLO بشكل صحيح"""
    try:
        # اختبار النموذج على صورة بسيطة
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        results = model(test_image)
        return True
    except Exception as e:
        st.error(f"خطأ في التحقق من نموذج YOLO: {str(e)}")
        return False

@st.cache_resource
def load_models():
    """تحميل النماذج مع تشخيص إضافي"""
    try:
        st.info("جاري تحميل النماذج...")
        
        # تحميل YOLO
        yolo_model = YOLO('yolov8n.pt')
        if not verify_yolo_model(yolo_model):
            raise RuntimeError("فشل التحقق من نموذج YOLO")
        st.success("تم تحميل نموذج YOLO بنجاح")
        
        # تحميل SAM
        model_type = "vit_b"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"استخدام جهاز: {device}")
        
        if not MODEL_PATH.exists():
            st.warning("نموذج SAM غير موجود، جاري التحميل...")
            if not download_with_progress(MODEL_URL, MODEL_PATH):
                raise RuntimeError("فشل تحميل نموذج SAM")
        
        sam = sam_model_registry[model_type](checkpoint=str(MODEL_PATH))
        sam.to(device=device)
        predictor = SamPredictor(sam)
        st.success("تم تحميل نموذج SAM بنجاح")
        
        return yolo_model, predictor
    
    except Exception as e:
        st.error(f"خطأ في تحميل النماذج: {str(e)}")
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        raise

def detect_objects_in_frame(frame, yolo_model):
    """تحسين الكشف عن الكائنات مع تشخيص إضافي"""
    try:
        st.info("جاري تحليل الإطار...")
        results = yolo_model(frame)
        detected_objects = []
        
        for r in results:
            for cls in r.boxes.cls:
                obj_name = yolo_model.names[int(cls)]
                if obj_name not in detected_objects:
                    detected_objects.append(obj_name)
        
        if not detected_objects:
            st.warning("لم يتم العثور على كائنات في الإطار")
        else:
            st.success(f"تم العثور على {len(detected_objects)} كائنات")
            
        return detected_objects
    
    except Exception as e:
        st.error(f"خطأ في الكشف عن الكائنات: {str(e)}")
        return []

def main():
    st.title("تطبيق إزالة الكائنات من الفيديو")
    st.write("قم بتحميل فيديو واختر الكائنات المراد إزالتها")
    
    try:
        # تحميل النماذج مع التشخيص
        yolo_model, predictor = load_models()
        
        video_file = st.file_uploader("اختر ملف فيديو", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if video_file:
            # حفظ الفيديو المرفوع
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_video.write(video_file.read())
            
            try:
                # عرض معلومات الفيديو
                cap = cv2.VideoCapture(temp_video.name)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps
                st.info(f"""
                معلومات الفيديو:
                - عدد الإطارات في الثانية: {fps:.2f}
                - عدد الإطارات الكلي: {total_frames}
                - مدة الفيديو: {duration:.2f} ثانية
                """)
                
                # قراءة الإطار الأول للكشف عن الكائنات
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    st.info("تم قراءة الإطار الأول بنجاح")
                    detected_objects = detect_objects_in_frame(frame, yolo_model)
                    
                    if detected_objects:
                        # عرض الفيديو الأصلي
                        st.video(temp_video.name)
                        
                        selected_objects = st.multiselect(
                            "اختر الكائنات المراد إزالتها",
                            options=detected_objects
                        )
                        
                        if selected_objects and st.button("معالجة الفيديو"):
                            with st.spinner("جاري معالجة الفيديو..."):
                                output_path = process_video(temp_video.name, selected_objects, yolo_model, predictor)
                                st.success("اكتملت المعالجة!")
                                st.video(output_path)
                                
                                # تنظيف الملفات المؤقتة
                                os.unlink(temp_video.name)
                                os.unlink(output_path)
                    else:
                        st.error("لم يتم العثور على أي كائنات في الفيديو")
                else:
                    st.error("فشل في قراءة الإطار الأول من الفيديو")
                    
            except Exception as e:
                st.error(f"خطأ في معالجة الفيديو: {str(e)}")
                if os.path.exists(temp_video.name):
                    os.unlink(temp_video.name)
            
    except Exception as e:
        st.error("حدث خطأ. الرجاء المحاولة مرة أخرى.")
        st.error(f"تفاصيل الخطأ: {str(e)}")

if __name__ == "__main__":
    main()