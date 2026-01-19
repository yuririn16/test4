import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image

st.title("Android顔認識カメラ")

# エラーを回避するための読み込み方
try:
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    st.error("Pythonのバージョンが3.13になっています。設定で3.11に変更してください。")
    st.stop()

img_file = st.camera_input("写真を撮る")
if img_file:
    image = Image.open(img_file)
    img_array = np.array(image)
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(img_array)
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(img_array, detection)
            st.success(f"{len(results.detections)}人の顔を検出！")
    st.image(img_array, use_container_width=True)
