import streamlit as st
import numpy as np
from PIL import Image
import cv2

# ここを修正：一番シンプルで確実な読み込み方に戻します
import mediapipe as mp

st.title("顔認識カメラ：完成版")

# MediaPipeの機能を準備
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# カメラ入力
img_file = st.camera_input("自撮りして顔を認識させてください")

if img_file is not None:
    image = Image.open(img_file)
    img_array = np.array(image)

    # 顔検出を実行
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_det:
        # Mediapipeは画像を処理
        results = face_det.process(img_array)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(img_array, detection)
            st.success(f"{len(results.detections)} 人の顔を検出しました！")
        else:
            st.warning("顔が検出されませんでした。")

    st.image(img_array, use_container_width=True)
