import streamlit as st
import numpy as np
from PIL import Image

# MediaPipeを特殊な方法で読み込む
import mediapipe as mp
from mediapipe.python.solutions import face_detection as mp_face_detection
from mediapipe.python.solutions import drawing_utils as mp_drawing

st.title("顔認識カメラ：最終解決版")

# カメラ入力
img_file = st.camera_input("自撮りして顔を認識させてください")

if img_file is not None:
    image = Image.open(img_file)
    img_array = np.array(image)

    # 直接クラスを呼び出す
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_det:
        results = face_det.process(img_array)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(img_array, detection)
            st.success(f"{len(results.detections)} 人の顔を検出しました！")
        else:
            st.warning("顔が検出されませんでした。")

    st.image(img_array, use_container_width=True)
