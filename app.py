import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image

# AIモデルの読み込み
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

st.title("顔認識カメラ：完成版")
st.write("Python 3.11 & MediaPipe 正常動作中")

# カメラ入力
img_file = st.camera_input("自撮りして顔を認識させてください")

if img_file is not None:
    image = Image.open(img_file)
    img_array = np.array(image)

    # 顔検出の実行
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        # MediapipeはRGB画像を期待するので、そのまま処理
        results = face_detection.process(img_array)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(img_array, detection)
            st.success(f"{len(results.detections)} 人の顔を検出しました！")
        else:
            st.warning("顔が検出されませんでした。明るい場所で試してください。")

    # 結果を画面に表示
    st.image(img_array, use_container_width=True)
