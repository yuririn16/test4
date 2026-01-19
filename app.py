import streamlit as st
import numpy as np
from PIL import Image

# 普通の読み込みがダメな場合のための「直接指定」読み込み
try:
    import mediapipe as mp
    # solutionsを通さず、内部のモジュールを直接インポート
    from mediapipe.python.solutions import face_detection as mp_face_detection
    from mediapipe.python.solutions import drawing_utils as mp_drawing
except (AttributeError, ImportError):
    # もし上記がダメなら、別の内部パスを試す
    try:
        import mediapipe.python.solutions.face_detection as mp_face_detection
        import mediapipe.python.solutions.drawing_utils as mp_drawing
    except:
        st.error("MediaPipeの内部構造にアクセスできません。")
        st.stop()

st.title("顔認識カメラ：最終解決ver")

# カメラ入力
img_file = st.camera_input("自撮りしてください")

if img_file is not None:
    image = Image.open(img_file)
    img_array = np.array(image)

    # 検出器を直接クラスから呼び出す
    with mp_face_detection.FaceDetection(
        model_selection=0, 
        min_detection_confidence=0.5
    ) as face_det:
        results = face_det.process(img_array)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(img_array, detection)
            st.success(f"{len(results.detections)} 人の顔を検出しました！")
        else:
            st.warning("顔が検出されませんでした。")

    st.image(img_array, use_container_width=True)
