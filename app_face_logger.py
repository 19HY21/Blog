
import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
import csv
import time
from datetime import datetime

# --- 設定 ---
KNOWN_DIR = "known"
SAVE_DIR = "detected_faces"
LOG_FILE = "recognition_log.csv"
os.makedirs(KNOWN_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# --- CSVログの初期化 ---
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "name", "confidence", "image_filename"])

# --- Streamlit UI ---
st.set_page_config(page_title="顔認識ログ付きアプリ", layout="wide")
st.title("🧠 顔認識ログ付きアプリ")
st.caption("顔画像をアップロードし、リアルタイムで認識・画像保存・CSV出力します。")

# 顔画像アップロード
uploaded_files = st.file_uploader("登録する顔画像（.jpg/.png）をアップロード", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
known_face_encodings = []
known_face_names = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(KNOWN_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

# 認識間隔
update_interval = st.number_input("🕒 映像更新の間隔（秒）", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

# 認識開始ボタン
if st.button("📸 顔認識を開始"):
    # 顔データの読み込み
    for filename in os.listdir(KNOWN_DIR):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(KNOWN_DIR, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])
                st.success(f"{filename} を登録しました")
            else:
                st.warning(f"{filename} に顔が見つかりませんでした")

    if not known_face_encodings:
        st.error("登録された顔がありません")
    else:
        video_capture = cv2.VideoCapture(0)
        frame_placeholder = st.empty()

        while True:
            ret, frame = video_capture.read()
            if not ret:
                st.error("❌ カメラの映像を取得できません")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = "Unknown"
                confidence = 0.0

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    confidence = 1 - face_distances[best_match_index]
                    if face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)[best_match_index]:
                        name = known_face_names[best_match_index]

                label = f"{name} ({confidence:.2f})"
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)

                # 顔画像の保存
                now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{name}_{now_str}.jpg"
                image_path = os.path.join(SAVE_DIR, filename)
                face_image = frame[top:bottom, left:right]
                cv2.imwrite(image_path, face_image)

                # ログへの書き込み
                with open(LOG_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, f"{confidence:.2f}", filename])

            # Streamlit上で表示
            frame_placeholder.image(frame, channels="BGR")

            # 更新間隔
            time.sleep(update_interval)

        video_capture.release()
        st.success("🎉 顔認識を終了しました")

# ログファイルの表示とダウンロード
st.markdown("### 📄 認識ログ（最新10件）")
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as f:
        rows = f.readlines()[-10:]
        for row in rows:
            st.text(row.strip())

    with open(LOG_FILE, "rb") as f:
        st.download_button("⬇️ 認識ログCSVをダウンロード", f, file_name="recognition_log.csv", mime="text/csv")
