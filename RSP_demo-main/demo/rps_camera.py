"""
剪刀石頭布即時辨識程式
結合 test.py（SVM 模型推論）與 carema.py（攝影機擷取），
透過攝影機即時辨識手勢為剪刀、石頭或布。

操作方式：
  - 按下 空白鍵 擷取畫面並進行辨識
  - 按下 q 鍵離開程式
"""

import os
import cv2
import numpy as np
import joblib


def load_model(model_path='rps_svm_model.pkl'):
    """載入已訓練好的 SVM 模型"""
    # 嘗試在同目錄下找模型
    if not os.path.exists(model_path):
        # 嘗試用腳本所在位置尋找
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'rps_svm_model.pkl')

    if not os.path.exists(model_path):
        print(f"❌ 錯誤：找不到模型檔案 '{model_path}'，請確認是否已放入 demo 資料夾。")
        return None

    print("⏳ 載入模型中...")
    clf = joblib.load(model_path)
    print("✅ 模型載入成功！")
    return clf


def preprocess_frame(frame):
    """
    將攝影機擷取的畫面進行前處理（與訓練時一致）
    1. 轉灰階
    2. 縮放至 64x64
    3. 攤平為一維陣列
    4. 正規化 (除以 255)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    flattened = resized.flatten()
    normalized = flattened / 255.0
    return normalized.reshape(1, -1)


def predict_gesture(clf, processed_frame):
    """使用 SVM 模型預測手勢"""
    label_map = {0: 'Rock 🪨 石頭', 1: 'Paper 📄 布', 2: 'Scissors ✂️ 剪刀'}
    prediction = clf.predict(processed_frame)[0]
    return label_map.get(prediction, '未知')


def draw_ui(frame, result_text, roi_box):
    """在畫面上繪製介面元素"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi_box

    # 繪製 ROI 框線（綠色虛線風格）
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ROI 上方提示文字
    cv2.putText(frame, "Place hand here", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 底部半透明黑色背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 100), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # 辨識結果文字
    cv2.putText(frame, f"Result: {result_text}", (10, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # 操作說明
    cv2.putText(frame, "[SPACE] Capture  |  [Q] Quit", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return frame


def open_camera():
    """嘗試多種方式開啟攝影機（解決 Windows 上無法以 index 開啟的問題）"""
    # 依序嘗試的 (index, backend) 組合
    attempts = []
    for idx in range(5):
        attempts.append((idx, cv2.CAP_DSHOW))   # DirectShow (Windows 最穩定)
    for idx in range(5):
        attempts.append((idx, cv2.CAP_MSMF))    # Media Foundation
    for idx in range(5):
        attempts.append((idx, cv2.CAP_ANY))      # 自動選擇

    for idx, backend in attempts:
        backend_name = {cv2.CAP_DSHOW: 'DSHOW', cv2.CAP_MSMF: 'MSMF', cv2.CAP_ANY: 'ANY'}.get(backend, str(backend))
        print(f"  嘗試 index={idx}, backend={backend_name} ...", end=" ")
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print("✅ 成功！")
                return cap
            else:
                cap.release()
                print("❌ 可開啟但無法讀取畫面")
        else:
            print("❌ 無法開啟")

    return None


def main():
    # 載入模型
    clf = load_model()
    if clf is None:
        return

    # 開啟攝影機
    print("🔍 正在搜尋可用的攝影機...")
    cap = open_camera()
    if cap is None:
        print("❌ 錯誤：無法開啟任何攝影機，請確認攝影機是否連接。")
        print("   提示：可嘗試在「裝置管理員」中確認攝影機驅動是否正常。")
        return

    print("\n🎮 剪刀石頭布辨識系統啟動！")
    print("   按 [空白鍵] 擷取畫面並辨識")
    print("   按 [Q] 鍵離開程式\n")

    result_text = "Waiting..."

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 無法讀取攝影機畫面")
            break

        # 水平翻轉（鏡像），讓使用者操作更直覺
        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        # 定義 ROI 區域（畫面中央偏右的正方形區域）
        roi_size = min(h, w) // 2
        roi_x1 = w // 2 - roi_size // 2
        roi_y1 = h // 2 - roi_size // 2
        roi_x2 = roi_x1 + roi_size
        roi_y2 = roi_y1 + roi_size
        roi_box = (roi_x1, roi_y1, roi_x2, roi_y2)

        # 繪製 UI
        display_frame = draw_ui(frame.copy(), result_text, roi_box)
        cv2.imshow("Rock Paper Scissors - RSP Detector", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("👋 程式結束，再見！")
            break
        elif key == ord(' '):
            # 擷取 ROI 區域進行辨識
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            processed = preprocess_frame(roi)
            result_text = predict_gesture(clf, processed)
            print(f"🔍 辨識結果: {result_text}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
