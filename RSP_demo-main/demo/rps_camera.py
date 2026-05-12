"""
剪刀石頭布即時辨識程式（ONNX + SVM 雙模型支援）
優先使用 YOLOv8 ONNX 模型推論（不需要 ultralytics），
若找不到 ONNX 模型則退回 SVM。
透過攝影機即時辨識手勢為剪刀、石頭或布。

操作方式：
  - 按下 空白鍵 擷取畫面並進行辨識
  - 按下 q 鍵離開程式

所需套件：pip install onnxruntime opencv-python numpy joblib scikit-learn
"""

import os
import cv2
import numpy as np
import joblib


# YOLO 分類模型的類別名稱（與訓練時的資料夾順序一致）
CLASS_NAMES = {0: 'paper', 1: 'rock', 2: 'scissors'}


# ============================================================
# ONNX 推論工具
# ============================================================

def softmax(x):
    """計算 softmax 機率"""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def preprocess_onnx(frame, imgsz=224):
    """將影像前處理為 ONNX 模型輸入格式 (1, 3, 224, 224)"""
    resized = cv2.resize(frame, (imgsz, imgsz))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


# ============================================================
# 模型載入
# ============================================================

def load_yolo_onnx():
    """嘗試載入 YOLO ONNX 模型"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  ⚠️ onnxruntime 未安裝，無法使用 YOLO 模型")
        print("     安裝方式：pip install onnxruntime")
        return None

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'rps_yolo_model.onnx')

    if not os.path.exists(model_path):
        print(f"  ⚠️ 找不到 ONNX 模型檔案 '{model_path}'")
        return None

    print("⏳ 載入 YOLO ONNX 模型中...")
    session = ort.InferenceSession(model_path)
    print("✅ YOLO ONNX 模型載入成功！")
    return session


def load_svm_model(model_path='rps_svm_model.pkl'):
    """載入已訓練好的 SVM 模型（備援）"""
    if not os.path.exists(model_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'rps_svm_model.pkl')

    if not os.path.exists(model_path):
        print(f"  ⚠️ 找不到 SVM 模型檔案 '{model_path}'")
        return None

    print("⏳ 載入 SVM 模型中...")
    clf = joblib.load(model_path)
    print("✅ SVM 模型載入成功！")
    return clf


def load_model():
    """優先載入 YOLO ONNX 模型，若失敗則退回 SVM"""
    print("🔎 正在尋找可用模型...")

    # 優先嘗試 YOLO ONNX
    onnx_session = load_yolo_onnx()
    if onnx_session is not None:
        return ('yolo', onnx_session)

    # 退回 SVM
    print("  → 嘗試載入 SVM 模型作為備援...")
    svm_model = load_svm_model()
    if svm_model is not None:
        return ('svm', svm_model)

    print("❌ 錯誤：找不到任何可用模型，請先訓練模型。")
    return None


# ============================================================
# 推論
# ============================================================

def predict_yolo(session, roi_frame):
    """使用 YOLO ONNX 模型預測手勢"""
    emoji_map = {'rock': '🪨 Rock 石頭', 'paper': '📄 Paper 布', 'scissors': '✂️ Scissors 剪刀'}

    input_tensor = preprocess_onnx(roi_frame)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor})[0][0]
    probs = softmax(output)
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    pred_name = CLASS_NAMES.get(pred_idx, str(pred_idx))
    display_name = emoji_map.get(pred_name, pred_name)
    return f"{display_name} ({confidence * 100:.1f}%)"


def predict_svm(clf, roi_frame):
    """使用 SVM 模型預測手勢"""
    label_map = {0: 'Rock 🪨 石頭', 1: 'Paper 📄 布', 2: 'Scissors ✂️ 剪刀'}
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    flattened = resized.flatten() / 255.0
    processed = flattened.reshape(1, -1)
    prediction = clf.predict(processed)[0]
    return label_map.get(prediction, '未知')


def predict_gesture(model_info, roi_frame):
    """根據模型類型分發推論"""
    model_type, model = model_info
    if model_type == 'yolo':
        return predict_yolo(model, roi_frame)
    else:
        return predict_svm(model, roi_frame)


# ============================================================
# UI 繪製
# ============================================================

def draw_ui(frame, result_text, roi_box, model_type='yolo'):
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
    cv2.rectangle(overlay, (0, h - 120), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # 模型類型標籤
    model_label = f"Model: {model_type.upper()}"
    cv2.putText(frame, model_label, (10, h - 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

    # 辨識結果文字
    cv2.putText(frame, f"Result: {result_text}", (10, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # 操作說明
    cv2.putText(frame, "[SPACE] Capture  |  [Q] Quit", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return frame


# ============================================================
# 攝影機開啟
# ============================================================

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


# ============================================================
# 主程式
# ============================================================

def main():
    # 載入模型
    model_info = load_model()
    if model_info is None:
        return

    model_type, model = model_info

    # 開啟攝影機
    print("\n🔍 正在搜尋可用的攝影機...")
    cap = open_camera()
    if cap is None:
        print("❌ 錯誤：無法開啟任何攝影機，請確認攝影機是否連接。")
        print("   提示：可嘗試在「裝置管理員」中確認攝影機驅動是否正常。")
        return

    print(f"\n🎮 剪刀石頭布辨識系統啟動！（使用 {model_type.upper()} 模型）")
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
        display_frame = draw_ui(frame.copy(), result_text, roi_box, model_type)
        cv2.imshow("Rock Paper Scissors - RSP Detector", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("👋 程式結束，再見！")
            break
        elif key == ord(' '):
            # 擷取 ROI 區域進行辨識
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            result_text = predict_gesture(model_info, roi)
            print(f"🔍 辨識結果: {result_text}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
