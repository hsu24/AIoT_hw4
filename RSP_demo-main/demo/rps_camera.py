"""
剪刀石頭布即時辨識程式（MediaPipe + YOLO + SVM 三模型支援）
支援動態切換模型。
透過攝影機即時辨識手勢為剪刀、石頭或布。

操作方式：
  - 按下 空白鍵 擷取畫面並進行辨識
  - 按下 m 鍵切換使用的模型 (MediaPipe -> YOLO -> SVM)
  - 按下 q 鍵離開程式

所需套件：pip install mediapipe ultralytics opencv-python numpy joblib scikit-learn
"""

import os
import cv2
import numpy as np
import joblib

# 嘗試載入 MediaPipe
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
except ImportError:
    mp_hands = None

# ============================================================
# 模型載入
# ============================================================

def load_mediapipe_model():
    """嘗試載入 MediaPipe 模型"""
    if mp_hands is None:
        print("  ⚠️ mediapipe 未安裝，無法使用 MediaPipe 模型")
        return None
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'rps_mp_model.pkl')
    
    if not os.path.exists(model_path):
        print(f"  ⚠️ 找不到 MediaPipe 模型檔案 '{model_path}'")
        return None
        
    print("⏳ 載入 MediaPipe 模型中...")
    model_data = joblib.load(model_path)
    print("✅ MediaPipe 模型載入成功！")
    return model_data


def load_yolo_model():
    """嘗試載入 YOLO 模型"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ⚠️ ultralytics 未安裝，無法使用 YOLO 模型")
        return None

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'rps_yolo_model.pt')

    if not os.path.exists(model_path):
        print(f"  ⚠️ 找不到 YOLO 模型檔案 '{model_path}'")
        return None

    print("⏳ 載入 YOLO 模型中...")
    model = YOLO(model_path)
    print("✅ YOLO 模型載入成功！")
    return model


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


def load_all_models():
    """載入所有可用的模型"""
    print("\n🔎 正在載入所有可用的模型...")
    models = {}
    
    mp_model = load_mediapipe_model()
    if mp_model:
        models['mediapipe'] = mp_model
        
    yolo_model = load_yolo_model()
    if yolo_model:
        models['yolo'] = yolo_model
        
    svm_model = load_svm_model()
    if svm_model:
        models['svm'] = svm_model
        
    return models

# ============================================================
# 推論
# ============================================================

def predict_mediapipe(model_data, roi_frame, hands_detector):
    """使用 MediaPipe + RF 模型預測手勢"""
    clf = model_data['model']
    class_names = model_data['class_names']
    emoji_map = {'rock': '🪨 Rock 石頭', 'paper': '📄 Paper 布', 'scissors': '✂️ Scissors 剪刀'}
    
    img_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)
    
    if not results.multi_hand_landmarks:
        return "No Hand Detected", roi_frame
        
    # 繪製骨架到 roi_frame
    annotated_roi = roi_frame.copy()
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_roi,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
        
    # 取第一隻手進行預測
    hand_landmarks = results.multi_hand_landmarks[0]
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z])
    coords = np.array(coords)
    
    base_coord = coords[0]
    coords = coords - base_coord
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords = coords / max_val
        
    features = coords.flatten()
    
    # 取得預測結果 (支援機率預測)
    if hasattr(clf, 'predict_proba'):
        probs = clf.predict_proba([features])[0]
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
    else:
        pred_idx = clf.predict([features])[0]
        confidence = 1.0
        
    pred_name = class_names[pred_idx]
    display_name = emoji_map.get(pred_name, pred_name)
    
    return f"{display_name} ({confidence * 100:.1f}%)", annotated_roi


def predict_yolo(model, roi_frame):
    """使用 YOLO 模型預測手勢"""
    emoji_map = {'rock': '🪨 Rock 石頭', 'paper': '📄 Paper 布', 'scissors': '✂️ Scissors 剪刀'}

    results = model.predict(roi_frame, verbose=False)
    if len(results) == 0:
        return "Unknown", roi_frame
        
    result = results[0]
    pred_idx = int(result.probs.top1)
    confidence = float(result.probs.top1conf)
    
    pred_name = model.names[pred_idx]
    display_name = emoji_map.get(pred_name, pred_name)
    return f"{display_name} ({confidence * 100:.1f}%)", roi_frame


def predict_svm(clf, roi_frame):
    """使用 SVM 模型預測手勢"""
    label_map = {0: 'Rock 🪨 石頭', 1: 'Paper 📄 布', 2: 'Scissors ✂️ 剪刀'}
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    flattened = resized.flatten() / 255.0
    processed = flattened.reshape(1, -1)
    prediction = clf.predict(processed)[0]
    return label_map.get(prediction, '未知'), roi_frame


def predict_gesture(model_type, model, roi_frame, hands_detector=None):
    """根據模型類型分發推論"""
    if model_type == 'mediapipe':
        return predict_mediapipe(model, roi_frame, hands_detector)
    elif model_type == 'yolo':
        return predict_yolo(model, roi_frame)
    else:
        return predict_svm(model, roi_frame)

# ============================================================
# UI 繪製
# ============================================================

def draw_ui(frame, result_text, roi_box, model_type):
    """在畫面上繪製介面元素"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi_box

    # 繪製 ROI 框線（綠色虛線風格）
    color = (255, 100, 100) if model_type == 'mediapipe' else (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # ROI 上方提示文字
    cv2.putText(frame, "Place hand here", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 底部半透明黑色背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 120), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # 模型類型標籤
    model_label = f"Model: {model_type.upper()}"
    cv2.putText(frame, model_label, (10, h - 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

    # 辨識結果文字
    cv2.putText(frame, f"Result: {result_text}", (10, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # 操作說明
    cv2.putText(frame, "[SPACE] Capture  |  [M] Switch Model  |  [Q] Quit", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


# ============================================================
# 攝影機開啟
# ============================================================

def open_camera():
    """嘗試多種方式開啟攝影機（解決 Windows 上無法以 index 開啟的問題）"""
    attempts = []
    for idx in range(5):
        attempts.append((idx, cv2.CAP_DSHOW))
    for idx in range(5):
        attempts.append((idx, cv2.CAP_MSMF))
    for idx in range(5):
        attempts.append((idx, cv2.CAP_ANY))

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
    # 載入所有模型
    available_models = load_all_models()
    if not available_models:
        print("❌ 錯誤：找不到任何可用模型，請先訓練模型。")
        return

    model_keys = list(available_models.keys())
    current_model_idx = 0
    
    # 初始化 MediaPipe Hands 偵測器 (針對即時預測)
    hands_detector = None
    if 'mediapipe' in model_keys:
        hands_detector = mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=1, 
            min_detection_confidence=0.5
        )

    # 開啟攝影機
    print("\n🔍 正在搜尋可用的攝影機...")
    cap = open_camera()
    if cap is None:
        print("❌ 錯誤：無法開啟任何攝影機。")
        return

    print(f"\n🎮 剪刀石頭布辨識系統啟動！")
    print(f"   目前支援的模型：{', '.join([k.upper() for k in model_keys])}")
    print("   按 [空白鍵] 擷取畫面並辨識")
    print("   按 [M] 鍵切換模型")
    print("   按 [Q] 鍵離開程式\n")

    result_text = "Waiting..."
    annotated_roi_cache = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 無法讀取攝影機畫面")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        roi_size = min(h, w) // 2
        roi_x1 = w // 2 - roi_size // 2
        roi_y1 = h // 2 - roi_size // 2
        roi_x2 = roi_x1 + roi_size
        roi_y2 = roi_y1 + roi_size
        roi_box = (roi_x1, roi_y1, roi_x2, roi_y2)

        # 替換 ROI 區域為有骨架的區域 (如果剛剛有辨識)
        if annotated_roi_cache is not None:
            frame[roi_y1:roi_y2, roi_x1:roi_x2] = annotated_roi_cache

        model_type = model_keys[current_model_idx]
        display_frame = draw_ui(frame.copy(), result_text, roi_box, model_type)
        cv2.imshow("Rock Paper Scissors - RSP Detector", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("👋 程式結束，再見！")
            break
        elif key == ord('m'):
            current_model_idx = (current_model_idx + 1) % len(model_keys)
            new_model = model_keys[current_model_idx]
            print(f"🔄 已切換至 {new_model.upper()} 模型")
            result_text = "Waiting..."
            annotated_roi_cache = None
        elif key == ord(' '):
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            model = available_models[model_type]
            
            # 進行預測
            result_text, new_roi = predict_gesture(model_type, model, roi.copy(), hands_detector)
            print(f"🔍 [{model_type.upper()}] 辨識結果: {result_text}")
            
            if model_type == 'mediapipe':
                annotated_roi_cache = new_roi
            else:
                annotated_roi_cache = None

    cap.release()
    cv2.destroyAllWindows()
    if hands_detector:
        hands_detector.close()

if __name__ == "__main__":
    main()
