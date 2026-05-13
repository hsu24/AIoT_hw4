"""
剪刀石頭布即時辨識程式（全 ONNX Runtime 三模型支援）
支援動態切換模型：MediaPipe / YOLO / SVM。
使用 ONNX Runtime 大幅降低資源消耗，非常適合樹莓派等邊緣裝置。

操作方式：
  - 按下 空白鍵 擷取畫面並進行辨識
  - 按下 m 鍵切換使用的模型 (MediaPipe -> YOLO -> SVM)
  - 按下 q 鍵離開程式

所需套件：pip install onnxruntime mediapipe opencv-python numpy
"""

import os
import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("❌ 錯誤：缺少 onnxruntime 套件。")
    print("   請執行: pip install onnxruntime")
    exit(1)

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

def load_onnx_model(filename, model_name):
    """通用的 ONNX 模型載入函式"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, filename)
    
    if not os.path.exists(model_path):
        print(f"  ⚠️ 找不到 {model_name} 模型檔案 '{model_path}'")
        return None
        
    print(f"⏳ 載入 {model_name} ONNX 模型中...")
    session = ort.InferenceSession(model_path)
    print(f"✅ {model_name} 模型載入成功！")
    return session

def load_all_models():
    """載入所有可用的模型"""
    print("\n🔎 正在載入所有可用的模型...")
    models = {}
    
    if mp_hands is not None:
        mp_session = load_onnx_model('rps_mp_model.onnx', 'MediaPipe(RF)')
        if mp_session:
            models['mediapipe'] = mp_session
    else:
        print("  ⚠️ mediapipe 未安裝，無法使用 MediaPipe 模型")
        
    yolo_session = load_onnx_model('rps_yolo_model.onnx', 'YOLOv8')
    if yolo_session:
        models['yolo'] = yolo_session
        
    svm_session = load_onnx_model('rps_svm_model.onnx', 'SVM')
    if svm_session:
        models['svm'] = svm_session
        
    return models

# ============================================================
# 推論
# ============================================================

def predict_mediapipe(session, roi_frame, hands_detector):
    """使用 MediaPipe + RF (ONNX) 預測手勢"""
    class_names = ['paper', 'rock', 'scissors']
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
        
    # 取第一隻手進行特徵擷取
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
        
    features = coords.flatten().reshape(1, -1).astype(np.float32)
    
    # ONNX 推論
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: features})
    
    pred_idx = int(outputs[0][0])
    confidence = 1.0
    if len(outputs) > 1:
        probs = outputs[1][0]
        confidence = float(np.max(probs))
        
    pred_name = class_names[pred_idx]
    display_name = emoji_map.get(pred_name, pred_name)
    
    return f"{display_name} ({confidence * 100:.1f}%)", annotated_roi


def predict_yolo(session, roi_frame):
    """使用 YOLOv8-cls (ONNX) 預測手勢"""
    class_names = {0: 'paper', 1: 'rock', 2: 'scissors'}
    emoji_map = {'rock': '🪨 Rock 石頭', 'paper': '📄 Paper 布', 'scissors': '✂️ Scissors 剪刀'}

    # YOLOv8 分類模型前處理: resize 224x224, BGR->RGB, HWC->CHW, /255.0
    img = cv2.resize(roi_frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img})[0]
    
    # 解析結果
    pred_idx = int(np.argmax(outputs[0]))
    confidence = float(outputs[0][pred_idx])
    
    pred_name = class_names.get(pred_idx, "Unknown")
    display_name = emoji_map.get(pred_name, pred_name)
    return f"{display_name} ({confidence * 100:.1f}%)", roi_frame


def predict_svm(session, roi_frame):
    """使用 SVM (ONNX) 預測手勢"""
    label_map = {0: 'Rock 🪨 石頭', 1: 'Paper 📄 布', 2: 'Scissors ✂️ 剪刀'}
    
    # SVM 前處理
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    flattened = resized.flatten() / 255.0
    processed = flattened.reshape(1, -1).astype(np.float32)
    
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: processed})
    
    pred_idx = int(outputs[0][0])
    return label_map.get(pred_idx, '未知'), roi_frame


def predict_gesture(model_type, session, roi_frame, hands_detector=None):
    """根據模型類型分發推論"""
    if model_type == 'mediapipe':
        return predict_mediapipe(session, roi_frame, hands_detector)
    elif model_type == 'yolo':
        return predict_yolo(session, roi_frame)
    else:
        return predict_svm(session, roi_frame)

# ============================================================
# UI 繪製與攝影機
# ============================================================

def draw_ui(frame, result_text, roi_box, model_type):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi_box

    color = (255, 100, 100) if model_type == 'mediapipe' else (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, "Place hand here", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 120), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    model_label = f"Model: {model_type.upper()} (ONNX)"
    cv2.putText(frame, model_label, (10, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
    cv2.putText(frame, f"Result: {result_text}", (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, "[SPACE] Capture  |  [M] Switch Model  |  [Q] Quit", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


def open_camera():
    attempts = []
    for idx in range(5): attempts.append((idx, cv2.CAP_DSHOW))
    for idx in range(5): attempts.append((idx, cv2.CAP_MSMF))
    for idx in range(5): attempts.append((idx, cv2.CAP_ANY))

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
    available_models = load_all_models()
    if not available_models:
        print("❌ 錯誤：找不到任何 ONNX 模型。")
        print("   提示：請先執行 train/convert_to_onnx.py 轉換模型！")
        return

    model_keys = list(available_models.keys())
    current_model_idx = 0
    
    hands_detector = None
    if 'mediapipe' in model_keys:
        hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

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
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        roi_size = min(h, w) // 2
        roi_x1 = w // 2 - roi_size // 2
        roi_y1 = h // 2 - roi_size // 2
        roi_x2 = roi_x1 + roi_size
        roi_y2 = roi_y1 + roi_size
        roi_box = (roi_x1, roi_y1, roi_x2, roi_y2)

        if annotated_roi_cache is not None:
            frame[roi_y1:roi_y2, roi_x1:roi_x2] = annotated_roi_cache

        model_type = model_keys[current_model_idx]
        display_frame = draw_ui(frame.copy(), result_text, roi_box, model_type)
        cv2.imshow("Rock Paper Scissors - ONNX Engine", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("👋 程式結束，再見！")
            break
        elif key == ord('m'):
            current_model_idx = (current_model_idx + 1) % len(model_keys)
            new_model = model_keys[current_model_idx]
            print(f"🔄 已切換至 {new_model.upper()} (ONNX) 模型")
            result_text = "Waiting..."
            annotated_roi_cache = None
        elif key == ord(' '):
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            session = available_models[model_type]
            
            result_text, new_roi = predict_gesture(model_type, session, roi.copy(), hands_detector)
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

