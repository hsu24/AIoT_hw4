"""
YOLOv8 疲勞駕駛即時偵測 Demo
==============================
使用訓練好的 YOLOv8 模型，透過攝影機即時偵測疲勞狀態。

使用前:
1. 安裝套件: pip install ultralytics opencv-python
2. 將訓練好的 yolo_ver1.pt 模型放到此資料夾 (demo/)
3. 執行: python camera.py

按 'q' 結束程式
"""

import os
import cv2
import time
from pathlib import Path

def main():
    # ---- 設定 ----
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    model_path = script_dir / 'yolo_ver1.pt'

    # 檢查模型是否存在
    if not model_path.exists():
        print(f"❌ 找不到模型: {model_path}")
        print("   請將訓練好的 yolo_ver1.pt 放到 demo/ 資料夾")
        return

    # 載入 YOLOv8 模型
    print("⏳ 載入 YOLOv8 模型中...")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ 請先安裝 ultralytics: pip install ultralytics")
        return

    model = YOLO(str(model_path))
    print("✅ 模型載入成功!")

    # 類別名稱與顏色
    CLASS_NAMES = ['drowsy_face', 'normal_face']
    CLASS_COLORS = {
        'drowsy_face': (0, 0, 255),    # 紅色 (BGR)
        'normal_face': (0, 255, 0),    # 綠色 (BGR)
    }

    # 疲勞警告設定
    DROWSY_THRESHOLD = 5  # 連續偵測到 drowsy 的秒數閾值
    drowsy_start_time = None
    is_alarm = False

    # 開啟攝影機
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 無法開啟攝影機")
        return

    print("\n🎥 攝影機已開啟！按 'q' 結束")
    print("=" * 40)

    # FPS 計算
    prev_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 預測
        results = model.predict(frame, conf=0.40, verbose=False)
        result = results[0]

        # 判斷是否偵測到 drowsy
        current_drowsy = False
        current_status = "正常 ✅"

        for box in result.boxes:
            # 取得座標
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = CLASS_NAMES[cls_id]
            color = CLASS_COLORS.get(cls_name, (255, 255, 255))

            # 繪製偵測框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 繪製標籤
            label = f"{cls_name} {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if cls_name == 'drowsy_face':
                current_drowsy = True

        # 疲勞計時邏輯
        if current_drowsy:
            if drowsy_start_time is None:
                drowsy_start_time = time.time()

            elapsed = time.time() - drowsy_start_time
            current_status = f"⚠️ 疲勞偵測中 ({elapsed:.1f}s)"

            if elapsed >= DROWSY_THRESHOLD:
                is_alarm = True
                current_status = "🚨 警告：疲勞駕駛！"
        else:
            drowsy_start_time = None
            is_alarm = False
            current_status = "正常 ✅"

        # 計算 FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # 繪製狀態面板
        panel_h = 80
        cv2.rectangle(frame, (0, 0), (frame.shape[1], panel_h), (0, 0, 0), -1)

        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # 狀態
        status_color = (0, 0, 255) if is_alarm else ((0, 200, 255) if current_drowsy else (0, 255, 0))
        cv2.putText(frame, current_status, (10, 60),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        # 若觸發警報，整個畫面閃紅框
        if is_alarm:
            cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 255), 8)

        # 顯示畫面
        cv2.imshow("YOLOv8 Drowsy Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 程式結束")


if __name__ == '__main__':
    main()