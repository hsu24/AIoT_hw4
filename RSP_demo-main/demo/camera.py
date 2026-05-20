import os
import cv2
import time
from pathlib import Path
from yolo_onnx import YOLOv8ONNX

def main():
    # ---- 設定 ----
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    model_path = script_dir / 'yolo_ver1.onnx'

    # 檢查模型是否存在
    if not model_path.exists():
        print(f"❌ 找不到模型: {model_path}")
        print("   請確認是否已順利轉出 yolo_ver1.onnx 並放在 demo/ 資料夾")
        return

    # 載入自建的 ONNX 推論模組
    print("⏳ 載入 YOLOv8 ONNX 模型中...")
    try:
        model = YOLOv8ONNX(str(model_path), conf_thres=0.5, iou_thres=0.45)
    except Exception as e:
        print(f"載入 ONNX 模型失敗: {e}")
        return

    # 疲勞警告設定
    DROWSY_THRESHOLD = 5  # 連續偵測到 drowsy 的秒數閾值
    drowsy_start_time = None
    is_alarm = False

    # 開啟攝影機
    print("🎥 開啟攝影機中...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 無法開啟攝影機！請檢查設備連線。")
        return

    print("✅ 啟動成功！按 'q' 結束。")

    # 用來計算 FPS
    prev_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 無法讀取攝影機畫面。")
            break

        # 1. 進行預測 (ONNX 推論與後處理)
        results = model.infer(frame)
        
        # 2. 判斷是否偵測到 drowsy
        current_drowsy = False
        for res in results:
            if res['class_name'] == 'drowsy_face':
                current_drowsy = True
                break

        # 判斷疲勞邏輯
        if current_drowsy:
            if drowsy_start_time is None:
                drowsy_start_time = time.time()
            elif time.time() - drowsy_start_time > DROWSY_THRESHOLD:
                is_alarm = True
        else:
            drowsy_start_time = None
            is_alarm = False

        # 3. 畫出預測框
        display_frame = model.draw_detections(frame, results)

        # 加上警告標語
        if is_alarm:
            cv2.putText(display_frame, "WARNING: DROWSY!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            # 在畫面四周畫紅色警告框
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (0, 0, 255), 15)

        # 4. 計算並顯示 FPS
        current_time = time.time()
        if current_time - prev_time > 0:
            fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(
            display_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        # 5. 顯示影像
        cv2.imshow("Drowsy Detection (ONNX Runtime)", display_frame)

        # 按 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()
    print("👋 程式結束。")

if __name__ == "__main__":
    main()