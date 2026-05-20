import os
import cv2
import numpy as np
from pathlib import Path
from yolo_onnx import YOLOv8ONNX

def test_yolo_onnx():
    """
    載入 ONNX 格式的 YOLOv8 模型，
    並使用 CPU/ONNXRuntime 進行快速預測測試。
    """
    # 1. 路徑設定
    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / 'yolo_ver1.onnx'
    
    # 使用稍早建立的 simple_dataset 作為測試來源 (挑選 drowsy/yawning 為例)
    test_source_dir = script_dir.parent / 'simple_dataset' / 'drowsy' / 'yawning'
    
    # 預測結果輸出路徑
    output_project = script_dir / 'runs'
    output_name = 'predict_onnx_test'
    save_dir = output_project / output_name

    print("==================================================")
    print("      YOLOv8 模型 (ONNXRuntime) 測試腳本          ")
    print("==================================================")

    # 2. 檢查模型檔案是否存在
    if not model_path.exists():
        print(f"\n❌ 錯誤：找不到 ONNX 模型檔案 '{model_path}'")
        print("💡 請確認您已經匯出 `yolo_ver1.onnx` 並放在 `demo/` 資料夾底下！")
        return

    # 3. 載入模型
    print(f"\n⏳ 正在載入 ONNX 模型: {model_path.name}")
    try:
        model = YOLOv8ONNX(str(model_path), conf_thres=0.5, iou_thres=0.45)
    except Exception as e:
        print(f"載入失敗: {e}")
        return

    # 4. 準備測試圖片
    if not test_source_dir.exists():
        print(f"\n⚠️ 找不到測試目錄 '{test_source_dir}'")
        return

    # 抓取最多 5 張圖片進行測試
    images = list(test_source_dir.glob('*.jpg'))[:5]
    if not images:
        print(f"\n❌ 目錄 '{test_source_dir}' 內沒有圖片！")
        return
    print(f"\n📂 找到測試圖片，將選取 {len(images)} 張進行預測。")

    save_dir.mkdir(parents=True, exist_ok=True)

    # 5. 進行推論預測 (Inference)
    print("\n🚀 開始執行 ONNX 預測...")
    
    for img_path in images:
        # 使用 numpy + imdecode 來支援讀取含中文路徑的檔案 (解決 cv2.imread 失敗問題)
        img_array = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            continue
            
        results = model.infer(img)
        display_frame = model.draw_detections(img, results)
        
        save_file = save_dir / img_path.name
        # 同樣地，存檔也要用 imencode 來支援中文路徑
        is_success, buffer = cv2.imencode(".jpg", display_frame)
        if is_success:
            buffer.tofile(str(save_file))
        print(f"已儲存: {save_file}")

    print("\n✅ 測試完成！")
    print(f"📦 畫好辨識框的預測圖片已儲存於: {save_dir}")

if __name__ == '__main__':
    test_yolo_onnx()
