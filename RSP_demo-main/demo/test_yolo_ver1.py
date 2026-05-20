import os
import cv2
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ 尚未安裝 ultralytics！請先執行: pip install ultralytics")
    exit(1)

def test_yolo_gpu():
    """
    載入由 Colab 訓練好的 YOLOv8 模型 (yolo_ver1.pt)，
    並使用 GPU (若有) 進行快速預測測試。
    """
    # 1. 路徑設定
    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / 'yolo_ver1.pt'
    
    # 使用稍早建立的 simple_dataset 作為測試來源 (挑選 drowsy/yawning 為例)
    test_source_dir = script_dir.parent / 'simple_dataset' / 'drowsy' / 'yawning'
    
    # 預測結果輸出路徑
    output_project = script_dir / 'runs'
    output_name = 'predict_gpu_test'

    print("==================================================")
    print("      YOLOv8 模型 (Colab 訓練版) GPU 測試腳本     ")
    print("==================================================")

    # 2. 檢查模型檔案是否存在
    if not model_path.exists():
        print(f"\n❌ 錯誤：找不到模型檔案 '{model_path}'")
        print("💡 請確認您已經將 `yolo_ver1.pt` 放在 `demo/` 資料夾底下！")
        return

    # 3. 載入模型
    print(f"\n⏳ 正在載入模型: {model_path.name}")
    model = YOLO(str(model_path))

    # 4. 準備測試圖片
    if not test_source_dir.exists():
        print(f"\n⚠️ 找不到測試目錄 '{test_source_dir}'")
        print("將改用攝影機進行即時推論測試 (按 'q' 離開)...")
        source = 0  # 使用 Webcam
        is_webcam = True
    else:
        # 抓取最多 5 張圖片進行測試
        images = list(test_source_dir.glob('*.jpg'))[:5]
        if not images:
            print(f"\n❌ 目錄 '{test_source_dir}' 內沒有圖片！")
            return
        print(f"\n📂 找到測試圖片，將選取 {len(images)} 張進行預測。")
        source = [str(img) for img in images]
        is_webcam = False

    # 5. 進行推論預測 (Inference)
    print("\n🚀 開始執行預測...")
    # Ultralytics 自動偵測硬體：若有 GPU 則會優先使用 GPU；否則退回 CPU。
    results = model.predict(
        source=source,
        conf=0.5,             # 信心度門檻 (大於 0.5 才顯示)
        show=True,            # 測試時彈出視窗顯示結果
        save=not is_webcam,   # 若為圖片則儲存結果
        project=str(output_project),
        name=output_name,
        exist_ok=True         # 覆寫既有資料夾
    )

    if not is_webcam:
        save_path = output_project / output_name
        print("\n✅ 測試完成！")
        print(f"📦 畫好辨識框的預測圖片已儲存於: {save_path}")

if __name__ == '__main__':
    test_yolo_gpu()
