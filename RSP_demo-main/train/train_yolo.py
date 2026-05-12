"""
YOLOv8 分類模型訓練腳本
使用 YOLOv8n-cls（Nano 分類版）訓練剪刀石頭布辨識模型

使用方式：
  cd RSP_demo-main
  python train/train_yolo.py
"""

import os
import shutil
from ultralytics import YOLO


def prepare_val_folder(dataset_dir):
    """
    將 validation/ 資料夾整理成 YOLO 分類格式的 val/ 資料夾。
    validation/ 中的檔案以 rock/paper/scissors 開頭命名，
    需要分到 val/rock/, val/paper/, val/scissors/ 子資料夾。
    """
    validation_dir = os.path.join(dataset_dir, 'validation')
    val_dir = os.path.join(dataset_dir, 'val')

    # 如果 val/ 已經存在且有子資料夾，就不需要重新建立
    if os.path.exists(val_dir):
        subdirs = [d for d in os.listdir(val_dir)
                   if os.path.isdir(os.path.join(val_dir, d))]
        if len(subdirs) >= 3:
            print("  ✅ val/ 資料夾已存在且結構正確")
            return True

    if not os.path.exists(validation_dir):
        print("  ⚠️ 找不到 validation/ 資料夾，將使用 test/ 作為驗證集")
        return False

    print("  📁 正在從 validation/ 建立 val/ 資料夾...")

    categories = ['rock', 'paper', 'scissors']
    for cat in categories:
        os.makedirs(os.path.join(val_dir, cat), exist_ok=True)

    # 根據檔名前綴分類
    for filename in os.listdir(validation_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        src_path = os.path.join(validation_dir, filename)
        if os.path.isdir(src_path):
            continue

        # 判斷類別（檔名以 rock, paper, scissors 開頭）
        fname_lower = filename.lower()
        target_cat = None
        for cat in categories:
            if fname_lower.startswith(cat):
                target_cat = cat
                break

        if target_cat:
            dst_path = os.path.join(val_dir, target_cat, filename)
            shutil.copy2(src_path, dst_path)
            print(f"    {filename} → val/{target_cat}/")
        else:
            print(f"    ⚠️ 無法判斷 {filename} 的類別，跳過")

    # 驗證結果
    for cat in categories:
        cat_dir = os.path.join(val_dir, cat)
        count = len(os.listdir(cat_dir))
        print(f"    val/{cat}: {count} 張圖片")

    return True


def main():
    # 動態取得專案根目錄（train_yolo.py 所在資料夾的上一層）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)

    # 資料夾路徑
    dataset_dir = os.path.join(base_dir, 'dataset')
    train_dir = os.path.join(dataset_dir, 'train')
    demo_dir = os.path.join(base_dir, 'demo')

    # 確認資料夾存在
    if not os.path.exists(train_dir):
        print(f"❌ 錯誤：找不到訓練資料集 '{train_dir}'")
        return

    # 整理驗證集資料夾
    print("=" * 50)
    print("📁 檢查資料集結構")
    print("=" * 50)
    prepare_val_folder(dataset_dir)

    # 顯示資料集統計
    print("\n" + "=" * 50)
    print("📊 資料集統計")
    print("=" * 50)
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        if os.path.exists(split_dir):
            total = 0
            for cls_name in os.listdir(split_dir):
                cls_path = os.path.join(split_dir, cls_name)
                if os.path.isdir(cls_path):
                    count = len([f for f in os.listdir(cls_path)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    total += count
            print(f"  {split}: {total} 張圖片")
    print()

    # ========== 訓練 ==========
    print("=" * 50)
    print("🚀 開始訓練 YOLOv8n-cls 分類模型")
    print("   （使用 CPU，預計需要 10~30 分鐘）")
    print("=" * 50)

    # 載入預訓練模型（會自動下載）
    model = YOLO('yolov8n-cls.pt')

    # 訓練
    # data 參數指向包含 train/ 和 val/ 子資料夾的父目錄
    results = model.train(
        data=dataset_dir,
        epochs=50,
        imgsz=224,
        batch=32,
        device='cpu',
        workers=0,        # Windows 上建議設為 0 以避免多進程問題
        patience=10,      # 若 10 個 epoch 沒改善就提前停止
        optimizer='Adam',
        lr0=0.001,
        project=os.path.join(base_dir, 'runs'),
        name='rps_yolo_cls',
        exist_ok=True,
        verbose=True,
    )

    # ========== 評估 ==========
    print("\n" + "=" * 50)
    print("📋 模型評估結果")
    print("=" * 50)

    # 在測試集上驗證
    test_dir = os.path.join(dataset_dir, 'test')
    if os.path.exists(test_dir):
        metrics = model.val(data=dataset_dir, split='test')
        print(f"\n🎯 測試集 Top-1 準確率: {metrics.top1 * 100:.2f}%")
        print(f"🎯 測試集 Top-5 準確率: {metrics.top5 * 100:.2f}%")

    # ========== 匯出最佳模型到 demo 資料夾 ==========
    # 找到最佳模型路徑
    best_model_path = os.path.join(base_dir, 'runs', 'rps_yolo_cls', 'weights', 'best.pt')
    if not os.path.exists(best_model_path):
        # 嘗試 last.pt
        best_model_path = os.path.join(base_dir, 'runs', 'rps_yolo_cls', 'weights', 'last.pt')

    if os.path.exists(best_model_path):
        os.makedirs(demo_dir, exist_ok=True)
        dest_path = os.path.join(demo_dir, 'rps_yolo_model.pt')
        shutil.copy2(best_model_path, dest_path)
        print(f"\n✅ 最佳模型已複製到: {dest_path}")
        print(f"   模型大小: {os.path.getsize(dest_path) / 1024 / 1024:.1f} MB")
    else:
        print(f"\n⚠️ 找不到訓練後的模型檔案，請檢查 runs/ 資料夾")

    print("\n🎉 訓練完成！可以執行以下指令測試：")
    print("   python demo/test_yolo.py")
    print("   python demo/rps_camera.py")


if __name__ == "__main__":
    main()
